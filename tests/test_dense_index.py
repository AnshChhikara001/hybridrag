"""Dense index tests.

Run against an in-memory Chroma with the deterministic TopicEmbedder, so they assert our
wiring -- that the right vectors reach Chroma and the right ids and similarities come back
-- rather than re-testing Chroma's vector search or a real model's judgement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hybridrag.embedding import Embedder
from hybridrag.indexing import DenseIndex
from hybridrag.models import Chunk, ChunkingStrategy


def make_chunk(chunk_id: str, text: str, relative_path: str = "guide.md") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        relative_path=relative_path,
        section_ids=["section"],
        heading_path=("Guide",),
        text=text,
        chunk_index=0,
        strategy=ChunkingStrategy.STRUCTURE,
        token_count=max(1, len(text.split())),
        char_count=len(text),
        start_char=0,
        end_char=len(text),
    )


@pytest.fixture
def index(topic_embedder: Embedder) -> DenseIndex:
    dense = DenseIndex.in_memory(topic_embedder)
    dense.add(
        [
            make_chunk("c1", "alpha content about the first topic"),
            make_chunk("c2", "beta content about the second topic"),
            make_chunk("c3", "gamma content about the third topic"),
        ]
    )
    return dense


class TestRetrieval:
    def test_the_matching_topic_ranks_first(self, index: DenseIndex) -> None:
        assert index.search("beta")[0][0] == "c2"

    def test_similarity_is_returned_not_distance(self, index: DenseIndex) -> None:
        """Chroma reports cosine distance; fusion needs larger-is-better, like BM25."""
        assert index.search("beta")[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_scores_are_descending(self, index: DenseIndex) -> None:
        scores = [score for _, score in index.search("beta", k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_k_limits_the_result_count(self, index: DenseIndex) -> None:
        assert len(index.search("alpha", k=2)) == 2

    def test_k_larger_than_the_collection_is_safe(self, index: DenseIndex) -> None:
        """Chroma errors if n_results exceeds what it holds."""
        assert len(index.search("alpha", k=50)) == 3

    def test_searching_an_empty_index_returns_nothing(self, topic_embedder: Embedder) -> None:
        assert DenseIndex.in_memory(topic_embedder).search("alpha") == []

    def test_non_positive_k_is_rejected(self, index: DenseIndex) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            index.search("alpha", k=0)


class TestIndexContents:
    def test_every_chunk_is_indexed(self, index: DenseIndex) -> None:
        assert len(index) == 3

    def test_chunk_ids_are_exposed_for_the_sync_check(self, index: DenseIndex) -> None:
        """The dense and sparse indexes are kept honest by comparing their id sets."""
        assert index.chunk_ids() == {"c1", "c2", "c3"}

    def test_adding_nothing_is_a_no_op(self, index: DenseIndex) -> None:
        index.add([])
        assert len(index) == 3

    def test_reindexing_the_same_chunk_does_not_duplicate_it(self, index: DenseIndex) -> None:
        """Re-running ingestion must be idempotent, not additive."""
        index.add([make_chunk("c1", "alpha content about the first topic")])
        assert len(index) == 3


class TestModes:
    def test_embedded_mode_persists_across_clients(
        self, topic_embedder: Embedder, tmp_path: Path
    ) -> None:
        """The single-container deployment reopens an index written by the seed script."""
        path = tmp_path / "chroma"
        DenseIndex.embedded(topic_embedder, path).add([make_chunk("c1", "alpha content")])
        assert DenseIndex.embedded(topic_embedder, path).chunk_ids() == {"c1"}

    def test_embedded_mode_creates_its_directory(
        self, topic_embedder: Embedder, tmp_path: Path
    ) -> None:
        DenseIndex.embedded(topic_embedder, tmp_path / "nested" / "chroma")
        assert (tmp_path / "nested" / "chroma").is_dir()

    def test_two_in_memory_indexes_are_isolated(self, topic_embedder: Embedder) -> None:
        """Regression: Chroma caches its in-process system by settings.

        Two EphemeralClients built with the same configuration share one database, so a
        second "fresh" index saw the first one's chunks. Isolation is restored by giving
        each in-memory index its own collection.
        """
        first = DenseIndex.in_memory(topic_embedder)
        first.add([make_chunk("c1", "alpha content")])
        assert DenseIndex.in_memory(topic_embedder).chunk_ids() == set()
        assert first.chunk_ids() == {"c1"}
