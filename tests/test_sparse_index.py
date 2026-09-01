"""Sparse index tests.

The behaviours pinned here are the ones rank fusion and evaluation depend on: identifier
queries must beat prose queries on the chunk that contains the identifier, non-matches must
not be ranked at all, and ordering must be identical on every run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybridrag.indexing import SparseIndex
from hybridrag.models import Chunk, ChunkingStrategy


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        relative_path="guide.md",
        section_ids=["section"],
        heading_path=("Guide",),
        text=text,
        chunk_index=0,
        strategy=ChunkingStrategy.FIXED,
        token_count=max(1, len(text.split())),
        char_count=len(text),
        start_char=0,
        end_char=len(text),
    )


@pytest.fixture
def index() -> SparseIndex:
    return SparseIndex.build(
        [
            make_chunk("c1", "Use response_model to filter the data you return."),
            make_chunk("c2", "The response of the model is returned to the client."),
            make_chunk("c3", "Deploy the application behind a reverse proxy."),
        ]
    )


class TestRetrieval:
    def test_exact_identifier_outranks_the_prose_decoy(self, index: SparseIndex) -> None:
        """The project's central claim, in miniature.

        c2 contains both "response" and "model" as ordinary words. Only c1 contains the
        identifier. A tokenizer that shredded response_model would rank these the same way.
        """
        assert index.search("response_model")[0][0] == "c1"

    def test_unrelated_chunks_are_not_ranked(self, index: SparseIndex) -> None:
        """Non-matches handed to rank fusion would vote on documents they never matched."""
        assert "c3" not in {cid for cid, _ in index.search("response_model")}

    def test_k_limits_the_result_count(self, index: SparseIndex) -> None:
        assert len(index.search("the", k=2)) <= 2

    def test_scores_are_descending(self, index: SparseIndex) -> None:
        scores = [score for _, score in index.search("response model", k=10)]
        assert scores == sorted(scores, reverse=True)

    def test_a_query_with_no_terms_returns_nothing(self, index: SparseIndex) -> None:
        assert index.search("!!!") == []

    def test_a_query_matching_nothing_returns_nothing(self, index: SparseIndex) -> None:
        assert index.search("kubernetes") == []

    def test_ordering_is_deterministic(self, index: SparseIndex) -> None:
        """Ties must break identically every run, or eval numbers move on their own."""
        assert index.search("the", k=10) == index.search("the", k=10)

    def test_non_positive_k_is_rejected(self, index: SparseIndex) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            index.search("anything", k=0)


class TestPersistence:
    def test_round_trip_preserves_results(self, index: SparseIndex, tmp_path: Path) -> None:
        path = tmp_path / "sparse.json"
        index.save(path)
        assert SparseIndex.load(path).search("response_model") == index.search("response_model")

    def test_the_file_is_plain_json_not_a_pickle(self, index: SparseIndex, tmp_path: Path) -> None:
        """Loading a pickle executes code in it; an index file must be inert data."""
        path = tmp_path / "sparse.json"
        index.save(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["chunk_ids"] == ["c1", "c2", "c3"]

    def test_a_future_format_version_fails_loudly(self, index: SparseIndex, tmp_path: Path) -> None:
        """A silently misread index would degrade retrieval with no error anywhere."""
        path = tmp_path / "sparse.json"
        index.save(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["format_version"] = 99
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="format version"):
            SparseIndex.load(path)

    def test_save_creates_missing_directories(self, index: SparseIndex, tmp_path: Path) -> None:
        index.save(tmp_path / "nested" / "deeper" / "sparse.json")
        assert (tmp_path / "nested" / "deeper" / "sparse.json").exists()


class TestEdges:
    def test_an_empty_index_searches_without_crashing(self) -> None:
        """BM25 divides by mean document length, which is undefined for an empty corpus."""
        assert SparseIndex.build([]).search("anything") == []

    def test_mismatched_inputs_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="correspond one to one"):
            SparseIndex(chunk_ids=["a", "b"], documents=[["term"]])
