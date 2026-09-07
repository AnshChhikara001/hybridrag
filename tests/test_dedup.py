"""Near-duplicate detection.

The load-bearing guarantees are that removal is deterministic and that ids never move.
Renumbering chunks to close the gaps would change every id after a removal, which silently
invalidates the embedding cache and every stored identifier -- a far more expensive bug
than the duplicates it would be tidying.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from hybridrag.indexing import DedupScope, deduplicate
from hybridrag.models import Chunk, ChunkingStrategy


class DirectionEmbedder:
    """Places each chunk on a unit vector chosen by its first word.

    Deterministic and readable in the test itself: two chunks starting with the same word
    are identical in vector space, and a "near" prefix sits at a controllable angle.
    """

    model_name = "direction-embedder"
    dimension = 8

    def _vector(self, text: str) -> NDArray[np.float32]:
        head = text.split()[0].lower() if text.split() else ""
        raw = np.zeros(self.dimension)
        if head == "alpha":
            raw[0] = 1.0
        elif head == "near":  # ~0.96 cosine to alpha
            raw[0], raw[1] = 1.0, 0.29
        elif head == "cousin":  # ~0.82 cosine to alpha
            raw[0], raw[1] = 1.0, 0.7
        else:
            # A distinct direction per unseen word. An earlier version sent every unknown
            # word to one shared vector, which made two unrelated chunks duplicates of each
            # other by construction and failed a test the code was passing correctly.
            seed = int(hashlib.sha1(head.encode()).hexdigest()[:8], 16)
            raw = np.random.default_rng(seed).normal(size=self.dimension)
        return (raw / np.linalg.norm(raw)).astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._vector(text)


def chunk(index: int, text: str, path: str = "guide.md") -> Chunk:
    return Chunk(
        chunk_id=Chunk.make_id(f"doc-{path}", ChunkingStrategy.STRUCTURE, index),
        doc_id=f"doc-{path}",
        relative_path=path,
        text=text,
        chunk_index=index,
        strategy=ChunkingStrategy.STRUCTURE,
        token_count=max(1, len(text.split())),
        char_count=len(text),
        start_char=0,
        end_char=len(text),
    )


@pytest.fixture
def embedder() -> DirectionEmbedder:
    return DirectionEmbedder()


class TestExactDuplicates:
    def test_an_identical_text_is_removed_and_the_first_kept(
        self, embedder: DirectionEmbedder
    ) -> None:
        chunks = [chunk(0, "alpha one"), chunk(1, "alpha one"), chunk(2, "beta two")]

        report = deduplicate(chunks, embedder)

        assert [c.chunk_index for c in report.kept] == [0, 2]
        assert report.removed[0].chunk_id == chunks[1].chunk_id
        assert report.removed[0].duplicate_of == chunks[0].chunk_id
        assert report.removed[0].exact

    def test_an_exact_match_is_found_without_consulting_the_vectors(
        self, embedder: DirectionEmbedder
    ) -> None:
        """The cheap stage first: most duplicates here are byte-identical."""
        report = deduplicate([chunk(0, "zzz same"), chunk(1, "zzz same")], embedder)

        assert report.exact_removals == 1
        assert report.removed[0].similarity == 1.0


class TestNearDuplicates:
    def test_a_chunk_above_the_threshold_is_removed(self, embedder: DirectionEmbedder) -> None:
        report = deduplicate(
            [chunk(0, "alpha one"), chunk(1, "near one")], embedder, threshold=0.95
        )

        assert [c.chunk_index for c in report.kept] == [0]
        assert not report.removed[0].exact
        assert report.removed[0].similarity == pytest.approx(0.96, abs=0.01)

    def test_a_chunk_below_the_threshold_survives(self, embedder: DirectionEmbedder) -> None:
        report = deduplicate(
            [chunk(0, "alpha one"), chunk(1, "cousin one")], embedder, threshold=0.95
        )

        assert len(report.kept) == 2

    def test_the_threshold_decides(self, embedder: DirectionEmbedder) -> None:
        chunks = [chunk(0, "alpha one"), chunk(1, "cousin one")]

        assert len(deduplicate(chunks, embedder, threshold=0.99).kept) == 2
        assert len(deduplicate(chunks, embedder, threshold=0.80).kept) == 1

    def test_a_removal_names_what_it_duplicated(self, embedder: DirectionEmbedder) -> None:
        """A removal that cannot be explained cannot be trusted or debugged."""
        chunks = [chunk(0, "alpha one"), chunk(1, "near one")]

        removed = deduplicate(chunks, embedder, threshold=0.95).removed[0]

        assert removed.duplicate_of == chunks[0].chunk_id
        assert removed.relative_path == "guide.md"


class TestScope:
    """Measured: corpus-wide removal deleted a golden question's only evidence."""

    def test_the_same_passage_on_another_page_is_kept(self, embedder: DirectionEmbedder) -> None:
        """Which page a passage sits on is part of its meaning and its citation.

        Removing the copy in `request-forms.md` because `request-files.md` carried a
        near-identical warning left one golden question with no evidence at all, and Tier 1
        recorded hybrid/structure Recall@5 falling 0.759 -> 0.724.
        """
        chunks = [chunk(0, "alpha one"), chunk(1, "alpha one", path="other.md")]

        report = deduplicate(chunks, embedder, threshold=0.95)

        assert len(report.kept) == 2
        assert report.removed == []

    def test_a_repeat_inside_one_document_is_still_removed(
        self, embedder: DirectionEmbedder
    ) -> None:
        """Include expansion emits the same example once per Python version; a reader
        gains nothing from three copies on one page."""
        chunks = [chunk(0, "alpha one"), chunk(1, "alpha one"), chunk(2, "beta two")]

        assert len(deduplicate(chunks, embedder).kept) == 2

    def test_corpus_scope_remains_available_for_comparison(
        self, embedder: DirectionEmbedder
    ) -> None:
        """Kept so the negative result stays reproducible rather than merely asserted."""
        chunks = [chunk(0, "alpha one"), chunk(1, "alpha one", path="other.md")]

        report = deduplicate(chunks, embedder, scope=DedupScope.CORPUS)

        assert len(report.kept) == 1
        assert report.removed[0].relative_path == "other.md"
        assert report.scope is DedupScope.CORPUS

    def test_the_summary_names_the_scope(self, embedder: DirectionEmbedder) -> None:
        summary = deduplicate([chunk(0, "alpha"), chunk(1, "alpha")], embedder).summary()

        assert "document scope" in summary


class TestInvariants:
    def test_ids_are_never_renumbered_to_close_a_gap(self, embedder: DirectionEmbedder) -> None:
        """Ids derive from chunk_index, so renumbering would change every later id."""
        chunks = [chunk(0, "alpha one"), chunk(1, "alpha one"), chunk(2, "beta two")]
        before = chunks[2].chunk_id

        report = deduplicate(chunks, embedder)

        assert report.kept[1].chunk_id == before
        assert report.kept[1].chunk_index == 2

    def test_corpus_order_is_preserved(self, embedder: DirectionEmbedder) -> None:
        chunks = [chunk(index, text) for index, text in enumerate(["beta a", "alpha b", "zeta c"])]

        kept = deduplicate(chunks, embedder).kept

        assert [c.text for c in kept] == ["beta a", "alpha b", "zeta c"]

    def test_deduplication_is_deterministic(self, embedder: DirectionEmbedder) -> None:
        chunks = [chunk(0, "alpha one"), chunk(1, "near one"), chunk(2, "alpha one")]

        first = deduplicate(chunks, embedder, threshold=0.95)
        second = deduplicate(chunks, embedder, threshold=0.95)

        assert [c.chunk_id for c in first.kept] == [c.chunk_id for c in second.kept]

    def test_nothing_in_means_nothing_out(self, embedder: DirectionEmbedder) -> None:
        assert deduplicate([], embedder).kept == []

    def test_an_impossible_threshold_is_refused(self, embedder: DirectionEmbedder) -> None:
        with pytest.raises(ValueError, match="threshold must be"):
            deduplicate([chunk(0, "alpha")], embedder, threshold=0.0)

    def test_the_summary_states_both_kinds(self, embedder: DirectionEmbedder) -> None:
        chunks = [chunk(0, "alpha one"), chunk(1, "alpha one"), chunk(2, "near one")]

        summary = deduplicate(chunks, embedder, threshold=0.95).summary()

        assert "1 exact" in summary
        assert "1 near-duplicate" in summary
