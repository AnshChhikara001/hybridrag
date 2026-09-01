"""Tests for the identity guarantees the rest of the pipeline depends on."""

from hybridrag.models import Chunk, ChunkingStrategy, Document, Section


class TestSectionId:
    """`section_id` is the unit of retrieval ground truth (decision D-Q1.2)."""

    def test_is_deterministic(self) -> None:
        path, heading = "tutorial/query-params.md", ("Tutorial", "Query Parameters")
        assert Section.make_id(path, heading) == Section.make_id(path, heading)

    def test_differs_by_heading_path(self) -> None:
        path = "tutorial/query-params.md"
        assert Section.make_id(path, ("Tutorial",)) != Section.make_id(path, ("Advanced",))

    def test_differs_by_document(self) -> None:
        heading = ("Tutorial",)
        assert Section.make_id("a.md", heading) != Section.make_id("b.md", heading)

    def test_survives_chunking_strategy_changes(self) -> None:
        """The whole point: the same section keeps one ID no matter how it is chunked."""
        expected = Section.make_id("index.md", ("Features",))
        for _ in ChunkingStrategy:
            assert Section.make_id("index.md", ("Features",)) == expected


class TestChunkId:
    def test_is_deterministic_so_reindexing_is_idempotent(self) -> None:
        args = ("doc1", ChunkingStrategy.FIXED, 3)
        assert Chunk.make_id(*args) == Chunk.make_id(*args)

    def test_strategies_do_not_collide_in_a_shared_store(self) -> None:
        ids = {Chunk.make_id("doc1", s, 0) for s in ChunkingStrategy}
        assert len(ids) == len(ChunkingStrategy)

    def test_differs_by_position(self) -> None:
        assert Chunk.make_id("doc1", ChunkingStrategy.FIXED, 0) != Chunk.make_id(
            "doc1", ChunkingStrategy.FIXED, 1
        )


class TestDocument:
    def test_id_is_derived_from_path(self) -> None:
        assert Document.make_id("a/b.md") == Document.make_id("a/b.md")
        assert Document.make_id("a/b.md") != Document.make_id("a/c.md")


class TestChunkSpans:
    """The span predicate is the Phase 4 relevance definition, so it is pinned here."""

    @staticmethod
    def _chunk(start: int, end: int) -> Chunk:
        return Chunk(
            chunk_id="c1",
            doc_id="d1",
            relative_path="a.md",
            text="x" * (end - start),
            chunk_index=0,
            strategy=ChunkingStrategy.FIXED,
            token_count=1,
            char_count=end - start,
            start_char=start,
            end_char=end,
        )

    def test_span_fully_inside_chunk_is_a_hit(self) -> None:
        assert self._chunk(0, 100).covers_span(10, 20)

    def test_span_outside_chunk_is_not_a_hit(self) -> None:
        assert not self._chunk(0, 100).covers_span(150, 160)

    def test_partial_overlap_fails_strict_containment(self) -> None:
        """Half a span is not the answer; strict mode must reject it."""
        assert not self._chunk(0, 100).covers_span(90, 110)

    def test_partial_overlap_passes_a_relaxed_ratio(self) -> None:
        """Tolerates an answer split across a chunk overlap boundary."""
        assert self._chunk(0, 100).covers_span(90, 110, min_ratio=0.5)

    def test_adjacent_but_disjoint_is_not_a_hit(self) -> None:
        assert self._chunk(0, 100).overlap_chars(100, 120) == 0

    def test_empty_span_is_never_a_hit(self) -> None:
        assert not self._chunk(0, 100).covers_span(50, 50)


class TestChunkSpanValidation:
    def test_span_inconsistent_with_char_count_is_rejected(self) -> None:
        """Catches a chunker whose offsets disagree with the text it emitted."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="char_count"):
            Chunk(
                chunk_id="c1",
                doc_id="d1",
                relative_path="a.md",
                text="hello",
                chunk_index=0,
                strategy=ChunkingStrategy.FIXED,
                token_count=1,
                char_count=5,
                start_char=0,
                end_char=99,
            )
