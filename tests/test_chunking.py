"""Chunker tests.

The invariants asserted here are what make the Phase 4 strategy comparison trustworthy:
chunk offsets must address the document exactly, no content may be dropped, and each
strategy must actually place boundaries differently from the others.
"""

from __future__ import annotations

from itertools import combinations, pairwise

import pytest

from conftest import TopicEmbedder
from hybridrag.chunking import Chunker, FixedChunker, SemanticChunker, StructureChunker
from hybridrag.embedding import Embedder
from hybridrag.loaders.base import SectionBlock, assemble_document
from hybridrag.models import ChunkingStrategy, Document, SourceFormat
from hybridrag.tokenization import TokenCounter


def build_document(blocks: list[tuple[tuple[str, ...], str]]) -> Document:
    return assemble_document(
        relative_path="test.md",
        source_format=SourceFormat.MARKDOWN,
        blocks=[SectionBlock(heading_path=h, level=len(h), text=t) for h, t in blocks],
    )


@pytest.fixture
def document() -> Document:
    return build_document(
        [
            (("Guide",), " ".join(f"intro{i}" for i in range(120))),
            (("Guide", "Tiny"), "short"),
            (("Guide", "Tiny2"), "also short"),
            (("Guide", "Big"), " ".join(f"body{i}" for i in range(300))),
        ]
    )


def all_chunkers(tokenizer: TokenCounter) -> list[Chunker]:
    return [
        FixedChunker(tokenizer, max_tokens=50, overlap_tokens=10),
        StructureChunker(tokenizer, max_tokens=50, overlap_tokens=10, min_tokens=10),
        SemanticChunker(tokenizer, TopicEmbedder(), max_tokens=50),
    ]


class TestSharedInvariants:
    """Properties every strategy must satisfy, or Phase 4's metrics are meaningless."""

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_offsets_slice_back_to_chunk_text(
        self, document: Document, word_tokenizer: TokenCounter, index: int
    ) -> None:
        for chunk in all_chunkers(word_tokenizer)[index].chunk(document):
            assert document.text[chunk.start_char : chunk.end_char] == chunk.text

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_no_content_is_dropped(
        self, document: Document, word_tokenizer: TokenCounter, index: int
    ) -> None:
        """A gap between chunks is text that can never be retrieved."""
        covered = bytearray(len(document.text))
        for chunk in all_chunkers(word_tokenizer)[index].chunk(document):
            for position in range(chunk.start_char, chunk.end_char):
                covered[position] = 1
        missed = [
            position
            for position, flag in enumerate(covered)
            if not flag and not document.text[position].isspace()
        ]
        assert not missed, f"{len(missed)} non-whitespace characters unreachable"

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_chunk_ids_are_unique(
        self, document: Document, word_tokenizer: TokenCounter, index: int
    ) -> None:
        chunks = all_chunkers(word_tokenizer)[index].chunk(document)
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_chunk_indices_are_contiguous(
        self, document: Document, word_tokenizer: TokenCounter, index: int
    ) -> None:
        chunks = all_chunkers(word_tokenizer)[index].chunk(document)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_strategies_place_different_boundaries(
        self, document: Document, word_tokenizer: TokenCounter
    ) -> None:
        """If two strategies agree everywhere, the comparison measures nothing."""
        boundaries = [
            {(c.start_char, c.end_char) for c in chunker.chunk(document)}
            for chunker in all_chunkers(word_tokenizer)
        ]
        for left, right in combinations(boundaries, 2):
            assert left != right


class TestFixedChunker:
    def test_respects_the_token_budget(
        self, document: Document, word_tokenizer: TokenCounter
    ) -> None:
        chunker = FixedChunker(word_tokenizer, max_tokens=50, overlap_tokens=10)
        assert all(c.token_count <= 50 for c in chunker.chunk(document))

    def test_consecutive_chunks_overlap(
        self, document: Document, word_tokenizer: TokenCounter
    ) -> None:
        chunks = FixedChunker(word_tokenizer, max_tokens=50, overlap_tokens=10).chunk(document)
        assert chunks[1].start_char < chunks[0].end_char

    def test_zero_overlap_is_allowed(
        self, document: Document, word_tokenizer: TokenCounter
    ) -> None:
        chunks = FixedChunker(word_tokenizer, max_tokens=50, overlap_tokens=0).chunk(document)
        assert chunks[1].start_char >= chunks[0].end_char

    def test_strategy_is_tagged_on_every_chunk(
        self, document: Document, word_tokenizer: TokenCounter
    ) -> None:
        chunks = FixedChunker(word_tokenizer, max_tokens=50, overlap_tokens=10).chunk(document)
        assert all(c.strategy is ChunkingStrategy.FIXED for c in chunks)


class TestStructureChunker:
    def test_undersized_sections_are_merged(self, word_tokenizer: TokenCounter) -> None:
        """Two 1-2 token sections must not become two separate retrieval units."""
        doc = build_document([(("A",), "short"), (("B",), "also short"), (("C",), "third one")])
        chunks = StructureChunker(
            word_tokenizer, max_tokens=50, overlap_tokens=10, min_tokens=10
        ).chunk(doc)
        assert len(chunks) == 1
        assert len(chunks[0].section_ids) == 3

    def test_substantial_sections_are_not_fused(self, word_tokenizer: TokenCounter) -> None:
        """Merging is for undersized sections only, not anything that happens to fit."""
        body = " ".join(f"w{i}" for i in range(30))
        doc = build_document([(("A",), body), (("B",), body)])
        chunks = StructureChunker(
            word_tokenizer, max_tokens=200, overlap_tokens=10, min_tokens=10
        ).chunk(doc)
        assert len(chunks) == 2

    def test_oversized_section_is_split(self, word_tokenizer: TokenCounter) -> None:
        doc = build_document([(("A",), " ".join(f"w{i}" for i in range(300)))])
        chunks = StructureChunker(
            word_tokenizer, max_tokens=50, overlap_tokens=10, min_tokens=10
        ).chunk(doc)
        assert len(chunks) > 1
        assert all(c.token_count <= 50 for c in chunks)

    def test_splits_prefer_paragraph_boundaries(self, word_tokenizer: TokenCounter) -> None:
        para = " ".join(f"w{i}" for i in range(20))
        doc = build_document([(("A",), f"{para}\n\n{para}\n\n{para}")])
        chunks = StructureChunker(
            word_tokenizer, max_tokens=25, overlap_tokens=5, min_tokens=5
        ).chunk(doc)
        assert all(not c.text.startswith(" ") for c in chunks)
        assert len(chunks) == 3

    def test_code_fence_is_kept_whole_within_the_overflow_allowance(
        self, word_tokenizer: TokenCounter
    ) -> None:
        """Half a code example helps neither retrieval nor generation.

        Sized to land in the overflow window: ~62 tokens against a 50-token budget with a
        1.5x allowance (75). Bigger than the budget, small enough to be worth keeping whole.
        """
        fence = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(20)) + "\n```"
        doc = build_document([(("A",), f"Intro text here.\n\n{fence}")])
        chunks = StructureChunker(
            word_tokenizer, max_tokens=50, overlap_tokens=10, min_tokens=5, fence_overflow=1.5
        ).chunk(doc)
        holding_fence = [c for c in chunks if "```python" in c.text]
        assert len(holding_fence) == 1
        assert holding_fence[0].text.count("```") == 2, "fence was split across chunks"

    def test_fence_overflow_is_bounded(self, word_tokenizer: TokenCounter) -> None:
        """An enormous fence still gets split rather than blowing the budget open."""
        fence = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(400)) + "\n```"
        doc = build_document([(("A",), fence)])
        chunks = StructureChunker(
            word_tokenizer, max_tokens=50, overlap_tokens=10, min_tokens=5, fence_overflow=1.5
        ).chunk(doc)
        assert len(chunks) > 1


class TestConfigurationGuards:
    def test_overlap_at_or_above_budget_is_rejected(self, word_tokenizer: TokenCounter) -> None:
        with pytest.raises(ValueError, match="cannot advance"):
            FixedChunker(word_tokenizer, max_tokens=50, overlap_tokens=50)

    def test_fence_overflow_bounds_are_enforced(self, word_tokenizer: TokenCounter) -> None:
        with pytest.raises(ValueError, match="fence_overflow"):
            StructureChunker(word_tokenizer, fence_overflow=9.0)


class TestTokenBudgetIsEnforcedAfterRetokenisation:
    """A window of N tokens does not always re-tokenise to N tokens.

    Sub-word tokenizers merge and split differently when text is cut at a boundary, so a
    512-token window can re-encode to 513 -- and the embedder, whose limit is 512, would
    silently truncate the tail out of the index. Uses the real tokenizer because the
    behaviour is a property of real sub-word vocabularies.
    """

    def test_no_fixed_chunk_exceeds_the_budget(self) -> None:
        from hybridrag.tokenization import HuggingFaceTokenCounter

        tokenizer = HuggingFaceTokenCounter("BAAI/bge-small-en-v1.5")
        prose = " ".join(
            f"The `param_{i}` option configures behaviour, e.g. `app.get('/x{i}')`."
            for i in range(400)
        )
        document = build_document([(("Doc",), prose)])
        chunks = FixedChunker(tokenizer, max_tokens=128, overlap_tokens=16).chunk(document)
        offenders = [(c.chunk_index, c.token_count) for c in chunks if c.token_count > 128]
        assert not offenders, f"chunks exceed the embedder input limit: {offenders}"


class TestTokenFreeTail:
    def test_zero_width_trailing_character_is_still_reachable(self) -> None:
        """U+FE0F (emoji variation selector) produces no token, so a naive token-window
        walk leaves it outside every chunk. Found on the real corpus, in a one-line file
        ending '... and more. ✈️'."""
        from hybridrag.tokenization import HuggingFaceTokenCounter

        tokenizer = HuggingFaceTokenCounter("BAAI/bge-small-en-v1.5")
        document = build_document([(("R",), "Additional resources and more. ✈️")])
        chunks = FixedChunker(tokenizer, max_tokens=128, overlap_tokens=16).chunk(document)
        assert max(c.end_char for c in chunks) == len(document.text)


def sentences(topic: str, count: int) -> str:
    return " ".join(f"The {topic} value is {index}." for index in range(count))


class TestSemanticChunker:
    """Boundary placement must come from the embeddings, not from the document's layout."""

    def test_boundary_lands_at_the_topic_shift(
        self, word_tokenizer: TokenCounter, topic_embedder: Embedder
    ) -> None:
        document = build_document([(("Guide",), f"{sentences('alpha', 8)} {sentences('beta', 8)}")])
        chunker = SemanticChunker(word_tokenizer, topic_embedder, max_tokens=500, percentile=95.0)
        chunks = chunker.chunk(document)

        assert len(chunks) == 2
        assert "beta" not in chunks[0].text
        assert "alpha" not in chunks[1].text

    def test_headings_are_ignored(
        self, word_tokenizer: TokenCounter, topic_embedder: Embedder
    ) -> None:
        """A section break with no topic change must not produce a chunk boundary.

        This is what keeps the strategy independent of the structure-aware one: if section
        boundaries leaked in, Phase 4 would be comparing two spellings of the same idea.
        """
        document = build_document(
            [
                (("Guide", "Intro"), sentences("alpha", 6)),
                (("Guide", "Details"), f"{sentences('alpha', 6)} {sentences('beta', 6)}"),
            ]
        )
        chunker = SemanticChunker(word_tokenizer, topic_embedder, max_tokens=500, percentile=95.0)
        chunks = chunker.chunk(document)

        boundary = document.sections[1].start_char
        assert not any(c.start_char == boundary for c in chunks), "cut at the section heading"
        assert any(c.start_char < boundary < c.end_char for c in chunks), "no chunk spans it"

    def test_a_lower_percentile_cuts_more_often(
        self, word_tokenizer: TokenCounter, topic_embedder: Embedder
    ) -> None:
        """The knob has to do something, or the Phase 4 sweep is measuring noise."""
        text = " ".join(sentences(topic, 4) for topic in ("alpha", "beta", "gamma", "delta"))
        document = build_document([(("Guide",), text)])
        counts = [
            len(
                SemanticChunker(word_tokenizer, topic_embedder, max_tokens=500, percentile=p).chunk(
                    document
                )
            )
            for p in (95.0, 50.0)
        ]
        assert counts[1] > counts[0]

    def test_respects_the_token_budget(
        self, word_tokenizer: TokenCounter, topic_embedder: Embedder
    ) -> None:
        """A topic longer than the budget still has to be cut, or the embedder truncates it."""
        document = build_document([(("Guide",), sentences("alpha", 60))])
        chunker = SemanticChunker(word_tokenizer, topic_embedder, max_tokens=50)
        assert all(c.token_count <= 50 for c in chunker.chunk(document))

    def test_an_oversized_single_sentence_falls_back_to_a_token_window(
        self, word_tokenizer: TokenCounter, topic_embedder: Embedder
    ) -> None:
        """Where no topic boundary exists, the budget still wins."""
        document = build_document(
            [(("Guide",), " ".join(f"alpha{i}" for i in range(200)) + ". alpha end.")]
        )
        chunker = SemanticChunker(word_tokenizer, topic_embedder, max_tokens=50)
        chunks = chunker.chunk(document)
        assert len(chunks) > 1
        assert all(c.token_count <= 50 for c in chunks)

    def test_chunks_do_not_overlap(
        self, word_tokenizer: TokenCounter, topic_embedder: Embedder
    ) -> None:
        """A topic boundary is not a boundary worth blurring, so overlap defaults to zero."""
        document = build_document([(("Guide",), f"{sentences('alpha', 8)} {sentences('beta', 8)}")])
        chunks = SemanticChunker(word_tokenizer, topic_embedder, max_tokens=500).chunk(document)
        assert all(a.end_char <= b.start_char for a, b in pairwise(chunks))

    def test_a_single_sentence_document_yields_one_chunk(
        self, word_tokenizer: TokenCounter, topic_embedder: Embedder
    ) -> None:
        document = build_document([(("Guide",), "One alpha sentence.")])
        assert len(SemanticChunker(word_tokenizer, topic_embedder).chunk(document)) == 1

    def test_percentile_out_of_range_is_rejected(
        self, word_tokenizer: TokenCounter, topic_embedder: Embedder
    ) -> None:
        with pytest.raises(ValueError, match="percentile"):
            SemanticChunker(word_tokenizer, topic_embedder, percentile=100.0)
