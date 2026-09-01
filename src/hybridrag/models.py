"""Domain models for documents, sections and chunks.

The central design constraint here is that Phase 4 compares three chunking strategies
against each other. Chunk identifiers cannot be the unit of ground truth, because every
strategy produces different chunks -- a golden set pinned to chunk IDs could only ever
evaluate the one strategy it was built against. `Section` therefore carries a
strategy-independent `section_id`, and every `Chunk` records which sections it covers.
Retrieval metrics compare section IDs, so all three strategies are measured on one scale.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, model_validator


class SourceFormat(StrEnum):
    """Input formats the ingestion pipeline accepts."""

    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    PDF = "pdf"


class ChunkingStrategy(StrEnum):
    """The three strategies Phase 4 compares.

    FIXED     -- fixed token window with overlap (the baseline).
    STRUCTURE -- recursive split that respects section headings.
    SEMANTIC  -- split at topic boundaries found via sentence-embedding distance.
    """

    FIXED = "fixed"
    STRUCTURE = "structure"
    SEMANTIC = "semantic"


def _sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


class Section(BaseModel):
    """A heading-delimited region of a document.

    `section_id` is the unit of retrieval ground truth and must stay stable across
    chunking strategies, re-indexing runs and machines -- so it is derived purely from
    the document path and the heading breadcrumb, never from chunk boundaries or
    ordering.
    """

    section_id: str
    relative_path: str
    heading_path: tuple[str, ...] = Field(
        default=(), description="Heading breadcrumb, e.g. ('Tutorial', 'Query Parameters')."
    )
    level: NonNegativeInt = Field(default=0, description="Heading depth; 0 = document preamble.")
    text: str
    start_char: NonNegativeInt = Field(
        default=0, description="Inclusive offset of this section within `Document.text`."
    )
    end_char: NonNegativeInt = Field(
        default=0, description="Exclusive offset of this section within `Document.text`."
    )
    page: int | None = Field(
        default=None,
        description="1-indexed page, for PDFs. PDFs expose no reliable heading structure, "
        "so page number is the fallback anchor when heading_path is empty.",
    )

    @staticmethod
    def make_id(relative_path: str, heading_path: tuple[str, ...], discriminator: str = "") -> str:
        """Build the stable section identifier (decision D-Q1.2).

        Format: sha1(relative_path + "#" + " > ".join(heading_path))

        `discriminator` disambiguates sections that share a breadcrumb and would
        otherwise collide: PDF pages (which have no headings at all, so every page would
        hash identically) and documents that repeat a heading at the same depth. It is
        omitted from the hash when empty, so the common case keeps the simple form.
        """
        base = f"{relative_path}#{' > '.join(heading_path)}"
        return _sha1(f"{base}#{discriminator}" if discriminator else base)


class Document(BaseModel):
    """A source file normalised to plaintext, with its sections extracted."""

    doc_id: str
    relative_path: str
    source_format: SourceFormat
    title: str | None = None
    text: str
    sections: list[Section] = Field(default_factory=list)
    content_hash: str = Field(
        description="sha1 of the normalised text. Lets re-ingestion skip unchanged files "
        "and gives deduplication a cheap exact-match pre-filter before cosine similarity."
    )

    @staticmethod
    def make_id(relative_path: str) -> str:
        return _sha1(relative_path)


class Chunk(BaseModel):
    """A retrievable unit of text produced by one chunking strategy.

    Carries every metadata field the brief requires (source document, chunk index,
    section heading, chunking strategy, character count) plus `token_count`, because
    context-window budgeting is denominated in tokens rather than characters, and this
    corpus mixes prose with code -- where characters-per-token differs sharply.
    """

    chunk_id: str
    doc_id: str
    relative_path: str
    section_ids: list[str] = Field(
        default_factory=list,
        description="Sections this chunk covers. A list because a chunk may span several.",
    )
    heading_path: tuple[str, ...] = ()
    text: str
    chunk_index: NonNegativeInt
    strategy: ChunkingStrategy
    token_count: PositiveInt
    char_count: PositiveInt
    start_char: NonNegativeInt = Field(
        description="Inclusive offset of this chunk within `Document.text`."
    )
    end_char: NonNegativeInt = Field(
        description="Exclusive offset of this chunk within `Document.text`."
    )
    page: int | None = None

    @model_validator(mode="after")
    def _span_matches_text(self) -> Chunk:
        """A chunk is exactly a contiguous span of `Document.text`.

        Holding chunkers to this makes the offsets trustworthy as retrieval ground truth:
        a chunker that merges or splits incorrectly fails here rather than producing
        offsets that quietly disagree with the text they claim to describe.
        """
        span = self.end_char - self.start_char
        if span != self.char_count:
            raise ValueError(
                f"chunk {self.chunk_id}: span {self.start_char}..{self.end_char} covers "
                f"{span} characters but char_count is {self.char_count}"
            )
        return self

    def overlap_chars(self, start: int, end: int) -> int:
        """Characters shared with the span [start, end)."""
        return max(0, min(self.end_char, end) - max(self.start_char, start))

    def covers_span(self, start: int, end: int, min_ratio: float = 1.0) -> bool:
        """Whether this chunk covers enough of an answer span to count as a retrieval hit.

        This is the Phase 4 relevance predicate, defined here so every metric shares one
        definition. Span containment is strategy-independent -- unlike section membership,
        which is derived from headings and therefore flatters the structure-aware chunker
        and admits false hits when a chunk clips a section without carrying its content.

        `min_ratio` below 1.0 tolerates a span split across an overlap boundary.
        """
        length = end - start
        if length <= 0:
            return False
        return self.overlap_chars(start, end) / length >= min_ratio

    @staticmethod
    def make_id(doc_id: str, strategy: ChunkingStrategy, chunk_index: int) -> str:
        """Deterministic per (document, strategy, position).

        Stable across runs so re-indexing is idempotent, and strategy-scoped so the three
        strategies can coexist in one store without colliding.
        """
        return _sha1(f"{doc_id}:{strategy.value}:{chunk_index}")
