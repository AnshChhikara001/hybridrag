"""Chunker interface.

Subclasses decide one thing only: where the boundaries go, as character spans into
`Document.text`. Turning spans into validated `Chunk` objects -- identity, token and
character counts, section provenance, heading path -- happens once, here. That keeps the
three strategies genuinely comparable: they differ in boundary placement and nothing else,
which is the variable Phase 4 is trying to isolate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from hybridrag.models import Chunk, ChunkingStrategy, Document, Section
from hybridrag.tokenization import TokenCounter


class Chunker(ABC):
    """Base for every chunking strategy."""

    strategy: ClassVar[ChunkingStrategy]

    def __init__(
        self,
        tokenizer: TokenCounter,
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be in [0, max_tokens={max_tokens}); "
                "otherwise the window cannot advance."
            )
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @abstractmethod
    def _spans(self, document: Document) -> list[tuple[int, int]]:
        """Boundary placement for this strategy, as spans into `document.text`."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Produce chunks for one document."""
        chunks: list[Chunk] = []
        seen: set[tuple[int, int]] = set()

        for start, end in self._spans(document):
            if end <= start or (start, end) in seen:
                continue
            text = document.text[start:end]
            if not text.strip():
                continue
            seen.add((start, end))

            covered = _sections_overlapping(document.sections, start, end)
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=Chunk.make_id(document.doc_id, self.strategy, index),
                    doc_id=document.doc_id,
                    relative_path=document.relative_path,
                    section_ids=[section.section_id for section in covered],
                    heading_path=covered[0].heading_path if covered else (),
                    text=text,
                    chunk_index=index,
                    strategy=self.strategy,
                    token_count=max(1, self.tokenizer.count(text)),
                    char_count=len(text),
                    start_char=start,
                    end_char=end,
                    page=covered[0].page if covered else None,
                )
            )
        return chunks

    def _token_window_spans(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        """Slide a fixed token window across [start, end).

        Shared by the fixed strategy and used as the structure-aware chunker's last resort,
        so an unsplittable run of text is broken the same way in both.
        """
        offsets = self.tokenizer.offsets(text[start:end])
        if not offsets:
            return [(start, end)]

        step = self.max_tokens - self.overlap_tokens
        spans: list[tuple[int, int]] = []
        for first in range(0, len(offsets), step):
            window = offsets[first : first + self.max_tokens]
            if not window:
                break

            # Re-encoding a slice is not token-count preserving: without whole-document
            # context the tokenizer re-segments the slice edges, so a window of N tokens
            # can encode to N+1. The embedder would then silently truncate the tail out of
            # the index. Shrink until the text the embedder will actually receive fits.
            span_start = start + window[0][0]
            length = len(window)
            while (
                length > 1
                and self.tokenizer.count(text[span_start : start + window[length - 1][1]])
                > self.max_tokens
            ):
                length -= 1

            spans.append((span_start, start + window[length - 1][1]))
            if first + length >= len(offsets):
                break

        # Characters the tokenizer emits no token for -- zero-width Unicode such as the
        # U+FE0F variation selector that renders an emoji -- fall outside every token span
        # and would be unreachable. Absorb such a tail into the final chunk so document
        # coverage is exactly complete. Only extended when the tail is genuinely
        # token-free, so this can never push a chunk over its budget.
        if spans and spans[-1][1] < end and not self.tokenizer.count(text[spans[-1][1] : end]):
            spans[-1] = (spans[-1][0], end)
        return spans


def _sections_overlapping(sections: list[Section], start: int, end: int) -> list[Section]:
    """Sections a chunk touches, for citation provenance.

    Provenance only: retrieval relevance is decided by answer-span coverage, because
    section membership is derived from headings and would flatter the structure-aware
    chunker while admitting chunks that clip a section without carrying its content.
    """
    return [s for s in sections if s.start_char < end and s.end_char > start]
