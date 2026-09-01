"""Structure-aware chunking.

Corpus measurements drive the two behaviours here. 139 sections (7.4%) are under 50 tokens
and would otherwise become chunks too thin to answer anything while still occupying a
retrieval slot; they are merged forward into their neighbours. 265 sections (14.2%) exceed
the budget and hold 50.5% of all corpus tokens; they are broken down.

The loader already splits at every heading, so an oversized section contains no headings to
recurse into. The descent is therefore paragraphs, then sentences, then a token window as a
last resort -- with code fences atomic throughout, and a bounded overflow allowance so a
single large code example stays whole rather than being cut in half.
"""

from __future__ import annotations

from typing import ClassVar

from hybridrag.chunking.base import Chunker
from hybridrag.chunking.segmentation import find_code_fences, paragraph_spans, sentence_spans
from hybridrag.models import ChunkingStrategy, Document
from hybridrag.tokenization import TokenCounter


class StructureChunker(Chunker):
    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.STRUCTURE

    def __init__(
        self,
        tokenizer: TokenCounter,
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        min_tokens: int = 50,
        fence_overflow: float = 1.5,
    ) -> None:
        super().__init__(tokenizer, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        if not 1.0 <= fence_overflow <= 3.0:
            raise ValueError(f"fence_overflow must be in [1.0, 3.0], got {fence_overflow}")
        self.min_tokens = min_tokens
        self.fence_overflow = fence_overflow

    def _spans(self, document: Document) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for start, end in self._merge_small_sections(document):
            spans.extend(self._fit(document.text, start, end))
        return spans

    def _merge_small_sections(self, document: Document) -> list[tuple[int, int]]:
        """Greedily absorb undersized sections into the following one.

        Merging forward keeps spans contiguous, which `Chunk` requires, and only fires
        while the accumulated span is still below `min_tokens` -- so two substantial
        sections are never fused just because they happen to fit.
        """
        if not document.sections:
            return [(0, len(document.text))]

        text = document.text
        merged: list[tuple[int, int]] = []
        start: int | None = None
        end = 0

        for section in document.sections:
            if start is None:
                start, end = section.start_char, section.end_char
                continue
            undersized = self.tokenizer.count(text[start:end]) < self.min_tokens
            if (
                undersized
                and self.tokenizer.count(text[start : section.end_char]) <= self.max_tokens
            ):
                end = section.end_char
                continue
            merged.append((start, end))
            start, end = section.start_char, section.end_char

        if start is not None:
            merged.append((start, end))
        return merged

    def _fit(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        """Break [start, end) down until every piece fits the budget."""
        if self.tokenizer.count(text[start:end]) <= self.max_tokens:
            return [(start, end)]

        units = paragraph_spans(text, start, end)
        if len(units) == 1:
            units = sentence_spans(text, start, end)
        if len(units) == 1:
            return self._emit(text, start, end)
        return self._pack(text, units)

    def _pack(self, text: str, units: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Greedily fill chunks with whole units, never straddling a unit boundary."""
        spans: list[tuple[int, int]] = []
        start: int | None = None
        end = 0

        for unit_start, unit_end in units:
            if start is None:
                start, end = unit_start, unit_end
                continue
            if self.tokenizer.count(text[start:unit_end]) <= self.max_tokens:
                end = unit_end
            else:
                spans.extend(self._emit(text, start, end))
                start, end = unit_start, unit_end

        if start is not None:
            spans.extend(self._emit(text, start, end))
        return spans

    def _emit(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        """Emit a span, breaking it down further if it is still oversized."""
        tokens = self.tokenizer.count(text[start:end])
        if tokens <= self.max_tokens:
            return [(start, end)]

        # A code example is worth more whole than budget-compliant, within a bound.
        if tokens <= self.max_tokens * self.fence_overflow and self._is_mostly_fence(
            text, start, end
        ):
            return [(start, end)]

        sentences = sentence_spans(text, start, end)
        if len(sentences) > 1:
            return self._pack(text, sentences)
        return self._token_window_spans(text, start, end)

    @staticmethod
    def _is_mostly_fence(text: str, start: int, end: int) -> bool:
        """Whether a span is dominated by a single fenced code block."""
        span = end - start
        if span <= 0:
            return False
        covered = max(
            (
                min(end, fence_end) - max(start, fence_start)
                for fence_start, fence_end in find_code_fences(text)
            ),
            default=0,
        )
        return covered / span >= 0.9
