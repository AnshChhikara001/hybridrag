"""Semantic chunking by percentile breakpoint.

Embed every sentence, measure the cosine distance between each consecutive pair, and cut
where that distance spikes -- the point at which the text stops being about one thing and
starts being about another. The threshold is a percentile of the distances *within the
document*, not a fixed number, so a uniformly technical page and a wide-ranging tutorial
both get a sensible number of cuts instead of the first getting none and the second getting
one per line.

Deliberately blind to headings. The loader knows exactly where every section starts, and
feeding that in would make this a variant of the structure-aware chunker rather than an
independent strategy -- and Phase 4 would then be comparing two spellings of the same idea.
Running it over raw document text is also the honest test of whether embeddings can find
structure that is really there, which is the interesting question.

No overlap, for the same reason the structure-aware chunker has none: a boundary chosen
because the topic changed is not a boundary worth blurring.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from hybridrag.chunking.base import Chunker
from hybridrag.chunking.segmentation import sentence_spans
from hybridrag.embedding import Embedder
from hybridrag.models import ChunkingStrategy, Document
from hybridrag.tokenization import TokenCounter


class SemanticChunker(Chunker):
    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.SEMANTIC

    def __init__(
        self,
        tokenizer: TokenCounter,
        embedder: Embedder,
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
        percentile: float = 95.0,
    ) -> None:
        super().__init__(tokenizer, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        if not 0.0 < percentile < 100.0:
            raise ValueError(f"percentile must be in (0, 100), got {percentile}")
        self.embedder = embedder
        self.percentile = percentile

    def _spans(self, document: Document) -> list[tuple[int, int]]:
        text = document.text
        units = [(s, e) for s, e in sentence_spans(text, 0, len(text)) if text[s:e].strip()]
        if len(units) < 2:
            return self._token_window_spans(text, 0, len(text))

        distances = self._distances(text, units)
        threshold = float(np.percentile(distances, self.percentile))

        spans: list[tuple[int, int]] = []
        for first, last in _groups(distances, threshold):
            spans.extend(self._enforce_budget(text, units, distances, first, last))
        # Budget enforcement subdivides depth-first, so restore document order for the
        # contiguous chunk_index the base class assigns.
        return sorted(spans)

    def _distances(self, text: str, units: list[tuple[int, int]]) -> NDArray[np.float32]:
        """Cosine distance between each consecutive pair of sentences.

        `Embedder` guarantees unit-length vectors, so the row-wise dot product *is* the
        cosine and no renormalisation is needed here.
        """
        vectors = self.embedder.embed_documents([text[s:e] for s, e in units])
        similarity: NDArray[np.float32] = np.einsum("ij,ij->i", vectors[:-1], vectors[1:])
        distances: NDArray[np.float32] = 1.0 - similarity
        return distances

    def _enforce_budget(
        self,
        text: str,
        units: list[tuple[int, int]],
        distances: NDArray[np.float32],
        first: int,
        last: int,
    ) -> list[tuple[int, int]]:
        """Subdivide an oversized topic until every piece fits the embedder's input limit.

        A topic can run longer than the budget, and a chunk over the budget is silently
        truncated at embed time -- so the budget wins. Where it has to cut, it cuts at the
        largest remaining distance inside the group: the next-best topic boundary, rather
        than an arbitrary one.

        Iterative rather than recursive on purpose. `argmax` can land at the edge of a
        group, so a pathological document -- and this corpus has a 10,611-token section --
        would recurse once per sentence and blow the stack.
        """
        spans: list[tuple[int, int]] = []
        stack = [(first, last)]
        while stack:
            lo, hi = stack.pop()
            start, end = units[lo][0], units[hi][1]
            if self.tokenizer.count(text[start:end]) <= self.max_tokens:
                spans.append((start, end))
            elif lo == hi:
                # One sentence over budget: no topic boundary left to cut on. This is
                # where an oversized code fence ends up, since fences are atomic units.
                spans.extend(self._token_window_spans(text, start, end))
            else:
                cut = lo + int(np.argmax(distances[lo:hi]))
                stack.extend(((lo, cut), (cut + 1, hi)))
        return spans


def _groups(distances: NDArray[np.float32], threshold: float) -> list[tuple[int, int]]:
    """Inclusive sentence-index ranges, cut wherever a distance exceeds the threshold.

    Strictly greater, so at the default 95th percentile only the top 5% of gaps become
    boundaries. `distances[i]` is the gap between sentence `i` and `i + 1`.
    """
    groups: list[tuple[int, int]] = []
    first = 0
    for index, distance in enumerate(distances):
        if distance > threshold:
            groups.append((first, index))
            first = index + 1
    groups.append((first, len(distances)))
    return groups
