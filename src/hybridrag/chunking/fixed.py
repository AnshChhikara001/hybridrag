"""Fixed-size chunking with overlap -- the baseline.

Deliberately structure-blind: one token window slid across the whole document, cutting
wherever it lands, including mid-sentence and mid-code-block. That is the point. It is the
control the other two strategies are measured against, so giving it any structural
awareness would blunt the comparison.
"""

from __future__ import annotations

from typing import ClassVar

from hybridrag.chunking.base import Chunker
from hybridrag.models import ChunkingStrategy, Document


class FixedChunker(Chunker):
    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.FIXED

    def _spans(self, document: Document) -> list[tuple[int, int]]:
        return self._token_window_spans(document.text, 0, len(document.text))
