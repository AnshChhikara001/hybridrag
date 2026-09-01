"""Plain-text loader.

Plain text carries no structure, so the whole file becomes a single section with an empty
heading path. That is honest rather than lossy: inventing headings from blank-line runs
would fabricate ground truth the source never asserted.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from hybridrag.loaders.base import DocumentLoader, SectionBlock, assemble_document
from hybridrag.models import Document, SourceFormat


class TextLoader(DocumentLoader):
    extensions: ClassVar[frozenset[str]] = frozenset({".txt"})
    source_format: ClassVar[SourceFormat] = SourceFormat.TEXT

    def load(self, path: Path, corpus_root: Path) -> Document:
        relative_path = self.relative_path_of(path, corpus_root)
        text = path.read_text(encoding="utf-8", errors="replace")
        return assemble_document(
            relative_path=relative_path,
            source_format=self.source_format,
            blocks=[SectionBlock(heading_path=(), level=0, text=text)],
            title=path.stem,
        )
