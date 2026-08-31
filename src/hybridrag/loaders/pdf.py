"""PDF loader.

PDFs expose no reliable heading structure, so page number is the anchor instead of a
heading breadcrumb: one section per page, `heading_path` empty, `page` populated. Section
identity therefore comes from the page discriminator rather than a breadcrumb -- without
it every page of a PDF would hash to the same `section_id`.

This makes PDF a deliberately second-class citizen: citations resolve to a page rather
than a section, and structure-aware chunking degrades to fixed-size. That is a real
limitation of the format, recorded rather than papered over.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from hybridrag.loaders.base import DocumentLoader, LoaderError, SectionBlock, assemble_document
from hybridrag.models import Document, SourceFormat


class PdfLoader(DocumentLoader):
    extensions: ClassVar[frozenset[str]] = frozenset({".pdf"})
    source_format: ClassVar[SourceFormat] = SourceFormat.PDF

    def load(self, path: Path, corpus_root: Path) -> Document:
        relative_path = self.relative_path_of(path, corpus_root)
        try:
            reader = PdfReader(str(path))
            blocks = [
                SectionBlock(heading_path=(), level=0, text=page.extract_text() or "", page=number)
                for number, page in enumerate(reader.pages, start=1)
            ]
        except (PyPdfError, OSError, ValueError) as exc:
            raise LoaderError(f"could not read PDF {relative_path}: {exc}") from exc

        title = None
        if reader.metadata is not None and reader.metadata.title:
            title = str(reader.metadata.title).strip() or None

        return assemble_document(
            relative_path=relative_path,
            source_format=self.source_format,
            blocks=blocks,
            title=title or path.stem,
        )
