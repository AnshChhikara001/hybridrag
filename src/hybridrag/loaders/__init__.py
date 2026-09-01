"""Format-dispatching corpus loading.

`CorpusLoader` owns the loader instances rather than exposing module-level singletons, so
ingestion statistics (such as unresolved include directives) belong to one run and cannot
leak between corpora or between tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from hybridrag.loaders.base import (
    DocumentLoader,
    LoaderError,
    UnsupportedFormatError,
    content_hash,
    normalise_text,
)
from hybridrag.loaders.html import HtmlLoader
from hybridrag.loaders.markdown import MarkdownLoader
from hybridrag.loaders.pdf import PdfLoader
from hybridrag.loaders.text import TextLoader
from hybridrag.models import Document

__all__ = [
    "CorpusLoader",
    "DocumentLoader",
    "HtmlLoader",
    "LoaderError",
    "MarkdownLoader",
    "PdfLoader",
    "TextLoader",
    "UnsupportedFormatError",
    "content_hash",
    "normalise_text",
]


class CorpusLoader:
    """Dispatches files to the right loader and walks a corpus directory."""

    def __init__(self, corpus_root: Path, *, expand_includes: bool = True) -> None:
        self.corpus_root = corpus_root.resolve()
        self._markdown = MarkdownLoader(expand_includes=expand_includes)
        loaders: tuple[DocumentLoader, ...] = (
            self._markdown,
            TextLoader(),
            HtmlLoader(),
            PdfLoader(),
        )
        self._by_extension = {ext: loader for loader in loaders for ext in loader.extensions}

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset(self._by_extension)

    @property
    def missing_includes(self) -> list[str]:
        """Include directives that could not be resolved during this run."""
        return list(self._markdown.missing_includes)

    def loader_for(self, path: Path) -> DocumentLoader:
        try:
            return self._by_extension[path.suffix.lower()]
        except KeyError:
            raise UnsupportedFormatError(
                f"no loader for '{path.suffix}'; supported: "
                f"{', '.join(sorted(self.supported_extensions))}"
            ) from None

    def load(self, path: Path) -> Document:
        return self.loader_for(path).load(path, self.corpus_root)

    def iter_documents(self) -> Iterator[Document]:
        """Load every supported file under the corpus root, in stable sorted order.

        Sorted so ingestion is deterministic: chunk indices, and therefore chunk IDs,
        must not depend on filesystem iteration order.
        """
        for path in sorted(self.corpus_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in self._by_extension:
                yield self.load(path)
