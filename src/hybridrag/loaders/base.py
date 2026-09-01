"""Loader interface and the text-handling shared by every format.

Each loader turns one file into a `Document` carrying `Section` objects. Sections are the
unit of retrieval ground truth, so assembling them correctly here is what makes the
Phase 4 strategy comparison possible at all.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from hybridrag.models import Document, Section, SourceFormat


class LoaderError(RuntimeError):
    """A document could not be loaded."""


class UnsupportedFormatError(LoaderError):
    """No registered loader handles this file extension."""


_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

# Sections are joined by this separator to form `Document.text`; section offsets are
# computed against the same layout, so the two can never drift apart.
_SECTION_SEPARATOR = "\n\n"


def normalise_text(text: str) -> str:
    """Normalise line endings and trailing whitespace, preserving indentation.

    Deliberately does NOT collapse runs of spaces: this corpus embeds Python source, where
    leading indentation is semantic. Collapsing it would corrupt every code example and
    strip exactly the tokens sparse retrieval depends on.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _EXCESS_BLANK_LINES.sub("\n\n", text).strip()


def content_hash(text: str) -> str:
    """sha1 of normalised text; identifies unchanged files across re-ingestion runs."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SectionBlock:
    """A section a loader has found, before it is given a stable identifier."""

    heading_path: tuple[str, ...]
    level: int
    text: str
    page: int | None = None


def assemble_document(
    *,
    relative_path: str,
    source_format: SourceFormat,
    blocks: list[SectionBlock],
    title: str | None = None,
) -> Document:
    """Turn a loader's section blocks into a validated `Document`.

    Assigns each section its stable ID and disambiguates repeats. Two sections in one
    document can legitimately share a breadcrumb -- a repeated `### Example`, or any two
    pages of a PDF (which have no headings at all). Left alone they would hash to the same
    `section_id`, silently merging distinct regions and corrupting retrieval ground truth.
    The occurrence index is appended only from the second occurrence onward, so the
    ordinary case keeps the clean identifier.
    """
    seen: dict[tuple[str, ...], int] = {}
    sections: list[Section] = []
    cursor = 0

    for block in blocks:
        text = normalise_text(block.text)
        if not text:
            continue

        occurrence = seen.get(block.heading_path, 0)
        seen[block.heading_path] = occurrence + 1
        if block.page is not None:
            discriminator = f"page={block.page}"
        elif occurrence:
            discriminator = f"occurrence={occurrence}"
        else:
            discriminator = ""

        # Offsets are assigned as the joined text is laid out, so they address
        # `Document.text` -- the text that actually gets chunked and indexed. Answer spans
        # in the golden set live in this same coordinate system, which is what lets a
        # retrieval hit be defined as span containment rather than section membership.
        start = cursor
        cursor = start + len(text) + len(_SECTION_SEPARATOR)

        sections.append(
            Section(
                section_id=Section.make_id(relative_path, block.heading_path, discriminator),
                relative_path=relative_path,
                heading_path=block.heading_path,
                level=block.level,
                text=text,
                start_char=start,
                end_char=start + len(text),
                page=block.page,
            )
        )

    full_text = _SECTION_SEPARATOR.join(section.text for section in sections)
    return Document(
        doc_id=Document.make_id(relative_path),
        relative_path=relative_path,
        source_format=source_format,
        title=title,
        text=full_text,
        sections=sections,
        content_hash=content_hash(full_text),
    )


class DocumentLoader(ABC):
    """Base for every format loader.

    `corpus_root` is passed to `load` rather than held on the instance so loaders stay
    stateless and reusable across corpora, and so relative paths -- which become document
    and section identities -- are always derived from an explicit root.
    """

    extensions: ClassVar[frozenset[str]]
    source_format: ClassVar[SourceFormat]

    @abstractmethod
    def load(self, path: Path, corpus_root: Path) -> Document:
        """Read one file and return it as a `Document`."""

    @staticmethod
    def relative_path_of(path: Path, corpus_root: Path) -> str:
        """POSIX-style path relative to the corpus root.

        Forced to POSIX so identifiers hash identically on Windows and Unix -- otherwise a
        golden set built on one platform silently fails to match on another.
        """
        return path.resolve().relative_to(corpus_root.resolve()).as_posix()
