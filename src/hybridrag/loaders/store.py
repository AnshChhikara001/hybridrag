"""Durable storage for parsed documents, so re-indexing need not re-parse.

The brief asks for raw documents kept alongside processed versions "so re-indexing needs no
re-upload". The speed saving is small and measured, not assumed: parsing this corpus takes
0.14s and loading the store takes 0.01s, a 0.12s saving per build (D44). The store earns its
place for the two reasons below, not for speed.

The corpus has already been lost once: it lived in a temp directory and vanished on a
reboot, which is what D19 pinned the fetch script for. A processed store is the second
copy -- self-contained, with include expansion already applied, so the indexes can be
rebuilt with no network and no `docs_src` checkout.

Staleness is keyed on a hash of the **raw bytes**, not of the parsed text, because checking
the parsed text would require the parse this exists to avoid. That has one honest limit,
recorded rather than hidden: FastAPI's `{* ... *}` includes pull in files this hash does not
cover, so editing an included example leaves the store stale. `--refresh` exists for that.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import BaseModel, Field

from hybridrag.models import Document

_FORMAT_VERSION = 1
_MANIFEST = "manifest.json"
_DOCUMENTS = "documents.jsonl"


def source_digest(path: Path) -> str:
    """sha1 of a file's raw bytes, which needs no parsing to compute."""
    return hashlib.sha1(path.read_bytes()).hexdigest()


class ProcessedManifest(BaseModel):
    """What the store holds and what produced it."""

    format_version: int = _FORMAT_VERSION
    corpus_root: str = ""
    include_root: str | None = None
    documents: int = 0
    # relative_path -> sha1 of the source file's raw bytes.
    sources: dict[str, str] = Field(default_factory=dict)


class DocumentStore:
    """Parsed documents on disk, one JSON object per line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.manifest_path = path / _MANIFEST
        self.documents_path = path / _DOCUMENTS

    def exists(self) -> bool:
        return self.manifest_path.is_file() and self.documents_path.is_file()

    def manifest(self) -> ProcessedManifest:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        version = payload.get("format_version")
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"{self.manifest_path} was written by format version {version!r}, but this "
                f"build reads version {_FORMAT_VERSION}. Re-run ingestion with --refresh."
            )
        return ProcessedManifest.model_validate(payload)

    def save(
        self,
        documents: Iterable[Document],
        *,
        corpus_root: Path,
        include_root: Path | None = None,
    ) -> int:
        """Write every document, plus the source hashes that detect staleness.

        JSON Lines rather than pickle, for the reason the rest of this project avoids
        pickle: loading a pickle executes arbitrary code, and a corpus cache is exactly the
        kind of file that gets copied between machines without being read first.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        sources: dict[str, str] = {}
        written = 0
        with self.documents_path.open("w", encoding="utf-8") as handle:
            for document in documents:
                handle.write(json.dumps(document.model_dump(mode="json")) + "\n")
                source = corpus_root / document.relative_path
                if source.is_file():
                    sources[document.relative_path] = source_digest(source)
                written += 1
        self.manifest_path.write_text(
            ProcessedManifest(
                corpus_root=str(corpus_root),
                include_root=str(include_root) if include_root else None,
                documents=written,
                sources=sources,
            ).model_dump_json(indent=2),
            encoding="utf-8",
        )
        return written

    def load(self) -> Iterator[Document]:
        """Every stored document, in the order it was written."""
        with self.documents_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield Document.model_validate_json(line)

    def stale_against(self, corpus_root: Path) -> list[str]:
        """Documents whose source file has changed, been added, or been removed.

        Only the documents' own bytes are checked. An edit to an included example is
        invisible here, which is stated in the module docstring and is why `--refresh`
        exists rather than being a silent correctness hole.
        """
        if not self.exists():
            return []
        recorded = self.manifest().sources
        changed = [
            relative_path
            for relative_path, digest in recorded.items()
            if not (corpus_root / relative_path).is_file()
            or source_digest(corpus_root / relative_path) != digest
        ]
        on_disk = {
            str(path.relative_to(corpus_root))
            for path in corpus_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".md", ".txt", ".html", ".pdf"}
        }
        return sorted(set(changed) | (on_disk - set(recorded)))
