"""Sparse (BM25) index over chunks.

Holds chunk identifiers and their analysed terms, and nothing else. Chunk content lives in
one place -- the chunk store -- so the dense and sparse indexes can be compared and kept in
sync by their identifier sets alone, and there is no second copy of the corpus to drift.

Persistence is JSON, deliberately, even though `BM25Okapi` is picklable. Loading a pickle
executes arbitrary code in it, so a pickled index is a code-execution vector wearing a data
file's clothes -- a bad thing to ship in a container and seed from a script. JSON is inert,
inspectable and diffable, and rebuilding the BM25 statistics on load costs milliseconds.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from hybridrag.indexing.tokenizer import tokenize
from hybridrag.models import Chunk

_FORMAT_VERSION = 1


class SparseIndex:
    """BM25 over analysed chunk text, returning (chunk_id, score)."""

    def __init__(self, chunk_ids: list[str], documents: list[list[str]]) -> None:
        if len(chunk_ids) != len(documents):
            raise ValueError(
                f"chunk_ids ({len(chunk_ids)}) and documents ({len(documents)}) must "
                "correspond one to one."
            )
        self.chunk_ids = chunk_ids
        self.documents = documents
        # BM25Okapi divides by the mean document length, so an empty corpus is undefined.
        self._bm25: BM25Okapi | None = BM25Okapi(documents) if documents else None

    @classmethod
    def build(cls, chunks: Iterable[Chunk]) -> SparseIndex:
        materialised = list(chunks)
        return cls(
            chunk_ids=[chunk.chunk_id for chunk in materialised],
            documents=[tokenize(chunk.text) for chunk in materialised],
        )

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k chunk ids by BM25 score, best first.

        Zero-scoring chunks share no term with the query and are dropped rather than
        ranked. Handing them to rank fusion would give non-matches an arbitrary position
        and let the sparse side vote on documents it never actually matched.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        terms = tokenize(query)
        if self._bm25 is None or not terms:
            return []

        scores = self._bm25.get_scores(terms)
        # Sorted by descending score, then by chunk id: ties must break the same way on
        # every run or evaluation numbers move without the retrieval changing.
        ranked = sorted(
            ((cid, float(s)) for cid, s in zip(self.chunk_ids, scores, strict=True) if s > 0.0),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[:k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": _FORMAT_VERSION,
            "chunk_ids": self.chunk_ids,
            "documents": self.documents,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SparseIndex:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        version = payload.get("format_version")
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"{path} was written by format version {version!r}, but this build reads "
                f"version {_FORMAT_VERSION}. Re-run ingestion to rebuild the index."
            )
        return cls(chunk_ids=payload["chunk_ids"], documents=payload["documents"])
