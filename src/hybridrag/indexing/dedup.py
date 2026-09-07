"""Near-duplicate detection, the last unimplemented piece of Phase 1.

Documentation corpora repeat themselves structurally: FastAPI ships the same example once
per supported Python version, and its `{* ... *}` include expansion copies each variant into
every page that references it. Those copies compete for the same ranking slots, so a query
matching one of them tends to match all of them, and the generator receives one passage
several times instead of several passages.

Measured on this corpus before building anything, because a feature that removes nothing is
worse than no feature: **83 exact duplicate chunk texts under structure-aware chunking and
114 under semantic**, against **1** under fixed-size. At cosine 0.95 the removable fraction
is 6.2% of the structure index and 1.2% of the fixed one. That asymmetry is itself a finding
-- a 512-token window absorbs a repeated snippet into surrounding prose that differs, while
a heading-shaped chunk isolates it into a chunk identical to its siblings.

Two stages, cheapest first: an exact text match, then cosine similarity. The exact stage is
free and catches most of what is here; the vector stage catches the near-misses that differ
by a version number or a comment.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field

from hybridrag.embedding import Embedder
from hybridrag.models import Chunk

DEFAULT_THRESHOLD = 0.95


class DedupScope(StrEnum):
    """How widely a chunk is allowed to be considered a duplicate.

    DOCUMENT is the default, and the reason is measured. Corpus-wide removal deleted a
    chunk of `tutorial/request-forms.md` because `tutorial/request-files.md` carried a
    near-identical warning at cosine 0.982 -- and that chunk was the entire evidence for a
    golden question, which became unanswerable from its own source. Tier 1 registered the
    damage immediately: hybrid/structure Recall@5 fell 0.759 -> 0.724.

    The lesson generalises past this corpus. Inside one document a repeated passage really
    is redundant: include expansion emits the same example once per Python version, and a
    reader gains nothing from three copies. Across documents the same text is *not*
    redundant, because which page it appears on is part of both its meaning and its
    citation -- a question about forms must be answerable from the forms page, whatever the
    files page happens to say.
    """

    DOCUMENT = "document"
    CORPUS = "corpus"


class Duplicate(BaseModel):
    """One chunk dropped, and the one it duplicates."""

    chunk_id: str
    duplicate_of: str
    similarity: float
    exact: bool = Field(description="True when the texts are byte-identical.")
    relative_path: str = ""


class DedupReport(BaseModel):
    """What survived, what did not, and why -- so a removal can always be explained."""

    kept: list[Chunk] = Field(default_factory=list)
    removed: list[Duplicate] = Field(default_factory=list)
    threshold: float = DEFAULT_THRESHOLD
    scope: DedupScope = DedupScope.DOCUMENT

    @property
    def examined(self) -> int:
        return len(self.kept) + len(self.removed)

    @property
    def exact_removals(self) -> int:
        return sum(1 for entry in self.removed if entry.exact)

    def summary(self) -> str:
        if not self.examined:
            return "no chunks to deduplicate"
        share = len(self.removed) / self.examined
        return (
            f"{len(self.removed)} of {self.examined} chunks removed ({share:.1%}): "
            f"{self.exact_removals} exact, {len(self.removed) - self.exact_removals} "
            f"near-duplicate at cosine >= {self.threshold} ({self.scope.value} scope)"
        )


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def deduplicate(
    chunks: Sequence[Chunk],
    embedder: Embedder,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    scope: DedupScope = DedupScope.DOCUMENT,
) -> DedupReport:
    """Drop chunks that repeat one already kept, keeping the first occurrence.

    Order is the tie-break: the earliest chunk in corpus order survives, which makes the
    result deterministic across runs and machines. Chunk ids and `chunk_index` are **never**
    renumbered to close the gaps -- ids are derived from the index, so renumbering would
    change every id after a removal, invalidating the embedding cache and every stored
    identifier for no benefit.

    Similarity is a plain dot product because `Embedder` guarantees L2-normalised vectors
    (D17); if that guarantee is ever dropped this becomes silently wrong, which is why it is
    stated here rather than assumed.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if not chunks:
        return DedupReport(threshold=threshold, scope=scope)

    vectors = np.asarray(
        embedder.embed_documents([chunk.text for chunk in chunks]), dtype=np.float32
    )
    # Pre-allocated and filled in place: growing an array by vstack per chunk copies the
    # whole thing every time, which turns a 2,000-chunk pass into a 2,000-copy pass.
    kept_vectors = np.empty((len(chunks), vectors.shape[1]), dtype=np.float32)
    kept: list[Chunk] = []
    removed: list[Duplicate] = []
    by_digest: dict[tuple[str, str], str] = {}
    # Positions in `kept_vectors` a chunk may be compared against. Under DOCUMENT scope a
    # chunk only ever sees its own document's kept vectors, which is both the correct
    # behaviour and considerably less arithmetic.
    comparable: dict[str, list[int]] = defaultdict(list)

    for chunk, vector in zip(chunks, vectors, strict=True):
        bucket = "" if scope is DedupScope.CORPUS else chunk.relative_path
        digest = _digest(chunk.text)
        original = by_digest.get((bucket, digest))
        if original is not None:
            removed.append(
                Duplicate(
                    chunk_id=chunk.chunk_id,
                    duplicate_of=original,
                    similarity=1.0,
                    exact=True,
                    relative_path=chunk.relative_path,
                )
            )
            continue

        rows = comparable[bucket]
        if rows:
            similarities = kept_vectors[rows] @ vector
            best = int(np.argmax(similarities))
            score = float(similarities[best])
            if score >= threshold:
                best = rows[best]
                removed.append(
                    Duplicate(
                        chunk_id=chunk.chunk_id,
                        duplicate_of=kept[best].chunk_id,
                        similarity=score,
                        exact=False,
                        relative_path=chunk.relative_path,
                    )
                )
                continue

        kept_vectors[len(kept)] = vector
        comparable[bucket].append(len(kept))
        kept.append(chunk)
        by_digest[(bucket, digest)] = chunk.chunk_id

    return DedupReport(kept=kept, removed=removed, threshold=threshold, scope=scope)
