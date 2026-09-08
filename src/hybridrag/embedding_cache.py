"""Persistent embedding cache.

Embedding is the slowest thing this project does and the only thing that can cost money.
Semantic chunking embeds every sentence in the corpus -- a measured 28 minutes -- and a
percentile sweep re-embeds the *same sentences* for every threshold it tries. Without a
cache, tuning that parameter costs half an hour per value. With one, it costs half an hour
once.

Implemented as a decorator over any `Embedder` rather than inside one, so it composes: the
local model gets faster, and the paid OpenAI adapter gets cheaper, from the same code. That
second case matters for a project with a $1 ceiling -- a re-run of the D3 benchmark should
not re-spend the budget.

Storage is SQLite: stdlib, so no new dependency; random access, so a 46 MB cache is never
loaded into memory; and a single file that is trivial to delete or ship.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import numpy as np

from hybridrag.embedding import Embedder, Vector

# SQLite's default limit on host parameters in one statement is 999.
_LOOKUP_BATCH = 500


class CachedEmbedder:
    """Wraps an `Embedder`, serving repeat texts from disk instead of recomputing them."""

    def __init__(self, inner: Embedder, path: Path) -> None:
        self.inner = inner
        self.model_name = inner.model_name
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        # WAL lets a reader proceed while a writer commits; NORMAL trades an fsync per
        # commit for speed, which is the right call for a cache that can always be rebuilt.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB)")
        self._db.commit()
        # Guards only the sqlite calls in `_fetch`/`_store`, not `_compute`: the API runs
        # concurrent requests in a threadpool, and holding this across an embedding call
        # would serialise every request on the model itself instead of just the cache.
        self._lock = threading.Lock()

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    def close(self) -> None:
        self._db.close()

    def _key(self, text: str, kind: str) -> str:
        """Content address for one text.

        The model name is part of the key, so two models never share a vector. So is the
        kind: `bge` prefixes queries with an instruction, and a query and a document with
        identical text embed to genuinely different vectors.
        """
        digest = sha256(f"{self.model_name}\x00{kind}\x00{text}".encode())
        return digest.hexdigest()

    def _fetch(self, keys: Sequence[str]) -> dict[str, Vector]:
        found: dict[str, Vector] = {}
        with self._lock:
            for start in range(0, len(keys), _LOOKUP_BATCH):
                batch = keys[start : start + _LOOKUP_BATCH]
                placeholders = ",".join("?" * len(batch))
                rows = self._db.execute(
                    f"SELECT key, vec FROM vectors WHERE key IN ({placeholders})", batch
                ).fetchall()
                for key, blob in rows:
                    found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def _store(self, pairs: list[tuple[str, Vector]]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO vectors (key, vec) VALUES (?, ?)",
                [(key, vector.astype(np.float32, copy=False).tobytes()) for key, vector in pairs],
            )
            self._db.commit()

    def _embed(self, texts: Sequence[str], kind: str) -> Vector:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        keys = [self._key(text, kind) for text in texts]
        cached = self._fetch(keys)

        # Deduplicated: a text repeated within one call is embedded once, not once per
        # occurrence. Corpora contain boilerplate, so this is not a theoretical saving.
        missing = list(dict.fromkeys(key for key in keys if key not in cached))
        if missing:
            index_of = {key: position for position, key in enumerate(keys)}
            fresh = self._compute([texts[index_of[key]] for key in missing], kind)
            new_pairs = list(zip(missing, fresh, strict=True))
            self._store(new_pairs)
            cached.update(dict(new_pairs))

        return np.vstack([cached[key] for key in keys]).astype(np.float32, copy=False)

    def _compute(self, texts: list[str], kind: str) -> Vector:
        if kind == "query":
            return np.vstack([self.inner.embed_query(text) for text in texts])
        return self.inner.embed_documents(texts)

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> Vector:
        row: Vector = self._embed([text], "query")[0]
        return row
