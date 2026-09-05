"""Persistent cache for model responses (D12).

The same reasoning that produced `CachedEmbedder`, applied to the other thing here that
costs money and time. An evaluation sweep asks the same 50 questions on every iteration
while a threshold or a prompt is tuned; without a cache each iteration re-spends the budget
and re-waits the latency. With one, only what actually changed is recomputed.

It matters more here than for embeddings, because generation is rate-limited by someone
else. Gemini's free tier returned 503 and then 429 during this project's own build, and D12
assumed exactly that: an evaluation interrupted at question 37 should resume free rather
than re-spend everything before it.

A decorator over any `LanguageModel`, not a feature inside one, so both providers gain it
from the same code -- and so the cache can be left out entirely when a run must be live.

**A cache is not a store of record**, and this file differs from `ChunkStore` accordingly.
A row that no longer parses is treated as a miss and overwritten, rather than raised: a
stale entry costs one regeneration, while refusing to start costs the run. `ChunkStore`
takes the opposite side, because a chunk that fails to load is data loss.
"""

from __future__ import annotations

import sqlite3
import time
from hashlib import sha256
from pathlib import Path

from pydantic import ValidationError

from hybridrag.generation.base import Completion, LanguageModel


class CachedLanguageModel:
    """Wraps a `LanguageModel`, serving repeated prompts from disk."""

    def __init__(self, inner: LanguageModel, path: Path | str, *, read_only: bool = False) -> None:
        self.inner = inner
        self.model_name = inner.model_name
        self.path = path
        # Set when a run must not reuse anything -- measuring true latency, or checking
        # that a prompt change actually changed the answers.
        self.read_only = read_only
        self.hits = 0
        self.misses = 0

        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        # WAL lets a reader proceed while a writer commits; NORMAL trades an fsync per
        # commit for speed, which is the right call for something always rebuildable.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS completions ("
            "  key TEXT PRIMARY KEY,"
            "  fingerprint TEXT NOT NULL,"
            "  payload TEXT NOT NULL,"
            "  created_at REAL NOT NULL"
            ")"
        )
        self._db.commit()

    @property
    def fingerprint(self) -> str:
        """The wrapped model's own, so caching never changes what a key means."""
        return self.inner.fingerprint

    def close(self) -> None:
        self._db.close()

    def __len__(self) -> int:
        count: int = self._db.execute("SELECT COUNT(*) FROM completions").fetchone()[0]
        return count

    def _key(self, prompt: str, system: str | None, max_output_tokens: int | None) -> str:
        """Content address for one call.

        Everything that can change the response is in here: the model's fingerprint (its
        id and its sampling settings), the system prompt, the user prompt, and the output
        cap, which decides where a long answer gets truncated. NUL separators keep two
        different splits from colliding into one key.
        """
        payload = "\x00".join(
            [
                self.inner.fingerprint,
                system or "",
                prompt,
                "" if max_output_tokens is None else str(max_output_tokens),
            ]
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def _load(self, key: str) -> Completion | None:
        row = self._db.execute("SELECT payload FROM completions WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        try:
            return Completion.model_validate_json(row[0])
        except ValidationError:
            # Written by an older `Completion` schema. Regenerating costs one request;
            # raising would strand the run behind a file it could simply overwrite.
            return None

    def _store(self, key: str, completion: Completion) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO completions (key, fingerprint, payload, created_at) "
            "VALUES (?, ?, ?, ?)",
            (key, self.inner.fingerprint, completion.model_dump_json(), time.time()),
        )
        self._db.commit()

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_output_tokens: int | None = None,
    ) -> Completion:
        key = self._key(prompt, system, max_output_tokens)

        if not self.read_only:
            started = time.perf_counter()
            stored = self._load(key)
            if stored is not None:
                self.hits += 1
                # Cost is zeroed because nothing was spent again, and latency becomes the
                # lookup rather than the generation. Token counts stay as first measured:
                # they describe the prompt and the response, not this particular call.
                return stored.model_copy(
                    update={
                        "cached": True,
                        "cost_usd": 0.0,
                        "latency_s": time.perf_counter() - started,
                    }
                )

        self.misses += 1
        # Only a successful call is stored. An exception propagates untouched, so a
        # rate-limit or a truncated response is never cached as if it were an answer.
        completion = self.inner.generate(prompt, system=system, max_output_tokens=max_output_tokens)
        self._store(key, completion)
        return completion
