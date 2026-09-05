"""The corpus of record: chunk text and provenance, addressed by chunk id.

Both indexes hold identifiers and nothing else -- Chroma stores vectors plus two filter
fields, BM25 stores analysed terms -- so retrieval returns ids that mean nothing on their
own. This is what turns them back into chunks. Keeping content in exactly one place is
what lets the two indexes be checked against each other by identifier set alone, and means
a re-chunk cannot leave one copy of the corpus stale while another moves on.

SQLite, matching `CachedEmbedder`: stdlib, one file to ship or delete, and random access by
id so hydrating ten chunks never loads the corpus. It is also inert data, unlike a pickle,
which is the same reason `SparseIndex` persists as JSON.

Each row keeps the whole `Chunk` as JSON in a payload column, with the few fields that get
filtered on lifted out beside it. A column per field would mean a schema migration every
time the model gains one; Pydantic already validates the round trip, and the lifted columns
are what queries actually index on.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from hybridrag.models import Chunk, ChunkingStrategy

# Bumped when the row format changes in a way that makes an existing file unreadable.
_SCHEMA_VERSION = 1

# SQLite's default ceiling on host parameters in one statement is 999.
_LOOKUP_BATCH = 500


class MissingChunkError(KeyError):
    """Raised when an index holds an id the store does not.

    Loud on purpose. A retrieved id with no chunk behind it means the index and the store
    were built from different runs, and silently dropping it would show up only as
    retrieval quality quietly degrading -- the hardest class of bug to notice in a system
    whose output is a ranked list.
    """


class ChunkStore:
    """Persistent chunk lookup by id."""

    def __init__(self, path: Path | str) -> None:
        self.path = path
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()

    @classmethod
    def in_memory(cls) -> ChunkStore:
        """Nothing persisted. For tests, which should not touch the filesystem."""
        return cls(":memory:")

    def _create_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id      TEXT PRIMARY KEY,
                doc_id        TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                strategy      TEXT NOT NULL,
                payload       TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunks_by_strategy ON chunks (strategy);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        row = self._db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            self._db.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )
        elif int(row[0]) != _SCHEMA_VERSION:
            raise ValueError(
                f"{self.path} was written by schema version {row[0]}, but this build reads "
                f"version {_SCHEMA_VERSION}. Re-run ingestion to rebuild the store."
            )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __len__(self) -> int:
        count: int = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return count

    def __contains__(self, chunk_id: str) -> bool:
        return (
            self._db.execute("SELECT 1 FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
            is not None
        )

    def add(self, chunks: Iterable[Chunk]) -> None:
        """Insert or replace chunks.

        Upsert rather than insert, so re-running ingestion over an unchanged corpus is
        idempotent instead of a primary-key error. Chunk ids are deterministic per
        (document, strategy, position), so a rebuild overwrites its own previous rows and
        leaves everything else alone.
        """
        self._db.executemany(
            "INSERT OR REPLACE INTO chunks "
            "(chunk_id, doc_id, relative_path, strategy, payload) VALUES (?, ?, ?, ?, ?)",
            (
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.relative_path,
                    chunk.strategy.value,
                    chunk.model_dump_json(),
                )
                for chunk in chunks
            ),
        )
        self._db.commit()

    def get(self, chunk_id: str) -> Chunk | None:
        """One chunk, or None when the id is unknown."""
        row = self._db.execute(
            "SELECT payload FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return None if row is None else Chunk.model_validate_json(row[0])

    def get_many(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Chunks for `chunk_ids`, **in the order requested**.

        Order is the contract. The caller's sequence is a ranking, and SQL returns rows in
        whatever order it finds them -- hydrating a result set through a plain `IN` query
        would silently reorder search results into something close to insertion order,
        which looks like working retrieval and scores like noise.
        """
        if not chunk_ids:
            return []

        found: dict[str, Chunk] = {}
        unique = list(dict.fromkeys(chunk_ids))
        for start in range(0, len(unique), _LOOKUP_BATCH):
            batch = unique[start : start + _LOOKUP_BATCH]
            placeholders = ",".join("?" * len(batch))
            rows = self._db.execute(
                f"SELECT chunk_id, payload FROM chunks WHERE chunk_id IN ({placeholders})",
                batch,
            ).fetchall()
            for chunk_id, payload in rows:
                found[chunk_id] = Chunk.model_validate_json(payload)

        missing = [chunk_id for chunk_id in unique if chunk_id not in found]
        if missing:
            raise MissingChunkError(
                f"{len(missing)} id(s) are in the index but not the chunk store, so the two "
                f"were built from different runs: {missing[:5]}. Re-run ingestion."
            )
        return [found[chunk_id] for chunk_id in chunk_ids]

    def chunk_ids(self, strategy: ChunkingStrategy | None = None) -> set[str]:
        """Every id held, so the indexes can be checked against the store and each other."""
        if strategy is None:
            rows = self._db.execute("SELECT chunk_id FROM chunks").fetchall()
        else:
            rows = self._db.execute(
                "SELECT chunk_id FROM chunks WHERE strategy = ?", (strategy.value,)
            ).fetchall()
        return {row[0] for row in rows}

    def iter_chunks(self, strategy: ChunkingStrategy | None = None) -> Iterator[Chunk]:
        """Stream chunks, optionally for one strategy.

        Streamed rather than returned as a list because Phase 4 holds three strategies in
        one store, and rebuilding an index should not require the whole corpus in memory.
        Ordered by id so a rebuild is reproducible.
        """
        if strategy is None:
            cursor = self._db.execute("SELECT payload FROM chunks ORDER BY chunk_id")
        else:
            cursor = self._db.execute(
                "SELECT payload FROM chunks WHERE strategy = ? ORDER BY chunk_id",
                (strategy.value,),
            )
        for (payload,) in cursor:
            yield Chunk.model_validate_json(payload)

    def delete_strategy(self, strategy: ChunkingStrategy) -> int:
        """Drop every chunk of one strategy, returning how many rows went.

        Needed because re-chunking with different parameters produces *fewer or more*
        chunks than last time: upsert alone would leave the tail of the previous run
        behind as chunks no index points at.
        """
        cursor = self._db.execute("DELETE FROM chunks WHERE strategy = ?", (strategy.value,))
        self._db.commit()
        return cursor.rowcount
