"""Dense (vector) index over chunks, backed by Chroma.

Mirrors `SparseIndex` deliberately: same `add`/`search` shape, same `(chunk_id, score)`
return, scores ascending in quality. Rank fusion can then treat the two retrievers
identically instead of special-casing either.

Dual-mode, per decision D6. `embedded` runs Chroma in-process against a directory, which is
what ships in the single-container deployment; `server` talks to a Chroma service over
HTTP, which is what docker-compose runs locally. Identical client API either way, so the
switch is configuration rather than a second code path -- and dev matches prod.

Embeddings are supplied by our own `Embedder`, never by Chroma's default embedding
function. Letting Chroma embed would silently download a second model, and the vectors in
the index would no longer match the ones the chunker and query path use.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from uuid import uuid4

from chromadb import EphemeralClient, HttpClient, PersistentClient
from chromadb.api import ClientAPI
from chromadb.config import Settings

from hybridrag.embedding import Embedder
from hybridrag.models import Chunk, ChunkingStrategy

# Chroma computes cosine distance as 1 - cosine_similarity, so this inverts back to a
# similarity where larger is better, matching BM25's direction.
_COSINE_SPACE = {"hnsw:space": "cosine"}

# Chroma phones home with usage statistics by default. Off: a documentation indexer has no
# business making network calls, and the container should run in an offline environment.
_SETTINGS = Settings(anonymized_telemetry=False)


class DenseIndex:
    """Vector search over chunk embeddings, returning (chunk_id, cosine similarity)."""

    def __init__(
        self, embedder: Embedder, client: ClientAPI, *, collection_name: str = "chunks"
    ) -> None:
        self.embedder = embedder
        self.client = client
        self.collection = client.get_or_create_collection(
            name=collection_name, metadata=_COSINE_SPACE
        )

    @classmethod
    def embedded(
        cls, embedder: Embedder, path: Path, *, collection_name: str = "chunks"
    ) -> DenseIndex:
        """In-process Chroma against a directory -- the single-container deployment."""
        path.mkdir(parents=True, exist_ok=True)
        return cls(
            embedder,
            PersistentClient(path=str(path), settings=_SETTINGS),
            collection_name=collection_name,
        )

    @classmethod
    def server(
        cls, embedder: Embedder, host: str, port: int, *, collection_name: str = "chunks"
    ) -> DenseIndex:
        """Chroma as a separate service -- what docker-compose runs locally."""
        return cls(
            embedder,
            HttpClient(host=host, port=port, settings=_SETTINGS),
            collection_name=collection_name,
        )

    @classmethod
    def in_memory(cls, embedder: Embedder, *, collection_name: str | None = None) -> DenseIndex:
        """Nothing persisted. Used by tests, which should not touch the filesystem.

        Chroma caches its in-process system by settings, so two EphemeralClients built
        with the same configuration share one database -- a fresh client sees the previous
        one's data. Defaulting to a unique collection name per instance restores the
        isolation the name `in_memory` implies.
        """
        return cls(
            embedder,
            EphemeralClient(settings=_SETTINGS),
            collection_name=collection_name or f"chunks-{uuid4().hex[:12]}",
        )

    def __len__(self) -> int:
        return self.collection.count()

    def chunk_ids(self) -> set[str]:
        """Every id in the collection, so the sparse index can be checked against it."""
        return set(self.collection.get(include=[])["ids"])

    def delete_strategy(self, strategy: ChunkingStrategy) -> int:
        """Drop every vector belonging to one strategy, returning how many went.

        Re-chunking with different parameters produces a different number of chunks, and
        `upsert` alone leaves the previous run's surplus behind: the collection grows by
        the difference and answers queries with vectors no chunk store can hydrate. This
        is why `strategy` is carried in the metadata at all.
        """
        doomed = self.collection.get(where={"strategy": strategy.value}, include=[])["ids"]
        if doomed:
            self.collection.delete(ids=doomed)
        return len(doomed)

    def delete_document(self, relative_path: str, strategy: ChunkingStrategy) -> int:
        """Drop one document's vectors under one strategy, returning how many went.

        Mirrors `delete_strategy` at document scope: re-ingesting a document that now
        chunks to fewer pieces than its previous version would otherwise leave the old
        version's surplus vectors behind, answering queries with ids no chunk store holds.
        """
        doomed = self.collection.get(
            where={"$and": [{"relative_path": relative_path}, {"strategy": strategy.value}]},
            include=[],
        )["ids"]
        if doomed:
            self.collection.delete(ids=doomed)
        return len(doomed)

    def add(self, chunks: Iterable[Chunk]) -> None:
        """Embed and index chunks, in batches Chroma will accept."""
        materialised = list(chunks)
        if not materialised:
            return
        limit = self.client.get_max_batch_size()
        for start in range(0, len(materialised), limit):
            self._add_batch(materialised[start : start + limit])

    def _add_batch(self, batch: Sequence[Chunk]) -> None:
        vectors = self.embedder.embed_documents([chunk.text for chunk in batch])
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in batch],
            embeddings=vectors,
            # Only what a retrieval filter would need. Chunk text and full provenance live
            # in the chunk store, so there is one copy of the corpus, not two.
            metadatas=[
                {"relative_path": chunk.relative_path, "strategy": chunk.strategy.value}
                for chunk in batch
            ],
        )

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k chunk ids by cosine similarity, best first.

        HNSW is an approximate index, so results for a large collection are not guaranteed
        to be the exact top-k. That is the standard speed/recall trade and is worth
        remembering when a retrieval metric moves by a fraction.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if self.collection.count() == 0:
            return []

        # Converted to a plain list: Chroma's signature accepts a float sequence, and
        # feeding it a float32 array trips the invariance of its declared list type.
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query).tolist()],
            n_results=min(k, self.collection.count()),
            include=["distances"],
        )
        distances = result["distances"]
        if distances is None:
            return []
        return [
            (chunk_id, 1.0 - float(distance))
            for chunk_id, distance in zip(result["ids"][0], distances[0], strict=True)
        ]
