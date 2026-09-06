"""Where each chunking strategy's index artefacts live.

Phase 4 compares three chunking strategies, which means three sets of indexes that must
never share storage -- for a different reason in each index:

* **BM25 statistics are corpus-relative.** IDF is document frequency over the index. Three
  strategies in one BM25 index triples the document count and reweights every term, so
  each arm would be scored against statistics the other two produced.
* **HNSW is one graph.** Filtering a mixed collection by metadata prunes *after* the
  approximate traversal, so recall degrades by however much of the graph belongs to the
  other strategies. That is a measurement artefact that would read as a chunking result.

The chunk store is exempt: it is keyed by chunk id and already records `strategy`, and
nothing about a lookup by primary key is affected by its neighbours.

Naming lives here rather than at each call site because the builder, the query script and
the evaluation harness must agree on it -- and a disagreement shows up as an empty index
rather than an error.
"""

from __future__ import annotations

from pathlib import Path

from hybridrag.models import ChunkingStrategy


def collection_name(strategy: ChunkingStrategy) -> str:
    """Chroma collection holding one strategy's vectors."""
    return f"chunks-{strategy.value}"


def chroma_path(index_dir: Path) -> Path:
    """Chroma's own directory. One database, one collection per strategy inside it."""
    return index_dir / "chroma"


def sparse_path(index_dir: Path, strategy: ChunkingStrategy) -> Path:
    """The BM25 index file for one strategy."""
    return index_dir / f"sparse-{strategy.value}.json"


def store_path(index_dir: Path) -> Path:
    """The chunk store: one file for every strategy (D21)."""
    return index_dir / "chunks.sqlite"
