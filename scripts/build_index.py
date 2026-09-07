"""Build the chunk store and both indexes from a corpus directory.

One script owns the whole ingestion path, so the three artefacts are always produced from
the same chunks in the same run. Building them separately is how a dense index ends up
holding ids the sparse index and the chunk store have never heard of, and that desync only
shows up later as retrieval quietly returning less than it should. The run ends by checking
all three identifier sets against each other and failing if they disagree.

Cost control: the embedder is always wrapped in the persistent cache, so re-running over an
unchanged corpus re-embeds nothing and costs nothing. `--budget` caps the run before a
request is sent, not after.

    uv run python scripts/build_index.py data/raw/fastapi/docs/en/docs
    uv run python scripts/build_index.py <corpus> --embedder local   # no API key needed
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from hybridrag.chunk_store import ChunkStore
from hybridrag.chunking import Chunker, FixedChunker, SemanticChunker, StructureChunker
from hybridrag.config import get_settings
from hybridrag.embedding import Embedder, FastEmbedEmbedder
from hybridrag.embedding_cache import CachedEmbedder
from hybridrag.embedding_openai import OpenAIEmbedder
from hybridrag.indexing import (
    DenseIndex,
    SparseIndex,
    chroma_path,
    collection_name,
    deduplicate,
    sparse_path,
    store_path,
)
from hybridrag.loaders import CorpusLoader, DocumentStore, default_include_root
from hybridrag.models import Chunk, ChunkingStrategy, Document
from hybridrag.tokenization import HuggingFaceTokenCounter

app = typer.Typer(add_completion=False)

CHUNKERS: dict[ChunkingStrategy, type[Chunker]] = {
    ChunkingStrategy.FIXED: FixedChunker,
    ChunkingStrategy.STRUCTURE: StructureChunker,
    ChunkingStrategy.SEMANTIC: SemanticChunker,
}


class EmbedderChoice(StrEnum):
    OPENAI = "openai"
    LOCAL = "local"


def _build_embedder(choice: EmbedderChoice, budget_tokens: int | None) -> Embedder:
    """The hosted adapter by default, the local model when there is no key (D18)."""
    settings = get_settings()
    if choice is EmbedderChoice.LOCAL:
        return FastEmbedEmbedder(settings.embedding_model)
    if settings.openai_api_key is None:
        raise typer.BadParameter(
            "OPENAI_API_KEY is not set. Put it in .env, or pass --embedder local to use "
            "the local model instead."
        )
    return OpenAIEmbedder(settings.openai_api_key, token_budget=budget_tokens)


def _build_chunker(strategy: ChunkingStrategy, embedder: Embedder) -> Chunker:
    """Construct one chunking strategy.

    The semantic chunker needs an embedder and the other two do not, which is why this is
    a branch rather than a uniform call -- and why the script previously could not build
    semantic at all: it passed the same three arguments to every strategy.

    Budgets must be counted in the embedding model's own tokens, so the tokenizer is the
    local model's even when embedding is hosted: it is what the 512-token budget was
    measured against, and changing it would renumber every chunk id.
    """
    settings = get_settings()
    tokenizer = HuggingFaceTokenCounter(settings.embedding_model)
    if strategy is ChunkingStrategy.SEMANTIC:
        return SemanticChunker(
            tokenizer,
            embedder,
            max_tokens=settings.chunk_tokens,
            overlap_tokens=0,  # D16: a boundary chosen for a topic change is not blurred.
            percentile=settings.semantic_percentile,
        )
    return CHUNKERS[strategy](
        tokenizer,
        max_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )


def _load_documents(
    corpus: Path, include_root: Path | None, store: DocumentStore, *, refresh: bool
) -> list[Document]:
    """Parsed documents, from the processed store when it is still valid.

    The store is written on every run, so the expensive path is paid once and a chunking
    sweep -- which re-runs this for each of three strategies -- parses the corpus once
    rather than three times.
    """
    if not refresh and store.exists():
        stale = store.stale_against(corpus)
        if not stale:
            documents = list(store.load())
            typer.echo(f"  {len(documents)} documents from the processed store (no parsing)")
            return documents
        typer.echo(f"  processed store is stale in {len(stale)} document(s); re-parsing")

    loader = CorpusLoader(corpus, include_root=include_root)
    documents = list(loader.iter_documents())
    if loader.missing_includes:
        # Loud: each unresolved directive is a code example missing from the corpus, and
        # code examples carry the identifiers sparse retrieval exists to match.
        typer.echo(f"  WARNING: {len(loader.missing_includes)} include(s) did not resolve")
    store.save(documents, corpus_root=corpus, include_root=include_root)
    typer.echo(f"  {len(documents)} documents parsed and written to {store.path}")
    return documents


def _chunk_corpus(
    documents: Sequence[Document], strategy: ChunkingStrategy, embedder: Embedder
) -> list[Chunk]:
    """Chunk every document with one strategy."""
    chunker = _build_chunker(strategy, embedder)
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunker.chunk(document))
    typer.echo(f"  {len(documents)} documents -> {len(chunks)} chunks ({strategy.value})")
    return chunks


@app.command()
def build(
    corpus: Annotated[Path, typer.Argument(help="Corpus directory, e.g. data/raw/fastapi")],
    strategy: Annotated[
        ChunkingStrategy, typer.Option(help="Chunking strategy to index.")
    ] = ChunkingStrategy.STRUCTURE,
    embedder: Annotated[
        EmbedderChoice, typer.Option(help="Hosted OpenAI, or the local model.")
    ] = EmbedderChoice.OPENAI,
    budget_tokens: Annotated[
        int | None, typer.Option(help="Refuse to send a request that would exceed this.")
    ] = 600_000,
    include_root: Annotated[
        Path | None, typer.Option(help="Root for {* ... *} includes. Detected if omitted.")
    ] = None,
    dedup: Annotated[bool, typer.Option(help="Drop near-duplicate chunks before indexing.")] = True,
    dedup_threshold: Annotated[
        float | None, typer.Option(help="Cosine at or above which a chunk is a duplicate.")
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option(
            help="Re-parse the corpus even if the processed store looks current. Needed "
            "after editing an included example, which source hashes cannot see."
        ),
    ] = False,
) -> None:
    """Chunk a corpus and build the chunk store, dense index and sparse index."""
    if not corpus.is_dir():
        typer.echo(f"error: {corpus} is not a directory. Run scripts/fetch_corpus.sh first.")
        raise typer.Exit(code=1)

    settings = get_settings()
    started = time.perf_counter()

    def elapsed() -> str:
        return f"[{time.perf_counter() - started:6.1f}s]"

    resolved_include_root = include_root or default_include_root(corpus)
    # Built before chunking, because the semantic strategy embeds every sentence to find
    # its boundaries. Those vectors go through the same cache as the chunk vectors, so a
    # second run over an unchanged corpus re-embeds nothing.
    inner = _build_embedder(embedder, budget_tokens)
    cached = CachedEmbedder(inner, settings.cache_dir / "embeddings.sqlite")

    typer.echo(f"{elapsed()} loading and chunking {corpus}")
    typer.echo(f"  includes resolve from {resolved_include_root or corpus}")
    documents = _load_documents(
        corpus,
        resolved_include_root,
        DocumentStore(settings.processed_dir),
        refresh=refresh,
    )
    chunks = _chunk_corpus(documents, strategy, cached)
    if not chunks:
        typer.echo("error: the corpus produced no chunks.")
        raise typer.Exit(code=1)

    if dedup:
        # Free: it reuses the very vectors the dense index is about to need, and they come
        # from the same cache, so deduplication costs no additional request.
        threshold = (
            dedup_threshold if dedup_threshold is not None else settings.dedup_threshold
        )
        report = deduplicate(chunks, cached, threshold=threshold)
        typer.echo(f"{elapsed()} dedup: {report.summary()}")
        for entry in report.removed[:5]:
            typer.echo(
                f"    {entry.relative_path} duplicates {entry.duplicate_of[:12]} "
                f"at {entry.similarity:.3f}"
            )
        if len(report.removed) > 5:
            typer.echo(f"    ... and {len(report.removed) - 5} more")
        chunks = report.kept

    store = ChunkStore(store_path(settings.index_dir))
    # Cleared first: re-chunking with different parameters yields a different number of
    # chunks, and upsert alone would leave the previous run's tail behind as orphans.
    store.delete_strategy(strategy)
    store.add(chunks)
    typer.echo(f"{elapsed()} chunk store: {len(store)} chunks total")

    # One collection and one BM25 file per strategy: the three arms of the chunking
    # comparison must not share an ANN graph or a set of IDF statistics (see
    # `indexing/layout.py`).
    dense = DenseIndex.embedded(
        cached, chroma_path(settings.index_dir), collection_name=collection_name(strategy)
    )
    dropped = dense.delete_strategy(strategy)
    dense.add(chunks)
    typer.echo(f"{elapsed()} dense index: {len(dense)} vectors ({dropped} stale dropped)")

    sparse = SparseIndex.build(chunks)
    sparse.save(sparse_path(settings.index_dir, strategy))
    typer.echo(f"{elapsed()} sparse index: {len(sparse)} documents")

    # Checked in both directions. A missing-only check passes while an index quietly
    # accumulates orphans from a previous run -- which is exactly what happened here:
    # 1,892 stale vectors survived a rebuild and the one-way check reported success.
    expected = {chunk.chunk_id for chunk in chunks}
    held = {
        "chunk store": store.chunk_ids(strategy),
        "dense index": dense.chunk_ids(),
        "sparse index": set(sparse.chunk_ids),
    }
    ok = True
    for name, ids in held.items():
        missing, extra = expected - ids, ids - expected
        if missing or extra:
            ok = False
            typer.echo(
                f"error: {name} is missing {len(missing)} and holds {len(extra)} extra id(s)"
            )
    if not ok:
        raise typer.Exit(code=1)
    typer.echo(
        f"{elapsed()} verified: all three artefacts hold exactly the same {len(expected)} ids"
    )

    if isinstance(inner, OpenAIEmbedder):
        typer.echo(
            f"{elapsed()} spend: {inner.requests_made} request(s), "
            f"{inner.tokens_used:,} tokens, ${inner.estimated_cost_usd:.4f}"
        )
    cached.close()
    store.close()


if __name__ == "__main__":
    sys.exit(app())
