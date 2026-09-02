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
from hybridrag.indexing import DenseIndex, SparseIndex
from hybridrag.loaders import CorpusLoader
from hybridrag.models import Chunk, ChunkingStrategy
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


def _default_include_root(corpus: Path) -> Path | None:
    """Where `{* ... *}` directives resolve from, for a corpus laid out like FastAPI's.

    Its prose lives in `docs/en/docs` and its 684 example files in `docs_src`, a sibling
    of `docs`. Walking from the repository root instead would sweep six `requirements*.txt`
    files into a documentation corpus and rewrite every relative path -- and every id
    derived from one -- so discovery stays narrow and only resolution widens.
    """
    for parent in corpus.resolve().parents:
        if (parent / "docs_src").is_dir():
            return parent
    return None


def _chunk_corpus(
    corpus: Path, strategy: ChunkingStrategy, include_root: Path | None
) -> list[Chunk]:
    """Load and chunk every document, with the tokenizer the embedder itself uses."""
    settings = get_settings()
    # Budgets must be counted in the embedding model's own tokens, so this is the local
    # model's tokenizer even when embedding is hosted -- it is what the 512-token budget
    # was measured against, and changing it would renumber every chunk id.
    chunker = CHUNKERS[strategy](
        HuggingFaceTokenCounter(settings.embedding_model),
        max_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    loader = CorpusLoader(corpus, include_root=include_root)

    chunks: list[Chunk] = []
    documents = 0
    for document in loader.iter_documents():
        documents += 1
        chunks.extend(chunker.chunk(document))

    typer.echo(f"  {documents} documents -> {len(chunks)} chunks ({strategy.value})")
    if loader.missing_includes:
        # Loud: each unresolved directive is a code example missing from the corpus, and
        # code examples carry the identifiers sparse retrieval exists to match.
        typer.echo(f"  WARNING: {len(loader.missing_includes)} include(s) did not resolve")
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
) -> None:
    """Chunk a corpus and build the chunk store, dense index and sparse index."""
    if not corpus.is_dir():
        typer.echo(f"error: {corpus} is not a directory. Run scripts/fetch_corpus.sh first.")
        raise typer.Exit(code=1)

    settings = get_settings()
    started = time.perf_counter()

    def elapsed() -> str:
        return f"[{time.perf_counter() - started:6.1f}s]"

    resolved_include_root = include_root or _default_include_root(corpus)
    typer.echo(f"{elapsed()} loading and chunking {corpus}")
    typer.echo(f"  includes resolve from {resolved_include_root or corpus}")
    chunks = _chunk_corpus(corpus, strategy, resolved_include_root)
    if not chunks:
        typer.echo("error: the corpus produced no chunks.")
        raise typer.Exit(code=1)

    inner = _build_embedder(embedder, budget_tokens)
    cached = CachedEmbedder(inner, settings.cache_dir / "embeddings.sqlite")

    store = ChunkStore(settings.index_dir / "chunks.sqlite")
    # Cleared first: re-chunking with different parameters yields a different number of
    # chunks, and upsert alone would leave the previous run's tail behind as orphans.
    store.delete_strategy(strategy)
    store.add(chunks)
    typer.echo(f"{elapsed()} chunk store: {len(store)} chunks total")

    dense = DenseIndex.embedded(cached, settings.index_dir / "chroma")
    dropped = dense.delete_strategy(strategy)
    dense.add(chunks)
    typer.echo(f"{elapsed()} dense index: {len(dense)} vectors ({dropped} stale dropped)")

    sparse = SparseIndex.build(chunks)
    sparse.save(settings.index_dir / "sparse.json")
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
