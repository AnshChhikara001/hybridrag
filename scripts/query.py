"""Query the indexes and show what each retriever contributes.

The point of the slice: one query, three rankings -- dense alone, sparse alone, and the two
fused -- printed side by side so the fusion's effect is visible rather than asserted. The
ablations are derived from the hybrid retriever with `ablation()`, so they differ from it in
exactly one variable.

    uv run python scripts/query.py "how do I run it with docker"
    uv run python scripts/query.py "HTTPException status_code" --k 5 --text
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from hybridrag.chunk_store import ChunkStore
from hybridrag.config import get_settings
from hybridrag.embedding import Embedder, FastEmbedEmbedder
from hybridrag.embedding_cache import CachedEmbedder
from hybridrag.embedding_openai import OpenAIEmbedder
from hybridrag.indexing import DenseIndex, SparseIndex
from hybridrag.retrieval import HybridRetriever, RetrievedChunk

app = typer.Typer(add_completion=False)


def _embedder() -> Embedder:
    """Whatever built the index must also embed the query, or the vectors do not compare."""
    settings = get_settings()
    if settings.openai_api_key is None:
        return FastEmbedEmbedder(settings.embedding_model)
    return OpenAIEmbedder(settings.openai_api_key)


def _label(result: RetrievedChunk) -> str:
    """One line of provenance: where the chunk came from and which retrievers found it."""
    heading = " > ".join(result.chunk.heading_path) or "(preamble)"
    votes = "+".join(result.retrievers)
    return f"{result.chunk.relative_path}  [{heading}]  ({votes})"


def _show(title: str, results: list[RetrievedChunk], *, text: bool) -> None:
    typer.echo(f"\n{title}")
    typer.echo("-" * len(title))
    if not results:
        typer.echo("  (nothing matched)")
        return
    for result in results:
        typer.echo(f"  {result.rank}. {result.score:.5f}  {_label(result)}")
        for name, hit in sorted(result.hits.items()):
            typer.echo(f"       {name:<7} rank {hit.rank:<3} score {hit.score:.4f}")
        if text:
            snippet = " ".join(result.chunk.text.split())[:200]
            typer.echo(f"       {snippet}...")


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="The question to retrieve for.")],
    k: Annotated[int, typer.Option(help="Results to show per retriever.")] = 5,
    text: Annotated[bool, typer.Option(help="Print a snippet of each chunk.")] = False,
    index_dir: Annotated[Path | None, typer.Option(help="Overrides the configured path.")] = None,
) -> None:
    """Retrieve for one question through hybrid, dense-only and sparse-only."""
    settings = get_settings()
    root = index_dir or settings.index_dir
    if not (root / "sparse.json").is_file():
        typer.echo(f"error: no index at {root}. Run scripts/build_index.py first.")
        raise typer.Exit(code=1)

    cached = CachedEmbedder(_embedder(), settings.cache_dir / "embeddings.sqlite")
    store = ChunkStore(root / "chunks.sqlite")
    retriever = HybridRetriever(
        {
            "dense": DenseIndex.embedded(cached, root / "chroma"),
            "sparse": SparseIndex.load(root / "sparse.json"),
        },
        store,
    )

    typer.echo(f'query: "{question}"   ({len(store)} chunks indexed)')
    _show("hybrid (RRF)", retriever.retrieve(question, k=k), text=text)
    _show("dense only", retriever.ablation("dense").retrieve(question, k=k), text=text)
    _show("sparse only", retriever.ablation("sparse").retrieve(question, k=k), text=text)

    cached.close()
    store.close()


if __name__ == "__main__":
    sys.exit(app())
