"""Sweep RRF's dense/sparse weighting on Tier 1 -- free, no language model.

The brief requires fusion to be "configurable so you can tune it" (D20 implemented the
knob; this exercises it and measures the result instead of leaving it theoretical).

Only the *ratio* between the two weights matters: RRF's score is a sum of
`weight / (rank_constant + rank)`, so scaling both weights by the same constant leaves
every ranking, and therefore every metric, unchanged. `sparse_weight` is held at 1.0 and
`dense_weight` is swept around it -- below 1.0 favours sparse, above favours dense, 1.0 is
the project's current, unweighted default.

    uv run python scripts/sweep_rrf_weights.py data/raw/fastapi/docs/en/docs
    uv run python scripts/sweep_rrf_weights.py <corpus> --strategy semantic
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Annotated

import typer

from hybridrag.chunk_store import ChunkStore
from hybridrag.config import get_settings
from hybridrag.embedding import Embedder, FastEmbedEmbedder
from hybridrag.embedding_cache import CachedEmbedder
from hybridrag.embedding_openai import OpenAIEmbedder
from hybridrag.evaluation import (
    DEFAULT_DEPTH,
    DEFAULT_TOKEN_BUDGET,
    GoldenSet,
    build_pools,
    locate_all,
    run_arm,
)
from hybridrag.evaluation.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_ci, paired_delta
from hybridrag.indexing import (
    DenseIndex,
    SparseIndex,
    chroma_path,
    collection_name,
    sparse_path,
    store_path,
)
from hybridrag.loaders import CorpusLoader, default_include_root
from hybridrag.models import ChunkingStrategy, Document
from hybridrag.retrieval import HybridRetriever

app = typer.Typer(add_completion=False)

# sparse_weight is fixed at 1.0 throughout; these are the dense_weight values tried
# against it. 1.0 is the project's current, unweighted default and the baseline every
# other point is compared against.
DENSE_WEIGHTS: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)

METRICS: tuple[str, ...] = ("recall@5", "recall@10", "ndcg@10", "recall@budget")


def _embedder() -> Embedder:
    settings = get_settings()
    if settings.openai_api_key is None:
        return FastEmbedEmbedder(settings.embedding_model)
    return OpenAIEmbedder(settings.openai_api_key)


def _load_corpus(corpus: Path) -> dict[str, Document]:
    loader = CorpusLoader(corpus, include_root=default_include_root(corpus))
    return {document.relative_path: document for document in loader.iter_documents()}


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cell.strip() for cell in row) + " |" for row in rows]
    return "\n".join(lines)


@app.command()
def sweep(
    corpus: Annotated[Path, typer.Argument(help="Corpus directory the index was built from.")],
    strategy: Annotated[
        ChunkingStrategy, typer.Option(help="Which chunking strategy's index to sweep.")
    ] = ChunkingStrategy.FIXED,
    golden_path: Annotated[Path, typer.Option(help="The golden set.")] = Path(
        "evals/golden_set.yaml"
    ),
    out_dir: Annotated[Path, typer.Option(help="Where the report is written.")] = Path(
        "evals/reports"
    ),
    depth: Annotated[int, typer.Option(help="How deep the retriever searches.")] = DEFAULT_DEPTH,
    budget: Annotated[int, typer.Option()] = DEFAULT_TOKEN_BUDGET,
    index_dir: Annotated[Path | None, typer.Option(help="Overrides the configured path.")] = None,
) -> None:
    """Score hybrid retrieval on one strategy across a grid of dense:sparse RRF weights."""
    settings = get_settings()
    root = index_dir or settings.index_dir
    started = time.perf_counter()

    def elapsed() -> str:
        return f"[{time.perf_counter() - started:6.1f}s]"

    if not corpus.is_dir():
        typer.echo(f"error: {corpus} is not a directory. Run scripts/fetch_corpus.sh first.")
        raise typer.Exit(code=1)

    golden = GoldenSet.load(golden_path)
    documents = _load_corpus(corpus)
    located = locate_all(golden, documents)
    typer.echo(f"{elapsed()} golden set: {len(golden.verified())} verified questions")

    inner = _embedder()
    cached = CachedEmbedder(inner, settings.cache_dir / "embeddings.sqlite")
    for question in golden.verified():
        cached.embed_query(question.question)

    store = ChunkStore(store_path(root))
    bm25_path = sparse_path(root, strategy)
    if not bm25_path.is_file():
        typer.echo(
            f"error: no {strategy.value} index at {root}. Build it first:\n"
            f"  uv run python scripts/build_index.py {corpus} --strategy {strategy.value}"
        )
        raise typer.Exit(code=1)

    dense = DenseIndex.embedded(
        cached, chroma_path(root), collection_name=collection_name(strategy)
    )
    sparse = SparseIndex.load(bm25_path)
    chunks = list(store.iter_chunks(strategy))
    pools = build_pools(chunks, located)
    typer.echo(f"{elapsed()} {strategy.value}: {len(chunks)} chunks, sweeping dense_weight")

    baseline_weight = 1.0
    results = {}
    for dense_weight in DENSE_WEIGHTS:
        retriever = HybridRetriever(
            {"dense": dense, "sparse": sparse},
            store,
            weights={"dense": dense_weight, "sparse": 1.0},
        )
        result = run_arm(
            f"dense={dense_weight}",
            strategy,
            retriever,
            golden,
            located,
            pools,
            depth=depth,
            budget=budget,
        )
        results[dense_weight] = result
        recall = sum(result.series("recall@5")) / max(1, len(result.questions))
        typer.echo(f"  dense_weight={dense_weight:<5} recall@5 {recall:.3f}")

    baseline = results[baseline_weight]

    rows = []
    for dense_weight in DENSE_WEIGHTS:
        result = results[dense_weight]
        cells = [f"{dense_weight}:1"]
        for metric in METRICS:
            point = bootstrap_ci(
                result.series(metric), resamples=DEFAULT_RESAMPLES, seed=DEFAULT_SEED
            )
            cells.append(f"{point.mean:.3f} [{point.low:.3f}, {point.high:.3f}]")
        if dense_weight == baseline_weight:
            cells.append("-- baseline --")
        else:
            d = paired_delta(
                result.series("recall@5"),
                baseline.series("recall@5"),
                resamples=DEFAULT_RESAMPLES,
                seed=DEFAULT_SEED,
            )
            mark = "*" if d.significant else ""
            cells.append(f"{d.difference:+.3f} [{d.low:+.3f}, {d.high:+.3f}]{mark}")
        rows.append(cells)

    header = ["dense:sparse", *METRICS, "recall@5 vs 1:1 [95% CI]"]
    table = _table(rows, header)

    report = f"""# RRF weight sweep — Tier 1

`{strategy.value}` chunking, hybrid retriever, dense:sparse weight ratio swept against the
project's unweighted 1:1 default. Only the ratio matters (D20): RRF's score is a sum of
`weight / (rank_constant + rank)`, so scaling both weights together changes nothing.

**Provenance** · corpus `{golden.corpus_ref}` · depth {depth} · bootstrap {DEFAULT_RESAMPLES:,}
resamples, seed {DEFAULT_SEED}

{table}

`*` marks a paired-bootstrap interval on Recall@5 that excludes zero against the 1:1
baseline -- the only circumstance under which a difference here is a claim rather than a
hint.
"""

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rrf_weights.md"
    out_path.write_text(report, encoding="utf-8")
    typer.echo(f"{elapsed()} wrote {out_path}")

    if isinstance(inner, OpenAIEmbedder):
        typer.echo(
            f"{elapsed()} spend: {inner.requests_made} request(s), "
            f"{inner.tokens_used:,} tokens, ${inner.estimated_cost_usd:.6f}"
        )
    cached.close()
    store.close()


if __name__ == "__main__":
    sys.exit(app())
