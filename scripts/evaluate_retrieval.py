"""Run the Tier-1 retrieval grid: every chunking strategy x every retriever.

Nine arms, one golden set, no language model -- so the whole thing is deterministic, costs
nothing beyond a handful of cached query embeddings, and can be re-run on every change to
retrieval to see what moved.

    uv run python scripts/evaluate_retrieval.py data/raw/fastapi/docs/en/docs
    uv run python scripts/evaluate_retrieval.py <corpus> --strategy structure   # one arm set
    uv run python scripts/evaluate_retrieval.py <corpus> --rerank               # + reranker.md

Every strategy named must already be indexed. A missing index fails the run rather than
being skipped: an arm silently absent from a comparison table is worse than no table.

`--rerank` adds a second, separate report (`reranker.md`): a cross-encoder rerank pass
(D8) against each strategy's plain-hybrid baseline, both rescored at a 5-wide horizon.
Still $0 and still no language model -- the cross-encoder is local -- but it downloads
~80 MB on a cold cache, which is why it is opt-in rather than run by default.
"""

from __future__ import annotations

import json
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
    ArmResult,
    ChunkStats,
    GoldenSet,
    GridResult,
    RerankerRun,
    answerable,
    build_pools,
    distractor_rate,
    locate_all,
    provenance,
    render,
    render_reranker_report,
    run_arm,
)
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
from hybridrag.retrieval.fusion import DEFAULT_RANK_CONSTANT
from hybridrag.retrieval.hybrid import DEFAULT_CANDIDATES
from hybridrag.retrieval.rerank import (
    DEFAULT_RERANK_DEPTH,
    DEFAULT_RERANK_MODEL,
    CrossEncoderReranker,
    RerankingRetriever,
)

app = typer.Typer(add_completion=False)

# The retriever arms. Each ablation is derived from the hybrid retriever rather than built
# separately, so the three differ in exactly one variable: which indexes they can see.
ARMS: tuple[str, ...] = ("hybrid", "dense", "sparse")


def _embedder() -> Embedder:
    """Whatever built the index must also embed the query, or the vectors do not compare."""
    settings = get_settings()
    if settings.openai_api_key is None:
        return FastEmbedEmbedder(settings.embedding_model)
    return OpenAIEmbedder(settings.openai_api_key)


def _load_corpus(corpus: Path) -> dict[str, Document]:
    """Load the corpus exactly as the builder did.

    Golden spans are character offsets into `Document.text`, and include expansion shifts
    those offsets by hundreds of lines, so a different include root here would silently
    point every span at the wrong text (D22, D30).
    """
    loader = CorpusLoader(corpus, include_root=default_include_root(corpus))
    documents = {document.relative_path: document for document in loader.iter_documents()}
    if loader.missing_includes:
        typer.echo(f"  WARNING: {len(loader.missing_includes)} include(s) did not resolve")
    return documents


def _verify_arm(
    strategy: ChunkingStrategy, store: ChunkStore, dense: DenseIndex, sparse: SparseIndex
) -> None:
    """Refuse to score an arm whose three artefacts disagree about what is indexed.

    A half-built arm does not error, it just retrieves less -- and would appear in the
    comparison as a chunking strategy that performs badly.
    """
    expected = store.chunk_ids(strategy)
    if not expected:
        raise typer.BadParameter(f"the chunk store holds no {strategy.value} chunks")
    for name, ids in (("dense index", dense.chunk_ids()), ("sparse index", set(sparse.chunk_ids))):
        missing, extra = expected - ids, ids - expected
        if missing or extra:
            typer.echo(
                f"error: {strategy.value} {name} is missing {len(missing)} and holds "
                f"{len(extra)} id(s) the chunk store does not. Rebuild it:\n"
                f"  uv run python scripts/build_index.py <corpus> --strategy {strategy.value}"
            )
            raise typer.Exit(code=1)


@app.command()
def evaluate(
    corpus: Annotated[Path, typer.Argument(help="Corpus directory the index was built from.")],
    strategy: Annotated[
        list[ChunkingStrategy] | None,
        typer.Option(help="Restrict the grid. Repeatable; all three by default."),
    ] = None,
    golden_path: Annotated[Path, typer.Option(help="The golden set.")] = Path(
        "evals/golden_set.yaml"
    ),
    out_dir: Annotated[Path, typer.Option(help="Where the report is written.")] = Path(
        "evals/reports"
    ),
    depth: Annotated[int, typer.Option(help="How deep each arm retrieves.")] = DEFAULT_DEPTH,
    budget: Annotated[
        int, typer.Option(help="Context budget for Recall@budget, in tokens.")
    ] = DEFAULT_TOKEN_BUDGET,
    min_ratio: Annotated[
        float, typer.Option(help="Span fraction that must be covered to count as retrieved.")
    ] = 1.0,
    index_dir: Annotated[Path | None, typer.Option(help="Overrides the configured path.")] = None,
    rerank: Annotated[
        bool,
        typer.Option(
            help="Also score a cross-encoder rerank pass (D8) against plain hybrid, "
            "per strategy, at a 5-wide horizon. Writes reranker.md separately."
        ),
    ] = False,
    rerank_model: Annotated[
        str, typer.Option(help="fastembed cross-encoder model id.")
    ] = DEFAULT_RERANK_MODEL,
    rerank_depth: Annotated[
        int, typer.Option(help="Candidates fetched and reranked before keeping the top 5.")
    ] = DEFAULT_RERANK_DEPTH,
    distractor_filename: Annotated[
        str,
        typer.Option(help="Corpus file whose top-5 share is reported as the distractor rate."),
    ] = "release-notes.md",
) -> None:
    """Score every (chunking strategy x retriever) arm against the golden set."""
    settings = get_settings()
    root = index_dir or settings.index_dir
    strategies = strategy or list(ChunkingStrategy)
    started = time.perf_counter()

    def elapsed() -> str:
        return f"[{time.perf_counter() - started:6.1f}s]"

    if not corpus.is_dir():
        typer.echo(f"error: {corpus} is not a directory. Run scripts/fetch_corpus.sh first.")
        raise typer.Exit(code=1)

    golden = GoldenSet.load(golden_path)
    typer.echo(f"{elapsed()} golden set: {len(golden.verified())} verified questions")

    typer.echo(f"{elapsed()} loading corpus {corpus}")
    documents = _load_corpus(corpus)
    located = locate_all(golden, documents)
    spans = sum(len(value) for value in located.values())
    typer.echo(f"  {len(documents)} documents, {spans} spans resolved")

    inner = _embedder()
    cached = CachedEmbedder(inner, settings.cache_dir / "embeddings.sqlite")

    # Warmed before any arm is timed. Without this the first arm to run pays the embedding
    # round-trip for all 35 queries and every later arm reads them from cache, so the
    # latency column would report arm *order* rather than retrieval cost.
    for question in golden.verified():
        cached.embed_query(question.question)
    typer.echo(f"{elapsed()} query embeddings warmed, so latencies are comparable")

    store = ChunkStore(store_path(root))
    arms: list[ArmResult] = []
    shape: list[ChunkStats] = []
    rerank_baselines: list[ArmResult] = []
    rerank_arms: list[ArmResult] = []
    distractor_rates: dict[str, float] = {}
    # One instance shared across strategies: the ONNX session loads on first use and stays
    # loaded, so the 80 MB download and session start happen once per run, not per strategy.
    reranker = CrossEncoderReranker(rerank_model) if rerank else None

    for chunking in strategies:
        bm25_path = sparse_path(root, chunking)
        if not bm25_path.is_file():
            typer.echo(
                f"error: no {chunking.value} index at {root}. Build it first:\n"
                f"  uv run python scripts/build_index.py {corpus} --strategy {chunking.value}"
            )
            raise typer.Exit(code=1)

        dense = DenseIndex.embedded(
            cached, chroma_path(root), collection_name=collection_name(chunking)
        )
        sparse = SparseIndex.load(bm25_path)
        _verify_arm(chunking, store, dense, sparse)

        hybrid = HybridRetriever({"dense": dense, "sparse": sparse}, store)
        chunks = list(store.iter_chunks(chunking))
        pools = build_pools(chunks, located)
        tokens = sorted(chunk.token_count for chunk in chunks)
        shape.append(
            ChunkStats(
                strategy=chunking,
                chunks=len(chunks),
                mean_tokens=sum(tokens) / len(tokens),
                median_tokens=float(tokens[len(tokens) // 2]),
                p95_tokens=float(tokens[int(len(tokens) * 0.95)]),
                total_tokens=sum(tokens),
            )
        )
        # no_answer questions have no spans, so their pool is empty by definition (D29);
        # counting them here reported "6 unreachable" on every strategy, every run.
        scored_ids = {question.question_id for question in answerable(golden)}
        empty = [qid for qid, pool in pools.items() if not pool and qid in scored_ids]
        typer.echo(
            f"{elapsed()} {chunking.value}: {len(chunks)} chunks indexed, "
            f"{len(empty)} question(s) with no chunk covering their span"
        )

        for name in ARMS:
            retriever = hybrid if name == "hybrid" else hybrid.ablation(name)
            result = run_arm(
                name,
                chunking,
                retriever,
                golden,
                located,
                pools,
                depth=depth,
                budget=budget,
                min_ratio=min_ratio,
            )
            arms.append(result)
            recall = sum(result.series("recall@5")) / max(1, len(result.questions))
            typer.echo(f"  {result.name:<20} recall@5 {recall:.3f}")

        if reranker is not None:
            # Rescored at ndcg_at=5, not the grid's default 10: 5 is what the reranker
            # actually keeps (D8) and what the generator actually reads, so this is the
            # only horizon "did the reranker earn its place" can be judged at fairly.
            baseline_at5 = run_arm(
                "hybrid",
                chunking,
                hybrid,
                golden,
                located,
                pools,
                depth=depth,
                ndcg_at=5,
                budget=budget,
                min_ratio=min_ratio,
            )
            reranking_retriever = RerankingRetriever(hybrid, reranker, rerank_depth=rerank_depth)
            reranked_result = run_arm(
                "reranked",
                chunking,
                reranking_retriever,
                golden,
                located,
                pools,
                depth=5,
                ndcg_at=5,
                budget=budget,
                min_ratio=min_ratio,
            )
            rerank_baselines.append(baseline_at5)
            rerank_arms.append(reranked_result)
            distractor_rates[baseline_at5.name] = distractor_rate(
                [baseline_at5], filename=distractor_filename
            )
            distractor_rates[reranked_result.name] = distractor_rate(
                [reranked_result], filename=distractor_filename
            )
            reranked_recall = sum(reranked_result.series("recall@5")) / max(
                1, len(reranked_result.questions)
            )
            typer.echo(
                f"  {reranked_result.name:<20} recall@5 {reranked_recall:.3f} "
                f"(plain hybrid at the same horizon: "
                f"{sum(baseline_at5.series('recall@5')) / max(1, len(baseline_at5.questions)):.3f})"
            )

    grid = GridResult(
        provenance=provenance(
            golden,
            documents=len(documents),
            embedding_model=inner.model_name,
            confidence_threshold=settings.retrieval_confidence_threshold,
            chunk_tokens=settings.chunk_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
            semantic_percentile=settings.semantic_percentile,
            rank_constant=DEFAULT_RANK_CONSTANT,
            candidates=DEFAULT_CANDIDATES,
            depth=depth,
            token_budget=budget,
            min_ratio=min_ratio,
        ),
        arms=arms,
        corpus=shape,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_results.json").write_text(
        json.dumps(grid.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    (out_dir / "retrieval.md").write_text(render(grid, golden), encoding="utf-8")
    typer.echo(f"{elapsed()} wrote {out_dir / 'retrieval.md'} and retrieval_results.json")

    if rerank_arms:
        reranker_run = RerankerRun(
            provenance=grid.provenance,
            baselines=rerank_baselines,
            reranked=rerank_arms,
            distractor_filename=distractor_filename,
            distractor_rates=distractor_rates,
        )
        (out_dir / "reranker_results.json").write_text(
            json.dumps(reranker_run.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        (out_dir / "reranker.md").write_text(render_reranker_report(reranker_run), encoding="utf-8")
        typer.echo(f"{elapsed()} wrote {out_dir / 'reranker.md'} and reranker_results.json")

    if isinstance(inner, OpenAIEmbedder):
        typer.echo(
            f"{elapsed()} spend: {inner.requests_made} request(s), "
            f"{inner.tokens_used:,} tokens, ${inner.estimated_cost_usd:.6f}"
        )
    cached.close()
    store.close()


if __name__ == "__main__":
    sys.exit(app())
