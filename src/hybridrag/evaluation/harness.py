"""Runs one arm of the evaluation grid and holds its results.

An *arm* is one (chunking strategy x retriever) pair -- nine of them in the full grid.
Every arm is scored on the same questions with the same code, and the retrievers are
derived from one another with `HybridRetriever.ablation`, so the only thing that varies
between two arms is the thing the comparison names.

The harness owns three jobs the metrics module deliberately does not: fetching rankings
from a live index, building each question's *ideal pool* from the chunk store, and keeping
every arm's per-question scores aligned so the paired bootstrap can compare them.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from statistics import median

from pydantic import BaseModel, Field

from hybridrag.evaluation.golden import (
    GoldenQuestion,
    GoldenSet,
    LocatedSpan,
    QuestionCategory,
)
from hybridrag.evaluation.metrics import (
    DEFAULT_CUTOFFS,
    DEFAULT_TOKEN_BUDGET,
    QuestionMetrics,
    evaluate_question,
)
from hybridrag.models import Chunk, ChunkingStrategy, Document
from hybridrag.retrieval import Retriever
from hybridrag.retrieval.fusion import RetrieverHit

# How deep each arm retrieves. Deeper than any reported cutoff on purpose: Recall@budget
# walks the ranking until a token budget is spent, and a strategy that emits small chunks
# fits many more of them, so a depth of 10 would cut that measurement short for exactly
# the strategy it is meant to reward.
DEFAULT_DEPTH = 50

# The metric keys every table and comparison addresses, in report order.
METRICS: tuple[str, ...] = (
    *(f"recall@{k}" for k in DEFAULT_CUTOFFS),
    "mrr@10",
    "ndcg@10",
    "recall@budget",
)


class RefusalProbe(BaseModel):
    """A no-answer question's retrieval signal.

    These have no retrieval ground truth (D29), but they are not useless to Tier 1: the
    best dense similarity a question with no answer attracts is exactly what the
    pre-generation refusal gate (D23) thresholds on, and having 6 of them beside 29
    answerable ones turns that threshold from a hand-picked constant into a measurement.
    """

    question_id: str
    category: QuestionCategory
    top_dense_score: float | None = None


class ArmResult(BaseModel):
    """Everything one arm produced, per question and in aggregate."""

    retriever: str = Field(description="hybrid, dense or sparse.")
    strategy: ChunkingStrategy
    questions: list[QuestionMetrics]
    refusals: list[RefusalProbe] = Field(default_factory=list)
    answerable_scores: list[RefusalProbe] = Field(
        default_factory=list,
        description="The same dense signal for answerable questions, so the two "
        "distributions can be compared rather than one being read alone.",
    )
    median_latency_ms: float = 0.0
    depth: int = DEFAULT_DEPTH
    top5_paths: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Each answerable question's top-5 `relative_path`s, best first. "
        "Deliberately generic -- this harness has no notion of any one corpus's "
        "distractor documents -- so a caller can compute a distractor rate for whatever "
        "file it wants (`distractor_rate`) without a second pass over the retriever.",
    )

    @property
    def name(self) -> str:
        return f"{self.retriever}/{self.strategy.value}"

    @property
    def unreachable(self) -> list[str]:
        """Questions whose spans no chunk of this strategy can cover.

        A chunking failure rather than a retrieval one, and it must be reported as such:
        these questions cap the arm's ceiling below 1.0 no matter how good the ranking is.
        """
        return [q.question_id for q in self.questions if not q.reachable]

    def series(self, metric: str, *, category: QuestionCategory | None = None) -> list[float]:
        """One score per question, in golden-set order.

        Order is the contract, exactly as it is for `ChunkStore.get_many`: the paired
        bootstrap subtracts two arms element by element, so a reordering silently compares
        one arm's easy question against another arm's hard one.
        """
        records = self.questions
        if category is not None:
            records = [record for record in records if record.category is category]
        return [_score(record, metric) for record in records]

    def question_ids(self, *, category: QuestionCategory | None = None) -> list[str]:
        records = self.questions
        if category is not None:
            records = [record for record in records if record.category is category]
        return [record.question_id for record in records]

    def misses(self, k: int = 10) -> list[str]:
        """Questions this arm failed to retrieve by rank k -- the failure list."""
        return [q.question_id for q in self.questions if not q.hits.get(k, False)]


def _score(record: QuestionMetrics, metric: str) -> float:
    if metric.startswith("recall@"):
        suffix = metric.removeprefix("recall@")
        if suffix == "budget":
            return float(record.hit_at_budget)
        cutoff = int(suffix)
        if cutoff not in record.hits:
            # A cutoff that was never measured must not be readable as a score: the arms
            # would agree on a number none of them computed.
            raise ValueError(
                f"unknown metric {metric!r}: this run scored cutoffs "
                f"{sorted(record.hits)}. Re-run with that cutoff to report it."
            )
        return float(record.hits[cutoff])
    if metric.startswith("mrr"):
        return record.reciprocal_rank
    if metric.startswith("ndcg"):
        return record.ndcg
    raise ValueError(f"unknown metric {metric!r}. Known: {METRICS}")


def locate_all(
    golden: GoldenSet, documents: Mapping[str, Document]
) -> dict[str, list[LocatedSpan]]:
    """Resolve every question's quotes to offsets, failing loudly on the first that cannot.

    Resolution happens once for the whole grid rather than per arm: the spans describe the
    corpus, not the index, so re-resolving nine times would only create nine chances for
    the arms to disagree about where an answer is.
    """
    return {question.question_id: question.locate(documents) for question in golden.verified()}


def build_pools(
    chunks: Iterable[Chunk], located: Mapping[str, Sequence[LocatedSpan]]
) -> dict[str, list[Chunk]]:
    """Every chunk that touches each question's answer spans.

    This is nDCG's ideal set, and it must come from the index rather than from what an arm
    retrieved: normalising against an arm's own results would score a retriever that found
    one of three needed chunks as perfect for ordering that one correctly.

    Built in a single pass over the store, because the alternative -- a scan per question
    -- is 29 passes over 1,892 chunks for the same answer.
    """
    by_path: dict[str, list[tuple[str, LocatedSpan]]] = {}
    for question_id, spans in located.items():
        for span in spans:
            by_path.setdefault(span.relative_path, []).append((question_id, span))

    pools: dict[str, list[Chunk]] = {question_id: [] for question_id in located}
    for chunk in chunks:
        for question_id, span in by_path.get(chunk.relative_path, ()):
            if chunk.overlap_chars(span.start_char, span.end_char) > 0:
                pools[question_id].append(chunk)
    return pools


def _top_dense(hits: Mapping[str, RetrieverHit]) -> float | None:
    """The raw dense cosine behind one result, if this arm has a dense index at all."""
    hit = hits.get("dense")
    return hit.score if hit is not None else None


def run_arm(
    retriever_name: str,
    strategy: ChunkingStrategy,
    retriever: Retriever,
    golden: GoldenSet,
    located: Mapping[str, Sequence[LocatedSpan]],
    pools: Mapping[str, Sequence[Chunk]],
    *,
    depth: int = DEFAULT_DEPTH,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    ndcg_at: int = 10,
    budget: int = DEFAULT_TOKEN_BUDGET,
    min_ratio: float = 1.0,
) -> ArmResult:
    """Retrieve for every verified question and score it. No LLM, no cost.

    `ndcg_at` also sets the horizon nDCG and MRR are read at (`evaluate_question`'s
    `ndcg_at`) -- it defaults to 10 to match every existing report, but the reranker
    comparison scores its arms at 5, matching what the reranker actually keeps.
    """
    scored: list[QuestionMetrics] = []
    refusals: list[RefusalProbe] = []
    answerable: list[RefusalProbe] = []
    latencies: list[float] = []
    top5_paths: dict[str, list[str]] = {}

    for question in golden.verified():
        started = time.perf_counter()
        results = retriever.retrieve(question.question, k=depth)
        latencies.append((time.perf_counter() - started) * 1000.0)
        dense_scores = [
            score for score in (_top_dense(result.hits) for result in results) if score is not None
        ]
        top_dense = max(dense_scores, default=None)
        probe = RefusalProbe(
            question_id=question.question_id,
            category=question.category,
            top_dense_score=top_dense,
        )
        if question.category is QuestionCategory.NO_ANSWER:
            refusals.append(probe)
            continue
        answerable.append(probe)
        top5_paths[question.question_id] = [result.chunk.relative_path for result in results[:5]]
        scored.append(
            evaluate_question(
                question,
                located[question.question_id],
                [result.chunk for result in results],
                pools.get(question.question_id, ()),
                cutoffs=cutoffs,
                ndcg_at=ndcg_at,
                budget=budget,
                min_ratio=min_ratio,
                top_dense_score=top_dense,
            )
        )

    return ArmResult(
        retriever=retriever_name,
        strategy=strategy,
        questions=scored,
        refusals=refusals,
        answerable_scores=answerable,
        median_latency_ms=median(latencies) if latencies else 0.0,
        depth=depth,
        top5_paths=top5_paths,
    )


def distractor_rate(arms: Iterable[ArmResult], *, filename: str) -> float:
    """Share of top-5 slots, across every question in every given arm, filled by `filename`.

    Deliberately takes a filename rather than knowing one: this harness has no notion of
    which document in a corpus is a distractor. Accepts an iterable of arms rather than one,
    because the natural unit to report this over is often several strategies or several
    questions' worth of slots at once, not a single arm scored in isolation.
    """
    slots = [path for arm in arms for paths in arm.top5_paths.values() for path in paths]
    if not slots:
        raise ValueError("cannot compute a distractor rate over zero top-5 slots")
    return sum(1 for path in slots if path == filename) / len(slots)


def answerable(golden: GoldenSet) -> list[GoldenQuestion]:
    """Verified questions that have retrieval ground truth."""
    return [q for q in golden.verified() if q.category is not QuestionCategory.NO_ANSWER]
