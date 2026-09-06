"""Tier-1 retrieval metrics, computed from answer-span coverage.

Every metric here reads the same ground truth: an answer *span* covered by the union of
the retrieved chunks (D13), judged by the question's own category rule (D29). Nothing in
this module performs I/O or calls a model, so the whole tier is deterministic, free, and
runs in a test.

The one thing that needed inventing is graded relevance. nDCG wants a gain per position,
and span truth is binary per span -- so a chunk's gain is defined as the **coverage it
adds** to what the higher-ranked chunks already covered. A chunk that repeats text already
retrieved earns nothing, which is the behaviour an overlapping chunker must not be paid
for, and the total gain over a perfect ranking is exactly 1.0.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from statistics import fmean

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt

from hybridrag.evaluation.golden import GoldenQuestion, LocatedSpan, QuestionCategory, covered_ratio
from hybridrag.models import Chunk

# Ranks reported by default. 1 and 3 matter because the generator is given 5 blocks, so a
# span arriving at rank 9 is retrieved but never read.
DEFAULT_CUTOFFS = (1, 3, 5, 10)

# The context budget Recall@budget is measured at (D14). Roughly four 512-token chunks,
# which is the order of what a grounded prompt can carry alongside its instructions.
DEFAULT_TOKEN_BUDGET = 2000

# Coverage is float arithmetic over character counts, so exact comparison against 1.0 is
# unsafe; a span may miss by one ULP and be scored a miss.
_EPSILON = 1e-9


class QuestionMetrics(BaseModel):
    """What one question scored under one arm."""

    question_id: str
    category: QuestionCategory
    hits: dict[int, bool] = Field(description="Whether the rule is satisfied by the top-k.")
    first_hit_rank: PositiveInt | None = Field(
        default=None, description="Smallest prefix that satisfies the rule, within the depth."
    )
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    ndcg: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0, description="Span coverage at the deepest cutoff.")
    hit_at_budget: bool
    budget_chunks: NonNegativeInt = Field(description="Chunks that fit the token budget.")
    budget_tokens: NonNegativeInt
    reachable: bool = Field(
        description="Whether any set of chunks in this index could satisfy the rule. False "
        "means the strategy cannot represent the span at all, which is a chunking result, "
        "not a retrieval failure -- and it caps nDCG at 0 for reasons the ranking did not "
        "cause."
    )
    top_dense_score: float | None = Field(
        default=None, description="Best raw dense cosine, for confidence calibration."
    )


def coverage(
    category: QuestionCategory, located: Sequence[LocatedSpan], chunks: Sequence[Chunk]
) -> float:
    """How much of what the question needs these chunks contain, in [0, 1].

    The graded form of `is_hit`, and deliberately consistent with it: averaging over spans
    reaches 1.0 only when *every* span is covered, and taking the maximum reaches 1.0 when
    *any* one is -- which is precisely multi_hop's rule and ambiguous's rule (D29). One
    definition therefore serves both the binary metrics and nDCG's gains, so they cannot
    drift into disagreeing about what a hit is.
    """
    if not located:
        return 0.0
    ratios = [covered_ratio(span, chunks) for span in located]
    if category is QuestionCategory.AMBIGUOUS:
        return max(ratios)
    return fmean(ratios)


def is_covered(
    category: QuestionCategory,
    located: Sequence[LocatedSpan],
    chunks: Sequence[Chunk],
    *,
    min_ratio: float = 1.0,
) -> bool:
    """The binary rule, expressed through `coverage` so the two share one definition."""
    return coverage(category, located, chunks) >= min_ratio - _EPSILON


def marginal_gains(
    category: QuestionCategory, located: Sequence[LocatedSpan], ranked: Sequence[Chunk]
) -> list[float]:
    """Coverage each ranked chunk adds beyond everything above it.

    This is what makes nDCG meaningful on span truth. A duplicate of an already-retrieved
    passage scores zero, so the fixed-size strategy earns nothing for its 64-token overlap
    and a strategy is rewarded only for reaching *new* answer text.
    """
    gains: list[float] = []
    previous = 0.0
    for depth in range(1, len(ranked) + 1):
        current = coverage(category, located, ranked[:depth])
        gains.append(max(0.0, current - previous))
        previous = current
    return gains


def dcg(gains: Iterable[float]) -> float:
    """Discounted cumulative gain, log2 discount, 1-based positions."""
    from math import log2

    return sum(gain / log2(position + 1) for position, gain in enumerate(gains, start=1))


def ideal_gains(
    category: QuestionCategory,
    located: Sequence[LocatedSpan],
    pool: Sequence[Chunk],
    limit: int,
) -> list[float]:
    """The best gain sequence any ranking of `pool` could produce, greedily.

    `pool` is every chunk in the index that touches a span, so the ideal is computed
    against what the corpus *can* deliver rather than against what this arm happened to
    return. Normalising by the retrieved set instead would let an arm that found one
    chunk out of three score 1.0 for ordering that one chunk correctly.

    Greedy is optimal enough here: coverage is submodular, and the pools are a handful of
    chunks around a 200-character span, not a search space.
    """
    chosen: list[Chunk] = []
    remaining = list(pool)
    gains: list[float] = []
    achieved = 0.0
    while remaining and len(gains) < limit:
        best: Chunk | None = None
        best_coverage = achieved
        for candidate in remaining:
            gained = coverage(category, located, [*chosen, candidate])
            if gained > best_coverage + _EPSILON:
                best, best_coverage = candidate, gained
        if best is None:  # Nothing left adds anything.
            break
        chosen.append(best)
        remaining.remove(best)
        gains.append(best_coverage - achieved)
        achieved = best_coverage
    return gains


def budget_prefix(
    ranked: Sequence[Chunk], budget: int = DEFAULT_TOKEN_BUDGET
) -> tuple[list[Chunk], int]:
    """The top of the ranking that fits a fixed context budget, and its token count.

    Chunks are taken in rank order and one that would overflow ends the prefix, rather
    than being skipped for a smaller one further down: the generator reads a prefix, not
    a knapsack. Overlapping text is counted every time it appears, because the context
    window pays for it every time.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    taken: list[Chunk] = []
    used = 0
    for chunk in ranked:
        if used + chunk.token_count > budget:
            break
        taken.append(chunk)
        used += chunk.token_count
    return taken, used


def evaluate_question(
    question: GoldenQuestion,
    located: Sequence[LocatedSpan],
    ranked: Sequence[Chunk],
    pool: Sequence[Chunk],
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    ndcg_at: int = 10,
    budget: int = DEFAULT_TOKEN_BUDGET,
    min_ratio: float = 1.0,
    top_dense_score: float | None = None,
) -> QuestionMetrics:
    """Score one question's ranking. `ranked` is best-first; `pool` is the ideal set."""
    if question.category is QuestionCategory.NO_ANSWER:
        raise ValueError(
            f"{question.question_id}: no_answer questions have no retrieval ground truth "
            "(D29). Score them on refusal rate instead."
        )
    if not located:
        raise ValueError(f"{question.question_id}: no located spans to score against")

    category = question.category
    hits = {k: is_covered(category, located, ranked[:k], min_ratio=min_ratio) for k in cutoffs}

    first_hit: int | None = None
    for depth in range(1, len(ranked) + 1):
        if is_covered(category, located, ranked[:depth], min_ratio=min_ratio):
            first_hit = depth
            break

    ideal = ideal_gains(category, located, pool, ndcg_at)
    ideal_dcg = dcg(ideal)
    achieved_dcg = dcg(marginal_gains(category, located, ranked[:ndcg_at]))

    fitted, tokens = budget_prefix(ranked, budget)
    return QuestionMetrics(
        question_id=question.question_id,
        category=category,
        hits=hits,
        first_hit_rank=first_hit,
        # Capped at the reporting horizon: a hit at rank 34 is not a usable retrieval, and
        # letting it contribute 1/34 would flatter an arm that never puts it in context.
        reciprocal_rank=1.0 / first_hit if first_hit is not None and first_hit <= ndcg_at else 0.0,
        ndcg=(achieved_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0,
        coverage=coverage(category, located, ranked[:ndcg_at]),
        hit_at_budget=is_covered(category, located, fitted, min_ratio=min_ratio),
        budget_chunks=len(fitted),
        budget_tokens=tokens,
        reachable=ideal_dcg > 0,
        top_dense_score=top_dense_score,
    )


def mean_of(records: Sequence[QuestionMetrics], field: str) -> float:
    """Mean of one numeric field, for the aggregate tables."""
    if not records:
        return 0.0
    return fmean(float(getattr(record, field)) for record in records)


def recall_at(records: Sequence[QuestionMetrics], k: int) -> float:
    if not records:
        return 0.0
    return fmean(float(record.hits[k]) for record in records)
