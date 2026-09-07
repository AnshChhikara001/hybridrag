"""Tier 2: answer quality, and where a bad answer actually went wrong.

Tier 1 measures whether the right text was retrieved. This measures whether the answer
built from it is right, grounded, and honestly cited -- and, when it is not, which half of
the system to blame.

That attribution is the point of joining the two tiers. "34% of answers are wrong" is not
actionable; "of the wrong answers, two thirds had the answer text in context and ignored
it" says to work on the prompt, while the reverse says to work on the retriever. The
classification below is therefore computed against Tier 1's own record of what each arm
retrieved, not guessed at.

Three of the four numbers here come from a judge whose agreement with a human was measured
first (D37). The fourth -- citation fabrication -- is deterministic: a citation either
resolves to a block that was in the prompt or it does not, and no model opinion is involved.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from statistics import fmean, median

from pydantic import BaseModel, Field

from hybridrag.evaluation.golden import GoldenQuestion, QuestionCategory
from hybridrag.evaluation.judge import Verdict
from hybridrag.models import ChunkingStrategy

# The metric keys the tables and paired comparisons address, in report order.
QUALITY_METRICS: tuple[str, ...] = ("correct", "correct_or_partial", "grounded", "cited_honestly")


class FailureKind(StrEnum):
    """Why an answer was not correct, attributed to a stage of the pipeline."""

    MISSING_REFUSAL = "missing_refusal"
    RETRIEVAL_MISS = "retrieval_miss"
    WRONG_REFUSAL = "wrong_refusal"
    IGNORED_CONTEXT = "ignored_context"


class AnswerRecord(BaseModel):
    """One question answered by one arm, judged and classified."""

    question_id: str
    category: QuestionCategory
    answered: bool = Field(description="False when either refusal path fired.")
    verdict: Verdict
    grounded: bool | None = None
    judge_reason: str = ""
    citations: int = 0
    fabricated_citations: int = Field(
        default=0, description="Bracketed numbers pointing at no context block."
    )
    span_retrieved: bool | None = Field(
        default=None,
        description="Whether Tier 1 found this question's answer span in the top 10 for "
        "this same arm. None for no-answer questions, which have no span.",
    )
    failure: FailureKind | None = None
    generation_cost_usd: float = Field(
        default=0.0, description="What this run actually paid. Zero on a cache hit."
    )
    modelled_cost_usd: float = Field(
        default=0.0,
        description="What producing this answer costs, priced from its token counts. "
        "Unlike the field above it does not collapse to zero on a cache hit, so unit "
        "economics survive a free re-run.",
    )
    judge_cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False

    @property
    def cited_honestly(self) -> bool:
        """No citation points at a block that was never in the prompt."""
        return self.fabricated_citations == 0


def classify(
    record_verdict: Verdict,
    *,
    answered: bool,
    category: QuestionCategory,
    span_retrieved: bool | None,
) -> FailureKind | None:
    """Attribute a non-correct answer to the stage that caused it.

    Order matters, and it is the reverse of what feels natural. A refusal on a question
    whose answer was never retrieved is **not** a wrong refusal -- refusing was the right
    response to the context the model was given, and blaming generation for it would hide
    a retrieval failure behind a prompt problem. So retrieval is checked first wherever
    Tier 1 has an opinion.
    """
    if record_verdict is Verdict.CORRECT:
        return None
    if category is QuestionCategory.NO_ANSWER:
        # Answering something unanswerable is the expected failure here. The other branch
        # means the system declined and the judge still marked it wrong -- a disagreement
        # about the wording of the decline, and still a failure, so it is attributed rather
        # than silently dropped. Every non-correct answer must land in some column, or the
        # failure table quietly stops summing to the error count.
        return FailureKind.MISSING_REFUSAL if answered else FailureKind.WRONG_REFUSAL
    if span_retrieved is False:
        return FailureKind.RETRIEVAL_MISS
    if not answered:
        return FailureKind.WRONG_REFUSAL
    return FailureKind.IGNORED_CONTEXT


class QualityArm(BaseModel):
    """Everything one arm produced, per question and in aggregate."""

    retriever: str
    strategy: ChunkingStrategy
    judge_model: str
    generation_model: str
    records: list[AnswerRecord] = Field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.retriever}/{self.strategy.value}"

    def subset(self, *, category: QuestionCategory | None = None) -> list[AnswerRecord]:
        if category is None:
            return self.records
        return [record for record in self.records if record.category is category]

    def answerable(self) -> list[AnswerRecord]:
        return [
            record for record in self.records if record.category is not QuestionCategory.NO_ANSWER
        ]

    def series(self, metric: str, records: Sequence[AnswerRecord] | None = None) -> list[float]:
        """One score per question, in golden-set order, for the paired bootstrap."""
        chosen = self.records if records is None else records
        return [_score(record, metric) for record in chosen]

    def question_ids(self) -> list[str]:
        return [record.question_id for record in self.records]

    @property
    def total_cost_usd(self) -> float:
        return sum(r.generation_cost_usd + r.judge_cost_usd for r in self.records)

    @property
    def modelled_cost_per_answer(self) -> float:
        """What one answer costs to produce, priced from tokens.

        Averaging what the run *paid* prices an arm by how much of it happened to be
        cached: in an earlier draft two arms running the identical model differed by 40%
        for that reason, and after the whole run went to cache both read $0.000000.
        """
        if not self.records:
            return 0.0
        return fmean(record.modelled_cost_usd for record in self.records)

    @property
    def live_answers(self) -> int:
        """How many answers this run actually generated, the rest being cache hits."""
        return sum(1 for record in self.records if not record.cached)

    @property
    def explicit_refusals(self) -> int:
        """Answers that fired a refusal path, by the sentinel or the confidence gate.

        Not the same as declining: a model can decline in prose without emitting the
        sentinel, which this counts as an answer. Measured on `no_answer-007`, where the
        reply said the documentation contains only specific CVE references and the judge
        correctly read it as a decline.
        """
        return sum(1 for record in self.records if not record.answered)

    @property
    def median_latency_s(self) -> float:
        """Live calls only. Including cache hits would report the cache, not the system."""
        live = [record.latency_s for record in self.records if not record.cached]
        return median(live) if live else 0.0

    def failures(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            if record.failure is not None:
                counts[record.failure.value] = counts.get(record.failure.value, 0) + 1
        return counts

    def refusal_rate(self, category: QuestionCategory) -> float:
        records = self.subset(category=category)
        if not records:
            return 0.0
        return fmean(float(not record.answered) for record in records)


def _score(record: AnswerRecord, metric: str) -> float:
    if metric == "correct":
        return float(record.verdict is Verdict.CORRECT)
    if metric == "correct_or_partial":
        # Reported beside `correct` rather than instead of it: partial credit flatters a
        # system, and a single blended number would hide which of the two moved.
        return float(record.verdict in (Verdict.CORRECT, Verdict.PARTIAL))
    if metric == "grounded":
        return float(bool(record.grounded))
    if metric == "cited_honestly":
        return float(record.cited_honestly)
    raise ValueError(f"unknown metric {metric!r}. Known: {QUALITY_METRICS}")


def reference_for(question: GoldenQuestion) -> str:
    """The judge's standard: the human answer, empty where the corpus has none."""
    return "" if question.category is QuestionCategory.NO_ANSWER else question.answer
