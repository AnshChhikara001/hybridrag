"""Evaluation: the golden set, retrieval metrics, and the comparison report."""

from __future__ import annotations

from hybridrag.evaluation.golden import (
    AnswerSpan,
    GoldenQuestion,
    GoldenSet,
    LocatedSpan,
    QuestionCategory,
    SpanNotFoundError,
    covered_ratio,
    is_hit,
)
from hybridrag.evaluation.harness import (
    DEFAULT_DEPTH,
    METRICS,
    ArmResult,
    RefusalProbe,
    answerable,
    build_pools,
    locate_all,
    run_arm,
)
from hybridrag.evaluation.metrics import (
    DEFAULT_CUTOFFS,
    DEFAULT_TOKEN_BUDGET,
    QuestionMetrics,
    coverage,
    evaluate_question,
    is_covered,
)
from hybridrag.evaluation.report import (
    ChunkStats,
    GridResult,
    Provenance,
    provenance,
    render,
)
from hybridrag.evaluation.stats import Delta, Interval, bootstrap_ci, paired_delta

__all__ = [
    "DEFAULT_CUTOFFS",
    "DEFAULT_DEPTH",
    "DEFAULT_TOKEN_BUDGET",
    "METRICS",
    "AnswerSpan",
    "ArmResult",
    "ChunkStats",
    "Delta",
    "GoldenQuestion",
    "GoldenSet",
    "GridResult",
    "Interval",
    "LocatedSpan",
    "Provenance",
    "QuestionCategory",
    "QuestionMetrics",
    "RefusalProbe",
    "SpanNotFoundError",
    "answerable",
    "bootstrap_ci",
    "build_pools",
    "coverage",
    "covered_ratio",
    "evaluate_question",
    "is_covered",
    "is_hit",
    "locate_all",
    "paired_delta",
    "provenance",
    "render",
    "run_arm",
]
