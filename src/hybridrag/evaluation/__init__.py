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

__all__ = [
    "AnswerSpan",
    "GoldenQuestion",
    "GoldenSet",
    "LocatedSpan",
    "QuestionCategory",
    "SpanNotFoundError",
    "covered_ratio",
    "is_hit",
]
