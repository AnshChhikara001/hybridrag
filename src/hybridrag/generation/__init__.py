"""Grounded answer generation with inline citations."""

from __future__ import annotations

from hybridrag.generation.answerer import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    Answerer,
    retrieval_confidence,
)
from hybridrag.generation.base import Completion, GenerationError, LanguageModel
from hybridrag.generation.cache import CachedLanguageModel
from hybridrag.generation.citations import parse_citation_numbers, resolve_citations
from hybridrag.generation.claims import Claim, split_claims, structural_coverage
from hybridrag.generation.confidence import (
    CALIBRATION_CEILING,
    CALIBRATION_FLOOR,
    Completeness,
    CompletenessScorer,
    CompositeScore,
    ConfidenceComponent,
    ConfidenceInputs,
    calibrated_retrieval,
    composite_confidence,
)
from hybridrag.generation.gemini import DEFAULT_MODEL, GeminiModel
from hybridrag.generation.models import (
    NO_ANSWER_TEXT,
    Answer,
    AnswerConfidence,
    Citation,
    CitationReport,
    NearMiss,
    Refusal,
    RefusalKind,
)
from hybridrag.generation.openai_model import BudgetExceededError, OpenAIModel
from hybridrag.generation.prompt import (
    REFUSAL_SENTINEL,
    SYSTEM_PROMPT,
    build_prompt,
    render_context,
)
from hybridrag.generation.refusal import build_refusal
from hybridrag.generation.verification import (
    CitationVerifier,
    ClaimVerification,
    SupportVerdict,
    VerificationReport,
)

__all__ = [
    "CALIBRATION_CEILING",
    "CALIBRATION_FLOOR",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_MODEL",
    "NO_ANSWER_TEXT",
    "REFUSAL_SENTINEL",
    "SYSTEM_PROMPT",
    "Answer",
    "AnswerConfidence",
    "Answerer",
    "BudgetExceededError",
    "CachedLanguageModel",
    "Citation",
    "CitationReport",
    "CitationVerifier",
    "Claim",
    "ClaimVerification",
    "Completeness",
    "CompletenessScorer",
    "Completion",
    "CompositeScore",
    "ConfidenceComponent",
    "ConfidenceInputs",
    "GeminiModel",
    "GenerationError",
    "LanguageModel",
    "NearMiss",
    "OpenAIModel",
    "Refusal",
    "RefusalKind",
    "SupportVerdict",
    "VerificationReport",
    "build_prompt",
    "build_refusal",
    "calibrated_retrieval",
    "composite_confidence",
    "parse_citation_numbers",
    "render_context",
    "resolve_citations",
    "retrieval_confidence",
    "split_claims",
    "structural_coverage",
]
