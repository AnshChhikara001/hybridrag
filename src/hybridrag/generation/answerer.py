"""Retrieve, ground, generate, verify: the whole question-to-answer path.

Two independent refusal paths, deliberately, because they catch different failures:

* **The retrieval gate**, before any generation. Cheap, deterministic, and the only one
  that can refuse without spending a request. It catches out-of-domain questions, where
  nothing in the corpus is close to the query.
* **The model's own refusal**, via the sentinel in the system prompt. It catches
  in-domain questions the corpus happens not to answer -- where retrieval returns
  plausibly-close chunks that do not contain the fact asked for. The gate cannot see this;
  only something that has read the chunks can.

That second path carries more weight than its description suggests. Measured over the
golden set, the six unanswerable questions retrieved top cosines of 0.379-0.640, sitting
inside the answerable range rather than below it, so the gate lets essentially all of them
through. It is the model's refusal that catches them, and the structured refusal both
paths return is what makes the outcome useful rather than merely correct.

The gate reads *dense cosine similarity*, never the fused RRF score. RRF is computed from
ranks alone, so its top result scores about 1/(k+1) whether that chunk is perfect or
useless -- a threshold on it would be a confidence number that moves for no reason.

**Verification is opt-in.** Claim-level citation checking and the completeness score each
cost a model call, which is right for an evaluation run and wrong for a dashboard that
must answer in under a second. Without them the answer still carries a composite, built
from what is free; `confidence.components` records which parts were actually measured, so
the cheap path is never mistaken for the thorough one.
"""

from __future__ import annotations

from hybridrag.generation.base import LanguageModel
from hybridrag.generation.citations import resolve_citations
from hybridrag.generation.claims import split_claims, structural_coverage
from hybridrag.generation.confidence import (
    CompletenessScorer,
    ConfidenceInputs,
    calibrated_retrieval,
)
from hybridrag.generation.models import (
    NO_ANSWER_TEXT,
    Answer,
    AnswerConfidence,
    Refusal,
    RefusalKind,
)
from hybridrag.generation.prompt import REFUSAL_SENTINEL, SYSTEM_PROMPT, build_prompt
from hybridrag.generation.refusal import build_refusal
from hybridrag.generation.verification import CitationVerifier
from hybridrag.retrieval import HybridRetriever, RetrievedChunk

DENSE = "dense"

# Measured on this corpus over 6 in-domain and 6 out-of-domain questions: in-domain top
# cosine ran 0.4602-0.6541, out-of-domain 0.1410-0.2414, a gap of 0.219. Set below the
# midpoint deliberately, because the two errors are not symmetric: a false refusal is final
# and the user simply gets nothing, while a false accept is caught downstream by the
# model's own refusal at the cost of one request.
#
# Re-measured against the golden set in Phase 4, which is the honest test: out-of-domain
# questions are the easy case, and questions that are *in*-domain but unanswered by the
# corpus score 0.379-0.640 -- indistinguishable from answerable ones. No threshold on this
# number separates them, so the gate is kept for the out-of-domain case it does handle and
# is not asked to do more than that.
DEFAULT_CONFIDENCE_THRESHOLD = 0.30


def retrieval_confidence(results: list[RetrievedChunk]) -> float | None:
    """Best dense cosine similarity across the retrieved chunks.

    None when no chunk carries a dense score, which means the retriever had no dense
    component. BM25 scores are unbounded and corpus-dependent, so there is no honest way
    to convert them into a confidence comparable across queries -- and inventing one is
    how a dashboard ends up displaying a number nobody can interpret.
    """
    scores = [result.hits[DENSE].score for result in results if DENSE in result.hits]
    return max(scores) if scores else None


class Answerer:
    """Answers a question from the corpus, with citations."""

    def __init__(
        self,
        retriever: HybridRetriever,
        model: LanguageModel,
        *,
        k: int = 5,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_output_tokens: int | None = None,
        verifier: CitationVerifier | None = None,
        completeness_scorer: CompletenessScorer | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.retriever = retriever
        self.model = model
        self.k = k
        self.confidence_threshold = confidence_threshold
        self.max_output_tokens = max_output_tokens
        self.verifier = verifier
        self.completeness_scorer = completeness_scorer

    def answer(self, question: str) -> Answer:
        """Retrieve context, generate a grounded answer, and check its citations."""
        if not question.strip():
            raise ValueError("question must not be empty")

        results = self.retriever.retrieve(question, k=self.k)
        cosine = retrieval_confidence(results)

        gate = self._gate(results, cosine)
        if gate is not None:
            kind, reason = gate
            return self._refuse(question, kind, reason, results, cosine)

        completion = self.model.generate(
            build_prompt(question, results),
            system=SYSTEM_PROMPT,
            max_output_tokens=self.max_output_tokens,
        )

        # The sentinel may arrive with punctuation or stray whitespace around it; anything
        # longer than the sentinel plus a little is the model answering *and* hedging,
        # which is an answer, not a refusal.
        stripped = completion.text.strip().strip(".\"'` ")
        if stripped == REFUSAL_SENTINEL:
            declined = self._refuse(
                question,
                RefusalKind.MODEL_DECLINED,
                "The model judged the retrieved context insufficient to answer.",
                results,
                cosine,
            )
            return declined.model_copy(
                update={
                    "model": completion.model,
                    "input_tokens": completion.input_tokens,
                    "output_tokens": completion.output_tokens + completion.thinking_tokens,
                    "cost_usd": completion.cost_usd,
                    "latency_s": completion.latency_s,
                    "cached": completion.cached,
                }
            )

        report = resolve_citations(completion.text, results)
        claims = split_claims(completion.text)

        # Verified coverage when a verifier is attached, structural otherwise. The two are
        # not interchangeable and `citation_coverage_verified` says which one this is.
        verified = self.verifier.verify(claims, results) if self.verifier else None
        coverage = verified.verified_coverage if verified else structural_coverage(claims)

        scored = (
            self.completeness_scorer.score(question, completion.text)
            if self.completeness_scorer
            else None
        )

        inputs = ConfidenceInputs(
            retrieval_cosine=cosine,
            citation_coverage=coverage,
            citation_coverage_verified=verified is not None,
            completeness=scored.score if scored else None,
        )
        composite = inputs.composite()

        return Answer(
            question=question,
            text=completion.text,
            answered=True,
            citations=report.citations,
            unresolved_citations=report.unresolved,
            uncited_blocks=report.uncited_blocks,
            claims=len(claims),
            unsupported_citations=(
                [item.claim_index for item in verified.unsupported] if verified else []
            ),
            retrieved=results,
            confidence=AnswerConfidence(
                retrieval=cosine,
                retrieval_calibrated=calibrated_retrieval(cosine),
                both_retrievers_agree=len(results[0].hits) > 1 if results else None,
                citation_coverage=coverage,
                citation_coverage_verified=verified is not None,
                citation_precision=verified.citation_precision if verified else None,
                completeness=scored.score if scored else None,
                composite=composite.value if composite else None,
                components=tuple(c.value for c in composite.components) if composite else (),
            ),
            model=completion.model,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens + completion.thinking_tokens,
            cost_usd=completion.cost_usd,
            verification_cost_usd=(verified.cost_usd if verified else 0.0)
            + (scored.cost_usd if scored else 0.0),
            latency_s=completion.latency_s,
            cached=completion.cached,
        )

    def _refuse(
        self,
        question: str,
        kind: RefusalKind,
        reason: str,
        results: list[RetrievedChunk],
        cosine: float | None,
    ) -> Answer:
        """A refusal, with the structure that makes it actionable.

        The composite here is built from retrieval alone. Coverage and completeness are
        undefined for an answer that was never given, and scoring them zero would report a
        low composite as if the answer had been checked and found wanting.
        """
        refusal: Refusal = build_refusal(kind, question, reason, results)
        composite = ConfidenceInputs(retrieval_cosine=cosine).composite()
        return Answer(
            question=question,
            text=refusal.render(),
            answered=False,
            refusal_reason=reason,
            refusal=refusal,
            retrieved=results,
            confidence=AnswerConfidence(
                retrieval=cosine,
                retrieval_calibrated=calibrated_retrieval(cosine),
                both_retrievers_agree=len(results[0].hits) > 1 if results else None,
                composite=composite.value if composite else None,
                components=tuple(c.value for c in composite.components) if composite else (),
            ),
            model=self.model.model_name,
        )

    def _gate(
        self, results: list[RetrievedChunk], confidence: float | None
    ) -> tuple[RefusalKind, str] | None:
        """The pre-generation refusal, or None to proceed."""
        if not results:
            return RefusalKind.NO_RESULTS, "Retrieval returned no chunks for this question."
        if confidence is None:
            # Sparse-only retrieval. Refusing here would reject every query on a retriever
            # that simply has no dense side, so the model's own refusal carries this case.
            return None
        if confidence < self.confidence_threshold:
            return RefusalKind.LOW_CONFIDENCE, (
                f"Retrieval confidence {confidence:.3f} is below the threshold "
                f"{self.confidence_threshold:.2f}: nothing in the corpus is close to "
                "this question."
            )
        return None


__all__ = ["DEFAULT_CONFIDENCE_THRESHOLD", "NO_ANSWER_TEXT", "Answerer", "retrieval_confidence"]
