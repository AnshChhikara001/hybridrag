"""Claim-level citation verification.

The verifier is a model grading a model, so these tests pin the things that must not
depend on its judgement: what gets sent, what never gets sent, how an unreadable reply is
handled, and how the two rates are counted.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from hybridrag.evaluation.report import (
    VerificationRun,
    VerifiedClaim,
    render_verification_report,
)
from hybridrag.generation.base import Completion
from hybridrag.generation.claims import Claim, split_claims
from hybridrag.generation.verification import (
    CitationVerifier,
    SupportVerdict,
    VerificationError,
    VerificationReport,
    build_verification_prompt,
    parse_support,
    render_cited_blocks,
)
from hybridrag.models import Chunk
from hybridrag.retrieval import RetrievedChunk

MakeChunk = Callable[..., Chunk]


class ScriptedModel:
    """Replies from a script, recording every prompt it was sent."""

    def __init__(self, *replies: str) -> None:
        self.model_name = "stub-verifier"
        self.replies = list(replies)
        self.prompts: list[str] = []

    @property
    def fingerprint(self) -> str:
        return f"stub|{self.model_name}"

    def generate(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int | None = None
    ) -> Completion:
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if self.replies else "SUPPORT: supported\nREASON: fine"
        return Completion(text=reply, model=self.model_name, cost_usd=0.001)


@pytest.fixture
def results(make_chunk: MakeChunk) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=make_chunk(i, f"block {i} body"), rank=i + 1, score=0.03, hits={})
        for i in range(3)
    ]


class TestParseSupport:
    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            ("SUPPORT: supported\nREASON: it says so", SupportVerdict.SUPPORTED),
            ("SUPPORT: unsupported\nREASON: absent", SupportVerdict.UNSUPPORTED),
            ("**SUPPORT:** supported\n**REASON:** bolded", SupportVerdict.SUPPORTED),
            ("support: SUPPORTED\nreason: case", SupportVerdict.SUPPORTED),
        ],
    )
    def test_readable_replies(self, reply: str, expected: SupportVerdict) -> None:
        verdict, _ = parse_support(reply)
        assert verdict is expected

    def test_reason_is_captured(self) -> None:
        _, reason = parse_support("SUPPORT: unsupported\nREASON: the passage never says it")
        assert reason == "the passage never says it"

    def test_the_bare_verdict_form_is_read(self) -> None:
        """Observed in production, mid-run: the model drops the label and writes the verdict.

        This exact reply crashed a paid verification run partway through, so it is pinned
        here rather than left to be rediscovered the next time one is paid for.
        """
        verdict, reason = parse_support(
            "UNSUPPORTED\nREASON: The cited passage discusses concurrency, not this."
        )
        assert verdict is SupportVerdict.UNSUPPORTED
        assert reason.startswith("The cited passage")

    def test_a_verdict_word_inside_prose_is_not_a_bare_verdict(self) -> None:
        """The bare form must be the whole line, or ordinary prose would parse as a verdict."""
        with pytest.raises(VerificationError):
            parse_support("I believe the claim is supported by the passage shown.")

    def test_missing_verdict_raises_rather_than_defaults(self) -> None:
        """A defaulted `supported` would inflate coverage with a judgement nobody made."""
        with pytest.raises(VerificationError, match="no SUPPORT line"):
            parse_support("I think this one is probably fine.")

    def test_unknown_verdict_raises(self) -> None:
        with pytest.raises(VerificationError, match="unknown support verdict"):
            parse_support("SUPPORT: maybe\nREASON: hedging")


class TestWhatTheVerifierSees:
    def test_only_cited_blocks_are_shown(self, results: list[RetrievedChunk]) -> None:
        """Showing the whole context would let a claim pass on a block it never cited."""
        rendered = render_cited_blocks((2,), results)
        assert "block 1 body" in rendered
        assert "block 0 body" not in rendered
        assert "block 2 body" not in rendered

    def test_prompt_carries_the_claim_and_its_blocks(self, results: list[RetrievedChunk]) -> None:
        claim = Claim(index=0, text="Workers are set with --workers [2].", citations=(2,))
        prompt = build_verification_prompt(claim, results)
        assert "--workers" in prompt
        assert "block 1 body" in prompt

    def test_multiple_citations_are_shown_together(self, results: list[RetrievedChunk]) -> None:
        """Judged jointly: evidence split across two blocks is legitimate."""
        rendered = render_cited_blocks((1, 3), results)
        assert "block 0 body" in rendered
        assert "block 2 body" in rendered


class TestVerify:
    def test_uncited_claims_are_never_sent(self, results: list[RetrievedChunk]) -> None:
        model = ScriptedModel("SUPPORT: supported\nREASON: yes")
        claims = split_claims("Cited claim here [1]. This one has no source at all.")
        report = CitationVerifier(model).verify(claims, results)
        assert len(model.prompts) == 1
        assert report.total_claims == 2
        assert report.cited_claims == 1

    def test_uncited_claims_still_lower_coverage(self, results: list[RetrievedChunk]) -> None:
        model = ScriptedModel("SUPPORT: supported\nREASON: yes")
        claims = split_claims("Cited claim here [1]. This one has no source at all.")
        report = CitationVerifier(model).verify(claims, results)
        assert report.verified_coverage == 0.5
        assert report.citation_precision == 1.0

    def test_unsupported_claims_are_flagged(self, results: list[RetrievedChunk]) -> None:
        model = ScriptedModel(
            "SUPPORT: supported\nREASON: yes", "SUPPORT: unsupported\nREASON: absent"
        )
        claims = split_claims("First claim [1]. Second claim [2].")
        report = CitationVerifier(model).verify(claims, results)
        assert [item.claim_index for item in report.unsupported] == [1]
        assert report.verified_coverage == 0.5

    def test_a_claim_citing_only_fabricated_blocks_costs_nothing(
        self, results: list[RetrievedChunk]
    ) -> None:
        """Unsupported by construction: there is no passage to send."""
        model = ScriptedModel()
        claims = split_claims("Asserted from nowhere [9].")
        report = CitationVerifier(model).verify(claims, results)
        assert model.prompts == []
        assert report.unsupported[0].verdict is SupportVerdict.UNSUPPORTED
        assert report.cost_usd == 0.0

    def test_cost_is_summed_across_claims(self, results: list[RetrievedChunk]) -> None:
        model = ScriptedModel("SUPPORT: supported\nREASON: a", "SUPPORT: supported\nREASON: b")
        claims = split_claims("One [1]. Two [2].")
        assert CitationVerifier(model).verify(claims, results).cost_usd == pytest.approx(0.002)


class TestRates:
    def test_no_claims_gives_no_rates(self) -> None:
        """A refusal asserts nothing, which is not the same as scoring zero."""
        report = VerificationReport()
        assert report.verified_coverage is None
        assert report.citation_precision is None

    def test_coverage_and_precision_differ_when_most_claims_are_uncited(
        self, results: list[RetrievedChunk]
    ) -> None:
        model = ScriptedModel("SUPPORT: supported\nREASON: yes")
        claims = split_claims("Cited [1]. Bare one. Bare two. Bare three.")
        report = CitationVerifier(model).verify(claims, results)
        assert report.verified_coverage == 0.25
        assert report.citation_precision == 1.0


class TestVerificationReport:
    """The renderer, checked for the claims it makes rather than its prose.

    The control section is the one that must never render as reassurance when the control
    did not run: a report that quietly omits it would present a supported rate that
    nothing licenses.
    """

    @staticmethod
    def _run(**overrides: object) -> VerificationRun:
        base: dict[str, object] = {
            "generated_at": "2026-09-07T00:00:00+00:00",
            "git_sha": "abc1234",
            "git_dirty": False,
            "arm": "hybrid",
            "strategy": "fixed",
            "generation_model": "stub-generator",
            "verifier_model": "stub-verifier",
            "questions": 2,
            "answered": 2,
            "coverage": [1.0, 0.5],
            "precision": [1.0, 1.0],
            "claims": [],
        }
        base.update(overrides)
        return VerificationRun.model_validate(base)

    def test_a_run_without_a_control_says_so(self) -> None:
        report = render_verification_report(self._run())
        assert "control was not run" in report

    def test_a_discriminating_control_is_reported_above_the_rates(self) -> None:
        claims = [
            VerifiedClaim(
                question_id=f"lookup-{index:03d}",
                category="lookup",
                claim_index=0,
                claim_text="a claim",
                cited=(1,),
                supported=True,
                control_supported=False,
            )
            for index in range(8)
        ]
        report = render_verification_report(self._run(claims=claims))
        assert report.index("Is the verifier reading") < report.index("Coverage and precision")
        assert "random blocks (control)" in report

    def test_unsupported_claims_are_listed_with_their_reason(self) -> None:
        claims = [
            VerifiedClaim(
                question_id="lookup-006",
                category="lookup",
                claim_index=2,
                claim_text="Body and Form can be mixed freely",
                cited=(3,),
                supported=False,
                control_supported=False,
                reason="the passage says the opposite",
            )
        ]
        report = render_verification_report(self._run(claims=claims))
        assert "lookup-006" in report
        assert "the passage says the opposite" in report
