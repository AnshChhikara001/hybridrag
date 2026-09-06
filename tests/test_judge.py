"""The LLM judge: prompt construction, reply parsing, and the labelling sheet.

Parsing carries the weight. A verdict that is silently defaulted when the reply is
malformed enters the agreement statistics as though a judge had actually made a decision,
which would inflate the very number this experiment exists to measure honestly. So an
unreadable reply is an error, and only formatting noise models genuinely produce -- bold
markers, stray case -- is tolerated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hybridrag.evaluation.golden import QuestionCategory
from hybridrag.evaluation.judge import (
    Judge,
    JudgeDecision,
    JudgeError,
    LabelItem,
    LabelSheet,
    Verdict,
    build_judge_prompt,
    parse_decision,
)
from hybridrag.generation import Completion

WELL_FORMED = "VERDICT: correct\nGROUNDED: yes\nREASON: it states the same rule.\n"


def item(**kwargs: object) -> LabelItem:
    defaults: dict[str, object] = {
        "question_id": "lookup-001",
        "category": QuestionCategory.LOOKUP,
        "question": "How do I declare a query parameter?",
        "reference_answer": "Declare it as a function argument.",
        "system_answer": "Declare it as a function argument [1].",
        "context": "[1] tutorial/query-params.md\nQuery parameters are function arguments.",
    }
    defaults.update(kwargs)
    return LabelItem.model_validate(defaults)


class StubJudgeModel:
    """Returns a scripted reply, so parsing is tested without a provider."""

    model_name = "stub-judge"

    def __init__(self, reply: str = WELL_FORMED) -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    @property
    def fingerprint(self) -> str:
        return "stub-judge"

    def generate(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int | None = None
    ) -> Completion:
        self.prompts.append(prompt)
        self.systems.append(system)
        return Completion(
            text=self.reply,
            model=self.model_name,
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.0001,
            latency_s=0.5,
        )


class TestParsing:
    def test_a_well_formed_reply_is_read(self) -> None:
        decision = parse_decision(WELL_FORMED)

        assert decision.verdict is Verdict.CORRECT
        assert decision.grounded is True
        assert decision.reason == "it states the same rule."

    def test_bold_markers_are_tolerated(self) -> None:
        """Models emit `**VERDICT:**` often enough that refusing it would lose real data."""
        decision = parse_decision("**VERDICT:** partial\n**GROUNDED:** no\n**REASON:** omits it.")

        assert decision.verdict is Verdict.PARTIAL
        assert decision.grounded is False

    def test_case_and_surrounding_prose_do_not_matter(self) -> None:
        decision = parse_decision("Here is my assessment.\n\nverdict: INCORRECT\ngrounded: NO\n")

        assert decision.verdict is Verdict.INCORRECT
        assert decision.grounded is False

    def test_a_missing_verdict_is_an_error_not_a_default(self) -> None:
        with pytest.raises(JudgeError, match="no VERDICT line"):
            parse_decision("The answer looks fine to me.")

    def test_an_unknown_verdict_is_refused(self) -> None:
        with pytest.raises(JudgeError, match="unknown verdict"):
            parse_decision("VERDICT: mostly-right\nGROUNDED: yes")

    def test_an_unknown_grounded_value_is_refused(self) -> None:
        with pytest.raises(JudgeError, match="unknown GROUNDED"):
            parse_decision("VERDICT: correct\nGROUNDED: partially")

    def test_a_missing_grounded_line_leaves_it_unknown(self) -> None:
        """None excludes the item from grounding agreement instead of faking a judgement."""
        assert parse_decision("VERDICT: correct").grounded is None

    def test_a_reason_containing_a_code_fence_survives(self) -> None:
        """Why this is not JSON: answers and context are full of quotes and fences."""
        reply = 'VERDICT: correct\nGROUNDED: yes\nREASON: it shows `Query("x")` as the ref does.'

        assert 'Query("x")' in parse_decision(reply).reason


class TestPrompt:
    def test_every_part_the_judge_needs_is_present(self) -> None:
        prompt = build_judge_prompt("Q?", "reference", "answer", "[1] context")

        for section in ("QUESTION", "REFERENCE ANSWER", "ASSISTANT'S ANSWER", "CONTEXT"):
            assert section in prompt

    def test_an_empty_reference_tells_the_judge_a_refusal_is_correct(self) -> None:
        """Without this the judge grades a correct refusal against a blank and calls it wrong."""
        prompt = build_judge_prompt("Q?", "", "I don't know.", "[1] context")

        assert "declining is correct" in prompt

    def test_the_judge_is_not_told_what_produced_the_answer(self) -> None:
        """It must grade the answer, not the pipeline's own confidence in it."""
        prompt = build_judge_prompt("Q?", "reference", "answer", "[1] context")

        for leak in ("hybrid", "dense", "sparse", "confidence", "cosine", "rank"):
            assert leak not in prompt.lower()


class TestJudge:
    def test_a_decision_carries_the_model_and_its_cost(self) -> None:
        judged = Judge(StubJudgeModel()).judge(item())

        assert judged.question_id == "lookup-001"
        assert judged.decision.verdict is Verdict.CORRECT
        assert judged.model == "stub-judge"
        assert judged.cost_usd == pytest.approx(0.0001)

    def test_the_reference_answer_reaches_the_judge(self) -> None:
        model = StubJudgeModel()
        Judge(model).judge(item(reference_answer="Declare it as a function argument."))

        assert "Declare it as a function argument." in model.prompts[0]

    def test_an_unreadable_reply_raises(self) -> None:
        with pytest.raises(JudgeError):
            Judge(StubJudgeModel("no verdict here")).judge(item())


class TestLabelSheet:
    def test_a_sheet_round_trips_through_yaml(self, tmp_path: Path) -> None:
        sheet = LabelSheet(
            generated_at="2026-09-06T00:00:00+00:00",
            generation_model="gpt-5-nano-2025-08-07",
            arm="hybrid/fixed",
            items=[item()],
        )
        path = tmp_path / "labels.yaml"
        sheet.save(path)

        assert LabelSheet.load(path) == sheet

    def test_unlabelled_items_are_excluded_rather_than_counted(self, tmp_path: Path) -> None:
        sheet = LabelSheet(
            generated_at="2026-09-06T00:00:00+00:00",
            generation_model="m",
            arm="hybrid/fixed",
            items=[
                item(question_id="a", human_verdict=Verdict.CORRECT),
                item(question_id="b"),
            ],
        )

        assert [entry.question_id for entry in sheet.labelled()] == ["a"]

    def test_a_sheet_from_another_format_version_refuses_to_load(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.yaml"
        path.write_text("format_version: 99\ngenerated_at: x\ngeneration_model: m\narm: a\n")

        with pytest.raises(ValueError, match="format version"):
            LabelSheet.load(path)

    def test_a_decision_defaults_to_no_grounding_claim(self) -> None:
        assert JudgeDecision(verdict=Verdict.CORRECT).grounded is None


class TestJudgeReport:
    """The report must make a good raw percentage impossible to misread."""

    def _sheet(self) -> LabelSheet:
        labels = [Verdict.CORRECT] * 9 + [Verdict.INCORRECT]
        return LabelSheet(
            generated_at="2026-09-06T00:00:00+00:00",
            generation_model="gpt-5-nano-2025-08-07",
            arm="hybrid/fixed",
            items=[
                item(question_id=f"q{index:03d}", human_verdict=verdict)
                for index, verdict in enumerate(labels)
            ],
        )

    def test_the_trivial_baseline_appears_beside_every_judge(self) -> None:
        from hybridrag.evaluation.report import JudgeRun, render_judge_report

        sheet = self._sheet()
        run = JudgeRun(model="candidate", verdicts=["correct"] * 10, cost_usd=0.01)

        markdown = render_judge_report(sheet, [run], resamples=500)

        assert "always answers" in markdown
        # 90% raw agreement from a judge that decided nothing: the number the baseline exists
        # to put in context.
        assert "90%" in markdown

    def test_disagreements_are_named_so_they_can_be_checked(self) -> None:
        from hybridrag.evaluation.report import JudgeRun, render_judge_report

        sheet = self._sheet()
        run = JudgeRun(model="candidate", verdicts=["correct"] * 10)

        markdown = render_judge_report(sheet, [run], resamples=500)

        assert "q009" in markdown

    def test_how_the_labels_were_made_is_on_the_record(self) -> None:
        """A model-proposed label the human rubber-stamped is a weaker standard, so the
        report states how many matched rather than leaving a reader to assume."""
        from hybridrag.evaluation.report import JudgeRun, render_judge_report

        sheet = self._sheet()
        for entry in sheet.items:
            entry.proposed_verdict = entry.human_verdict
        run = JudgeRun(model="candidate", verdicts=["correct"] * 10)

        markdown = render_judge_report(sheet, [run], resamples=500)

        assert "How these labels were made" in markdown
        assert "**10**" in markdown
