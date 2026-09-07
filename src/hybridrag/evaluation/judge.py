"""The LLM judge, and the hand-labelled sheet that decides whether to trust it.

Tier 2 grades answers, and grading answers automatically means an LLM grading an LLM --
which is circular unless the judge is itself measured. So before any answer-quality number
is reported, the judge is validated: a human labels a sample by hand, each candidate judge
labels the same sample, and the agreement between them is reported alongside every result
it produces (D11). A judge that disagrees with the human is not a cheap evaluator, it is a
random number generator with a plausible explanation attached.

Two things the judge is deliberately not shown: which system produced the answer, and any
confidence or retrieval score. Both would let it grade the pipeline's self-assessment
instead of the answer. It sees the question, the human reference, the answer, and the exact
context the answer was generated from -- nothing else.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from hybridrag.evaluation.golden import QuestionCategory
from hybridrag.generation.base import Completion, LanguageModel

_FORMAT_VERSION = 1


class _BlockDumper(yaml.SafeDumper):
    """A dumper that writes multi-line text as a readable block, not as escapes."""


def _literal_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    # PyYAML falls back to quoting when a line has trailing whitespace, which block style
    # cannot represent. That is the right trade: the text stays byte-exact, and only the
    # few scalars that cannot be written as blocks stay ugly.
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_BlockDumper.add_representer(str, _literal_str)


class Verdict(StrEnum):
    """Answer correctness against the human reference answer.

    Three levels rather than two because the interesting failure in a RAG system is
    rarely a confident lie -- it is an answer that is right about the thing it covers and
    silent about half the question. Collapsing that into "incorrect" overstates the damage
    and into "correct" hides it.
    """

    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"


class JudgeError(ValueError):
    """The judge's reply could not be read as a decision."""


class JudgeDecision(BaseModel):
    """One judgement, from a human or a model."""

    verdict: Verdict
    grounded: bool | None = Field(
        default=None,
        description="Whether every claim is supported by the shown context. None when the "
        "labeller skipped it, which excludes that item from grounding agreement rather "
        "than silently counting as a disagreement.",
    )
    reason: str = ""


JUDGE_SYSTEM_PROMPT = """\
You grade the answers of a documentation assistant. For each item you are given the \
question, a reference answer a human wrote from the documentation, the assistant's answer, \
and the numbered context passages the assistant was shown.

Judge two things independently.

VERDICT -- correctness against the reference answer:
* correct   -- conveys what the reference conveys.
* partial   -- gets part of it, but omits something the reference treats as essential, or \
adds a claim that is wrong.
* incorrect -- contradicts the reference, or does not answer the question.

Wording, length, formatting, ordering and extra correct detail do not matter. Only \
substance does. The assistant's answer may be longer and more detailed than the reference \
and still be correct.

If the reference answer is empty, the documentation does not answer this question. \
Declining to answer is then **correct**, and answering it anyway is **incorrect**.

GROUNDED -- support in the context shown:
* yes -- every factual claim in the answer is supported by the numbered passages.
* no  -- at least one claim is not.

Judge grounding against the passages only. Do not use your own knowledge of the subject: a \
claim that is true in the world but absent from the passages is not grounded. An answer \
that declines is grounded by default.

Reply in exactly this form and nothing else:

VERDICT: correct|partial|incorrect
GROUNDED: yes|no
REASON: one sentence
"""

_VERDICT_LINE = re.compile(
    r"^\s*\**\s*VERDICT\s*\**\s*:\s*\**\s*(\w+)", re.IGNORECASE | re.MULTILINE
)
_GROUNDED_LINE = re.compile(
    r"^\s*\**\s*GROUNDED\s*\**\s*:\s*\**\s*(\w+)", re.IGNORECASE | re.MULTILINE
)
_REASON_LINE = re.compile(r"^\s*\**\s*REASON\s*\**\s*:\s*(.+)", re.IGNORECASE | re.MULTILINE)

_AFFIRMATIVE = {"yes", "true", "grounded", "y"}
_NEGATIVE = {"no", "false", "ungrounded", "n"}


def build_judge_prompt(
    question: str, reference_answer: str, system_answer: str, context: str
) -> str:
    """Assemble one grading task.

    The reference answer comes first and the assistant's answer second, so the standard is
    read before the thing being measured against it.
    """
    reference = reference_answer.strip() or (
        "(empty -- the documentation does not answer this question, so declining is correct)"
    )
    return (
        f"QUESTION\n{question.strip()}\n\n"
        f"REFERENCE ANSWER\n{reference}\n\n"
        f"ASSISTANT'S ANSWER\n{system_answer.strip()}\n\n"
        f"CONTEXT THE ASSISTANT WAS SHOWN\n{context.strip()}\n"
    )


def parse_decision(text: str) -> JudgeDecision:
    """Read a verdict out of the judge's reply.

    Line-prefixed rather than JSON, for the reason D32 records: answers and context here
    contain quote characters and fenced code, which models fail to escape inside JSON
    strings and which break any regex that strips a markdown fence. `VERDICT:` cannot
    collide with documentation, and the fields needing no escaping are the only ones parsed
    strictly -- the free-text reason is taken as-is.

    Bold markers are tolerated because models emit `**VERDICT:**` about a third of the
    time; anything else unreadable is an error rather than a guess, since a
    silently-defaulted verdict would enter the agreement statistics as a real judgement.
    """
    verdict_match = _VERDICT_LINE.search(text)
    if verdict_match is None:
        raise JudgeError(f"no VERDICT line in the judge's reply. Began: {text[:120]!r}")
    raw = verdict_match.group(1).strip().lower()
    try:
        verdict = Verdict(raw)
    except ValueError as error:
        raise JudgeError(
            f"unknown verdict {raw!r}; expected one of {[v.value for v in Verdict]}"
        ) from error

    grounded: bool | None = None
    grounded_match = _GROUNDED_LINE.search(text)
    if grounded_match is not None:
        token = grounded_match.group(1).strip().lower()
        if token in _AFFIRMATIVE:
            grounded = True
        elif token in _NEGATIVE:
            grounded = False
        else:
            raise JudgeError(f"unknown GROUNDED value {token!r}; expected yes or no")

    reason_match = _REASON_LINE.search(text)
    return JudgeDecision(
        verdict=verdict,
        grounded=grounded,
        reason=reason_match.group(1).strip() if reason_match else "",
    )


class LabelItem(BaseModel):
    """One answer to be graded, by a human and by each candidate judge.

    Carries the full context the answer was generated from, because grounding cannot be
    judged -- by a person or by a model -- without seeing exactly what the generator saw.
    """

    question_id: str
    category: QuestionCategory
    question: str
    reference_answer: str
    system_answer: str
    refused: bool = False
    citations: list[str] = Field(default_factory=list)
    context: str = ""
    selected_because: str = Field(
        default="", description="Why this item is in the sample, so the sampling is auditable."
    )
    proposed_verdict: Verdict | None = Field(
        default=None,
        description="A model's suggested label, for a human to adjudicate. Kept in a "
        "separate field from `human_verdict` on purpose: the agreement statistic is only "
        "meaningful if what a person decided and what a model suggested never merge.",
    )
    proposed_reason: str = ""
    human_verdict: Verdict | None = None
    human_grounded: bool | None = None
    notes: str | None = None

    @property
    def labelled(self) -> bool:
        return self.human_verdict is not None


class LabelSheet(BaseModel):
    """The hand-labelling sheet: what was generated, and room for the human's judgement."""

    format_version: int = _FORMAT_VERSION
    generated_at: str
    generation_model: str
    arm: str = Field(description="Which retrieval arm produced these answers.")
    items: list[LabelItem] = Field(default_factory=list)

    def labelled(self) -> list[LabelItem]:
        return [item for item in self.items if item.labelled]

    @classmethod
    def load(cls, path: Path) -> LabelSheet:
        payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        version = payload.get("format_version")
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"{path} was written by format version {version!r}, but this build reads "
                f"version {_FORMAT_VERSION}."
            )
        return cls.model_validate(payload)

    def save(self, path: Path) -> None:
        """Written so a person can actually read it.

        Multi-line strings use YAML's literal block style, because the default quotes them
        and turns 7,000 characters of retrieved context into one line of `\n` escapes. The
        human labelling this has to read that context to judge grounding at all, so
        legibility here is not cosmetic -- it decides whether the labels can be trusted.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        path.write_text(
            yaml.dump(
                payload,
                Dumper=_BlockDumper,
                sort_keys=False,
                allow_unicode=True,
                width=96,
            ),
            encoding="utf-8",
        )


class JudgedItem(BaseModel):
    """What one judge said about one item, and what it cost to ask."""

    question_id: str
    decision: JudgeDecision
    model: str
    cost_usd: float = 0.0
    cached: bool = False


class Judge:
    """Grades answers with a language model, one item at a time."""

    def __init__(self, model: LanguageModel, *, max_output_tokens: int | None = None) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens

    def judge(self, item: LabelItem) -> JudgedItem:
        completion: Completion = self.model.generate(
            build_judge_prompt(
                item.question, item.reference_answer, item.system_answer, item.context
            ),
            system=JUDGE_SYSTEM_PROMPT,
            max_output_tokens=self.max_output_tokens,
        )
        return JudgedItem(
            question_id=item.question_id,
            decision=parse_decision(completion.text),
            model=completion.model,
            cost_usd=completion.cost_usd,
            cached=completion.cached,
        )

    def judge_all(self, items: Sequence[LabelItem]) -> list[JudgedItem]:
        return [self.judge(item) for item in items]
