"""The golden question set: schema, storage, and the relevance predicate.

Ground truth is an answer **span** (D13), and a span is stored as an exact **quote** rather
than a pair of character offsets. Offsets are derived when the set is loaded, against the
corpus as the loader produces it.

That is the important choice here. Offsets written into the file go stale the moment
anything upstream shifts them -- a re-fetched corpus, a change to include expansion, a
loader fix -- and they go stale *silently*, still parsing fine while pointing at the wrong
text, which would corrupt every retrieval metric without failing anything. A quote either
still occurs in the document or it does not, so the same drift becomes a loud error. It is
also the form a human can actually verify, which is the entire point of D11.

Offsets must be resolved against `Document.text`, never the raw file, because `{* ... *}`
include expansion inserts hundreds of lines of code examples and moves everything after
them. Chunk offsets live in that same expanded coordinate system.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from hybridrag.models import Chunk, Document

_FORMAT_VERSION = 1

_WHITESPACE = re.compile(r"\s+")


class QuestionCategory(StrEnum):
    """The four kinds the brief asks for, each scored by its own rule."""

    LOOKUP = "lookup"
    MULTI_HOP = "multi_hop"
    AMBIGUOUS = "ambiguous"
    NO_ANSWER = "no_answer"


class SpanNotFoundError(ValueError):
    """A quote does not occur exactly once in the document it names."""


class AnswerSpan(BaseModel):
    """A passage that answers a question, addressed by its text."""

    relative_path: str = Field(description="Document path, matching `Chunk.relative_path`.")
    quote: str = Field(min_length=1, description="Verbatim text from the document.")


class LocatedSpan(BaseModel):
    """An `AnswerSpan` resolved to character offsets in `Document.text`."""

    relative_path: str
    quote: str
    start_char: int
    end_char: int

    @property
    def length(self) -> int:
        return self.end_char - self.start_char


class GoldenQuestion(BaseModel):
    """One hand-verified question, its reference answer, and where the answer lives."""

    question_id: str
    category: QuestionCategory
    question: str = Field(min_length=1)
    answer: str = Field(
        description="Reference answer, for Tier 2 correctness judging. Empty for no_answer."
    )
    spans: list[AnswerSpan] = Field(default_factory=list)
    verified: bool = Field(
        default=False,
        description="Hand-checked by a human (D11). Unverified questions are excluded "
        "from reported metrics rather than silently counted.",
    )
    notes: str | None = None

    @model_validator(mode="after")
    def _spans_match_category(self) -> GoldenQuestion:
        """Each category has a different span shape, and a wrong shape is a scoring bug.

        A multi-hop question with one span is really a lookup, and would be scored under
        the strict all-spans rule while testing nothing multi-hop. Caught here rather than
        producing a quietly meaningless number.
        """
        count = len(self.spans)
        if self.category is QuestionCategory.NO_ANSWER:
            if count:
                raise ValueError(
                    f"{self.question_id}: a no_answer question must have no spans, got {count}. "
                    "If the corpus answers it, it belongs in another category."
                )
            return self
        if not count:
            raise ValueError(f"{self.question_id}: {self.category.value} needs at least one span")
        if self.category is QuestionCategory.LOOKUP and count != 1:
            raise ValueError(f"{self.question_id}: a lookup has exactly one span, got {count}")
        if self.category in (QuestionCategory.MULTI_HOP, QuestionCategory.AMBIGUOUS) and count < 2:
            raise ValueError(
                f"{self.question_id}: {self.category.value} needs two or more spans, got {count}"
            )
        if self.category is QuestionCategory.MULTI_HOP:
            documents = {span.relative_path for span in self.spans}
            if len(documents) < 2:
                raise ValueError(
                    f"{self.question_id}: multi_hop must span two or more documents, got "
                    f"{sorted(documents)}. One document is a lookup with a long answer."
                )
        return self

    def locate(self, documents: Mapping[str, Document]) -> list[LocatedSpan]:
        """Resolve every quote to offsets, or say precisely which one failed."""
        return [_locate(span, documents, self.question_id) for span in self.spans]


class GoldenSet(BaseModel):
    """The whole question set, plus what corpus it was written against."""

    format_version: int = _FORMAT_VERSION
    corpus_ref: str = Field(description="Corpus tag the quotes were taken from, e.g. 0.115.6.")
    questions: list[GoldenQuestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids_are_unique(self) -> GoldenSet:
        seen = [q.question_id for q in self.questions]
        duplicates = sorted({qid for qid in seen if seen.count(qid) > 1})
        if duplicates:
            raise ValueError(f"duplicate question_id(s): {duplicates}")
        return self

    def by_category(self, category: QuestionCategory) -> list[GoldenQuestion]:
        return [q for q in self.questions if q.category is category]

    def verified(self) -> list[GoldenQuestion]:
        """Only hand-checked questions, which are the only ones metrics may report."""
        return [q for q in self.questions if q.verified]

    @classmethod
    def load(cls, path: Path) -> GoldenSet:
        payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        version = payload.get("format_version")
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"{path} was written by format version {version!r}, but this build reads "
                f"version {_FORMAT_VERSION}."
            )
        return cls.model_validate(payload)

    def save(self, path: Path) -> None:
        """Written block-style so quotes stay readable, because a human reviews all of them."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json", exclude_none=True)
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=96),
            encoding="utf-8",
        )


def _normalised_with_map(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed text, plus each character's index in the original.

    The map is the point: a match is found in the collapsed text and then reported as
    offsets into the real document, so the resulting span still addresses exact bytes.
    """
    out: list[str] = []
    origin: list[int] = []
    previous_was_space = True
    for index, character in enumerate(text):
        if character.isspace():
            if not previous_was_space:
                out.append(" ")
                origin.append(index)
                previous_was_space = True
        else:
            out.append(character)
            origin.append(index)
            previous_was_space = False
    while out and out[-1] == " ":
        out.pop()
        origin.pop()
    return "".join(out), origin


def _locate_normalised(
    quote: str, document: Document, question_id: str, relative_path: str
) -> LocatedSpan:
    """Locate a quote whose whitespace does not match the document's.

    Language models reliably collapse blank lines when copying: where the document has a
    paragraph break, the reply has a single newline. That was the largest single cause of
    rejected candidates here -- quotes otherwise character-perfect for hundreds of
    characters, diverging only at a `\n\n`. Matching on collapsed whitespace tolerates
    exactly that, while still resolving to real offsets and still demanding a unique match,
    so nothing about the strength of the ground truth is given up.
    """
    haystack, origin = _normalised_with_map(document.text)
    needle = _WHITESPACE.sub(" ", quote).strip()
    if not needle:
        raise SpanNotFoundError(f"{question_id}: empty quote for {relative_path}.")

    occurrences = haystack.count(needle)
    if occurrences == 0:
        raise SpanNotFoundError(
            f"{question_id}: quote not found in {relative_path}, even ignoring whitespace. "
            f"It must be copied from the document, not paraphrased. Began: {quote[:60]!r}"
        )
    if occurrences > 1:
        raise SpanNotFoundError(
            f"{question_id}: quote occurs {occurrences} times in {relative_path}, so the "
            f"span is ambiguous. Extend it until it is unique. Began: {quote[:60]!r}"
        )

    start = haystack.index(needle)
    return LocatedSpan(
        relative_path=relative_path,
        # Stored as the document's own text, so the span is exact regardless of how the
        # quote was typed, and a later re-read of the file shows the real passage.
        quote=document.text[origin[start] : origin[start + len(needle) - 1] + 1],
        start_char=origin[start],
        end_char=origin[start + len(needle) - 1] + 1,
    )


def _locate(span: AnswerSpan, documents: Mapping[str, Document], question_id: str) -> LocatedSpan:
    """Find a quote's one unambiguous position in its document.

    Zero occurrences means the quote was paraphrased rather than copied, and two means the
    span is ambiguous -- neither can produce trustworthy ground truth, so both are refused
    instead of guessed at. This check is what keeps a generated candidate honest.
    """
    document = documents.get(span.relative_path)
    if document is None:
        raise SpanNotFoundError(f"{question_id}: no document {span.relative_path!r} in the corpus.")

    occurrences = document.text.count(span.quote)
    if occurrences == 0:
        # Falls back to whitespace-tolerant matching, which is what rescues a quote that
        # collapsed a paragraph break. Exact is tried first so the common case stays cheap
        # and unambiguous.
        return _locate_normalised(span.quote, document, question_id, span.relative_path)
    if occurrences > 1:
        raise SpanNotFoundError(
            f"{question_id}: quote occurs {occurrences} times in {span.relative_path}, so "
            f"the span is ambiguous. Extend it until it is unique. Began: {span.quote[:60]!r}"
        )

    start = document.text.index(span.quote)
    return LocatedSpan(
        relative_path=span.relative_path,
        quote=span.quote,
        start_char=start,
        end_char=start + len(span.quote),
    )


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse overlapping ranges into disjoint ones.

    Necessary because the fixed-size strategy overlaps its chunks by 64 tokens. Summing
    per-chunk overlap without merging would count those shared characters twice and report
    coverage above 100%, quietly inflating recall for the one strategy that overlaps.
    """
    ordered = sorted(intervals)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def covered_ratio(span: LocatedSpan, chunks: Sequence[Chunk]) -> float:
    """How much of `span` the retrieved chunks contain, between 0.0 and 1.0.

    Coverage is measured against the *union* of the chunks, not chunk by chunk. A span
    that straddles a chunk boundary is fully present in the model's context when both
    chunks are retrieved, and scoring it per chunk would call that a miss -- penalising
    every strategy for where its boundaries happen to fall, which is the one thing this
    comparison must not do.
    """
    if span.length <= 0:
        return 0.0
    intervals = _merge_intervals(
        (chunk.start_char, chunk.end_char)
        for chunk in chunks
        if chunk.relative_path == span.relative_path
    )
    covered = sum(
        max(0, min(end, span.end_char) - max(start, span.start_char)) for start, end in intervals
    )
    return covered / span.length


def is_hit(
    question: GoldenQuestion,
    located: Sequence[LocatedSpan],
    chunks: Sequence[Chunk],
    *,
    min_ratio: float = 1.0,
) -> bool:
    """Whether retrieval succeeded for this question, by its category's own rule.

    * lookup    -- its one span is covered.
    * multi_hop -- **every** span is covered: a question that genuinely needs two documents
                   is not half-answered by finding one of them.
    * ambiguous -- **any** acceptable span is covered, since several answers are correct.
    * no_answer -- has no span to retrieve; scored on refusal instead, never on recall.
    """
    if question.category is QuestionCategory.NO_ANSWER:
        raise ValueError(
            f"{question.question_id}: no_answer questions have no retrieval ground truth. "
            "Score them with the refusal rate instead."
        )
    covered = (covered_ratio(span, chunks) >= min_ratio for span in located)
    if question.category is QuestionCategory.AMBIGUOUS:
        return any(covered)
    return all(covered)
