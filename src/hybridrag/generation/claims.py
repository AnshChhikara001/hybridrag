"""Splitting an answer into the claims that citation verification checks.

A citation only means anything beside the sentence it supports, so verifying citations
starts by deciding what the units of assertion are. That decision is made here, and made
deterministically: no model call, no judgement, just the text. Everything downstream --
citation coverage, the composite confidence score, the unsupported-citation flag -- counts
these units, so a sloppy split shows up as a confidently wrong percentage.

Three rules earn their place, each from a way the obvious split goes wrong on this corpus:

* **A code fence attaches to the claim that introduces it and is never a claim itself.**
  "Create a Dockerfile [1]:" followed by twelve lines of Dockerfile is one assertion
  carrying its evidence, and that evidence is exactly what a verifier has to see. Split it
  off and the prose is left asserting nothing while the code cites no one.
* **A single newline ends a claim.** Answers here are markdown, where bullets and table
  rows are separate assertions on consecutive lines. The paragraph splitter used for
  chunking breaks on blank lines only, so it would fuse an entire list into one claim and
  report a single citation as covering all of it.
* **Headings, rules and table separators are not claims.** They assert nothing, and
  counting them as uncited claims would drag coverage down for formatting alone.
* **A citation with no sentence of its own belongs to the sentence before it.** This
  generator ends answers with a trailing block -- `... for operations. [1] [2] [4] [5]` --
  and puts a bare `[4]` on the line after a code fence. Both are non-assertions by every
  rule above, so an earlier version of this module discarded them and reported an answer
  with four resolved citations as having none at all.

One consequence is worth stating plainly rather than hiding: citations are attached to the
sentence they follow, so an answer that dumps every source at the end scores low coverage
however sound those sources are. That is the metric behaving correctly. Per-claim
attribution is the thing being measured, and a reader who cannot tell which passage backs
which sentence has not been given it.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt

from hybridrag.chunking.segmentation import find_code_fences, sentence_spans
from hybridrag.generation.citations import CITATION_PATTERN, parse_citation_numbers

_NEWLINE = re.compile(r"\n")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_BLOCKQUOTE = re.compile(r"^\s*>\s?")
_HEADING = re.compile(r"^\s*#{1,6}\s")
_RULE = re.compile(r"^\s*(?:[-*_]\s*){3,}$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|[\s:|-]*$")
_LETTER = re.compile(r"[A-Za-z]")

# "Yes [1]." is a real assertion at three letters; "**" and "|" are not assertions at zero.
_MIN_LETTERS = 3


class Claim(BaseModel):
    """One assertion from an answer, with the blocks it cites.

    `text` includes any code fence that followed the prose, because the code is part of
    what is being asserted. `citations` is parsed from the prose alone -- a bracketed
    number inside the fence is console output, not a source.
    """

    index: NonNegativeInt = Field(description="0-based position in the answer.")
    text: str
    citations: tuple[PositiveInt, ...] = ()

    @property
    def is_cited(self) -> bool:
        return bool(self.citations)


def _line_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split [start, end) on newlines. Callers pass ranges already outside code fences."""
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in _NEWLINE.finditer(text, start, end):
        if match.start() > cursor:
            spans.append((cursor, match.start()))
        cursor = match.end()
    if cursor < end:
        spans.append((cursor, end))
    return spans


def _strip_markup(line: str) -> str:
    """Remove the list marker or quote arrow, leaving the assertion itself."""
    stripped = _BLOCKQUOTE.sub("", line.strip())
    return _LIST_MARKER.sub("", stripped).strip()


def _is_assertion(text: str) -> bool:
    """Whether this line says something a citation could support."""
    if not text or _RULE.fullmatch(text) or _TABLE_SEPARATOR.fullmatch(text):
        return False
    return len(_LETTER.findall(CITATION_PATTERN.sub("", text))) >= _MIN_LETTERS


def _regions(text: str) -> list[tuple[int, int, bool]]:
    """The text as alternating prose and fence regions, in order."""
    regions: list[tuple[int, int, bool]] = []
    cursor = 0
    for start, end in find_code_fences(text):
        if start > cursor:
            regions.append((cursor, start, False))
        regions.append((start, end, True))
        cursor = end
    if cursor < len(text):
        regions.append((cursor, len(text), False))
    return regions


def split_claims(answer: str) -> list[Claim]:
    """Split an answer into its claims, in order.

    A fence with no prose before it is dropped rather than kept as a claim of its own:
    there is no assertion to check it against, and inventing one would mean asking a
    verifier whether a passage supports a block of code.
    """
    prose: list[str] = []
    extras: list[list[str]] = []
    trailing: list[list[int]] = []

    for start, end, is_fence in _regions(answer):
        if is_fence:
            if prose:
                extras[-1].append(answer[start:end].strip())
            continue
        for line_start, line_end in _line_spans(answer, start, end):
            if _HEADING.match(answer[line_start:line_end]):
                continue
            for piece_start, piece_end in sentence_spans(answer, line_start, line_end):
                cleaned = _strip_markup(answer[piece_start:piece_end])
                if _is_assertion(cleaned):
                    prose.append(cleaned)
                    extras.append([])
                    trailing.append([])
                elif prose:
                    # Not an assertion, but it may still be a citation block belonging to
                    # the sentence it follows. Dropping these loses real attributions.
                    trailing[-1].extend(parse_citation_numbers(cleaned))

    return [
        Claim(
            index=index,
            text="\n".join([body, *extra]),
            # Parsed from the prose only, deliberately: see the class docstring. Trailing
            # blocks are merged in, first appearance winning, so a citation repeated in
            # both places is counted once.
            citations=tuple(dict.fromkeys([*parse_citation_numbers(body), *attached])),
        )
        for index, (body, extra, attached) in enumerate(zip(prose, extras, trailing, strict=True))
    ]


def structural_coverage(claims: list[Claim]) -> float | None:
    """Share of claims carrying any citation at all. None when there are no claims.

    The free half of citation coverage: it asks whether the model attributed each
    assertion, not whether the attribution holds. `verification.verified_coverage` answers
    the second question and costs a model call per cited claim.
    """
    if not claims:
        return None
    return sum(claim.is_cited for claim in claims) / len(claims)
