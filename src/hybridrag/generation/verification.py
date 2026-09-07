"""Claim-level citation verification: does the cited block actually support the claim?

`citations.py` checks that `[3]` points at a block the model was given. That is necessary
and cheap, and it is not what a reader assumes a citation means. A citation that resolves
perfectly can still be attached to a sentence the block says nothing about, which is the
more dangerous failure: it renders as a working source link under a fabricated claim. This
module asks the semantic question, and it needs a model to do it.

**Verification is per claim, against every block that claim cites together.** The brief
frames this as one call per citation-claim pair; done literally, a claim citing [1][3]
because half the evidence is in each block is judged twice and marked unsupported both
times. Splitting evidence across blocks is legitimate model behaviour, so judging the
cited set jointly measures the thing a reader cares about -- is this sentence backed by
what it points at -- instead of manufacturing false alarms. When the set fails, every
citation in it is flagged, since nothing here can say which half was at fault.

The verifier is a language model grading a language model, so the honesty rule that
governs the Tier-2 judge governs it too: its agreement is a measured quantity, not an
assumption. `scripts/verify_citations.py` runs a negative control -- the same claims
paired with blocks chosen at random -- and a verifier that cannot separate those from the
real ones is rubber-stamping, whatever its supported rate says.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveInt

from hybridrag.generation.base import LanguageModel
from hybridrag.generation.claims import Claim
from hybridrag.retrieval import RetrievedChunk


class SupportVerdict(StrEnum):
    """Whether the cited blocks back the claim.

    Binary on purpose. The three-level scale the correctness judge uses exists because a
    RAG answer is often right about half of what was asked; a citation has no such middle
    -- either the passage says the thing or it does not -- and offering an "unclear" level
    is an invitation to use it whenever reading is hard.
    """

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class VerificationError(ValueError):
    """The verifier's reply could not be read as a verdict."""


VERIFIER_SYSTEM_PROMPT = """\
You check citations in a documentation assistant's answer. You are given one claim the \
assistant made and the numbered passages it cited for that claim.

Decide whether the passages support the claim.

* supported   -- the passages state the claim, or state something the claim follows \
directly from. Different wording, summarising, and dropping detail are all fine.
* unsupported -- the passages do not state it, contradict it, or the claim asserts \
something specific that is simply absent from them.

Judge against the passages alone. A claim that is true of the software but not present in \
these passages is unsupported: that is the failure you exist to catch. If the claim cites \
several passages, they count together -- the claim is supported when they jointly support \
it.

Code inside the claim is part of it. A command or snippet that does not appear in the \
passages, and does not follow from them, is unsupported.

Reply in exactly this form and nothing else:

SUPPORT: supported|unsupported
REASON: one sentence
"""

_SUPPORT_LINE = re.compile(
    r"^\s*\**\s*SUPPORT\s*\**\s*:\s*\**\s*(\w+)", re.IGNORECASE | re.MULTILINE
)
# The verdict alone on its own line, which this model emits often enough to have killed a
# paid run: `UNSUPPORTED\nREASON: ...`. Anchored to the whole line so it cannot match a
# verdict word occurring inside prose.
_BARE_VERDICT = re.compile(r"^\s*\**\s*(supported|unsupported)\s*\**\s*$", re.IGNORECASE)
_REASON_LINE = re.compile(r"^\s*\**\s*REASON\s*\**\s*:\s*(.+)", re.IGNORECASE | re.MULTILINE)

_NO_REAL_BLOCK = "Every block this claim cites was fabricated, so nothing supports it."


class ClaimVerification(BaseModel):
    """One claim, the blocks it cited, and whether they hold it up."""

    claim_index: NonNegativeInt
    claim_text: str
    cited: tuple[PositiveInt, ...]
    verdict: SupportVerdict
    reason: str = ""
    model: str = ""
    cost_usd: NonNegativeFloat = 0.0
    cached: bool = False

    @property
    def is_supported(self) -> bool:
        return self.verdict is SupportVerdict.SUPPORTED


class VerificationReport(BaseModel):
    """What verification found across one answer.

    Two rates, because they answer different questions and averaging them would hide
    both. Coverage is the brief's definition and the one the composite score reads: of
    everything the answer asserted, how much is backed by a checked citation -- so an
    uncited sentence lowers it. Precision asks only about the citations actually made:
    of those, how many hold. An answer that cites one sentence out of six and gets it
    right scores 0.17 coverage and 1.00 precision, and both facts are worth knowing.
    """

    verifications: list[ClaimVerification] = Field(default_factory=list)
    total_claims: NonNegativeInt = 0

    @property
    def cited_claims(self) -> int:
        return len(self.verifications)

    @property
    def supported_claims(self) -> int:
        return sum(item.is_supported for item in self.verifications)

    @property
    def unsupported(self) -> list[ClaimVerification]:
        """The citations to flag. This is the output the brief asks for."""
        return [item for item in self.verifications if not item.is_supported]

    @property
    def verified_coverage(self) -> float | None:
        """Share of all claims backed by a verified citation. None when there are none."""
        if not self.total_claims:
            return None
        return self.supported_claims / self.total_claims

    @property
    def citation_precision(self) -> float | None:
        """Share of cited claims whose citations hold. None when nothing was cited."""
        if not self.cited_claims:
            return None
        return self.supported_claims / self.cited_claims

    @property
    def cost_usd(self) -> float:
        return sum(item.cost_usd for item in self.verifications)


def render_cited_blocks(cited: Sequence[int], results: Sequence[RetrievedChunk]) -> str:
    """The cited passages, numbered as the answer numbers them.

    Only the cited blocks are shown. Handing the verifier the whole context would let it
    mark a claim supported on the strength of a block the answer never pointed at, which
    is precisely the mis-attribution being looked for.
    """
    blocks: list[str] = []
    for number in cited:
        if 1 <= number <= len(results):
            chunk = results[number - 1].chunk
            heading = " > ".join(chunk.heading_path)
            source = f"{chunk.relative_path}{' > ' + heading if heading else ''}"
            blocks.append(f"[{number}] {source}\n{chunk.text.strip()}")
    return "\n\n".join(blocks)


def build_verification_prompt(claim: Claim, results: Sequence[RetrievedChunk]) -> str:
    """One verification task: the claim first, then only what it cited."""
    return (
        f"CLAIM\n{claim.text.strip()}\n\n"
        f"CITED PASSAGES\n{render_cited_blocks(claim.citations, results).strip()}\n"
    )


def _bare_verdict(text: str) -> str | None:
    """The verdict written on its own line without the `SUPPORT:` label, if present."""
    for line in text.splitlines():
        found = _BARE_VERDICT.match(line)
        if found:
            return found.group(1).lower()
    return None


def parse_support(text: str) -> tuple[SupportVerdict, str]:
    """Read a verdict out of the verifier's reply.

    Line-prefixed rather than JSON for the reason D32 records: claims here carry fenced
    code and quote characters that models fail to escape inside JSON strings. An
    unreadable reply raises instead of defaulting, because a silently-defaulted
    `supported` would inflate coverage with a judgement nobody made.

    The bare form -- the verdict alone on its first line, with no `SUPPORT:` label -- is
    accepted, because the model emits it often enough to have crashed a paid run partway
    through. Accepting it is not defaulting: the verdict is still one the model actually
    wrote, and the line must consist of nothing else.
    """
    match = _SUPPORT_LINE.search(text)
    raw = match.group(1).strip().lower() if match else _bare_verdict(text)
    if raw is None:
        raise VerificationError(f"no SUPPORT line in the verifier's reply. Began: {text[:120]!r}")
    try:
        verdict = SupportVerdict(raw)
    except ValueError as error:
        raise VerificationError(
            f"unknown support verdict {raw!r}; expected one of {[v.value for v in SupportVerdict]}"
        ) from error
    reason = _REASON_LINE.search(text)
    return verdict, reason.group(1).strip() if reason else ""


class CitationVerifier:
    """Checks each cited claim against the blocks it cites."""

    def __init__(self, model: LanguageModel, *, max_output_tokens: int | None = None) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens

    def verify_claim(self, claim: Claim, results: Sequence[RetrievedChunk]) -> ClaimVerification:
        """Verify one cited claim. Callers filter out uncited claims first."""
        real = [number for number in claim.citations if 1 <= number <= len(results)]
        if not real:
            # Structurally unresolvable already, so there is nothing to send and no
            # judgement to make: a claim citing only blocks that do not exist is
            # unsupported by construction, and paying a model to confirm that is waste.
            return ClaimVerification(
                claim_index=claim.index,
                claim_text=claim.text,
                cited=claim.citations,
                verdict=SupportVerdict.UNSUPPORTED,
                reason=_NO_REAL_BLOCK,
            )

        completion = self.model.generate(
            build_verification_prompt(claim, results),
            system=VERIFIER_SYSTEM_PROMPT,
            max_output_tokens=self.max_output_tokens,
        )
        verdict, reason = parse_support(completion.text)
        return ClaimVerification(
            claim_index=claim.index,
            claim_text=claim.text,
            cited=claim.citations,
            verdict=verdict,
            reason=reason,
            model=completion.model,
            cost_usd=completion.cost_usd,
            cached=completion.cached,
        )

    def verify(
        self, claims: Sequence[Claim], results: Sequence[RetrievedChunk]
    ) -> VerificationReport:
        """Verify every cited claim in an answer.

        Uncited claims are never sent -- there is no citation to check -- but they stay in
        `total_claims`, so an answer that avoids citing cannot buy coverage by asserting
        things bare.
        """
        return VerificationReport(
            verifications=[self.verify_claim(claim, results) for claim in claims if claim.is_cited],
            total_claims=len(claims),
        )
