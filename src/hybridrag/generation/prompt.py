"""The grounded prompt: numbered context blocks, and the rules for citing them.

Each block carries its source path and heading breadcrumb above the text. That costs a few
tokens per block and buys two things: the model has something concrete to attribute a claim
*to*, and a reader comparing the answer against the retrieved chunks can see immediately
which page a citation points at without resolving an id.

The refusal is a literal sentinel rather than a phrase to match. Detecting "I don't know"
by searching prose is guesswork -- the model may hedge, apologise, or answer partially in
the same breath -- whereas a sentinel is either present or it is not, which is what a
downstream confidence score and an evaluation harness both need.

The refusal condition is deliberately narrow: *no block relates to the question at all*.
An earlier draft told the model that "partial information is not an answer", and with
reasoning disabled that produced a false refusal on "how do I run FastAPI in Docker" --
with five chunks of `deployment/docker.md` in context, one of them a complete Dockerfile.
The cause was an interaction rather than either half alone: the same strict prompt answers
correctly when reasoning is on, and a plain prompt answers correctly with reasoning off. A
model with no reasoning budget cannot weigh whether context suffices, so heavily-worded
refusal rules win by default. Chunked retrieval always delivers partial context, so the
instruction has to say when to answer, not only when to decline.
"""

from __future__ import annotations

from collections.abc import Sequence

from hybridrag.retrieval import RetrievedChunk

REFUSAL_SENTINEL = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""\
You answer questions about a technical documentation corpus, using the numbered context \
blocks supplied with each question.

How to answer:
1. Base every statement on the context blocks. They are your only source.
2. Cite the block each claim came from, in square brackets at the end of the sentence: \
"Workers are configured with --workers [2]." Cite several when several apply: \
"... is set on the decorator [1][3]."
3. Only cite numbers that appear in the context above.
4. Answer whenever the context addresses the question, even when it covers only part of \
it. Give what the context supports, and say plainly which part it does not cover.
5. Reply with exactly {REFUSAL_SENTINEL}, and nothing else, only when no block relates to \
the question at all.
6. Be concise and factual. No preamble, no restating of the question, no closing offer of \
further help.
"""


def render_context(results: Sequence[RetrievedChunk]) -> str:
    """Numbered context blocks, in retrieval order.

    Numbering is 1-based and follows the fused ranking, so citation `[1]` always means the
    top-ranked chunk. That mapping is the only thing tying an answer back to the corpus,
    so it is produced in one place and read back in one place.
    """
    blocks: list[str] = []
    for position, result in enumerate(results, start=1):
        heading = " > ".join(result.chunk.heading_path)
        source = f"{result.chunk.relative_path}{' > ' + heading if heading else ''}"
        blocks.append(f"[{position}] {source}\n{result.chunk.text.strip()}")
    return "\n\n".join(blocks)


def build_prompt(question: str, results: Sequence[RetrievedChunk]) -> str:
    """The user-turn prompt: context first, question last.

    Question last on purpose. With the question at the top it competes with the corpus for
    attention across a few thousand tokens of context; at the bottom it is the last thing
    read before generation begins.
    """
    return f"CONTEXT\n{render_context(results)}\n\nQUESTION\n{question.strip()}"
