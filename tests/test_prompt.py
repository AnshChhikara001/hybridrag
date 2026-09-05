"""The grounded prompt.

Block numbering is the contract that ties an answer back to the corpus: `[1]` must always
mean the top-ranked chunk. If rendering and resolution ever disagree about the offset,
every citation in the system points one place to the left, so both ends are pinned here.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from hybridrag.generation.prompt import (
    REFUSAL_SENTINEL,
    SYSTEM_PROMPT,
    build_prompt,
    render_context,
)
from hybridrag.models import Chunk
from hybridrag.retrieval import RetrievedChunk

MakeChunk = Callable[..., Chunk]


def retrieved(chunk: Chunk, rank: int) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, rank=rank, score=0.03, hits={})


@pytest.fixture
def results(make_chunk: MakeChunk) -> list[RetrievedChunk]:
    return [
        retrieved(make_chunk(0, "Query parameters are function arguments."), 1),
        retrieved(
            make_chunk(1, "Raise HTTPException to return an error.", heading_path=()),
            2,
        ),
    ]


def test_blocks_are_numbered_from_one_in_retrieval_order(results: list[RetrievedChunk]) -> None:
    rendered = render_context(results)

    assert rendered.index("[1]") < rendered.index("[2]")
    assert "Query parameters are function arguments." in rendered.split("[2]")[0]


def test_a_block_carries_its_path_and_heading_breadcrumb(results: list[RetrievedChunk]) -> None:
    """The model needs something concrete to attribute a claim to."""
    assert "[1] tutorial/query-params.md > Tutorial > Query Parameters" in render_context(results)


def test_a_block_without_headings_shows_only_its_path(results: list[RetrievedChunk]) -> None:
    assert "[2] tutorial/query-params.md\n" in render_context(results)


def test_no_context_renders_empty(results: list[RetrievedChunk]) -> None:
    assert render_context([]) == ""


def test_the_question_comes_after_the_context(results: list[RetrievedChunk]) -> None:
    """At the top it competes with a few thousand tokens of corpus for attention."""
    prompt = build_prompt("How do I declare a query parameter?", results)

    assert prompt.index("CONTEXT") < prompt.index("QUESTION")
    assert prompt.rstrip().endswith("How do I declare a query parameter?")


def test_the_system_prompt_names_the_refusal_sentinel_exactly() -> None:
    """The answerer matches on this string, so the prompt has to contain that same string."""
    assert REFUSAL_SENTINEL in SYSTEM_PROMPT
    assert "only cite numbers that appear" in SYSTEM_PROMPT.lower()
