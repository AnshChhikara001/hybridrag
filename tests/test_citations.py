"""Citation parsing and structural verification.

The case that matters most is a citation pointing at a block that does not exist: it
renders as a plausible source and goes nowhere. It must be reported and never quietly
removed, because stripping it would hide the exact failure this layer exists to expose.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from hybridrag.generation.citations import parse_citation_numbers, resolve_citations
from hybridrag.models import Chunk
from hybridrag.retrieval import RetrievedChunk

MakeChunk = Callable[..., Chunk]


@pytest.fixture
def results(make_chunk: MakeChunk) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=make_chunk(i, f"block {i}"), rank=i + 1, score=0.03, hits={})
        for i in range(3)
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Workers are set with --workers [2].", [2]),
        ("Both apply [1][3].", [1, 3]),
        ("Grouped [1, 3] form.", [1, 3]),
        ("Tight grouping [1,3].", [1, 3]),
        ("No citations at all.", []),
        ("Repeated [2] and again [2].", [2]),
        ("Order of first appearance [3] then [1].", [3, 1]),
    ],
)
def test_citation_numbers_are_parsed(text: str, expected: list[int]) -> None:
    assert parse_citation_numbers(text) == expected


def test_bracketed_non_numbers_are_not_citations() -> None:
    """Documentation prose is full of brackets that are not citations."""
    assert parse_citation_numbers("Use `Annotated[str, Query()]` here.") == []
    assert parse_citation_numbers("See [the docs] for more.") == []


def test_citations_resolve_to_the_chunk_they_point_at(results: list[RetrievedChunk]) -> None:
    report = resolve_citations("First [1] and third [3].", results)

    assert [c.number for c in report.citations] == [1, 3]
    assert report.citations[0].chunk_id == results[0].chunk.chunk_id
    assert report.citations[1].chunk_id == results[2].chunk.chunk_id
    assert report.is_structurally_sound


def test_a_citation_beyond_the_context_is_reported_not_dropped(
    results: list[RetrievedChunk],
) -> None:
    """A fabricated source is the failure this whole layer exists to catch."""
    report = resolve_citations("Supported [1], invented [9].", results)

    assert report.unresolved == [9]
    assert [c.number for c in report.citations] == [1]
    assert not report.is_structurally_sound


def test_block_zero_is_out_of_range(results: list[RetrievedChunk]) -> None:
    """Numbering is 1-based, so [0] is a fabrication rather than the first block."""
    assert resolve_citations("Cited [0].", results).unresolved == [0]


def test_blocks_that_were_never_cited_are_reported(results: list[RetrievedChunk]) -> None:
    """Separates 'retrieval was too broad' from 'generation ignored good context'."""
    report = resolve_citations("Only the second [2].", results)

    assert report.uncited_blocks == [1, 3]


def test_an_answer_citing_nothing_leaves_every_block_uncited(
    results: list[RetrievedChunk],
) -> None:
    report = resolve_citations("An uncited assertion.", results)

    assert report.citations == []
    assert report.uncited_blocks == [1, 2, 3]
    assert report.is_structurally_sound  # nothing fabricated; just nothing attributed


def test_a_citation_resolves_to_readable_provenance(results: list[RetrievedChunk]) -> None:
    citation = resolve_citations("See [1].", results).citations[0]

    assert citation.source == "tutorial/query-params.md > Tutorial > Query Parameters"


class TestCodeIsNotCited:
    """Bracketed numbers inside code are not citations.

    Found on the real corpus rather than imagined: a cached answer quoting `fastapi dev`
    console output carried the process id `[2248755]`, which the parser read as a citation
    to block 2,248,755 and reported as a fabricated source. `sys.argv[1]` is the quieter
    half of the same defect -- it resolves, so it silently attributes a sentence to
    whichever chunk ranked first.
    """

    def test_process_id_in_a_console_fence_is_not_a_citation(self) -> None:
        text = "Run the dev server [1].\n\n```console\n$ fastapi dev main.py\n[2248755]\n```"
        assert parse_citation_numbers(text) == [1]

    def test_array_index_in_a_fence_does_not_resolve_to_a_block(
        self, results: list[RetrievedChunk]
    ) -> None:
        text = "Read the argument [2].\n\n```python\nname = sys.argv[1]\n```"
        report = resolve_citations(text, results)
        assert [citation.number for citation in report.citations] == [2]

    def test_inline_code_span_is_not_a_citation(self) -> None:
        assert parse_citation_numbers("Access it with `argv[1]` as shown [3].") == [3]

    def test_unterminated_fence_still_protects_to_end_of_text(self) -> None:
        assert parse_citation_numbers("See [2].\n\n```\n[999]\n") == [2]

    def test_a_real_citation_after_a_fence_is_still_read(self) -> None:
        text = "```\n[999]\n```\n\nThe setting lives on the decorator [1]."
        assert parse_citation_numbers(text) == [1]
