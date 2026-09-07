"""The structured refusal.

"I don't know" is honest and nearly worthless. These tests pin the three facts that make
it actionable -- what was found, what was missing, which pages to open -- and the one
compatibility guarantee everything downstream leans on: the sentinel phrase still opens it.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from hybridrag.generation.models import NO_ANSWER_TEXT, RefusalKind
from hybridrag.generation.refusal import build_refusal
from hybridrag.models import Chunk
from hybridrag.retrieval import RetrievedChunk, RetrieverHit

MakeChunk = Callable[..., Chunk]


@pytest.fixture
def results(make_chunk: MakeChunk) -> list[RetrievedChunk]:
    """Four chunks over three files, so de-duplication has something to do."""
    paths = ["tutorial/body.md", "tutorial/body.md", "advanced/forms.md", "deploy/docker.md"]
    return [
        RetrievedChunk(
            chunk=make_chunk(index, f"body {index}", relative_path=path),
            rank=index + 1,
            score=0.03,
            hits={
                "dense": RetrieverHit(rank=index + 1, score=0.41 - index * 0.05, contribution=0.03)
            },
        )
        for index, path in enumerate(paths)
    ]


class TestWhatItReports:
    def test_documents_to_check_are_distinct_files_in_rank_order(
        self, results: list[RetrievedChunk]
    ) -> None:
        """Three chunks of one page is one page to open."""
        refusal = build_refusal(RefusalKind.LOW_CONFIDENCE, "q?", "reason", results)
        assert refusal.documents_to_check == [
            "tutorial/body.md",
            "advanced/forms.md",
            "deploy/docker.md",
        ]

    def test_near_misses_carry_their_similarity(self, results: list[RetrievedChunk]) -> None:
        """The scores are the reason for a gate refusal, so a reader should see them."""
        refusal = build_refusal(RefusalKind.LOW_CONFIDENCE, "q?", "reason", results)
        assert refusal.what_was_found[0].similarity == pytest.approx(0.41)

    def test_what_was_missing_names_the_question(self, results: list[RetrievedChunk]) -> None:
        refusal = build_refusal(
            RefusalKind.LOW_CONFIDENCE, "what is the capital of France", "r", results
        )
        assert "what is the capital of France" in refusal.what_was_missing

    def test_the_two_refusal_paths_say_different_things(
        self, results: list[RetrievedChunk]
    ) -> None:
        """A gate refusal means nothing came close; a model refusal means it came close
        and did not say it. Collapsing them hides a retrieval fault behind a prompt fault."""
        gate = build_refusal(RefusalKind.LOW_CONFIDENCE, "q?", "r", results)
        declined = build_refusal(RefusalKind.MODEL_DECLINED, "q?", "r", results)
        assert gate.what_was_missing != declined.what_was_missing

    def test_no_results_still_produces_a_refusal(self) -> None:
        refusal = build_refusal(RefusalKind.NO_RESULTS, "q?", "nothing retrieved", [])
        assert refusal.what_was_found == []
        assert refusal.documents_to_check == []
        assert "q?" in refusal.what_was_missing


class TestRendering:
    def test_the_sentinel_phrase_still_opens_it(self, results: list[RetrievedChunk]) -> None:
        """Downstream string checks and the Tier-2 judge both key off this line."""
        rendered = build_refusal(RefusalKind.LOW_CONFIDENCE, "q?", "r", results).render()
        assert rendered.startswith(NO_ANSWER_TEXT)

    def test_the_rendering_names_the_pages_to_open(self, results: list[RetrievedChunk]) -> None:
        rendered = build_refusal(RefusalKind.MODEL_DECLINED, "q?", "r", results).render()
        assert "deploy/docker.md" in rendered
        assert "Worth checking by hand" in rendered

    def test_an_empty_refusal_renders_without_dangling_headings(self) -> None:
        rendered = build_refusal(RefusalKind.NO_RESULTS, "q?", "r", []).render()
        assert "Worth checking by hand" not in rendered
        assert "What the search did find" not in rendered
