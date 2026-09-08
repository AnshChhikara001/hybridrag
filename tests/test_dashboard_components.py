"""The dashboard's HTML-fragment renderers.

These take plain dicts shaped like the API's JSON (not the pydantic models themselves --
the dashboard is a separate process that only ever sees `/v1/ask`'s response body) and
return HTML strings for `st.markdown(..., unsafe_allow_html=True)`. Three things matter:
they must not crash on the range of shapes the API actually returns (composite present or
absent, a chunk found by one index or both, a similarity of `None`); anything sourced from
the corpus or a language model must come back HTML-escaped; and `render_answer_card` must
produce one balanced string, never a tag meant to be closed by a later, separate
`st.markdown` call (see its docstring for why that silently renders as an empty box).
"""

from __future__ import annotations

from components import (
    render_answer_card,
    render_answer_text,
    render_chunks,
    render_citations,
    render_confidence_meter,
    render_refusal_body,
    render_telemetry,
)


class TestConfidenceMeter:
    def test_renders_a_hero_tile_for_the_composite_and_one_tile_per_component(self) -> None:
        html = render_confidence_meter(
            {"composite": 0.82, "retrieval": 0.55, "citation_coverage": 1.0}, "#B355F0"
        )
        assert html.count("stat-tile") == 3
        assert "hero-row" in html
        assert "0.82" in html

    def test_a_missing_component_is_skipped_not_rendered_as_zero(self) -> None:
        html = render_confidence_meter({"composite": 0.5, "completeness": None}, "#5EE6C9")
        assert "complete" not in html

    def test_no_signal_at_all_falls_back_to_a_message_instead_of_an_empty_grid(self) -> None:
        html = render_confidence_meter({}, "#5EE6C9")
        assert "no confidence signal" in html

    def test_a_component_value_sets_the_ring_fill_proportionally(self) -> None:
        html = render_confidence_meter({"composite": 0.4}, "#5EE6C9")
        assert "40%" in html


class TestCitations:
    def test_no_citations_renders_a_placeholder_not_an_empty_string(self) -> None:
        assert "no citations" in render_citations([])

    def test_a_citation_shows_its_number_and_source_with_heading_breadcrumb(self) -> None:
        html = render_citations(
            [{"number": 2, "relative_path": "a.md", "heading_path": ["Intro", "Setup"]}]
        )
        assert "[2]" in html
        assert "a.md" in html
        assert "Intro" in html and "Setup" in html

    def test_a_citation_with_no_heading_path_shows_just_the_file(self) -> None:
        html = render_citations([{"number": 1, "relative_path": "a.md", "heading_path": []}])
        assert "a.md" in html
        assert " > " not in html


class TestChunks:
    def _chunk(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "chunk": {"relative_path": "a.md", "heading_path": [], "text": "hello world"},
            "score": 0.0296,
            "hits": {"dense": {"rank": 1}},
        }
        base.update(overrides)
        return base

    def test_nothing_retrieved_renders_a_placeholder(self) -> None:
        assert "nothing retrieved" in render_chunks([])

    def test_unwraps_the_nested_chunk_for_source_and_text(self) -> None:
        html = render_chunks([self._chunk()])
        assert "a.md" in html
        assert "hello world" in html

    def test_long_chunk_text_is_truncated_with_an_ellipsis(self) -> None:
        html = render_chunks([self._chunk(chunk={"relative_path": "a.md", "text": "x" * 500})])
        assert "…" in html
        assert "x" * 500 not in html

    def test_found_only_by_dense_gets_the_dense_badge(self) -> None:
        html = render_chunks([self._chunk(hits={"dense": {"rank": 1}})])
        assert "dense" in html
        assert "sparse" not in html

    def test_found_only_by_sparse_gets_the_sparse_badge(self) -> None:
        html = render_chunks([self._chunk(hits={"sparse": {"rank": 1}})])
        assert "sparse" in html

    def test_found_by_both_gets_the_fused_badge(self) -> None:
        html = render_chunks([self._chunk(hits={"dense": {"rank": 1}, "sparse": {"rank": 3}})])
        assert "both" in html

    def test_a_missing_score_renders_n_a_rather_than_crashing(self) -> None:
        html = render_chunks([self._chunk(score=None)])
        assert "n/a" in html


class TestRefusalBody:
    """`render_refusal_body` returns content only, deliberately with no wrapping box --
    see `render_answer_card`'s docstring for why the outer card supplies the box."""

    def test_no_structured_refusal_falls_back_to_the_plain_text(self) -> None:
        html = render_refusal_body(None, "I don't know based on the indexed documentation.")
        assert "I don&#x27;t know" in html or "I don't know" in html
        assert "refusal-card" not in html

    def test_renders_the_kind_and_what_was_missing(self) -> None:
        refusal = {
            "kind": "low_confidence",
            "reason": "below threshold",
            "what_was_missing": "nothing close enough",
            "what_was_found": [],
            "documents_to_check": [],
        }
        html = render_refusal_body(refusal, "fallback")
        assert "low confidence" in html
        assert "nothing close enough" in html

    def test_a_near_miss_with_no_similarity_does_not_crash(self) -> None:
        refusal = {
            "kind": "low_confidence",
            "reason": "r",
            "what_was_missing": "m",
            "what_was_found": [{"source": "a.md", "similarity": None}],
            "documents_to_check": [],
        }
        html = render_refusal_body(refusal, "fallback")
        assert "a.md" in html

    def test_a_near_miss_with_a_similarity_shows_it_formatted(self) -> None:
        refusal = {
            "kind": "low_confidence",
            "reason": "r",
            "what_was_missing": "m",
            "what_was_found": [{"source": "a.md", "similarity": 0.1968}],
            "documents_to_check": [],
        }
        assert "0.20" in render_refusal_body(refusal, "fallback")

    def test_documents_to_check_are_listed(self) -> None:
        refusal = {
            "kind": "no_results",
            "reason": "r",
            "what_was_missing": "m",
            "what_was_found": [],
            "documents_to_check": ["b.md"],
        }
        assert "b.md" in render_refusal_body(refusal, "fallback")


class TestTelemetry:
    def test_shows_latency_cost_and_model(self) -> None:
        html = render_telemetry(
            {"latency_s": 1.234, "cost_usd": 0.00042, "model": "gemini-3.8-flash", "cached": False}
        )
        assert "1.23s" in html
        assert "0.00042" in html
        assert "gemini-3.8-flash" in html
        assert "cached" not in html

    def test_a_cached_answer_is_flagged(self) -> None:
        assert "cached" in render_telemetry({"cached": True})


class TestAnswerCard:
    """The one function where a bug reads as "the page looks empty" instead of a crash --
    see the module docstring for the Streamlit HTML-splitting pitfall this guards against.
    """

    def test_produces_one_balanced_div_not_a_dangling_open_tag(self) -> None:
        html = render_answer_card(
            css_class="dense", title="dense only", body_html="<p>hi</p>", telemetry_html=""
        )
        assert html.startswith('<div class="answer-card dense">')
        assert html.endswith("</div>")
        assert html.count("<div") == html.count("</div>")

    def test_the_title_is_escaped_and_the_body_is_passed_through_verbatim(self) -> None:
        html = render_answer_card(
            css_class="hybrid",
            title="<b>hybrid</b>",
            body_html="<p>already-safe body</p>",
            telemetry_html="",
        )
        assert "<b>hybrid</b>" not in html
        assert "<p>already-safe body</p>" in html


class TestEscaping:
    """Answer text and chunk text are untrusted as far as HTML goes: the corpus and the
    model both produce plain text that must never be interpreted as markup."""

    def test_answer_text_with_html_like_content_is_escaped(self) -> None:
        html = render_answer_text("<script>alert(1)</script>")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_chunk_text_with_html_like_content_is_escaped(self) -> None:
        chunk = {
            "chunk": {"relative_path": "a.md", "text": "<img src=x onerror=alert(1)>"},
            "score": 0.1,
            "hits": {},
        }
        html = render_chunks([chunk])
        assert "<img" not in html
        assert "&lt;img" in html

    def test_citation_source_path_is_escaped(self) -> None:
        html = render_citations([{"number": 1, "relative_path": "<b>a.md</b>", "heading_path": []}])
        assert "<b>a.md</b>" not in html
