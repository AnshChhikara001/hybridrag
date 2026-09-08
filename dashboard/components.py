"""HTML fragments for the dashboard's hand-drawn widgets.

Streamlit has no built-in confidence-meter or citation-tag widget, so these are plain
HTML/CSS strings injected via `st.markdown(..., unsafe_allow_html=True)`. Every piece of
dynamic text passed in here is escaped: chunk text and document titles come from the
corpus, and the answer text comes from a language model -- neither is trusted HTML.

`render_answer_card` is the one function that matters for correctness, not just looks: a
Streamlit `st.markdown(unsafe_allow_html=True)` call is its own isolated DOM container, so
opening a `<div>` in one call and closing it in a later one does not nest the content
between them -- the browser silently closes the orphaned tag at the end of its own call,
leaving an empty box and the real content adrift outside it. Every card is therefore built
as one fully-balanced HTML string and handed to the caller for a single `st.markdown` call.
"""

from __future__ import annotations

from html import escape
from typing import Any

from theme import BORDER, DENSE, FUSION, SPARSE, TEXT_MUTED

# `Answer.confidence`'s field order (models.py): composite is rendered separately, first
# and bold; these are the components that can feed it, in the order they're computed.
_COMPONENT_LABELS = {
    "retrieval": "retrieval",
    "retrieval_calibrated": "retrieval*",
    "citation_coverage": "citations",
    "citation_precision": "precision",
    "completeness": "complete",
}


def _source(item: dict[str, Any]) -> str:
    heading = " > ".join(item.get("heading_path") or ())
    path = item["relative_path"]
    return f"{path} > {heading}" if heading else path


def _ring_tile(label: str, value: float, color: str, *, hero: bool = False) -> str:
    """A donut-gauge stat tile: a conic-gradient ring, punched to a ring by an inset
    circle the same colour as the tile, with the value centred inside."""
    pct = max(0.0, min(1.0, value)) * 100
    tile_class = "stat-tile hero" if hero else "stat-tile"
    ring_style = f"background:conic-gradient({color} {pct:.0f}%, {BORDER} {pct:.0f}% 100%)"
    return (
        f'<div class="{tile_class}">'
        f'<span class="stat-ring" style="{ring_style}">'
        f'<span class="stat-ring-hole"><span class="stat-value">{value:.2f}</span></span>'
        "</span>"
        f'<span class="stat-label">{escape(label)}</span>'
        "</div>"
    )


def render_confidence_meter(confidence: dict[str, Any], accent: str) -> str:
    composite = confidence.get("composite")
    component_tiles = [
        _ring_tile(label, value, accent)
        for field, label in _COMPONENT_LABELS.items()
        if (value := confidence.get(field)) is not None
    ]

    if composite is None and not component_tiles:
        return '<div class="stat-empty">no confidence signal for this answer</div>'

    html = ""
    if composite is not None:
        hero_tile = _ring_tile("confidence", composite, accent, hero=True)
        html += f'<div class="stat-grid hero-row">{hero_tile}</div>'
    if component_tiles:
        html += f'<div class="stat-grid">{"".join(component_tiles)}</div>'
    return html


def render_answer_text(text: str) -> str:
    return f'<div class="answer-text">{escape(text)}</div>'


def render_citations(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return f'<span style="color:{TEXT_MUTED};font-size:0.8rem">no citations</span>'
    tags = [
        f'<span class="citation-tag"><span class="n">[{citation["number"]}]</span>'
        f"{escape(_source(citation))}</span>"
        for citation in citations
    ]
    return "".join(tags)


def _hits_badge(hits: dict[str, Any]) -> str:
    """Which index (or both) actually surfaced this chunk -- straight from RRF's inputs."""
    found_dense = "dense" in hits
    found_sparse = "sparse" in hits
    if found_dense and found_sparse:
        return f'<span style="color:{FUSION}">both</span>'
    if found_dense:
        return f'<span style="color:{DENSE}">dense</span>'
    if found_sparse:
        return f'<span style="color:{SPARSE}">sparse</span>'
    return ""


def render_chunks(chunks: list[dict[str, Any]]) -> str:
    """`chunks` is a list of `RetrievedChunk`: the actual chunk nests under `chunk`,
    alongside the RRF-fused `score` and a `hits` map of which index(es) surfaced it.
    """
    if not chunks:
        return f'<span style="color:{TEXT_MUTED};font-size:0.8rem">nothing retrieved</span>'
    cards = []
    for retrieved in chunks:
        chunk = retrieved["chunk"]
        score = retrieved.get("score")
        score_text = f"{score:.4f}" if score is not None else "n/a"
        preview = chunk.get("text", "")
        if len(preview) > 220:
            preview = preview[:220].rstrip() + "…"
        badge = _hits_badge(retrieved.get("hits") or {})
        cards.append(
            '<div class="chunk-card">'
            f'<div class="chunk-meta"><span>{escape(_source(chunk))}</span>'
            f'<span class="score">{badge} {score_text}</span></div>'
            f'<div class="chunk-text">{escape(preview)}</div>'
            "</div>"
        )
    return "".join(cards)


def render_refusal_body(refusal: dict[str, Any] | None, fallback_text: str) -> str:
    """The refusal's content only -- no wrapping box. The caller's outer `.answer-card`
    (styled with the `.refused` modifier) supplies the box, so a refusal never nests one
    bordered card inside another."""
    if refusal is None:
        return render_answer_text(fallback_text)

    parts = [
        f'<div class="refusal-kind">{escape(str(refusal["kind"]).replace("_", " "))}</div>',
        render_answer_text(refusal.get("what_was_missing") or refusal["reason"]),
    ]

    near_misses = refusal.get("what_was_found") or []
    if near_misses:
        parts.append('<div style="margin-top:0.5rem">')
        for miss in near_misses:
            similarity = miss.get("similarity")
            sim_text = (
                f' <span class="sim">({similarity:.2f})</span>' if similarity is not None else ""
            )
            parts.append(f'<div class="near-miss">· {escape(miss["source"])}{sim_text}</div>')
        parts.append("</div>")

    documents_to_check = refusal.get("documents_to_check") or []
    if documents_to_check:
        parts.append('<div style="margin-top:0.5rem">')
        for path in documents_to_check:
            parts.append(f'<div class="near-miss">▸ worth checking: {escape(path)}</div>')
        parts.append("</div>")

    return "".join(parts)


def render_telemetry(answer: dict[str, Any]) -> str:
    cached = " · cached" if answer.get("cached") else ""
    latency = answer.get("latency_s", 0.0)
    cost = answer.get("cost_usd", 0.0)
    model = escape(str(answer.get("model", "")))
    return (
        '<div class="telemetry">'
        f"<span>{latency:.2f}s</span>"
        f"<span>${cost:.5f}</span>"
        f"<span>{model}{cached}</span>"
        "</div>"
    )


def render_answer_card(*, css_class: str, title: str, body_html: str, telemetry_html: str) -> str:
    """The whole card as one balanced HTML string -- see the module docstring for why
    this must never be split across multiple `st.markdown` calls."""
    return (
        f'<div class="answer-card {css_class}">'
        f'<div class="channel-label">{escape(title)}</div>'
        f"{body_html}"
        f"{telemetry_html}"
        "</div>"
    )
