"""HybridRAG query dashboard.

Run with: uv run --group dashboard streamlit run dashboard/app.py

Every question fires both retrieval modes -- hybrid and dense-only -- concurrently against
the FastAPI service from Phase 5.1, so the two answers, their citations, retrieved chunks
and confidence breakdowns sit side by side: the brief's required hybrid-vs-dense-only
comparison, shown on every question rather than behind a toggle someone has to remember
to flip (settled in the Phase 5.2 grill).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import streamlit as st
from client import API_BASE_URL, AskResult, ask, documents, health
from components import (
    render_answer_card,
    render_answer_text,
    render_chunks,
    render_citations,
    render_confidence_meter,
    render_refusal_body,
    render_telemetry,
)
from theme import CUSTOM_CSS, DENSE, FUSION, ROBOT_SVG

st.set_page_config(page_title="HybridRAG", page_icon="◆", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _render_header() -> None:
    st.markdown(
        '<div class="console-header">'
        '<div class="brand"><span class="brand-orb"></span><h1>HYBRIDRAG</h1></div>'
        '<div class="thesis"><span class="dense">dense</span> + '
        '<span class="sparse">sparse</span> fused via <span class="fusion">RRF</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    status = health()
    if status is None:
        st.markdown(
            f'<div class="status-strip"><span><span class="dot down"></span>'
            f"API unreachable at {API_BASE_URL}</span></div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div class="status-strip">'
        '<span><span class="dot"></span>live</span>'
        f"<span>strategy: {status['strategy']}</span>"
        f"<span>{status['documents_indexed']} documents</span>"
        f"<span>{status['chunks_indexed']} chunks</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_answer_column(title: str, css_class: str, accent: str, result: AskResult) -> None:
    if not result.ok or result.data is None:
        body = '<div class="refusal-kind">request failed</div>' + render_answer_text(
            result.error or "unknown error"
        )
        st.markdown(
            render_answer_card(
                css_class=f"{css_class} refused", title=title, body_html=body, telemetry_html=""
            ),
            unsafe_allow_html=True,
        )
        return

    answer = result.data
    if not answer["answered"]:
        body = render_refusal_body(answer.get("refusal"), answer["text"])
        card_class = f"{css_class} refused"
    else:
        body = (
            render_answer_text(answer["text"])
            + render_confidence_meter(answer["confidence"], accent)
            + render_citations(answer["citations"])
        )
        card_class = css_class

    st.markdown(
        render_answer_card(
            css_class=card_class,
            title=title,
            body_html=body,
            telemetry_html=render_telemetry(answer),
        ),
        unsafe_allow_html=True,
    )

    if answer["answered"]:
        with st.expander(f"retrieved chunks ({len(answer['retrieved'])})"):
            st.markdown(render_chunks(answer["retrieved"]), unsafe_allow_html=True)


def _render_empty_state() -> None:
    """Shown before the first question -- the reference mockup's onboarding screen turned
    into an invitation to act rather than a blank page."""
    st.markdown(
        '<div class="empty-state">'
        f'<div class="robot">{ROBOT_SVG}</div>'
        "<h2>Ask HybridRAG anything about the FastAPI docs</h2>"
        "<p>Every question runs through dense (vector) search, sparse (keyword) search, "
        "and both fused via RRF -- shown side by side, so you can see what the fusion "
        "actually bought you.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_documents_panel() -> None:
    listing = documents()
    if listing is None:
        return
    docs: list[dict[str, Any]] = listing["documents"]
    with st.expander(f"indexed documents ({len(docs)})"):
        for doc in docs:
            st.markdown(
                f'<div class="doc-row"><span class="path">{doc["relative_path"]}</span>'
                f"<span>{doc['chunks']} chunks</span></div>",
                unsafe_allow_html=True,
            )


def main() -> None:
    _render_header()

    question = st.text_input(
        "question",
        placeholder="> ask something about fastapi docs_",
        label_visibility="collapsed",
    )
    asked = st.button("ASK  >")

    if asked and question.strip():
        with ThreadPoolExecutor(max_workers=2) as pool:
            hybrid_future = pool.submit(ask, question, "hybrid")
            dense_future = pool.submit(ask, question, "dense_only")
            hybrid_result = hybrid_future.result()
            dense_result = dense_future.result()

        left, right = st.columns(2)
        with left:
            _render_answer_column("dense only", "dense", DENSE, dense_result)
        with right:
            _render_answer_column(
                "hybrid (dense + sparse, rrf-fused)", "hybrid", FUSION, hybrid_result
            )
    elif asked:
        st.warning("type a question first.")
    else:
        _render_empty_state()

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    _render_documents_panel()


if __name__ == "__main__":
    main()
