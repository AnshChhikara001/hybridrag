"""Thin HTTP client for the HybridRAG API.

The dashboard is a separate process from the API (Phase 5.2's grill settled on talking to
5.1 over HTTP rather than importing its library code), so this is the only module that
knows the API's shape. Everything else in the dashboard works with the plain dicts these
functions return.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import requests

API_BASE_URL = os.environ.get("HYBRIDRAG_API_URL", "http://localhost:8000")

# A dense-only or hybrid /v1/ask can involve a real LLM call; a healthy provider still
# takes a few seconds, so this is generous rather than tight.
_ASK_TIMEOUT_S = 60


@dataclass
class AskResult:
    """Either a parsed `Answer` response, or a human-readable reason it didn't happen."""

    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None


def health() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result
    except requests.RequestException:
        return None


def documents() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_BASE_URL}/v1/documents", timeout=10)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result
    except requests.RequestException:
        return None


def ask(question: str, mode: Literal["hybrid", "dense_only"]) -> AskResult:
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/ask",
            json={"question": question, "mode": mode},
            timeout=_ASK_TIMEOUT_S,
        )
    except requests.RequestException as error:
        return AskResult(ok=False, error=f"could not reach the API: {error}")

    if response.status_code == 200:
        return AskResult(ok=True, data=response.json())

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return AskResult(ok=False, error=f"{response.status_code}: {detail}")
