"""Gemini adapter.

Free tier, which is why D4 chose it: a full Phase 4 evaluation run costs $0 here against
roughly $0.11 on the cheapest paid alternative, and that difference repeats on every run.

Two things this file pins deliberately:

* **An exact model id, never `gemini-flash-latest`.** An alias silently changes what
  produced a number, and evaluation results have to stay comparable across weeks. Verified
  while building this: `gemini-2.5-flash`, which D4 originally named, now returns 404 for
  new keys -- exactly the drift a pinned id makes visible instead of silent.
* **Thinking off by default.** Grounded answering from supplied context is extractive
  work; reasoning tokens are billed as output, add latency, and are never shown to us.
  Some models reject a zero budget, so it is configurable rather than hardcoded.

Two defences against the free tier, which D12 assumed would be hostile and which measured
out at **5 requests per minute** for `gemini-3.8-flash` (429 `RESOURCE_EXHAUSTED`, quota
`GenerateRequestsPerMinutePerProjectPerModel-FreeTier`, value 5).

* A **client-side throttle** paces requests to that quota, so a long run never trips it.
  Reacting to 429s with backoff alone does not work for a queue of dozens of calls: the
  budget is spent on retries that were always going to fail. Pacing is what makes a 50
  question sweep finish.
* **Backoff** stays as the safety net for what pacing cannot predict -- 503s when the
  shared model is under load, and any quota narrower than the one configured here.

Cache hits never reach `generate`, so a re-run over unchanged prompts is not throttled.
"""

from __future__ import annotations

import threading
import time

from google.genai import Client, types
from pydantic import SecretStr

from hybridrag.generation.base import Completion, GenerationError

# Free tier at the time of writing, so every entry is 0.0 -- but the table exists, and is
# read rather than assumed, so switching to a paid model reports real money immediately
# instead of continuing to claim the run was free.
_PRICE_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gemini-3.8-flash": (0.0, 0.0),
    "gemini-3.7-flash": (0.0, 0.0),
    "gemini-3.5-flash": (0.0, 0.0),
    "gemini-3.5-flash-lite": (0.0, 0.0),
}

DEFAULT_MODEL = "gemini-3.8-flash"

# 429 is the free tier's rate limit; 503 is the shared model under load. Both are worth
# waiting out. 4xx codes other than 429 mean the request itself is wrong, so retrying one
# just repeats the same mistake more slowly.
_RETRY_STATUS_CODES = [429, 500, 502, 503, 504]
_RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=8,
    initial_delay=2.0,
    max_delay=64.0,
    exp_base=2.0,
    jitter=1.0,
    http_status_codes=_RETRY_STATUS_CODES,
)

# Measured, not guessed: the free tier returned 429 with quotaValue 5 for this model.
FREE_TIER_REQUESTS_PER_MINUTE = 5


class GeminiModel:
    """Google Gemini behind the `LanguageModel` protocol."""

    def __init__(
        self,
        api_key: SecretStr | str,
        model_name: str = DEFAULT_MODEL,
        *,
        temperature: float = 0.0,
        thinking_budget: int | None = 0,
        max_output_tokens: int = 1024,
        timeout_ms: int = 120_000,
        requests_per_minute: int | None = FREE_TIER_REQUESTS_PER_MINUTE,
    ) -> None:
        secret = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not secret:
            raise ValueError("A Gemini API key is required. Set GEMINI_API_KEY in .env.")
        if model_name not in _PRICE_PER_1M_TOKENS:
            raise ValueError(
                f"Unknown model {model_name!r}. Add it to {__name__}._PRICE_PER_1M_TOKENS "
                "with its price before using it, so spend is never silently unaccounted."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.input_tokens = 0
        self.output_tokens = 0
        self.requests_made = 0
        self.requests_per_minute = requests_per_minute
        # A lock, because a threaded evaluation runner would otherwise pace each worker
        # separately and collectively blow the same quota this exists to respect.
        self._pace_lock = threading.Lock()
        self._next_allowed_at = 0.0
        self._client = Client(
            api_key=secret,
            http_options=types.HttpOptions(timeout=timeout_ms, retry_options=_RETRY_OPTIONS),
        )

    @property
    def fingerprint(self) -> str:
        return (
            f"gemini|{self.model_name}|t={self.temperature}"
            f"|think={self.thinking_budget}|max={self.max_output_tokens}"
        )

    @property
    def estimated_cost_usd(self) -> float:
        """Cumulative spend, from the published rate for this model."""
        input_rate, output_rate = _PRICE_PER_1M_TOKENS[self.model_name]
        return (self.input_tokens * input_rate + self.output_tokens * output_rate) / 1_000_000

    def _config(
        self, system: str | None, max_output_tokens: int | None
    ) -> types.GenerateContentConfig:
        thinking = (
            types.ThinkingConfig(thinking_budget=self.thinking_budget)
            if self.thinking_budget is not None
            else None
        )
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=max_output_tokens or self.max_output_tokens,
            thinking_config=thinking,
        )

    def _await_slot(self) -> None:
        """Hold the caller until the next request would be within quota."""
        if not self.requests_per_minute:
            return
        interval = 60.0 / self.requests_per_minute
        with self._pace_lock:
            wait = self._next_allowed_at - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._next_allowed_at = time.monotonic() + interval

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_output_tokens: int | None = None,
    ) -> Completion:
        self._await_slot()
        started = time.perf_counter()
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._config(system, max_output_tokens),
        )
        latency = time.perf_counter() - started

        usage = response.usage_metadata
        # Counted from the provider's own numbers rather than a local estimate, so the
        # ledger matches the bill the moment this stops being a free tier.
        prompt_tokens = (usage.prompt_token_count or 0) if usage else 0
        candidate_tokens = (usage.candidates_token_count or 0) if usage else 0
        thinking_tokens = (usage.thoughts_token_count or 0) if usage else 0
        self.input_tokens += prompt_tokens
        self.output_tokens += candidate_tokens + thinking_tokens
        self.requests_made += 1

        text = response.text
        if not text or not text.strip():
            # A truncated or filtered response arrives as a successful call with no text.
            # Raising keeps that from reaching the citation parser as an empty answer.
            reason = response.candidates[0].finish_reason if response.candidates else None
            raise GenerationError(
                f"{self.model_name} returned no text (finish_reason={reason}). "
                f"A max_output_tokens of {max_output_tokens or self.max_output_tokens} "
                "can cause this when reasoning is enabled."
            )

        input_rate, output_rate = _PRICE_PER_1M_TOKENS[self.model_name]
        return Completion(
            text=text.strip(),
            model=response.model_version or self.model_name,
            input_tokens=prompt_tokens,
            output_tokens=candidate_tokens,
            thinking_tokens=thinking_tokens,
            cost_usd=(
                prompt_tokens * input_rate + (candidate_tokens + thinking_tokens) * output_rate
            )
            / 1_000_000,
            latency_s=latency,
        )
