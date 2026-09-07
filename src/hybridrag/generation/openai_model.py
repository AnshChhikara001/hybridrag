"""OpenAI adapter: the second `LanguageModel`, and the one that proves D4's claim.

D4 said the provider was reversible in a config line. This file is that claim tested rather
than asserted -- and it was written because Gemini's free tier started returning 429 mid-
build, exactly the hostile behaviour D12 assumed. A project with one hardcoded provider
would have stopped there.

Reasoning effort is set to minimal by default. Reasoning tokens are billed as output and
never shown, and grounded answering from supplied context is extractive work: the same
measurement that led to disabling Gemini's thinking applies here. Non-reasoning models such
as `gpt-4.1-nano` ignore the setting, so both families work through this one class.

Every model here is priced, and an unpriced model is refused rather than guessed at, so a
$1 ceiling is never breached by a model whose cost nobody looked up.
"""

from __future__ import annotations

import time

from openai import OpenAI, omit
from pydantic import SecretStr

from hybridrag.generation.base import Completion, GenerationError

# USD per 1M tokens, (input, output). Pinned to dated snapshots: an undated alias silently
# changes which model produced an evaluation number, the same reasoning that pins the
# corpus tag in D19 and the Gemini model id.
_PRICE_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-5-nano-2025-08-07": (0.05, 0.40),
    "gpt-5-mini-2025-08-07": (0.25, 2.00),
    "gpt-4.1-nano-2025-04-14": (0.10, 0.40),
    "gpt-4.1-mini-2025-04-14": (0.40, 1.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
}

DEFAULT_MODEL = "gpt-5-nano-2025-08-07"

# Reasoning models reject `temperature`; non-reasoning models reject `reasoning_effort`.
# Deciding by prefix keeps one class serving both instead of two near-identical adapters.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def price_of(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """What a completion of this size costs to produce, whoever paid for it.

    Needed because a cache hit truthfully reports `cost_usd = 0.0` -- nothing was spent
    again -- which is right for spend accounting and wrong for "what does one answer
    cost". Token counts survive a cache hit, so the price can be recomputed from them and
    an evaluation re-run stays honest about unit economics without re-spending.
    """
    if model_name not in _PRICE_PER_1M_TOKENS:
        raise ValueError(f"Unknown model {model_name!r}; add it to _PRICE_PER_1M_TOKENS.")
    input_rate, output_rate = _PRICE_PER_1M_TOKENS[model_name]
    return input_tokens / 1_000_000 * input_rate + output_tokens / 1_000_000 * output_rate


class BudgetExceededError(RuntimeError):
    """Raised before a request that would take spending past its ceiling."""


class OpenAIModel:
    """OpenAI chat models behind the `LanguageModel` protocol."""

    def __init__(
        self,
        api_key: SecretStr | str,
        model_name: str = DEFAULT_MODEL,
        *,
        temperature: float = 0.0,
        reasoning_effort: str = "minimal",
        max_output_tokens: int = 1024,
        cost_budget_usd: float | None = None,
        timeout: float = 120.0,
        max_retries: int = 5,
    ) -> None:
        secret = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not secret:
            raise ValueError("An OpenAI API key is required. Set OPENAI_API_KEY in .env.")
        if model_name not in _PRICE_PER_1M_TOKENS:
            raise ValueError(
                f"Unknown model {model_name!r}. Add it to {__name__}._PRICE_PER_1M_TOKENS "
                "with its published price before using it, so spend is never unaccounted."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.cost_budget_usd = cost_budget_usd
        self.input_tokens = 0
        self.output_tokens = 0
        self.requests_made = 0
        # The SDK retries with exponential backoff on rate limits and 5xx, which is what
        # D12 asks for and what the free-tier experience showed is not optional.
        self._client = OpenAI(api_key=secret, timeout=timeout, max_retries=max_retries)

    @property
    def _is_reasoning_model(self) -> bool:
        return self.model_name.startswith(_REASONING_PREFIXES)

    @property
    def fingerprint(self) -> str:
        # Temperature is omitted from the request on reasoning models, so it is omitted
        # from the fingerprint too -- otherwise changing a field the API ignores would
        # discard a cache that is still perfectly valid.
        effort = self.reasoning_effort if self._is_reasoning_model else "n/a"
        temperature = "n/a" if self._is_reasoning_model else self.temperature
        return (
            f"openai|{self.model_name}|t={temperature}|effort={effort}|max={self.max_output_tokens}"
        )

    @property
    def estimated_cost_usd(self) -> float:
        input_rate, output_rate = _PRICE_PER_1M_TOKENS[self.model_name]
        return (self.input_tokens * input_rate + self.output_tokens * output_rate) / 1_000_000

    def _check_budget(self) -> None:
        """Refuse *before* sending, so a runaway loop fails loudly rather than expensively."""
        if self.cost_budget_usd is not None and self.estimated_cost_usd >= self.cost_budget_usd:
            raise BudgetExceededError(
                f"{self.model_name} has spent ${self.estimated_cost_usd:.4f}, at or past "
                f"its ${self.cost_budget_usd:.4f} budget. Raise cost_budget_usd to continue."
            )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_output_tokens: int | None = None,
    ) -> Completion:
        self._check_budget()

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore[arg-type]
            max_completion_tokens=max_output_tokens or self.max_output_tokens,
            temperature=omit if self._is_reasoning_model else self.temperature,
            reasoning_effort=self.reasoning_effort if self._is_reasoning_model else omit,  # type: ignore[arg-type]
        )
        latency = time.perf_counter() - started

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        details = usage.completion_tokens_details if usage else None
        reasoning_tokens = (details.reasoning_tokens or 0) if details else 0
        # Counted from the API's own usage block rather than a local estimate, so the
        # ledger matches the invoice.
        self.input_tokens += prompt_tokens
        self.output_tokens += completion_tokens
        self.requests_made += 1

        text = response.choices[0].message.content if response.choices else None
        if not text or not text.strip():
            finish = response.choices[0].finish_reason if response.choices else None
            raise GenerationError(
                f"{self.model_name} returned no text (finish_reason={finish}). On a "
                "reasoning model this usually means max_output_tokens was consumed by "
                "reasoning before any answer was produced."
            )

        input_rate, output_rate = _PRICE_PER_1M_TOKENS[self.model_name]
        return Completion(
            text=text.strip(),
            model=response.model,
            # Visible output only, so `output_tokens` and `thinking_tokens` never
            # double-count; the API reports reasoning inside completion_tokens.
            output_tokens=max(0, completion_tokens - reasoning_tokens),
            thinking_tokens=reasoning_tokens,
            input_tokens=prompt_tokens,
            cost_usd=(prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000,
            latency_s=latency,
        )
