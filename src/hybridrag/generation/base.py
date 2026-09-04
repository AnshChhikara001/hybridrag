"""The language-model interface, and what one call costs.

A protocol rather than a concrete client, for the same reason `Embedder` is one: the
provider is a decision the project should be able to reverse in a config line (D4), and
every consumer downstream of this file should be unable to tell which provider answered.

Every implementation reports tokens and cost per call and cumulatively. Generation is the
second thing in this project that can spend money, and a $1 ceiling is only enforceable if
spending is counted where it happens rather than estimated afterwards.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt


class Completion(BaseModel):
    """One model response, with what it cost to produce."""

    text: str
    model: str = Field(description="The exact model id that answered, never an alias.")
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    # Reasoning tokens are billed as output but never returned to us, so they are counted
    # separately: an answer that looks cheap by its visible length can still be expensive.
    thinking_tokens: NonNegativeInt = 0
    cost_usd: NonNegativeFloat = 0.0
    latency_s: NonNegativeFloat = 0.0


@runtime_checkable
class LanguageModel(Protocol):
    """Generates text from a prompt."""

    model_name: str

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_output_tokens: int | None = None,
    ) -> Completion: ...


class GenerationError(RuntimeError):
    """The model returned nothing usable.

    Separate from the provider's own exceptions so callers handle one failure type
    whichever provider is configured -- and so an empty response, which most SDKs report
    as a successful call with no text, is raised rather than passed on as an empty answer.
    """
