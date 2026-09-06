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
    cached: bool = Field(
        default=False,
        description=(
            "Served from the response cache. Then `cost_usd` is 0 because nothing was "
            "spent again, and `latency_s` is lookup time rather than generation time -- "
            "so any latency benchmark has to exclude these."
        ),
    )


@runtime_checkable
class LanguageModel(Protocol):
    """Generates text from a prompt."""

    model_name: str

    @property
    def fingerprint(self) -> str:
        """Every setting that changes what this model returns, as one stable string.

        Exists for the response cache. Hashing the prompt alone would keep serving
        temperature-0 answers after temperature was raised, or non-reasoning answers after
        reasoning was enabled -- a silent wrong result during exactly the parameter sweeps
        an evaluation run consists of. Making each implementation declare its own
        output-affecting settings puts that correctness in one place per adapter rather
        than in the cache's guesswork.
        """
        ...

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
