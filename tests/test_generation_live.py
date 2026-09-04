"""Live provider tests.

Marked `slow` and excluded from CI, which has no keys and should never depend on someone
else's uptime. They exist because the stubs in `test_answerer.py` verify our wiring and
cannot verify the thing that actually broke during this build: that a real model, given a
real prompt, answers instead of refusing.

Cost per run is about $0.0002 on `gpt-5-nano`, and $0 on Gemini's free tier.
"""

from __future__ import annotations

import pytest

from hybridrag.config import get_settings
from hybridrag.generation import GeminiModel, OpenAIModel
from hybridrag.generation.prompt import REFUSAL_SENTINEL, SYSTEM_PROMPT

pytestmark = pytest.mark.slow

CONTEXT = """\
CONTEXT
[1] deployment/docker.md > FastAPI in Containers - Docker
When deploying FastAPI applications a common approach is to build a Linux container image.

```Dockerfile
FROM python:3.9
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app
CMD ["fastapi", "run", "app/main.py", "--port", "80"]
```

[2] tutorial/first-steps.md > First Steps
Create a file main.py and declare an app with `app = FastAPI()`.

QUESTION
{question}"""


def _openai() -> OpenAIModel:
    key = get_settings().openai_api_key
    if key is None:
        pytest.skip("OPENAI_API_KEY is not set")
    return OpenAIModel(key, "gpt-5-nano-2025-08-07", max_output_tokens=512)


def test_a_real_model_answers_from_context_rather_than_refusing() -> None:
    """The regression for this build's worst bug: a false refusal with perfect context.

    Five chunks of `deployment/docker.md` were in context, one a complete Dockerfile, and
    the model still returned the refusal sentinel -- because the prompt's refusal rules
    outweighed its answering rules when reasoning was disabled.
    """
    completion = _openai().generate(
        CONTEXT.format(question="How do I run FastAPI in Docker?"), system=SYSTEM_PROMPT
    )

    assert REFUSAL_SENTINEL not in completion.text
    assert "[1]" in completion.text  # grounded in the block that holds the Dockerfile
    assert completion.input_tokens > 0


def test_a_real_model_refuses_when_nothing_in_context_relates() -> None:
    completion = _openai().generate(
        CONTEXT.format(question="What is the capital of France?"), system=SYSTEM_PROMPT
    )

    assert completion.text.strip().strip(".\"'` ") == REFUSAL_SENTINEL


def test_spend_is_counted_from_the_providers_own_usage_numbers() -> None:
    model = _openai()
    model.generate(CONTEXT.format(question="How do I run FastAPI in Docker?"))

    assert model.requests_made == 1
    assert model.input_tokens > 0
    assert 0.0 < model.estimated_cost_usd < 0.01


def test_gemini_answers_the_same_prompt() -> None:
    """The provider swap D4 promised, exercised rather than asserted.

    Skipped on 429 and 503 rather than failed: the free tier's limits are unpublished and
    the shared model returns 503 under load, so a red suite here would mean someone else's
    capacity, not our code.
    """
    key = get_settings().gemini_api_key
    if key is None:
        pytest.skip("GEMINI_API_KEY is not set")

    from google.genai.errors import APIError

    try:
        completion = GeminiModel(key, "gemini-3.8-flash", max_output_tokens=512).generate(
            CONTEXT.format(question="How do I run FastAPI in Docker?"), system=SYSTEM_PROMPT
        )
    except APIError as error:
        if error.code in (429, 503):
            pytest.skip(f"Gemini unavailable ({error.code})")
        raise

    assert REFUSAL_SENTINEL not in completion.text
    assert "[1]" in completion.text
