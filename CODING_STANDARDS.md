# Coding standards

For the reviewer (`/code-review` reads this file). Only judgement calls live here. Ruff, ruff format and mypy strict already enforce style and types, so this file does not repeat them. Terms are from `GLOSSARY.md`.

## Measurement
- **The Held-out set never chooses a setting.** Code that tunes, sweeps or picks a default must read only the Tuning set. A script that reads `Split.HELD_OUT` and then changes a setting is a hard violation.
- **Locked settings change only with a recorded decision.** Fixed chunking 512/64, RRF weights 1:1, rank constant 60, k = 5 and the refusal gate threshold 0.30 are locked for the Held-out run. A diff that changes one of their defaults must point to the issue or decision that allows it.
- **Published numbers come from committed reports.** README and docs numbers must match a file in `evals/reports/<split>/` that was made from a clean commit. Flag a number with no report behind it.
- **Report the negative result.** A component that did not beat the baseline (the reranker) stays measured and off by default. It is not quietly removed, and it is not described as an improvement.

## Cost
- **Every paid model call goes through a cache.** Generation goes through `generation/cache.py` and embeddings through `embedding_cache.py`. The project's total API budget is $1.00, so a new uncached call path is a hard violation.

## Design
- **New providers implement the existing protocol.** A new model, embedder or index implements `LanguageModel`, `Embedder` or `SearchIndex`. Callers do not branch on the provider name.
- **The module docstring states the module's role**, plus the reason for any decision that is not obvious. Keep it true when the module changes.
- **Comments explain why, not what.** A comment that only repeats the code is noise. A choice with a rejected alternative gets one line saying why.
- **Use the glossary's words in names.** Use `Split`, `Span`, `Arm` and `Judge`, not synonyms, in code, tests and report headings.
- **Write plain code a reviewer can explain line by line.** Prefer a clear loop over a clever one-liner.

## Tests
- **Test through the public interface**, not private helpers.
- **Use stand-ins, not mocks of our own code.** A small fake that implements the protocol is better than patching internals.
- **Mark tests that use a real model or a live provider with `slow`**, so CI's `-m "not slow"` run needs no API key.
