# CONTEXT: hybridrag

A cold-session map. Read it before the code. Terms: `GLOSSARY.md`. Review rules: `CODING_STANDARDS.md`. Past decisions (D1, D2, ...): `docs/PROJECT_STATE.md`.

## Purpose
Hybrid RAG over the FastAPI docs. BM25 and dense retrieval are fused with Reciprocal Rank Fusion (RRF). Answers cite numbered context blocks, and each claim's citation is checked. The edge is the eval method: a Held-out set of real user questions is the only source of published numbers.

## Run it
- `uv sync --locked`, then `git config core.hooksPath .githooks` once per clone (pre-commit gate: ruff, format, mypy).
- `uv run pytest -m "not slow"`: the CI test run. `slow` tests use a real model or a live provider.
- Retrieval eval ($0 from the cache): `uv run python scripts/evaluate_retrieval.py data/raw/raw/fastapi/docs/en/docs --split tuning --out-dir <scratch>`.
- Eval scripts stop on a dirty tree. Commit first, so each report names a real commit.

## Layout (package `hybridrag`, in `src/`)
| Folder | Role | Key files |
|---|---|---|
| `loaders/` | Parse Markdown, HTML, PDF and text into Documents | `markdown.py`, `store.py` |
| `chunking/` | Split Documents into Chunks; fixed 512/64 is the locked default | `fixed.py`, `structure.py`, `semantic.py` |
| `indexing/` | BM25 (sparse) and Chroma (dense) indexes, dedup | `sparse.py`, `dense.py`, `tokenizer.py` |
| `retrieval/` | Search every index, fuse with RRF, hydrate Chunks | `hybrid.py`, `fusion.py`, `rerank.py` (measured, off) |
| `generation/` | Gate, prompt, model call, citations, claim checks, refusal | `answerer.py`, `prompt.py`, `citations.py`, `verification.py`, `cache.py` |
| `evaluation/` | Golden set, Arms, metrics, bootstrap intervals, Judge, reports | `golden.py`, `harness.py`, `metrics.py`, `stats.py`, `report.py`, `cli.py` |
| `api/` | FastAPI service | `app.py` |
| `scripts/` | CLI entry points: index, ask, eval, Golden set tools | `evaluate_retrieval.py`, `evaluate_answers.py`, `generate_golden.py` |
| `evals/` | Golden set and committed reports | `golden_set.yaml`, `reports/<split>/` |
| `dashboard/` | Streamlit UI, to be removed (#10) | `app.py` |

## One question, start to end
1. `POST /v1/ask`: `api/app.py:104` builds an `Answerer`.
2. `Answerer.answer` (`generation/answerer.py:104`) calls `HybridRetriever.retrieve` (`retrieval/hybrid.py:137`).
3. `retrieve` searches each index and fuses them with `reciprocal_rank_fusion` (`retrieval/fusion.py:61`, rank constant 60), then loads the top-k Chunks from the chunk store.
4. The refusal gate (`answerer.py:234` `_gate`): a top dense cosine under 0.30 (`config.py:68`) refuses before any model call.
5. The model answers from the numbered blocks (`prompt.py`). A refusal sentinel in its reply becomes a `MODEL_DECLINED` refusal.
6. `resolve_citations` (`citations.py:66`) maps each `[n]` to a block. `CitationVerifier.verify` (`verification.py:254`) checks that each cited block supports its claim.
7. A composite confidence score is computed, and the `Answer` is returned.

## One retrieval eval run
1. `scripts/evaluate_retrieval.py:180` `exit_if_dirty`, then `load_split` reads one Split from `evals/golden_set.yaml`.
2. `run_arm` (`evaluation/harness.py:188`) runs each Arm (hybrid, dense, BM25) over the questions.
3. A hit means the retrieved Chunks cover a Span (`golden.py:355` `is_hit`). `stats.py` adds bootstrap intervals and paired deltas.
4. Results go to `evals/reports/<split>/` (`retrieval_results.json`, `retrieval.md`).

## Conventions
- GitHub Issues via `gh`, one ticket per feature branch. Labels: `docs/agents/triage-labels.md`.
- Python 3.12, ruff, mypy strict, pytest.

## Gotchas
- The corpus sits at `data/raw/raw/`, nested by an old move.
- A cache miss calls a paid API. The project's total budget is $1.00.
- Two chunking tests download the embedding model (~67 MB) on a cold cache.
- Local `mypy` needs the dashboard group (`uv sync --group dashboard`), as CI does, until #10 removes Streamlit.
- An LLM-written question reuses the docs' words, which inflates recall (51% overlap measured on v0). Held-out questions are real users' words.
