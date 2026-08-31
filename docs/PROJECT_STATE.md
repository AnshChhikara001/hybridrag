# Project State

Recovery file. Current architecture, locked decisions and their rationale, progress, and
the next step. Updated whenever a decision or milestone lands.

**Last updated:** 2026-08-31 · **Status:** Phase 1 in progress (loaders + 2 of 3 chunkers done)

---

## What this project is

A hybrid-retrieval RAG system over the public FastAPI documentation, standing in for "a
company's internal docs." Dense vectors + BM25, fused with Reciprocal Rank Fusion, then
reranked by a cross-encoder. Answers are grounded, carry inline citations, and every
citation is verified. Architecture decisions are made from a measured evaluation suite
rather than asserted.

Full requirements: `docs/project-brief.md`. Operating rules: `CLAUDE.md`.

---

## Locked decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Corpus: FastAPI docs** (`fastapi/fastapi`, `docs/en/docs`, MIT) | Measured: 155 md files, 1.47 MB, ~400K tokens, 2,100 headings, **797 unique inline-code identifiers**. The identifier density is what gives BM25 a fair chance to beat dense retrieval — the project's core thesis. Verified before committing to it. |
| D2 | **Default embedder: local `bge-small-en-v1.5`** (384-dim, ~130 MB) | Free and unlimited, so chunking parameter sweeps cost nothing. Runs offline → CI needs no secrets and no funding. Reviewers can index with zero API keys. |
| D3 | **OpenAI `text-embedding-3-small` as a second adapter**, run **once** on the winning config | Turns "which embedder?" into a measured result (Recall@10, nDCG@10, latency, cost) instead of an assertion. Cost ≈ **$0.009** against a $0.10 allowance. |
| D4 | **LLM: Gemini 2.5 Flash, free tier**, behind a provider-agnostic adapter | $0. Rejected gpt-5-nano ($0.05/$0.40): a full eval run would cost ~$0.13 **and** nano-class judgement is weaker than Flash — paying more for worse. Adapter makes this reversible in one config line. |
| D5 | **Two-tier evaluation** | **Tier 1** (retrieval: Recall@k, MRR, nDCG@k) needs no LLM — deterministic, free, seconds, runs in CI on every push. **Tier 2** (LLM-judged answer quality) runs periodically on the Tier-1 winner plus contrasts. The brief's approach — full LLM eval × 3 strategies — costs **$2.19–$5.05 per run** on the cheapest paid models, i.e. 2–5× the entire project budget, per run. |
| D6 | **Vector store: Chroma, dual-mode** | Identical client API embedded (single container → HF Spaces) and server (a real service in compose → Docker networking to learn). One code path, config switch, dev/prod parity. Qdrant rejected: no true embedded mode, so deploy would diverge from dev. |
| D7 | **Sparse: `rank_bm25` + hand-written tokenizer** | A default tokenizer shreds `response_model` → `response`+`model`, destroying the exact-match edge that justifies hybrid at all. `rank_bm25` accepts pre-tokenized input, so the tokenizer is ours and unit-testable. |
| D8 | **Reranker: local cross-encoder `ms-marco-MiniLM-L-6-v2`** (~90 MB) | Free, fast, deterministic. Strictly better than LLM-as-judge reranking on cost, latency and reproducibility. |
| D9 | **Chunkers hand-rolled; no LangChain** | Heavy, churny dependency that bloats the image, for ~150–200 lines of testable code. Comparing chunking strategies is this project's subject matter — importing all three undercuts the claim. |
| D10 | **Deployment: Hugging Face Spaces (Docker SDK)** | Free, public URL, 16 GB / 2 vCPU. Secrets via Spaces Secrets. Single container → ships a deploy `Dockerfile` alongside `docker-compose.yml` for local dev. |
| D11 | **Golden set: LLM-generated candidates, every pair hand-verified**, 50 pairs, mixed difficulty | What real teams do. Plus: hand-label ~20 judge decisions and report **judge–human agreement %** — defuses LLM-judge circularity for $0. |
| D12 | **Persistent LLM response cache** keyed on (prompt hash, model) + exponential backoff | Google no longer publishes free-tier RPM/RPD (per-account, visible in AI Studio), so rate limits must be assumed hostile. An eval interrupted at Q37 resumes free instead of re-spending. |

| D13 | **Retrieval ground truth is an answer *span*, not a `section_id`** | `section_id` is derived from headings, and the structure-aware chunker splits on headings -- grading it against that rubric measures its own assumptions. It also admits false hits: a chunk clipping a section's edge lists that section without carrying its answer. `Section` and `Chunk` therefore carry `start_char`/`end_char`, and `Chunk.covers_span()` is the single relevance predicate. `section_ids` remain, demoted to citation provenance. |
| D14 | **Report Recall@token-budget beside Recall@k** | Recall@k rewards strategies that emit larger chunks for being larger. Fixing the retrieved-context budget equalises what the generator actually sees. |
| D16 | **Semantic chunking stays heading-blind, overlap-free, and pure** | The loader knows every section boundary; feeding that in would make this a variant of the structure-aware chunker and Phase 4 would compare two spellings of one idea. No overlap, because a boundary chosen for a topic change is not worth blurring. Where a topic exceeds the token budget it is subdivided at the *next-largest* distance inside it, iteratively (not recursively -- `argmax` can land at a group edge, and a 10,611-token section would recurse once per sentence). |
| D17 | **`Embedder` protocol owns the query/document asymmetry** | `bge` was trained with an instruction prefix on queries only; omitting it costs retrieval quality and applying it to documents costs it again. Keeping it inside the embedder means no caller can get it wrong, and the OpenAI adapter simply carries an empty instruction. The protocol also guarantees L2-normalised vectors, so cosine reduces to a dot product for Chroma, dedup, and semantic chunking alike. |
| D15 | **Chunkers own boundary placement only** | `Chunker.chunk()` turns spans into validated chunks once, in the base class, so the three strategies differ in boundary placement and nothing else -- the variable Phase 4 isolates. |

## Environment (measured)

Apple M1, **8 GB RAM**, 69 GB free · Python 3.12.5 (`python3.11` absent; 3.12 satisfies the
brief's 3.11+) · Docker 29.6.1, Compose v5.2.0 · `uv` 0.11.21 · Node v26.5.0 · Ollama absent.

**8 GB is the binding constraint.** Small local models (embedder ~130 MB, reranker ~90 MB)
are comfortable. A local generation LLM is not — hence the hosted free tier.

## Stack

`uv` + `pyproject.toml` · Python 3.12 · FastAPI · Chroma · `rank_bm25` ·
`sentence-transformers` · Streamlit · Docker + Compose · pytest · ruff · mypy ·
GitHub Actions.

## Budget

Ceiling **$1.00**. Committed so far: **$0.00**. Planned: **~$0.009** (D3, one-time).
Everything else runs on free tiers or locally.

## Progress

- [x] Requirements captured, brief transcribed, corpus verified
- [x] Architecture decisions D1–D12 locked
- [x] Phase 1.1 — scaffolding, config, domain models
- [x] Phase 1.2 — multi-format loaders (155 docs, 1872 sections, 0 ID collisions)
- [x] Phase 1.3a — fixed + structure-aware chunkers (0 offset drift, 0 content lost)
- [x] Phase 1.3b — `Embedder` protocol + semantic chunker (all three strategies validated)
- [ ] Phase 1.4 — dense + sparse indexing, dedup, ingest CLI ← *current*
- [ ] Phase 2 — hybrid retrieval · [ ] Phase 3 — generation & citations
- [ ] Phase 4 — evaluation · [ ] Phase 5 — API & dashboard · [ ] Phase 6 — polish

## Known risks

1. **Hybrid may not beat dense-only.** D1's identifier density makes it plausible, not
   certain. A documented negative result with a diagnosis ships instead of a fudged table.
2. **Free-tier rate limits are unpublished and may throttle Tier-2 eval.** Mitigated by D12.
3. **8 GB during multi-container compose** — keep the service count and image sizes lean.
4. **LLM-judge circularity** — mitigated by D11's human-agreement measurement.
5. **Semantic chunking is 28 minutes per corpus pass** and will block Phase 4's sweep unless
   sentence embeddings are cached in Phase 1.4.

## Corpus facts (measured, not estimated)

155 markdown files · 1.47 MB raw -> 1.68 MB processed (include expansion) ·
**574,274 tokens** · 1,872 sections · 2,100 headings · 797 unique inline-code identifiers ·
839 code fences · 265 sections (14.2%) exceed a 512-token budget and hold **50.5%** of all
corpus tokens · 139 sections (7.4%) fall under 50 tokens.

155 markdown files, of which **154 carry content** (`newsletter.md` is frontmatter only).
Sentence units are short: median 63 tokens, **p10 15**, and 33% fall under 50 tokens --
list items, table rows and one-line prose. That fact drives the semantic result below.

Chunker output at 512/64, whole corpus, 0 offset drift and 0 unreachable characters in all
three:

| strategy | chunks | max | median | p10 | fences intact | wall time |
|---|---|---|---|---|---|---|
| fixed | 1,337 | 512 | 512 | 464 | 82.1% | 2.6 s |
| structure | 2,305 | 750 | 210 | 67 | **95.4%** | 5.1 s |
| semantic | 2,767 | 512 | 167 | **17** | 93.6% | **1,671.8 s** |

Two measured findings, both to be reported rather than smoothed over:

* **Semantic costs ~330x more wall time** (28 minutes vs 5 seconds) because it embeds every
  sentence in the corpus. Phase 1.4 should persist sentence embeddings, or Phase 4's
  configuration sweep is impractical.
* **Semantic produces thin chunks**: p10 is 17 tokens against structure-aware's 67. Traced
  to source on a 20-document sample -- chunks emitted straight from a topic group are 39%
  under 50 tokens, while chunks produced by budget subdivision are only 14%. So the cause is
  genuine adjacent percentile breakpoints over short sentence units, **not** the budget
  code. Left uncorrected on purpose: the known remedy (embedding each sentence with a
  neighbour buffer to damp the noise from short units) is a tuning change that belongs in
  Phase 4, where there are retrieval metrics to judge it against.

## Next step

Phase 1.4 — dense (Chroma) and sparse (`rank_bm25` + identifier-preserving tokenizer)
indexes built over the same chunks and kept in sync, near-duplicate detection at cosine
> 0.95, and the ingest CLI. Grill the design first.

Open question carried into Phase 4: whether to damp semantic chunking's thin-chunk problem
with a neighbour buffer (see Corpus facts). Decide with retrieval metrics, not by taste.
