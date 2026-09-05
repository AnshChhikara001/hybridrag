# Project State

Recovery file. Current architecture, locked decisions and their rationale, progress, and
the next step. Updated whenever a decision or milestone lands.

**Last updated:** 2026-09-05 · **Status:** Slice merged to main, green in CI; response cache in

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
| D4 | ~~Gemini 2.5 Flash~~ → **`gemini-3.8-flash` free tier, with `gpt-5-nano-2025-08-07` as the paid fallback**, both behind one adapter | **Revised 2026-09-04: `gemini-2.5-flash` now returns 404 for new keys.** Exactly why an exact model id is pinned and never a `-latest` alias — the retirement was visible immediately instead of silently changing what produced a number. Free tier still preferred at $0/run versus ~$0.11; the fallback exists because the free tier is unreliable, not because it is cheap. |
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
| D18 | **OpenAI `text-embedding-3-small` is the working default; local `bge-small` stays supported** | Embedding the corpus locally took 28 minutes on one core and made the machine unusable; hosted takes 13 seconds for $0.0067. Local is kept as a documented zero-key path so a reviewer can run the project without an API key, and CI stays keyless because real-model tests are marked `slow` and excluded. Superseded the D2 default; D2's reasoning still holds for the fallback. |
| D19 | **The corpus is fetched, pinned, and never vendored** | It lived only in a temp directory and vanished on a reboot. `scripts/fetch_corpus.sh` pins tag `0.115.6` so a re-fetch reproduces the same corpus and evaluation numbers stay comparable. Checking out `docs_src` and `fastapi` alongside the docs took unresolved includes from 1 to 0. |
| D16 | **Semantic chunking stays heading-blind, overlap-free, and pure** | The loader knows every section boundary; feeding that in would make this a variant of the structure-aware chunker and Phase 4 would compare two spellings of one idea. No overlap, because a boundary chosen for a topic change is not worth blurring. Where a topic exceeds the token budget it is subdivided at the *next-largest* distance inside it, iteratively (not recursively -- `argmax` can land at a group edge, and a 10,611-token section would recurse once per sentence). |
| D17 | **`Embedder` protocol owns the query/document asymmetry** | `bge` was trained with an instruction prefix on queries only; omitting it costs retrieval quality and applying it to documents costs it again. Keeping it inside the embedder means no caller can get it wrong, and the OpenAI adapter simply carries an empty instruction. The protocol also guarantees L2-normalised vectors, so cosine reduces to a dot product for Chroma, dedup, and semantic chunking alike. |
| D26 | **The response cache keys on a model *fingerprint*, not just the prompt** | Hashing the prompt alone keeps serving temperature-0 answers after temperature is raised — a silent wrong result during exactly the parameter sweeps an evaluation consists of. Each adapter declares the settings that change its output (`LanguageModel.fingerprint`), so correctness lives with the adapter rather than in the cache's guesswork. Measured: warm call $0.000000 / 0.00s against $0.000141 / 3.85s cold, 6.4x faster end to end. |
| D27 | **On reasoning models the cache is what makes a run reproducible** | `gpt-5-nano` rejects `temperature`, so "temperature 0 for determinism" does not apply to it: the same prompt returned 172 output tokens once and 259 the next time. Re-running an evaluation would therefore move the numbers without any code changing. The cache pins the answer to the prompt, which is the only reproducibility available on that model family. A deliberate `--no-cache` / `read_only` path exists for when a fresh answer is the point. |
| D23 | **Two independent refusal paths, not one** | The pre-generation gate reads dense cosine and catches *out-of-domain* questions without spending a request. The model's own sentinel catches *in-domain but unanswerable* ones, which the gate cannot see. Measured: "capital of France" scores 0.141 and is refused for $0.000000 in 0.00s; "FastAPI's enterprise support SLA" scores 0.495, clears the gate correctly, and is refused by the model after reading the chunks. Either layer alone gets one of these wrong. |
| D24 | **The prompt must say when to answer, not only when to refuse** | An early draft said "partial information is not an answer" and produced a false refusal on "how do I run FastAPI in Docker" with five chunks of `docker.md` in context, one a complete Dockerfile. Cause was an interaction, isolated by testing one variable at a time: the strict prompt answers correctly *with* reasoning enabled, and a plain prompt answers correctly *without* it. A model with no reasoning budget cannot weigh sufficiency, so refusal wording wins by default. Chunked retrieval always delivers partial context, so the refusal condition is now narrow — "only when no block relates to the question at all". |
| D25 | **Two generation providers, because one was not enough** | Gemini's free tier returned 503 then 429 mid-build, exactly the hostility D12 assumed. `LanguageModel` made the OpenAI adapter a drop-in, so the slice finished on `gpt-5-nano` at **$0.000125 per query** while Gemini was unavailable. D4's "reversible in one config line" is now demonstrated rather than asserted. |
| D20 | **Fusion reads ranks, never scores** | BM25 is unbounded and corpus-dependent; cosine sits in [-1, 1]. Min-max normalising per query makes the fused ranking depend on each list's *spread*, so one outlier rescales everything under it. RRF reads only positions, so it needs no per-corpus calibration. `rank_constant` (RRF's `k`, renamed because `k` already means "how many results") is 60: rank 1 and rank 2 differ by 1.6%, so agreement between retrievers outranks either one's first place. |
| D21 | **The chunk store is the corpus of record** | Chroma holds vectors plus two filter fields, BM25 holds analysed terms, and neither holds text. One SQLite file holds the chunks, so the indexes are comparable by identifier set alone and a re-chunk cannot leave one copy stale. `get_many` returns chunks **in the order requested**: the caller's sequence is a ranking, and SQL's own row order would silently reorder search results into something that reads as working retrieval and measures as noise. |
| D22 | **Document discovery and include resolution use separate roots** | FastAPI keeps prose in `docs/en/docs` and its 684 example files in `docs_src`, a sibling. One root cannot serve both: narrow loses all 383 includes (and the identifiers that justify BM25), wide sweeps six `requirements*.txt` files into the corpus and rewrites every `relative_path` — and every id derived from one. |
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

Ceiling **$1.00**. **Spent to date: ~$0.0089.** Generation allowance raised to $0.50 by the user; almost none of it is needed while the free tier is up.

| item | cost |
|---|---|
| OpenAI adapter smoke test (32 tokens) | $0.000001 |
| Full corpus index, structure-aware (334,955 tokens) | $0.0067 |
| Wasted run: corpus root missing its code examples (44,152 tokens) | $0.0009 |
| Generation: ~8 `gpt-5-nano` calls while Gemini was rate-limited | ~$0.0010 |
| Cache verification (3 calls) | $0.0003 |
| Gemini generation (free tier) | $0.0000 |
| **Total** | **~$0.0089** |

The wasted run is recorded rather than quietly dropped: it embedded a prose-only corpus
built from the wrong root, and is what led to D22. The corrected rebuild cost **$0.0000** —
every chunk hit the cache, which also proves the rebuild reproduces the original chunk text
byte for byte.

Re-indexing the same configuration is $0 -- the embedding cache is keyed on model and
text. What costs money is each genuinely new chunking configuration, at ~$0.007 each.
BM25 costs nothing: only the dense half of hybrid retrieval spends.

## Progress

- [x] Requirements captured, brief transcribed, corpus verified
- [x] Architecture decisions D1–D12 locked
- [x] Phase 1.1 — scaffolding, config, domain models
- [x] Phase 1.2 — multi-format loaders (155 docs, 1872 sections, 0 ID collisions)
- [x] Phase 1.3a — fixed + structure-aware chunkers (0 offset drift, 0 content lost)
- [x] Phase 1.3b — `Embedder` protocol + semantic chunker (all three strategies validated)
- [x] Vertical slice — tokenizer, BM25 index, Chroma index, embedding cache, OpenAI adapter
- [x] Vertical slice — chunk store, RRF retriever, reproducible build and query scripts
- [x] Vertical slice — **complete**: grounded generation, inline citations, two refusal paths
- [x] CI green on a clean Linux runner; slice merged to `main` (PR #3)
- [x] LLM response cache (D12) — fingerprint-keyed, read-only bypass
- [ ] Phase 4 — evaluation: golden set + Tier-1 metrics ← *current*
- [ ] Then Phase 2's reranker, measured against that baseline
- [ ] Still open from Phase 1: **near-duplicate detection** (`dedup_threshold` is wired to nothing)
- [ ] Phase 2 — hybrid retrieval · [ ] Phase 3 — generation & citations
- [ ] Phase 4 — evaluation · [ ] Phase 5 — API & dashboard · [ ] Phase 6 — polish

## Known risks

1. ~~**Hybrid may not beat dense-only.**~~ First evidence in, on three hand-picked queries:
   hybrid matches the better retriever every time and beats both on the identifier query.
   Not yet a measurement — three queries chosen by hand are an illustration, and the golden
   set in Phase 4 is what turns this into a number that can be reported.
6. **Dense returns near-duplicate chunks from one document.** All three top results for a
   query often come from the same page, so the generator sees one section three times
   instead of three sources. Diversity is a Phase 2 concern, once metrics can judge it.
7. **The retrieval comparison is only as good as the corpus.** A one-directional index
   check let 1,892 stale vectors survive a rebuild undetected; verification is now
   two-way, in `scripts/build_index.py`.
2. **Free-tier rate limits are unpublished and may throttle Tier-2 eval.** Mitigated by D12.
3. **8 GB during multi-container compose** — keep the service count and image sizes lean.
4. **LLM-judge circularity** — mitigated by D11's human-agreement measurement.
5. **Semantic chunking is 28 minutes per corpus pass** and will block Phase 4's sweep unless
   sentence embeddings are cached in Phase 1.4.

## Corpus facts (measured, not estimated)

Pinned at FastAPI **0.115.6**. 141 markdown files, **140 with content** · 1,570 sections ·
**0 unresolved includes** · 349 sections carry `@app.` examples only because `{* ... *}`
directives are expanded. Token counts differ by tokenizer, which matters because one is
billed: **360,501** OpenAI (cl100k) vs **464,184** bge.

Sentence units are short: median 63 tokens, p10 15, 33% under 50 tokens.

Chunker output at 512/64:

| strategy | chunks | billable tokens | cost/run | overlap |
|---|---|---|---|---|
| fixed | 1,097 | 408,132 | $0.0082 | 1.13x |
| structure-aware | 1,892 | 363,642 | $0.0073 | 1.01x |

Fixed costs more because its 64-token overlap re-embeds 13% of the corpus.

**Indexed, structure-aware, OpenAI embeddings:** 1,892 chunks · 2 requests ·
**334,955 tokens billed** · **$0.0067** · dense build **13.1s**, sparse **0.2s**,
end to end **17.4s**. Dense and sparse id sets match exactly. On disk: 28 MB Chroma,
2.2 MB sparse JSON, 15 MB embedding cache.

Reproduced by `scripts/build_index.py` in **7.6s** at **$0.0000** (fully cached), with all
three artefacts verified to hold exactly the same 1,892 ids.

## Retrieval, first end-to-end results

Three hand-picked queries, top 3 per retriever, counting results from the canonical page:

| query | dense only | sparse only | hybrid (RRF) |
|---|---|---|---|
| `how do I run it with docker` | **3/3** `docker.md` | 1/3 — wrong page first | **3/3** |
| `response_model` | **3/3** `response-model.md` | 1/3 — changelog noise | **3/3** |
| `HTTPException status_code` | 0/3 — misses the page | 1/3 `handling-errors.md` | **3/3** |

Hybrid matches the stronger retriever on every query and beats both on the third. The
mechanism is visible in the scores: hybrid's top result there was **dense rank 6 + sparse
rank 11** — neither retriever's own first choice. Agreement promoted it, which is precisely
what RRF is for.

Caveat: three queries chosen by hand illustrate the mechanism, they do not measure it.
Phase 4's golden set is what produces a number worth reporting.

Against the same work locally at ~28 minutes, that is **~96x faster**.


## Next step

Phase 1.4 — dense (Chroma) and sparse (`rank_bm25` + identifier-preserving tokenizer)
indexes built over the same chunks and kept in sync, near-duplicate detection at cosine
> 0.95, and the ingest CLI. Grill the design first.

Open question carried into Phase 4: whether to damp semantic chunking's thin-chunk problem
with a neighbour buffer (see Corpus facts). Decide with retrieval metrics, not by taste.

## Generation, first end-to-end results

`scripts/ask.py "how do I run FastAPI in Docker"` on `gpt-5-nano-2025-08-07`:
**926 in / 198 out · $0.000125 · 3.77s**, three resolved citations all from `docker.md`,
zero fabricated citations.

**Retrieval confidence, measured** (6 questions each, top dense cosine):

| | range |
|---|---|
| in-domain | 0.4602 – 0.6541 |
| out-of-domain | 0.1410 – 0.2414 |
| gap | **+0.2188** |

Threshold set at **0.30**, below the midpoint on purpose: the two errors are not
symmetric. A false refusal is final and the user gets nothing; a false accept is caught
downstream by the model's own refusal for the price of one request.

**Known weakness:** `gpt-5-nano` answers are verbose and repetitive — it restates the same
claim across sentences. D4 predicted nano-class judgement would be weaker, and it is. Worth
re-measuring on Gemini once the free tier is reachable, and a good Phase 4 comparison.

**Not built yet, and labelled as such everywhere it appears:** citation coverage,
completeness and the composite confidence score. The fields exist in `AnswerConfidence`
and return `None`, so the API contract is settled without anything reporting a score it
did not compute.
