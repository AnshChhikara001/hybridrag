# Project State

Recovery file. Current architecture, locked decisions and their rationale, progress, and
the next step. Updated whenever a decision or milestone lands.

**Last updated:** 2026-09-08 · **Status:** Phases 1 and 4 **complete**; Phase 2 **closed** — the reranker was built, measured on Tier 1, and does not ship as the default (a negative result); RRF weights are now swept and configurable. Phase 5.1 (API) and 5.2 (dashboard) are **built**; default generation provider switched from Gemini to `gpt-5-mini` (D55) after the dashboard's doubled request load exposed the free tier's 5 req/min pace gate. Phase 6 (README, architecture diagram, numbers-first case study) is **built**. Phase 5.3 (Docker) is the one piece of the original six-phase brief not shipped — this is a local-only portfolio deliverable by choice, recorded as a known gap rather than a silent omission.

---

## What this project is

A hybrid-retrieval RAG system over the public FastAPI documentation, standing in for "a
company's internal docs." Dense vectors + BM25, fused with Reciprocal Rank Fusion; a
cross-encoder reranker was built and measured but does not currently sit in the default
path (D53). Answers are grounded, carry inline citations, and every citation is verified.
Architecture decisions are made from a measured evaluation suite rather than asserted.

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
| D28 | **The changelog stays in the corpus; it is never a question source** | `release-notes.md` is 666 of 1,892 chunks — 35% of the corpus. Excluding it would lift every retrieval number, which is exactly why it stays: real internal corpora contain changelogs, and removing one to flatter your own metrics is undisclosable cherry-picking. It also gives the reranker a real job (distractor suppression) to be measured on. Project narrative (`history-design-future.md`, `alternatives.md`, `fastapi-people.md`, …) is likewise kept but excluded as a source — the first three lookup candidates were "which libraries did the author contribute to?", which measures nothing a docs assistant is for. |
| D29 | **Each question category is scored by its own rule** | lookup: its one span covered. multi_hop: **all** spans covered, because a question needing two documents is not half-answered by one, and its spans must be in different documents. ambiguous: **any** acceptable span, since several answers are legitimately correct. no_answer: **excluded from recall entirely** and scored on refusal rate — scoring it 0 would drag the number down meaninglessly. |
| D30 | **Golden spans are stored as quotes, resolved to offsets at load, matched whitespace-tolerantly** | Offsets in the file go stale silently when anything upstream shifts them, still parsing while pointing at the wrong text. A quote fails loudly instead, and is the only form a human can verify (D11). Matching then had to tolerate whitespace: models reliably collapse `\n\n` to `\n` when copying, which was the single largest cause of rejected candidates — quotes character-perfect for hundreds of characters diverging only at a paragraph break. Exact match is tried first; the fallback still demands a unique match and still resolves to the document's own bytes. Recovered lookups 18→24 and multi-hop 13→15 at **$0**. |
| D31 | **Gemini's free tier cannot run the evaluation** | Measured, where D12 could only assume: `gemini-3.8-flash` free tier is **5 requests/minute and 20 per day** (429 `RESOURCE_EXHAUSTED`, quotaValue 5 then 20). Quotas are per-model, so other models still answer, but 20/day cannot serve a 50-question sweep. A client-side throttle now paces to the RPM quota — reacting with backoff alone spends the retry budget on calls that were always going to fail. **Supersedes Q8:** OpenAI runs evaluation; Gemini stays the $0 interactive-demo path, where 20/day is ample. |
| D32 | **Verbatim documentation is transported in sentinel blocks, not JSON** | 52 of 130 replies were unparseable, for two reasons the corpus itself causes: the docs contain `"` characters (`a "context manager"`) that models fail to escape inside JSON strings, and they contain ``` code fences, so a regex stripping a markdown fence around a JSON envelope strips a Python example out of the middle of a quote instead. `<<<SPAN>>>`/`<<<ENDSPAN>>>` need no escaping and cannot collide with documentation. Multi-hop acceptance went 5 → 13 on that change alone. |
| D26 | **The response cache keys on a model *fingerprint*, not just the prompt** | Hashing the prompt alone keeps serving temperature-0 answers after temperature is raised — a silent wrong result during exactly the parameter sweeps an evaluation consists of. Each adapter declares the settings that change its output (`LanguageModel.fingerprint`), so correctness lives with the adapter rather than in the cache's guesswork. Measured: warm call $0.000000 / 0.00s against $0.000141 / 3.85s cold, 6.4x faster end to end. |
| D27 | **On reasoning models the cache is what makes a run reproducible** | `gpt-5-nano` rejects `temperature`, so "temperature 0 for determinism" does not apply to it: the same prompt returned 172 output tokens once and 259 the next time. Re-running an evaluation would therefore move the numbers without any code changing. The cache pins the answer to the prompt, which is the only reproducibility available on that model family. A deliberate `--no-cache` / `read_only` path exists for when a fresh answer is the point. |
| D23 | **Two independent refusal paths, not one** | The pre-generation gate reads dense cosine and catches *out-of-domain* questions without spending a request. The model's own sentinel catches *in-domain but unanswerable* ones, which the gate cannot see. Measured: "capital of France" scores 0.141 and is refused for $0.000000 in 0.00s; "FastAPI's enterprise support SLA" scores 0.495, clears the gate correctly, and is refused by the model after reading the chunks. Either layer alone gets one of these wrong. |
| D24 | **The prompt must say when to answer, not only when to refuse** | An early draft said "partial information is not an answer" and produced a false refusal on "how do I run FastAPI in Docker" with five chunks of `docker.md` in context, one a complete Dockerfile. Cause was an interaction, isolated by testing one variable at a time: the strict prompt answers correctly *with* reasoning enabled, and a plain prompt answers correctly *without* it. A model with no reasoning budget cannot weigh sufficiency, so refusal wording wins by default. Chunked retrieval always delivers partial context, so the refusal condition is now narrow — "only when no block relates to the question at all". |
| D25 | **Two generation providers, because one was not enough** | Gemini's free tier returned 503 then 429 mid-build, exactly the hostility D12 assumed. `LanguageModel` made the OpenAI adapter a drop-in, so the slice finished on `gpt-5-nano` at **$0.000125 per query** while Gemini was unavailable. D4's "reversible in one config line" is now demonstrated rather than asserted. |
| D20 | **Fusion reads ranks, never scores** | BM25 is unbounded and corpus-dependent; cosine sits in [-1, 1]. Min-max normalising per query makes the fused ranking depend on each list's *spread*, so one outlier rescales everything under it. RRF reads only positions, so it needs no per-corpus calibration. `rank_constant` (RRF's `k`, renamed because `k` already means "how many results") is 60: rank 1 and rank 2 differ by 1.6%, so agreement between retrievers outranks either one's first place. |
| D21 | **The chunk store is the corpus of record** | Chroma holds vectors plus two filter fields, BM25 holds analysed terms, and neither holds text. One SQLite file holds the chunks, so the indexes are comparable by identifier set alone and a re-chunk cannot leave one copy stale. `get_many` returns chunks **in the order requested**: the caller's sequence is a ranking, and SQL's own row order would silently reorder search results into something that reads as working retrieval and measures as noise. |
| D22 | **Document discovery and include resolution use separate roots** | FastAPI keeps prose in `docs/en/docs` and its 684 example files in `docs_src`, a sibling. One root cannot serve both: narrow loses all 383 includes (and the identifiers that justify BM25), wide sweeps six `requirements*.txt` files into the corpus and rewrites every `relative_path` — and every id derived from one. |
| D33 | **One index set per chunking strategy**, never a shared index with a filter | The nine-arm grid needs three dense collections and three BM25 files. Sharing would corrupt the comparison in a different way per index: BM25's IDF is document frequency *over the index*, so three strategies in one file triples N and reweights every term; and HNSW is one graph, where a metadata filter prunes after the approximate traversal, so recall degrades by however much of the graph belongs to the other strategies. Naming lives in `indexing/layout.py` so the builder, the query script and the harness cannot disagree — a disagreement shows up as an empty index, not an error. |
| D34 | **nDCG's gains are *marginal* span coverage, and its ideal comes from the index** | Span truth is binary per span, and nDCG needs a graded gain. A chunk is therefore worth the coverage it *adds* to what outranks it, which pays a strategy nothing for re-delivering text already retrieved — the fixed-size strategy's 64-token overlap earns zero. The ideal ranking is built greedily from every chunk in that strategy's index that touches a span, never from what the arm returned: normalising against an arm's own results would score a retriever that found one of three needed chunks as perfect for ordering that one correctly. |
| D35 | **Every comparison is a paired bootstrap; arms that cannot be separated are reported as unseparated** | At 29 answerable questions a five-point gap is roughly one question changing its mind. Resampling the arms independently would mostly measure which questions each draw contained, because question difficulty dominates the variance — so the same resampled questions are scored under both arms, cancelling that term. Reported as a difference with its interval plus P(>0), and where the interval spans zero the report says so rather than calling the larger mean a win. |
| D36 | **MRR is the rank at which a question *becomes answerable*, not the rank of the first relevant chunk** | For a lookup these are the same number. For multi-hop they are not: a question that genuinely needs two documents is not answered at the rank of the first one, and scoring it there would report the strict all-spans rule (D29) as satisfied by half the evidence. Reciprocal rank is also zeroed beyond rank 10, since a span found at rank 34 never reaches the generator's context. |
| D37 | **`gpt-5-mini` judges Tier 2; `gpt-5-nano` was disqualified by measurement** | The cheap judge is not merely weaker, it is **indistinguishable from chance**: kappa **0.007** [-0.040, +0.077] against 20 human labels, versus mini's **0.730** [+0.459, +1.000]. Two failure modes, both systematic: nano marked **5 of 6 correct refusals `incorrect`**, so it cannot apply the rule that declining an unanswerable question is right; and it downgraded 8 correct answers to `partial`. Measured rather than assumed, for $0.016 — and the 5x cost difference that made nano attractive buys nothing if the number it produces is noise. |
| D38 | **A trivial baseline is printed beside every agreement figure** | The human labels ran 18 correct / 2 incorrect, and on a split that lopsided a judge answering "correct" unconditionally scores **90% raw agreement**. Raw agreement alone would therefore have made even nano look defensible at a glance. Cohen's kappa scores that judge 0.000, which is why it is the headline number and why the baseline row stays in the report. |
| D39 | **Every failure is attributed to a stage by joining Tier 2 to Tier 1** | "34% of answers are wrong" says nothing about what to fix. Each non-correct answer is attributed against Tier 1's own record of whether *that same arm* retrieved the answer span: evidence missing = retrieval, evidence present = generation. Retrieval is checked **first**, because a refusal on a question whose evidence never arrived is the right response to bad context, and filing it under "wrong refusal" would hide a retrieval problem behind a prompt problem. |
| D40 | **Cost per answer is priced from token counts, not from what the run paid** | A cache hit truthfully reports `cost_usd = 0.0`, which is correct for spend accounting and useless for unit economics. Two arms running the identical model differed by 40% in one draft purely because one was partly cached, and after a full re-run both read **$0.000000** — a report claiming the system is free. Pricing from stored token counts survives a free re-run: **$0.000144 per answer**, against $0.000143 measured live. |
| D41 | **Grounding is reported but explicitly not validated** | The judge's *correctness* agreement with a human was measured (kappa 0.730). Its *grounding* judgement never was — no human labelled grounding — so the 100% grounding figure is published with that stated, not folded in under the same kappa. A metric with no variance across 70 answers is also not discriminating anything, which is itself worth saying rather than presenting as a triumph. |
| D42 | **Deduplication is scoped to within a document, never across the corpus** | Corpus-wide removal was built first and **Tier 1 caught it before it shipped**: it deleted a chunk of `tutorial/request-forms.md` because `tutorial/request-files.md` carried a near-identical warning at cosine 0.982, leaving one golden question (`lookup-006`) with no evidence at all, and hybrid/structure Recall@5 fell 0.759 → 0.724. Inside one document a repeat really is redundant — include expansion emits the same example once per Python version — but across documents the same text is **not** redundant, because which page it sits on is part of both its meaning and its citation. 99 of 116 structure removals were within-document anyway, so almost nothing of value is given up. |
| D43 | **Dedup is kept although it does not improve retrieval, because it was measured** | Across 18 paired comparisons (9 arms x 2 metrics) **not one difference is significant** — every interval spans zero, and each apparent ±0.034 is one question of 29. So the brief's deduplication requirement buys index size, not quality: 1.0% of the fixed index, 5.2% of structure, 5.6% of semantic, at $0 because it reuses the vectors the dense index already needs. Reported as a null result rather than dressed up. |
| D44 | **The processed store exists for durability, not for speed** | Measured before claiming otherwise: parsing the corpus takes **0.14s** and loading the store **0.01s**, so the saving is 0.12s per build — negligible, against an assumed ~2.5s. What it actually buys is the brief's stated reason: the indexes rebuild with no raw checkout and no network, include expansion already applied, on a corpus that has been lost to a temp directory once already (D19). Round-trip verified identical to a fresh parse. Staleness is keyed on raw bytes, which cannot see an edited *included* file — stated, with `--refresh` for it, rather than left as a silent hole. |
| D45 | **Citations are never parsed from code, fenced or inline** | The corpus documents a CLI, so answers quote console output — `fastapi dev` prints a process id as `[2248755]` — and Python such as `sys.argv[1]`. Read naively both become citations, and the two failures differ: the first is reported to the user as a **fabricated source** the model never claimed, and the second silently attributes a sentence to whichever chunk happens to rank first. Found in the response cache, not in theory — 2 bracketed numbers inside 155 stored code fences. Fenced blocks and inline spans are excluded from citation parsing. |
| D46 | **A trailing citation block belongs to the sentence before it** | This generator ends answers with `... for operations. [1] [2] [4] [5]`, and puts a bare `[4]` on the line after a code fence. Both are non-assertions under every claim rule, so the first splitter **discarded them**: `lookup-001` carries four resolved citations and was reported as having none, coverage 0.000. Caught by running the splitter over the whole golden set for **$0** before paying for verification — the run that would have measured a fiction was stopped 46 calls in. One consequence is published rather than hidden: attaching citations to the sentence they follow means an answer that dumps every source at the end scores low coverage. That is the metric working. Per-claim attribution is exactly what it measures, and a reader who cannot tell which passage backs which sentence has not been given it. |
| D47 | **The citation verifier is validated by a negative control, not by human labels** | A model grading a model is worth nothing unless the grader is checked (D11, D37), but a second hand-labelling round is expensive and slow. Instead every cited claim is verified **twice** — against the blocks it cited, and against blocks from the same corpus it never cited. A verifier that says "supported" out of politeness cannot separate those. Measured: **0.900 supported on real citations against 0.050 on random ones, a separation of +0.850 [+0.725, +0.950]**, interval excluding zero, for $0.013 and no human time. This is a *weaker* claim than kappa and is published as such: it shows the verifier reads, not that it agrees with a person. |
| D48 | **A claim is judged against all the blocks it cites, together** | The brief says to send each citation-claim pair to the judge. Done literally, a claim citing `[1][3]` because half the evidence sits in each block is judged twice and marked unsupported both times. Splitting evidence across blocks is legitimate model behaviour, so the cited set is judged jointly — measuring what a reader cares about instead of manufacturing false alarms. When a set fails, every citation in it is flagged, because nothing available can say which half was at fault. |
| D49 | **Retrieval confidence is calibrated on a measured range, and the gate cannot do what it appears to** | Raw cosine occupies a narrow band, so averaging it against two rates that genuinely span 0–1 would let it drag every composite to the middle. Measured over the golden set: answerable questions score **0.361–0.721** (median 0.566, p90 0.664), so the ramp runs from the refusal threshold 0.30 to 0.70 — almost everything lands in its upper half and exactly one question saturates it. The same measurement produced an uncomfortable finding, kept: the six **unanswerable** questions score **0.379–0.640**, *inside* the answerable range. D23's threshold was calibrated on out-of-domain questions (0.14–0.24), which are the easy case; no threshold on this number separates in-domain-but-unanswered ones. The gate is kept for what it does handle and is not asked to do more, and the model's own refusal carries that category — which the 100% refusal rate on `no_answer` confirms. |
| D50 | **Composite weights are equal, unfitted, and renormalised over what was actually measured** | With 29 answerable questions and two failures there is nothing to fit weights on that would not simply memorise this corpus, and a tuned weighting invites a question about overfitting that the sample size cannot answer. Equal weights make no claim to defend. An unmeasured component is renormalised away rather than scored zero: scoring it zero would penalise an answer for a check nobody ran, so the free path would look worse than the verified one on an identical answer — a number measuring the caller's configuration rather than the answer. `confidence.components` records which dimensions fed each composite, so a cheap composite and a thorough one are never mistaken for each other. |
| D51 | **A refusal returns what was found, what could not be established, and which pages to open** | "I don't know" is honest and nearly worthless — the reader learns only that this route failed. All three facts come free from the retrieval results with no extra model call, so the cheapest possible response, the one a user is most likely to receive on a bad day, is also actionable. The sentinel phrase still opens the rendered text, because the Tier-2 judge and every downstream string check key off that line: the structure is an addition, never a replacement. The two refusal paths say different things, since a gate refusal means nothing came close and a model refusal means several pages came close and none of them said it. |
| D15 | **Chunkers own boundary placement only** | `Chunker.chunk()` turns spans into validated chunks once, in the base class, so the three strategies differ in boundary placement and nothing else -- the variable Phase 4 isolates. |
| D52 | **Reranker built on fastembed's ONNX cross-encoder, not `sentence-transformers`** | D8 named the model (`ms-marco-MiniLM-L-6-v2`), not the runtime. fastembed -- already this project's embedding dependency -- ships `Xenova/ms-marco-MiniLM-L-6-v2`, an ONNX build of the identical weights `sentence-transformers` would load through PyTorch. Adding `sentence-transformers` would mean a second inference runtime and a `torch` install running to hundreds of MB to a few GB on the 8 GB dev machine (see Environment, below); fastembed serves the same weights in ~80 MB over onnxruntime, already installed. Same model, same $0, one fewer heavy dependency. |
| D53 | **The reranker does not separate from plain hybrid on Tier 1, and does not ship as the default** | Measured, paired bootstrap, 5-wide horizon (top-20 in, top-5 out, matching what the reranker actually returns): Recall@5 moves **-0.069 [-0.172, +0.000]** on fixed, **+0.000 [-0.103, +0.103]** on structure, **-0.034 [-0.172, +0.103]** on semantic -- no interval separates from zero, and fixed's own point estimate is a net loss. The distractor rate (top-5 slots from `release-notes.md`, D28) does not consistently improve either: fixed **4.1% → 6.9%** (worse), structure **9.7% → 10.3%** (worse), semantic **10.3% → 8.3%** (better). Full report: `evals/reports/reranker.md`. Leading candidate explanation, not proven: RRF fusion already combines two complementary signals -- BM25's exact-match on this corpus's 797 code identifiers (D1) and dense semantic similarity -- and a general-purpose MS MARCO cross-encoder, trained on web-passage QA rather than API documentation, is a *third*, weaker opinion on exactly the queries where lexical match already wins. Reported as a negative result, the way D43 reported dedup's, rather than shipped for the portfolio's sake. |
| D54 | **RRF weights are configurable and swept; the equal 1:1 default is kept** | The brief requires fusion to be "configurable so you can tune it" (D20 built the knob; this exercises it). Dense:sparse swept 0.25:1 to 4:1 on the winning arm (hybrid/fixed), $0, no language model. Recall@5 is flat at 0.897 from 0.5:1 through 4:1 and drops only when sparse dominates outright (0.25:1 → 0.828). nDCG@10 is highest at 0.5:1 (favouring sparse 2x) at 0.819 against 1:1's 0.794, but only Recall@5 was paired-bootstrapped here, and every non-baseline interval on it includes zero -- so the apparent nDCG edge is not established as more than noise, and the unweighted default stays. Full table: `evals/reports/rrf_weights.md`. |
| D55 | **Supersedes D31's "$0 interactive-demo path": default generation switched from Gemini free tier to `gpt-5-mini-2025-08-07`**, via `HYBRIDRAG_GENERATION_PROVIDER`/`_MODEL` in `.env` (code default in `config.py` untouched, so D4/D31 still describe what a clean checkout does) | D31 assumed 20 req/day was ample for interactive use; the Phase 5.2 dashboard breaks that assumption by design (every question fires hybrid *and* dense-only concurrently, 2 generation calls each, to show the comparison the brief asks for on every question rather than behind a toggle). Measured directly: `GeminiModel`'s client-side pacer (`gemini.py:137`) hard-caps the *process* to 5 req/min regardless of concurrency, so two concurrent dashboard calls already serialise behind a 12s gate — and a single **unpaced, unthrottled** call still measured **28.5s** (free-tier backend load, not just quota). The same question pair on `gpt-5-mini`: **3.8s and 3.9s, running genuinely concurrently**, both answers correct and cited, at **$0.0012 + $0.0011 = ~$0.0023/question-pair** — the $0.50 generation allowance (already raised for this) covers ~200 such pairs. Gemini stays available and free (`HYBRIDRAG_GENERATION_PROVIDER=gemini`) for anyone running this without spending money; it is no longer what the dashboard uses by default. |

## Environment (measured)

Apple M1, **8 GB RAM**, 69 GB free · Python 3.12.5 (`python3.11` absent; 3.12 satisfies the
brief's 3.11+) · Docker 29.6.1, Compose v5.2.0 · `uv` 0.11.21 · Node v26.5.0 · Ollama absent.

**8 GB is the binding constraint.** Small local models (embedder ~130 MB, reranker ~90 MB)
are comfortable. A local generation LLM is not — hence the hosted free tier.

## Stack

`uv` + `pyproject.toml` · Python 3.12 · FastAPI · Chroma · `rank_bm25` ·
`fastembed` (embeddings *and*, since Phase 2, cross-encoder reranking, D52 -- no
`sentence-transformers`/`torch` dependency was added) · Streamlit · Docker + Compose ·
pytest · ruff · mypy · GitHub Actions.

## Budget

Ceiling **$1.00**. **Spent to date: ~$0.383.** Generation allowance raised to $0.50 by the user; Phase 5.2 (D55) is now the first thing actually spending from it, at ~$0.0023/question-pair on `gpt-5-mini`.

| item | cost |
|---|---|
| OpenAI adapter smoke test (32 tokens) | $0.000001 |
| Full corpus index, structure-aware (334,955 tokens) | $0.0067 |
| Wasted run: corpus root missing its code examples (44,152 tokens) | $0.0009 |
| Generation: ~8 `gpt-5-nano` calls while Gemini was rate-limited | ~$0.0010 |
| Cache verification (3 calls) | $0.0003 |
| Golden-set drafting, `gpt-5-mini` (44 billed calls over 5 runs) | $0.2538 |
| Gemini generation (free tier) | $0.0000 |
| Citation verification + negative control (96 calls, `gpt-5-mini`) | $0.0130 |
| Tier-2 re-run after the refusal changed (structured refusal) | $0.0117 |
| Phase 3 `--verify` demos and the free-scan cache misses | $0.0021 |
| Phase 2: reranker build, 9-arm Tier-1 rerank comparison, RRF weight sweep (local cross-encoder + BM25/dense, no LLM, ~168s wall-clock) | $0.0000 |
| Phase 5.2 debugging: raw `gpt-5-mini` smoke test + one live hybrid/dense-only pair through the dashboard's own code path (D55) | $0.0024 |
| **Total** | **~$0.292** |

The wasted run is recorded rather than quietly dropped: it embedded a prose-only corpus
built from the wrong root, and is what led to D22. The corrected rebuild cost **$0.0000** —
every chunk hit the cache, which also proves the rebuild reproduces the original chunk text
byte for byte.

Re-indexing the same configuration is $0 -- the embedding cache is keyed on model and
text. What costs money is each genuinely new chunking configuration. Measured, for the
two indexes Phase 4 needed: **fixed $0.0081** (405K tokens) and **semantic $0.0117**
(587K tokens -- higher because it embeds every *sentence* to find topic boundaries, then
the chunks). Rebuilding **structure cost $0.0000**, every vector served from cache.
BM25 costs nothing, and **the whole nine-arm evaluation grid costs $0.000000** once the
query embeddings are cached: Tier 1 involves no language model at all.

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
- [x] Phase 4 — golden-set schema, span resolution, relevance rules (28 tests)
- [x] Phase 4 — 59 candidates drafted (24 lookup / 15 multi-hop / 10 ambiguous / 10 no-answer)
- [x] Phase 4 — golden set hand-verified and cut to 35 (18/6/5/6), 40 spans, 0 failures
- [x] Phase 4 — **Tier-1 metrics, the 9-arm grid, bootstrap CIs and the report** (43 tests)
- [x] Phase 4 — Tier 2: judge selection by measured human agreement, then answer quality
- [x] Phase 1 **complete** — near-duplicate detection (document-scoped) and the processed store
- [x] Phase 3 — claim segmentation, claim-level citation verification with a negative control
- [x] Phase 3 — composite confidence scorer, structured refusal (**complete**, 505 tests)
- [x] Phase 2 **complete** — cross-encoder reranker built and measured on Tier 1 (D52, D53);
      does not separate from plain hybrid and does not ship as the default -- a reported
      negative result, not a gap. RRF weights swept and configurable (D54). 532 tests.
- [x] Phase 5.1 — FastAPI service (`/health`, `/v1/documents`, `/v1/ask` with hybrid/dense-only
      ablation, `/v1/ingest` with document-scoped re-ingest and a full BM25 rebuild); 14 offline
      tests, `Resources` built once at startup, per-store locks for thread safety
- [x] Phase 5.2 — Streamlit dashboard (`dashboard/`): every question runs hybrid and
      dense-only concurrently and shows both, with citations, retrieved chunks (dense/sparse/
      both provenance), the confidence breakdown, and cost/latency telemetry; 32 tests
- [ ] Phase 5.3 — Docker + seed script (not built; see "Known risks" and Next step)
- [x] Phase 6 — README, Mermaid architecture diagram, LICENSE, numbers-first case study
      pulling every headline figure straight from `evals/reports/`

## Known risks

1. ~~**Hybrid may not beat dense-only.**~~ **Settled, measured.** Across all three chunking
   strategies hybrid leads both single retrievers on Recall@5 (+0.069 to +0.103). Only one
   of those leads clears a 95% paired interval at n=29, and one contrast goes the other way
   (dense/structure beats hybrid/structure on Recall@budget, −0.069). Reported as such.
6. **Retrieved context concentrates on a few documents** — measured, and not the problem
   first assumed. Across the 29 answerable questions the top 5 chunks come from a mean of
   **3.28 distinct documents**, and 10 of 29 questions draw 3+ chunks from a single page.
   But they are *not* near-duplicates: **0 of 290 retrieved pairs** reach cosine 0.95, so
   deduplication cannot address this. **Tried the reranker on it (D53) and it did not
   move the distractor rate consistently** -- the diversity problem stands, unresolved,
   an open item rather than a closed one with a wrong fix quietly shipped.
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


## Tier-1 retrieval results (2026-09-06, 29 answerable questions)

Full report: `evals/reports/retrieval.md`; raw records: `evals/reports/retrieval_results.json`.
The grid is nine arms — three chunking strategies x three retrievers — scored on span
coverage, every number carrying a 95% percentile bootstrap interval, every comparison a
*paired* bootstrap over the same resampled questions.

| arm | Recall@5 | nDCG@10 | Recall@2000 tok |
|---|---|---|---|
| **hybrid/fixed** | **0.897** [0.793, 1.000] | 0.792 | **0.897** |
| hybrid/semantic | 0.828 [0.690, 0.966] | 0.788 | 0.862 |
| dense/fixed | 0.828 [0.690, 0.966] | 0.742 | 0.793 |
| hybrid/structure | 0.759 [0.586, 0.897] | 0.738 | 0.759 |
| sparse/fixed | 0.793 [0.621, 0.931] | 0.779 | 0.724 |

Four results worth keeping, two of them uncomfortable:

1. **Hybrid beats both of its parts in all three strategies** on Recall@5 — the project's
   central claim, now a number rather than three hand-picked queries. Honest caveat: at
   n=29 only `hybrid − sparse` on fixed chunking clears a 95% paired interval
   (+0.172 [+0.034, +0.310]). The rest are leads, not proofs, and the report says so.
2. **The dumbest chunker wins.** Fixed-size beats structure-aware by +0.138 [+0.034, +0.276]
   on both Recall@5 and Recall@budget — the one chunking difference that *is* significant.
   The elaborate strategies lose to a sliding window. Reported rather than buried.

   The *mechanism* took two attempts, and the first was wrong. The obvious explanation —
   heading-shaped chunks split answer spans — is false: measured, fixed holds **100%** of
   the 40 spans whole, structure **98%**, semantic **95%**. A 2-point difference cannot
   produce a 14-point recall gap, and equalising the token budget (D14) does not close it
   either, so it is not simply that fixed retrieves more text.

   What is measured is **distractor suppression**. Across the 145 top-5 slots the 29
   questions fill, chunks from `release-notes.md` occupy 4% under fixed against **9% under
   structure** — twice as often — even though fixed's index is a *higher* proportion
   changelog (40% against 35%). Chunking the changelog into 512-token windows merges many
   small release entries into fewer, topically diffuse chunks that match a specific query
   less strongly, where structure gives each entry its own tight, precisely matchable
   chunk. This is a measured contributor, not a proof that it is the whole gap; the honest
   statement is that fixed wins, that span splitting is ruled out, and that distractor
   competition is the leading surviving explanation. It is also exactly the job D28 kept
   the changelog in the corpus to give the reranker.
3. **Semantic chunking is not worth its cost.** It sits between the two, separated from
   neither (fixed − semantic: +0.069 [−0.103, +0.241]), while costing $0.0117 and 3 minutes
   to build against fixed's $0.0081 and 15 seconds.
4. **The pre-generation refusal gate catches 0 of 6 unanswerable questions** — and produces
   0 false refusals. Exactly what D23 predicted: these are *in-domain but unanswerable*
   questions, so they retrieve real FastAPI chunks and score like real questions (no_answer
   max 0.640 against answerable min 0.361). The gate's value is against *out-of-domain*
   queries, where it does separate; everything in this set is the model's own sentinel to
   catch. The threshold of 0.30 is confirmed safe — nothing answerable falls below it.

Remaining failures for the leading arm are three questions at rank 10, all ranking failures
rather than chunking ones: `lookup-017`, `multi_hop-005`, `multi_hop-013`. Multi-hop is the
weak category at 0.667 against lookup's 0.944, which is the reranker's target in Phase 2.

Two instrumentation bugs were found and fixed while producing this, both of which would
have been reported as findings: the first arm measured 398 ms against every later arm's
3 ms because it paid the query-embedding round trip the others read from cache (queries are
now warmed before any arm is timed), and "questions with no covering chunk" reported 6 on
every strategy because it counted the six `no_answer` questions, which have no span by
definition.

**Correction to an earlier note:** semantic chunking is not 28 minutes per corpus pass. That
figure was local embedding; hosted, it is **177 seconds**. The 28-minute number stands only
for the keyless path.

## Judge validation (2026-09-06, 20 hand-adjudicated items)

Full report: `evals/reports/judge_agreement.md`. Answers came from `gpt-5-nano` on the
Tier-1 winning arm (`hybrid/fixed`); the sample was stratified toward failures on purpose,
so its correctness rate says nothing about the system.

| judge | raw agreement | kappa [95% CI] | cost |
|---|---|---|---|
| `gpt-5-nano` | 30% | 0.007 [-0.040, +0.077] | $0.0029 |
| **`gpt-5-mini`** | **95%** | **0.730** [+0.459, +1.000] | $0.0135 |
| *always answers "correct"* | 90% | 0.000 | $0 |

`gpt-5-mini` disagreed with the human on exactly one item — `lookup-017`, where the answer
describes `openapi_tags` while the reference gives the `tags=` parameter, and the judge
called `partial` what the human called `incorrect`. That is a genuine boundary case rather
than a failure of comprehension.

**Honest caveat carried into every Tier-2 number:** the kappa interval reaches to +0.459,
below the conventional 0.60 bar for "substantial". At n=20 with lopsided labels the point
estimate is the claim and the interval is the caveat. What the sample *does* settle beyond
doubt is the negative: a judge at chance level is unmistakable.

**Label provenance:** labels were proposed by a stronger model (Claude Desktop) and
adjudicated by the author, who agreed with all 20 proposals. Proposals live in a separate
field and never enter the statistics. The defensible wording is "judge-human agreement,
n=20, labels model-proposed and author-adjudicated" -- not "hand-labelled from scratch".

### Two findings that came out of this for free

1. **Tier 1 is a lower bound on usefulness, but not for the reason first claimed.** Of the
   three questions Tier 1 scores as retrieval failures at rank 10, **two produced answers
   the human judged correct** (`multi_hop-005`, `multi_hop-013`). The first explanation
   written here -- that `min_ratio=1.0` was too strict -- was **tested and is wrong**:
   Recall@5 is flat at 0.897 for every threshold from 1.0 down to 0.6, moving only at 0.5.
   The spans are either fully covered or barely covered, never marginally. What actually
   happens is that the generator produces a correct answer from *partially relevant*
   context that contains no complete answer span at all. Retrieval recall and answerability
   are therefore genuinely different quantities, not the same one measured strictly.
2. **A confirmed false refusal.** `multi_hop-002` retrieved successfully and the model
   refused anyway; both the human and the mini judge call that incorrect. This is D23's
   second refusal layer misfiring on a hard multi-hop question, and it is the first
   reproducible instance of the failure D24 was written to prevent.

**Tier 2 is cheaper than estimated.** Measured judge cost is **$0.000677 per call**, and
the judge returns correctness *and* grounding in one call rather than three. So 35 questions
x 2 arms = 70 generations (~$0.010) plus 70 judge calls (~$0.047) is about **$0.057**, not
the $0.16-0.22 estimated before the judge existed.

## Tier-2 answer quality (re-run 2026-09-07, 35 questions x 2 arms)

Full report: `evals/reports/answers.md`. Generator `gpt-5-nano`, judge `gpt-5-mini`
(kappa 0.730 against human labels). Cost **$0.0477** originally, **$0.0117** to re-run.

| arm | correct | correct or partial | grounded | cited honestly | cost/answer |
|---|---|---|---|---|---|
| hybrid/fixed | 0.943 [0.857, 1.000] | **1.000** | 1.000 *(unvalidated)* | 1.000 | $0.000146 |
| dense/fixed | 0.943 [0.857, 1.000] | 0.971 | 1.000 *(unvalidated)* | 1.000 | $0.000142 |

**Re-run because the refusal text changed, and the numbers moved for a different reason.**
Hybrid went 0.914 → 0.943 and its one wrong refusal (`multi_hop-002`) disappeared. That is
**not** attributable to the Phase 3 work: the generator, the prompt and `k` are unchanged,
so only the retrieved context can stop a model refusing — and the index was rebuilt with
document-scoped deduplication (D42) after the original Tier-2 run. The gain belongs to
Phase 1, and is recorded here rather than claimed for the phase that happened to re-measure
it. Three answers were cache misses for the same reason.

**The headline is still a negative result, and it is the most interesting number in the
project.** Tier 1 showed hybrid retrieval beating dense-only on Recall@5 in all three
chunking strategies. At the *answer* level the two are now **identical** on `correct`
(+0.000, interval [-0.086, +0.086]); hybrid leads only on `correct_or_partial` by one
question. Better retrieval did not produce better answers here.

The mechanism is visible in the failure table: at 91-94% correct there is very little room
left, and the generator recovers from imperfect context more often than the retrieval
metric implies. It is the same effect Tier 1's own caveat predicted — two of three
questions scored as retrieval misses still produced correct answers. **A reranker (Phase 2)
should therefore be judged on Tier-1 metrics and on the failure attribution, not on this
number, which is too near its ceiling to move.**

### Failure attribution (the table that says what to fix)

| arm | failures | missing refusal | retrieval miss | wrong refusal | ignored context | fabricated citations |
|---|---|---|---|---|---|---|
| hybrid/fixed | 3 | 0 | 1 | 1 | 1 | 0 |
| dense/fixed | 2 | 0 | 0 | 1 | 1 | 0 |

* **Zero fabricated citations across 70 answers.** Deterministic, no judge involved: every
  bracketed number resolved to a block that was actually in the prompt.
* **Both arms produce exactly one false refusal** — `multi_hop-002` on hybrid, `lookup-018`
  on dense. The confirmed instance of D24's failure mode, now reproducible.
* **Multi-hop and ambiguous questions scored 1.000 correct**, lookup 0.889. The category the
  retrieval metrics find hardest is the one the generator handles best, because a multi-hop
  answer can be assembled from partial evidence in a way a specific lookup fact cannot.

### What the refusal column actually counts

`no_answer-007` is answered rather than refused on the **dense-only** arm (hybrid refuses
it), and the judge still called it correct. Reading the answer explains why, and corrects an
earlier note here that called this a brittle-detection bug: the reply states what the corpus
*does* contain -- two specific CVE references -- and then says explicitly that no block
describes a broader disclosure process or audit history. That is a more useful response than
a bare refusal, and the judge was right to accept it.

So this is not a detection defect to fix. It is a real behaviour the binary `answered` flag
cannot represent: partial information plus an explicit statement of what is missing is
neither an answer nor a refusal. The report therefore says what the column counts -- explicit
refusals, meaning the confidence gate firing or the sentinel being emitted -- rather than
claiming to measure declining.

## Defect audit before the reranker (2026-09-07)

The instruction was to fix what makes retrieval or evaluation misleading before building
anything new. Each suspected defect was measured first, and **two of the four turned out not
to be defects at all** — both were claims written into this file the day before.

| suspected defect | measured verdict |
|---|---|
| Near-duplicate detection unwired | **Real.** Fixed 1 exact duplicate; structure 83; semantic 114. Now implemented (D42, D43). |
| Retrieved context is full of near-duplicates | **Wrong shape.** 0 of 290 retrieved pairs reach cosine 0.95. The redundancy is document concentration, 3.28 distinct documents per 5 chunks. |
| Tier 1's `min_ratio=1.0` understates answerability | **Wrong.** Recall@5 is flat at 0.897 for every threshold from 1.0 to 0.6, moving only at 0.5. Corrected above. |
| Refusal detection misses prose declines | **Mischaracterised.** A legitimate answer the binary `answered` flag cannot represent, not a bug. Corrected above. |
| Processed-document storage unimplemented | **Real**, and required by the brief. Now implemented (D44). |

**The most valuable thing that happened here was the evaluation catching a bug in a feature
before it shipped.** Corpus-wide dedup looked correct, passed its unit tests, and removed
exactly the duplicates it was supposed to — and Tier 1 reported a golden question losing all
its evidence and Recall@5 dropping. That is the harness paying for itself, and it is the
strongest single argument in the project that the evaluation is worth its weight.

### Guidance for the reranker, now that there is a baseline

* **Judge it on Tier 1 and on failure attribution, not on Tier-2 correctness**, which sits at
  0.914-0.943 and is too near its ceiling to move (D39).
* **Its measurable job is diversity and distractor suppression**: 3.28 distinct documents per
  5 chunks, and `release-notes.md` still taking 4% of the top-5 slots under fixed chunking.
* **The bar to beat**: hybrid/fixed at Recall@5 0.897, Recall@budget 0.862, nDCG@10 0.794.
* Multi-hop is the weak retrieval category (0.667 at rank 5) while being the strongest
  generation category (1.000 correct) — a reranker that helps multi-hop retrieval has the
  clearest room to show it.

## Phase 3 completed (2026-09-07) — the three requirements that were stubs

The vertical slice shipped a grounded prompt, structural citation resolution and two
refusal paths, and `docs/PROJECT_STATE.md` recorded Phase 3 as done. Reading the brief
against the code showed three of its four requirements were not built: claim-level citation
verification, the composite confidence scorer, and the structured "I don't know". All three
are now built and measured.

### Citation verification — does `[3]` support the claim attached to it?

Full report: `evals/reports/citation_verification.md`. Verifier `gpt-5-mini`, 29 answerable
questions, 56 cited claims out of 108 total.

| measure | value |
|---|---|
| citation coverage (share of **all** claims with a citation that holds) | 0.525 [0.425, 0.630] |
| citation precision (share of **cited** claims that hold) | 0.917 [0.822, 0.991] |
| unsupported citations found | **6 of 56** |
| verifier on real citations vs random blocks | **0.900 vs 0.050**, separation +0.850 [+0.725, +0.950] |

The two rates are reported apart because they answer different questions: an answer citing
one sentence in six and getting it right scores 0.17 coverage and 1.00 precision, and both
facts matter. Six real mis-attributions were found — citations that resolve perfectly to a
block that does not make the claim, the failure structural checking cannot see. This is the
layer the brief calls "the quality layer most RAG systems skip entirely", and it is the
one place in the project where a citation is checked for what a reader assumes it means.

### Answer confidence — three dimensions, never one opaque number

`retrieval` (calibrated, D49), `citation_coverage` (verified when a verifier is attached,
structural otherwise) and `completeness` (LLM-scored). Verification is **opt-in**: right for
an evaluation run, wrong for a dashboard that must answer in under a second. Without it an
answer still carries a composite built from what is free, and `confidence.components`
records which dimensions fed it, so the two are never confused. On the demo question the
free path reports 0.532 from two components and the verified path 0.521 from three.

### Graceful "I don't know"

A refusal now returns what the search found (with similarity scores), what it could not
establish, and up to three distinct pages worth opening by hand — all derived from the
retrieval results already computed, at no extra model call.

### Three defects this phase surfaced in earlier work

1. **Process ids parsed as citations** (D45). Present in the response cache before this
   phase started; it would have reported a fabricated source to a user.
2. **Trailing citation blocks discarded** (D46). Caught for $0 by measuring the splitter
   over the golden set before paying for a verification run, which was killed 46 calls in.
3. **One unreadable reply killed a paid 125-call run.** The verifier sometimes drops the
   `SUPPORT:` label and writes the verdict bare. The parser now accepts that exact form —
   which is reading a real judgement, not defaulting — and the script counts unparseable
   replies and continues, the way the judge already counted its own.

## Next step

Phases 1, 2, 3, 4 and 6 are complete. Phase 5.1 (FastAPI service) and 5.2 (Streamlit
dashboard) are built and tested. **~$0.383 spent of $1.00**; every retrieval report still
re-runs at $0, and the dashboard's real per-question spend is ~$0.0023 on the current
default (D55).

The user decided this ships as a local-only portfolio piece — no deployment. Phase 5.3
(Docker + seed script) is therefore **not planned**, not merely deferred: `README.md`
states plainly that this wasn't built, rather than implying a container that doesn't
exist. If that decision changes, Docker gets explained in depth per `CLAUDE.md` when it's
picked back up.

### Phase 6 — README and case study (2026-09-08)

`README.md` pulls every headline figure directly from `evals/reports/` — Tier 1 retrieval
(hybrid vs. dense vs. sparse), Tier 2 answer quality, the citation-verification negative
control, and the reranker's negative result (D53) reported beside the wins rather than
omitted, exactly the discipline `CLAUDE.md` asks for: "prefer measured numbers... report
negative results honestly." A Mermaid architecture diagram renders natively on GitHub.
Added a root `LICENSE` (MIT, matching the FastAPI corpus it indexes) and hardened
`.gitignore` against `.streamlit/secrets.toml` before anything used it.

### Phase 5.1 + 5.2 — built (2026-09-08)

The FastAPI service (`src/hybridrag/api/`) wraps the existing retrieval/generation pipeline
in three routes: `/v1/ask` (with a `mode` field for hybrid vs. dense-only ablation, settled
as a cross-cutting exception ahead of 5.2 needing it), `/v1/documents`, and `/v1/ingest`
(chunk, dedupe, embed into the dense index, then a full BM25 rebuild -- D33's IDF-is-
corpus-relative constraint applies to a single new document exactly as it does to the whole
corpus). `Resources` is built once at process startup rather than per request; the chunk
store, embedding cache and completion cache each gained a `threading.Lock` scoped to just
their SQLite calls, so concurrent requests still hit the model provider in parallel. 14
offline tests, `TestClient` with `build_resources()` bypassed via `dependency_overrides`.

The dashboard (`dashboard/`, not part of the installed package -- a separate `uv` dependency
group so the API never needs Streamlit) is a design deliberately built around the system's
own mechanism: dense (teal) and sparse (amber) are the two literal RRF inputs, fused
(violet) is what hybrid mode is coloured, and each retrieved chunk shows which index(es)
actually surfaced it, straight from `RetrievedChunk.hits`. Every question fires hybrid and
dense-only concurrently (`ThreadPoolExecutor`) against the running API and shows both.

Debugging this in real use (D55) is what surfaced Gemini's free-tier pace gate as a genuine
UX problem, not just an eval-time one: `GeminiModel`'s pacer holds the whole process to one
call every 12 seconds regardless of concurrency, and the dashboard's own design (both modes,
every question) doubles the calls per question. Switched the default provider to
`gpt-5-mini` via `.env`, not the code default, so a clean checkout with no `.env` still runs
$0 on Gemini -- D55 has the measured before/after.

### Phase 2 — closed (2026-09-07)

Built the reranker (D8, D52), judged it on Tier 1 only per the settled brief (never Tier 2,
which sits at its ceiling, D39), and it did not earn a place as the default: see D53 for the
measured paired-bootstrap comparison and the distractor-rate table, and
`evals/reports/reranker.md` for the full report. RRF weights are now configurable and swept
(D54): `evals/reports/rrf_weights.md`. Both add $0 to spend -- no language model, no
external index rebuild, ~168s and ~4s wall-clock respectively.

The reranker's code stays in the tree (`hybridrag.retrieval.rerank`, opt-in via
`--rerank` on `scripts/evaluate_retrieval.py`) because a negative result is only credible
if the thing that lost is inspectable, not deleted. It is not wired into the default query
path (`scripts/ask.py`), since nothing measured justifies routing production queries
through it.

Phase 1 is now complete: loaders, three chunking strategies, synchronised dense and sparse
indexes, deduplication, and processed-document storage.

**Closed by the Tier-1 results:** the open question of whether to damp semantic chunking's
thin chunks with a neighbour buffer. Semantic is not separable from fixed and costs more to
build, so tuning it further would be spending effort on the arm the measurement does not
favour. Decided with metrics, as intended.

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

## Golden-set drafting: what the $0.25 bought

Five runs. Most of the spend went on two bugs rather than on questions, both worth
recording because they are corpus-shaped rather than model-shaped:

| run | change | accepted | spend |
|---|---|---|---|
| 1 | first attempt, JSON transport | 30 | $0.039 |
| 2 | narrative docs excluded, no_answer without a document | 34 | $0.068 |
| 3 | output cap 4k → 16k (wrong diagnosis: replies were not truncated) | 34 | $0.068 |
| 4 | JSON → sentinel blocks | 31 | $0.068 |
| 5 | whitespace-tolerant matching, then the last two prompts | **59** | $0.011 |

Run 3 was a wasted $0.068 spent on a hypothesis I had not confirmed — the "truncated
mid-JSON" counter was my own detector misreporting an unescaped-quote error. Reading one
failing reply out of the cache, which cost nothing, was what actually found both causes.

The cache earned its keep here: run 5 changed two prompts out of four categories and paid
for 6 calls instead of 44.
