# Retrieval evaluation — Tier 1

Deterministic retrieval metrics over the hand-verified golden set. No language model is
involved anywhere in this report, so it costs nothing to reproduce and cannot drift with a
model release.

**Provenance** · generated 2026-09-07T01:22:16+00:00 · commit `552fc03` **(uncommitted changes)** ·
corpus `0.115.6` · embedder `text-embedding-3-small` ·
chunks 512 tokens / 64 overlap ·
semantic percentile 95.0 · RRF rank constant 60 ·
candidate depth 50 · scored to depth 50 ·
budget 2000 tokens · coverage threshold 1.0 ·
bootstrap 10,000 resamples, seed 20260906

**Questions** · 35 verified, of which **29 are scored here**
(5 ambiguous, 18 lookup, 6 multi_hop, 6 no_answer).
`no_answer` questions carry no retrieval ground truth (D29) and appear only in the
refusal-calibration section.

## What each strategy produced

The same 141 documents, cut three ways. These numbers explain most of the gap between Recall@k and Recall@budget below.

| chunking | chunks | mean tokens | median | p95 | total tokens |
|---|---|---|---|---|---|
| fixed | 1,086 | 479 | 512 | 512 | 519,785 |
| structure | 1,793 | 238 | 204 | 502 | 426,955 |
| semantic | 2,087 | 204 | 162 | 490 | 424,912 |

## Headline

Mean over questions, with a 95% percentile bootstrap interval.

| arm | recall@5 | recall@10 | ndcg@10 | recall@budget |
|---|---|---|---|---|
| hybrid/fixed | 0.897 [0.793, 1.000] | 0.897 [0.793, 1.000] | 0.794 [0.688, 0.889] | 0.862 [0.724, 0.966] |
| dense/fixed | 0.828 [0.690, 0.966] | 0.897 [0.793, 1.000] | 0.742 [0.631, 0.845] | 0.793 [0.655, 0.931] |
| sparse/fixed | 0.793 [0.621, 0.931] | 0.897 [0.759, 1.000] | 0.779 [0.651, 0.893] | 0.724 [0.552, 0.862] |
| hybrid/structure | 0.759 [0.586, 0.897] | 0.759 [0.586, 0.897] | 0.728 [0.609, 0.840] | 0.759 [0.586, 0.897] |
| dense/structure | 0.655 [0.483, 0.828] | 0.828 [0.690, 0.966] | 0.678 [0.556, 0.796] | 0.828 [0.690, 0.966] |
| sparse/structure | 0.655 [0.483, 0.828] | 0.793 [0.655, 0.931] | 0.631 [0.527, 0.734] | 0.759 [0.586, 0.897] |
| hybrid/semantic | 0.828 [0.690, 0.966] | 0.897 [0.793, 1.000] | 0.757 [0.667, 0.842] | 0.897 [0.793, 1.000] |
| dense/semantic | 0.724 [0.552, 0.862] | 0.897 [0.793, 1.000] | 0.718 [0.604, 0.825] | 0.862 [0.724, 0.966] |
| sparse/semantic | 0.690 [0.517, 0.862] | 0.724 [0.552, 0.862] | 0.643 [0.512, 0.767] | 0.724 [0.552, 0.862] |

## Every metric, point estimates

`recall@budget` fixes the retrieved-context budget instead of the number of chunks (D14), so a strategy earns nothing for emitting larger chunks. Latency is the median wall-clock time of one retrieval, query embeddings served from cache.

| arm | recall@1 | recall@3 | recall@5 | recall@10 | mrr@10 | ndcg@10 | recall@budget | latency |
|---|---|---|---|---|---|---|---|---|
| hybrid/fixed | 0.552 | 0.793 | 0.897 | 0.897 | 0.674 | 0.794 | 0.862 | 8.1 ms |
| dense/fixed | 0.448 | 0.724 | 0.828 | 0.897 | 0.591 | 0.742 | 0.793 | 4.3 ms |
| sparse/fixed | 0.586 | 0.690 | 0.793 | 0.897 | 0.662 | 0.779 | 0.724 | 2.9 ms |
| hybrid/structure | 0.483 | 0.655 | 0.759 | 0.759 | 0.593 | 0.728 | 0.759 | 11.2 ms |
| dense/structure | 0.379 | 0.621 | 0.655 | 0.828 | 0.520 | 0.678 | 0.828 | 6.4 ms |
| sparse/structure | 0.276 | 0.655 | 0.655 | 0.793 | 0.468 | 0.631 | 0.759 | 4.4 ms |
| hybrid/semantic | 0.414 | 0.690 | 0.828 | 0.897 | 0.575 | 0.757 | 0.897 | 14.4 ms |
| dense/semantic | 0.379 | 0.517 | 0.724 | 0.897 | 0.515 | 0.718 | 0.862 | 8.7 ms |
| sparse/semantic | 0.414 | 0.586 | 0.690 | 0.724 | 0.517 | 0.643 | 0.724 | 4.7 ms |

## Does hybrid actually beat its parts?

Paired bootstrap: the same resampled questions scored under both arms, which cancels question difficulty. `*` marks an interval that excludes zero — the only circumstance under which a difference here is a claim rather than a hint.

| contrast | chunking | metric | difference [95% CI] | P(>0) |
|---|---|---|---|---|
| hybrid - dense | fixed | recall@5 | +0.069 [+0.000, +0.172] | 88% |
| hybrid - dense | fixed | recall@budget | +0.069 [-0.069, +0.207] | 78% |
| hybrid - sparse | fixed | recall@5 | +0.103 [+0.000, +0.241] | 96% |
| hybrid - sparse | fixed | recall@budget | +0.138 [+0.034, +0.276]* | 99% |
| hybrid - dense | structure | recall@5 | +0.103 [-0.034, +0.241] | 88% |
| hybrid - dense | structure | recall@budget | -0.069 [-0.207, +0.069] | 9% |
| hybrid - sparse | structure | recall@5 | +0.103 [-0.034, +0.241] | 88% |
| hybrid - sparse | structure | recall@budget | +0.000 [-0.103, +0.103] | 35% |
| hybrid - dense | semantic | recall@5 | +0.103 [+0.000, +0.241] | 96% |
| hybrid - dense | semantic | recall@budget | +0.034 [+0.000, +0.103] | 64% |
| hybrid - sparse | semantic | recall@5 | +0.138 [+0.000, +0.310] | 94% |
| hybrid - sparse | semantic | recall@budget | +0.172 [+0.034, +0.310]* | 100% |

## Does the chunking strategy matter?

The same paired comparison across chunking strategies, holding the retriever fixed at hybrid.

| contrast | metric | difference [95% CI] | P(>0) |
|---|---|---|---|
| fixed - structure | recall@5 | +0.138 [+0.034, +0.276]* | 99% |
| fixed - structure | recall@budget | +0.103 [+0.000, +0.207] | 96% |
| fixed - semantic | recall@5 | +0.069 [-0.103, +0.241] | 73% |
| fixed - semantic | recall@budget | -0.034 [-0.172, +0.103] | 24% |
| structure - semantic | recall@5 | -0.069 [-0.241, +0.103] | 14% |
| structure - semantic | recall@budget | -0.138 [-0.310, +0.000] | 2% |

## By question category — hybrid/fixed

Each category is scored by its own rule (D29): a lookup needs its one span, multi-hop needs **every** span, ambiguous needs any acceptable one.

| category | n | recall@5 | recall@10 | nDCG@10 |
|---|---|---|---|---|
| lookup | 18 | 0.944 | 0.944 | 0.799 |
| multi_hop | 6 | 0.667 | 0.667 | 0.608 |
| ambiguous | 5 | 1.000 | 1.000 | 1.000 |

## What hybrid/fixed still misses at rank 10


| id | category | cause | question |
|---|---|---|---|
| lookup-017 | lookup | ranking | How do I add tags to group related path operations and have them appea |
| multi_hop-005 | multi_hop | ranking | If I want to accept partial updates without accidentally overwriting s |
| multi_hop-013 | multi_hop | ranking | How can I both set Field-level validation like gt=0 for a price attrib |

## Refusal calibration

`no_answer` questions have nothing to retrieve, but they measure the pre-generation refusal gate (D23), which thresholds the best dense similarity a query attracts. A threshold is defensible only if these two distributions separate.

| chunking | answerable min | answerable mean | no_answer max | no_answer mean | gap | false refusals | gate catches |
|---|---|---|---|---|---|---|---|
| fixed | 0.361 | 0.566 | 0.640 | 0.503 | -0.279 | 0 | 0/6 |
| structure | 0.390 | 0.633 | 0.640 | 0.509 | -0.250 | 0 | 0/6 |
| semantic | 0.443 | 0.613 | 0.640 | 0.521 | -0.197 | 0 | 0/6 |

A positive gap would mean one threshold separates the two sets perfectly. A negative one means no threshold can -- which is the expected result here and the reason two refusal paths exist (D23). These `no_answer` questions are *in-domain but unanswerable*: they are about FastAPI, so they retrieve FastAPI chunks and score like real questions. The pre-generation gate was measured against *out-of-domain* questions, which it does separate. **false refusals** counts answerable questions that the configured threshold of 0.3 would wrongly refuse; **gate catches** is how many unanswerable ones it stops before spending a request. The rest are the model's own refusal sentinel to catch.

## How to read this, and what it does not say

* **29 questions.** The intervals are wide because the sample is small, and they
  are reported rather than hidden. Most differences between adjacent arms here are not
  separable at 95%.
* **Coverage threshold 1.0.** A span counts as retrieved only when the union of
  the retrieved chunks contains **all** of it. Relaxing this would raise every number.
* **Approximate search.** Chroma's HNSW is approximate, so a fraction of a point of
  movement between runs is the index, not the code.
* **The golden set was drafted by a model and verified by hand** (D11). The questions are
  therefore ones a model found answerable in this corpus, which is a real selection effect
  even after human verification.
* Tier 2 — answer correctness, faithfulness and citation accuracy — is judged separately
  and is the only part of the evaluation that costs money.
