# Reranker evaluation — Tier 1

Does adding a cross-encoder rerank pass on top of hybrid RRF improve retrieval? Every
number below is scored at a 5-wide horizon, because that is what the reranker keeps and
what the generator reads -- scoring at 10 would credit the baseline for chunks the
reranked pipeline never returns to a caller.

**Provenance** · generated 2026-09-07T14:57:48+00:00 · commit `e4beff8` **(uncommitted changes)** ·
corpus `0.115.6` · bootstrap 10,000 resamples, seed 20260906

## Headline: paired bootstrap, reranked vs plain hybrid

Same resampled questions scored under both arms, which cancels question difficulty. `*` marks an interval that excludes zero -- the only circumstance under which a difference here is a claim rather than a hint.

| chunking | metric | hybrid | hybrid + rerank | difference [95% CI] | P(>0) |
|---|---|---|---|---|---|
| fixed | recall@5 | 0.897 [0.793, 1.000] | 0.828 [0.690, 0.966] | -0.069 [-0.172, +0.000] | 0% |
| fixed | ndcg@5 | 0.794 [0.688, 0.889] | 0.728 [0.606, 0.840] | -0.066 [-0.175, +0.044] | 12% |
| fixed | mrr@5 | 0.674 [0.531, 0.807] | 0.590 [0.449, 0.725] | -0.083 [-0.207, +0.046] | 10% |
| structure | recall@5 | 0.759 [0.586, 0.897] | 0.759 [0.586, 0.897] | +0.000 [-0.103, +0.103] | 35% |
| structure | ndcg@5 | 0.725 [0.606, 0.837] | 0.689 [0.565, 0.805] | -0.035 [-0.151, +0.085] | 27% |
| structure | mrr@5 | 0.593 [0.436, 0.748] | 0.564 [0.411, 0.713] | -0.029 [-0.144, +0.093] | 30% |
| semantic | recall@5 | 0.828 [0.690, 0.966] | 0.793 [0.655, 0.931] | -0.034 [-0.172, +0.103] | 25% |
| semantic | ndcg@5 | 0.737 [0.634, 0.830] | 0.724 [0.615, 0.819] | -0.014 [-0.100, +0.069] | 38% |
| semantic | mrr@5 | 0.563 [0.420, 0.701] | 0.537 [0.393, 0.677] | -0.026 [-0.113, +0.061] | 27% |

## Distractor rate: share of top-5 slots from `release-notes.md`

The corpus's changelog is kept in deliberately (D28) so the reranker has real distractor content to suppress. A lower rate here is that job being done.

| chunking | hybrid | hybrid + rerank |
|---|---|---|
| fixed | 4.1% | 6.9% |
| structure | 9.7% | 10.3% |
| semantic | 10.3% | 8.3% |

## Verdict, per chunking strategy

The reranker ships as the default only where this separates in its favour; otherwise this is reported as a negative result rather than decorated.

- **fixed**: recall@5 -0.069 [-0.172, +0.000]  -- **not separated from zero**
- **structure**: recall@5 +0.000 [-0.103, +0.103]  -- **not separated from zero**
- **semantic**: recall@5 -0.034 [-0.172, +0.103]  -- **not separated from zero**
