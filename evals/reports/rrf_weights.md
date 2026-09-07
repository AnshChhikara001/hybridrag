# RRF weight sweep — Tier 1

`fixed` chunking, hybrid retriever, dense:sparse weight ratio swept against the
project's unweighted 1:1 default. Only the ratio matters (D20): RRF's score is a sum of
`weight / (rank_constant + rank)`, so scaling both weights together changes nothing.

**Provenance** · corpus `0.115.6` · depth 50 · bootstrap 10,000
resamples, seed 20260906

| dense:sparse | recall@5 | recall@10 | ndcg@10 | recall@budget | recall@5 vs 1:1 [95% CI] |
|---|---|---|---|---|---|
| 0.25:1 | 0.828 [0.690, 0.966] | 0.862 [0.724, 0.966] | 0.798 [0.673, 0.908] | 0.724 [0.552, 0.862] | -0.069 [-0.172, +0.000] |
| 0.5:1 | 0.897 [0.793, 1.000] | 0.897 [0.793, 1.000] | 0.819 [0.707, 0.918] | 0.828 [0.690, 0.966] | +0.000 [+0.000, +0.000] |
| 1.0:1 | 0.897 [0.793, 1.000] | 0.897 [0.793, 1.000] | 0.794 [0.688, 0.889] | 0.862 [0.724, 0.966] | -- baseline -- |
| 1.5:1 | 0.897 [0.793, 1.000] | 0.897 [0.793, 1.000] | 0.785 [0.679, 0.880] | 0.897 [0.793, 1.000] | +0.000 [+0.000, +0.000] |
| 2.0:1 | 0.897 [0.793, 1.000] | 0.897 [0.793, 1.000] | 0.769 [0.663, 0.868] | 0.828 [0.690, 0.966] | +0.000 [+0.000, +0.000] |
| 3.0:1 | 0.897 [0.793, 1.000] | 0.897 [0.793, 1.000] | 0.776 [0.672, 0.871] | 0.828 [0.690, 0.966] | +0.000 [+0.000, +0.000] |
| 4.0:1 | 0.897 [0.793, 1.000] | 0.897 [0.793, 1.000] | 0.776 [0.672, 0.871] | 0.828 [0.690, 0.966] | +0.000 [+0.000, +0.000] |

`*` marks a paired-bootstrap interval on Recall@5 that excludes zero against the 1:1
baseline -- the only circumstance under which a difference here is a claim rather than a
hint.
