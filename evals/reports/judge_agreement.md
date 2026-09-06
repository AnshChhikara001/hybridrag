# Judge validation — does an LLM judge agree with a human?

Tier 2 grades answers with a language model, which is circular unless the judge is itself
measured. Before any answer-quality number is reported, a human labelled a sample by hand
and each candidate judge labelled the same sample; what follows is how far they agreed
(D11).

**Provenance** · commit `eba0301` **(uncommitted changes)** ·
answers from `gpt-5-nano-2025-08-07` on arm `hybrid/fixed` · generated 2026-09-06T10:23:10+00:00 ·
20 labelled items · bootstrap 10,000 resamples, seed 20260906

**Human label distribution** · 18 correct, 2 incorrect

## Agreement

Cohen's kappa removes the agreement two labellers would reach by chance alone. The last row
is the reason it is reported: a judge that replies "correct" to everything needs no
intelligence at all and still scores the raw agreement shown.

| judge | raw agreement | kappa [95% CI] | cost | unreadable |
|---|---|---|---|---|
| `gpt-5-nano-2025-08-07` | 30% | 0.007 [-0.040, +0.077] | $0.0000 | 0 |
| `gpt-5-mini-2025-08-07` | 95% | 0.730 [+0.459, +1.000] | $0.0000 | 0 |
| *always answers “correct”* | 90% | 0.000 | $0.0000 | 0 |

## Where each judge parted company with the human

**`gpt-5-nano-2025-08-07`** — 14 of 20

| id | category | human | judge |
|---|---|---|---|
| lookup-017 | lookup | incorrect | partial |
| multi_hop-013 | multi_hop | correct | partial |
| no_answer-003 | no_answer | correct | incorrect |
| no_answer-006 | no_answer | correct | incorrect |
| no_answer-007 | no_answer | correct | incorrect |
| no_answer-008 | no_answer | correct | incorrect |
| no_answer-009 | no_answer | correct | incorrect |
| lookup-001 | lookup | correct | partial |
| multi_hop-002 | multi_hop | incorrect | partial |
| lookup-002 | lookup | correct | partial |
| multi_hop-006 | multi_hop | correct | partial |
| ambiguous-002 | ambiguous | correct | partial |
| lookup-003 | lookup | correct | partial |
| multi_hop-008 | multi_hop | correct | partial |

**`gpt-5-mini-2025-08-07`** — 1 of 20

| id | category | human | judge |
|---|---|---|---|
| lookup-017 | lookup | incorrect | partial |

## How these labels were made

20 of 20 items carried a model-proposed label for the human to
adjudicate, and the human's final label matched the proposal on **20** of them.
Proposals are stored in a separate field and never enter the statistics above; only the
human's decision does. That distinction is what the numbers here rest on, so it is recorded
rather than assumed.

## What this licenses

`gpt-5-mini-2025-08-07` clears the conventional 0.6 bar for substantial agreement (kappa 0.730). Its interval reaches below that bar, so at this sample size the point estimate is the claim and the interval is the honest caveat. Every Tier-2 number
produced by this judge should be reported with that agreement figure beside it, not
without.

Twenty items is a small sample and the labels are lopsided, which is why the interval is
wide rather than why it should be ignored: a judge indistinguishable from chance is still
conclusively identified as such, and that is the decision this experiment existed to make.
