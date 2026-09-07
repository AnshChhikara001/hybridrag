# Answer evaluation — Tier 2

What the system answers, whether the answer is grounded in what it retrieved, and where a
wrong answer went wrong.

**Provenance** · commit `552fc03` **(uncommitted changes)** ·
generator `gpt-5-nano-2025-08-07` · judge `gpt-5-mini-2025-08-07` ·
35 questions · bootstrap 10,000 resamples, seed 20260906

**Correctness** is judged by `gpt-5-mini-2025-08-07`, whose agreement with a human was measured before it was trusted: **kappa 0.730** [+0.459, +1.000] on 20 hand-adjudicated items (D37). Every correctness figure below inherits that uncertainty on top of its own sampling error.

**Grounding is not validated.** The same judge produces it, but no human labelled grounding, so its kappa is unknown and the number below is reported for completeness rather than as a claim. **Citation honesty is deterministic** -- a bracketed number either resolves to a block that was in the prompt or it does not -- and involves no judge at all.

## Answer quality

`correct` and `correct or partial` are reported side by side deliberately: partial credit
flatters a system, and one blended score would hide which of the two moved. Cost per
answer is priced from token counts rather than from what this run paid, so it stays
meaningful when the answers come from cache; latency can only be measured on answers
actually generated, so the live count is shown beside it.

| arm | correct | correct_or_partial | grounded | cited_honestly | cost/answer | median latency |
|---|---|---|---|---|---|---|
| hybrid/fixed | 0.943 [0.857, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | $0.000146 | — (0/35 live) |
| dense/fixed | 0.943 [0.857, 1.000] | 0.971 [0.914, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | $0.000142 | — (0/35 live) |

## Does hybrid retrieval produce better *answers*?

Tier 1 showed hybrid retrieves better. This is the question that matters to a user, and it
is not the same question: better context only helps if the generator uses it. Paired
bootstrap over the same resampled questions; `*` marks an interval excluding zero.

| contrast | metric | difference [95% CI] | P(>0) |
|---|---|---|---|
| dense/fixed - hybrid/fixed | correct | +0.000 [-0.086, +0.086] | 35% |
| dense/fixed - hybrid/fixed | correct_or_partial | -0.029 [-0.086, +0.000] | 0% |
| dense/fixed - hybrid/fixed | grounded | +0.000 [+0.000, +0.000] | 0% |

## By question category — hybrid/fixed

| category | n | correct | correct or partial | grounded | refused |
|---|---|---|---|---|---|
| lookup | 18 | 0.889 | 1.000 | 1.000 | 0.000 |
| multi_hop | 6 | 1.000 | 1.000 | 1.000 | 0.000 |
| ambiguous | 5 | 1.000 | 1.000 | 1.000 | 0.000 |
| no_answer | 6 | 1.000 | 1.000 | 1.000 | 1.000 |

## Where the failures come from

The join between the two tiers, and the only table here that says what to fix. A wrong
answer is attributed to retrieval whenever Tier 1 records that this same arm failed to
retrieve the answer span -- refusing or fumbling a question whose evidence never arrived is
not a generation fault, and counting it as one would hide a retrieval problem behind a
prompt problem.

| arm | failures | missing_refusal | retrieval_miss | wrong_refusal | ignored_context | fabricated citations |
|---|---|---|---|---|---|---|
| hybrid/fixed | 2 | 0 | 1 | 0 | 1 | 0 |
| dense/fixed | 2 | 0 | 0 | 1 | 1 | 0 |

## Every answer hybrid/fixed got wrong

| id | category | verdict | attributed to | judge's reason |
|---|---|---|---|---|
| lookup-006 | lookup | partial | ignored_context | The assistant correctly explains that Form parameters cause the request to be encoded as f |
| lookup-017 | lookup | partial | retrieval_miss | The assistant mentions using openapi_tags metadata (which is supported by the provided con |

## Refusal behaviour

The two errors are not symmetric and are counted apart. These columns count **explicit** refusals -- the confidence gate firing, or the model emitting its sentinel. A model can also decline in prose without the sentinel, which is counted here as an answer and by the judge as a correct decline; that is why an arm can show fewer refusals than it has correct no-answer verdicts.

| arm | refused when unanswerable | refused when answerable | which ones |
|---|---|---|---|
| hybrid/fixed | 100% (6 asked) | 0/29 | - |
| dense/fixed | 83% (6 asked) | 1/29 | lookup-018 |

## Limits of this measurement

* **29 answerable questions and 6
  unanswerable ones.** Intervals are wide at this size and are reported rather than hidden.
* **The judge is imperfect and its imperfection is measured**, not assumed away: kappa
  0.730 against a human on 20 adjudicated items.
* **Answers are served from a response cache.** That is what makes a re-run free and
  reproducible on a model that rejects `temperature` (D27), but it means the numbers
  describe one sampled generation per question, not an average over several.
* **`cited_honestly` here checks resolution, not support.** A citation pointing at a real
  block that does not actually contain the claim is counted honest in this column. Whether
  the cited block supports the claim attached to it is measured separately and per claim,
  in `citation_verification.md`: on this arm 0.917 of cited claims hold, and the six that
  do not are listed there by name.
