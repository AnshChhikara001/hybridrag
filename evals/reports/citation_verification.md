# Citation verification — does the cited block support the claim?

Structural citation checking asks whether `[3]` points at a block that exists.
This asks the question a reader assumes is already answered: whether block 3
actually says the thing the sentence attached to it claims. A citation can resolve
perfectly and still be attached to a passage that never makes the claim, which
renders as a working source link under a fabricated statement.

**Provenance** · commit `552fc03` **(uncommitted changes)** ·
arm `hybrid` / `fixed` · generator `gpt-5-nano-2025-08-07` ·
verifier `gpt-5-mini-2025-08-07` · 29 questions, 29 answered ·
bootstrap 10,000 resamples, seed 20260906

## Is the verifier reading, or agreeing?

Every cited claim was checked twice: once against the blocks it actually cited,
and once against blocks drawn from the corpus that it never cited. A verifier that
rubber-stamps cannot tell those apart, and its supported rate means nothing however
high it is. This is the check that licenses every number below it, and it needs no
human labelling to run.

| pairing | supported rate | n |
|---|---|---|
| cited blocks (real) | 0.900 | 40 |
| random blocks (control) | 0.050 | 40 |
| **separation** | **+0.850 [+0.725, +0.950]*** | 40 |

The interval on the separation excludes zero.

## Coverage and precision

Coverage counts every claim the answer made, so an unattributed sentence lowers
it. Precision counts only the claims that were cited. They are reported apart
because an answer that cites one sentence in six and gets it right scores badly
on the first and perfectly on the second, and both facts matter.

| measure | mean [95% CI] | n |
|---|---|---|
| citation coverage | 0.525 [0.425, 0.630] | 29 |
| citation precision | 0.917 [0.822, 0.991] | 29 |

## Every claim whose citations did not hold

6 of 56 cited claims.

| id | cited | claim | why it does not hold |
|---|---|---|---|
| lookup-002 | 5 | When called, they return instances of classes of the same name (e.g., | The passage states that imported Query/Path are functions that return |
| lookup-017 | 1 | [1] You can also add additional metadata for the different tags used t | The passage only points to docs about adding tags to path operations a |
| lookup-017 | 4 | [4] Create metadata for your tags and pass it to the `openapi_tags` pa | Passage [4] discusses using tags on path operations, enums for tags, a |
| multi_hop-008 | 3 | The cookies can be set via a Response object (e.g., in a parameter or | The passage explains setting cookies via a Response parameter or retur |
| ambiguous-006 | 4 | [4] You can use Pydantic's model configuration to forbid any extra fie | The cited passage discusses Header parameter handling (underscore conv |
| ambiguous-006 | 4 | [4] The behavior is described under Forbid Extra Headers. | Passage [4] describes "Automatic conversion" of underscores to hyphens |

## What this cost

Two figures, because they answer different questions and D40 records what happens
when only the first is reported: a cached call truthfully bills $0, so a re-run of
a fully cached check reads as free and is not.

* **what this run paid**: $0.0000 ($0.0000 verification + $0.0000 control), with 96 of 96 calls served from cache
* **what the check costs**: $0.0334 — 96 calls at the measured $0.000348 each

## Limits of this measurement

* **The verifier's agreement with a human is not measured.** The negative control
  shows it discriminates rather than rubber-stamps, which is a weaker claim than
  the kappa reported for the correctness judge and is not a substitute for it.
* **A claim citing several blocks is judged against them together.** When such a
  set fails, every citation in it is flagged, because nothing here can say which
  half was at fault.
* **Claim splitting is deterministic and imperfect.** Coverage is a ratio over
  units this project defines; a different splitter would move the denominator.
