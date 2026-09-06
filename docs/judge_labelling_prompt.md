# Labelling the judge-agreement sample with a model's help

The judge-agreement experiment (D11) asks whether a cheap LLM judge agrees with a **human**.
That is what defuses the circularity of an LLM grading an LLM, so the final labels must be
a person's. Model-proposed labels are a labour-saving step, never the standard:
`proposed_verdict` and `human_verdict` are separate fields for that reason, and only
`human_verdict` enters the agreement statistic.

Using a second, *stronger* model to propose labels is defensible and normal in annotation
work, on two conditions: it is a different model from either judge candidate, and a human
adjudicates every item before the number is reported.

## The procedure

1. Build the compact review document:
   ```bash
   uv run python scripts/judge_agreement.py extract
   ```
   That writes `evals/judge_labels_review.md` — about 18 KB, everything a labeller needs and
   nothing else. Add `--context` to include the retrieved passages, which are needed only
   if you also want grounding labels.

2. Attach that file in Claude Desktop and send the prompt below.

3. Save the YAML block it returns to `evals/proposed_verdicts.yaml`, then:
   ```bash
   uv run python scripts/judge_agreement.py apply evals/proposed_verdicts.yaml --field proposed
   uv run python scripts/judge_agreement.py label
   ```
   The second command shows each item with its proposal and waits for **your** key:
   `c`/`p`/`i` to label, `s` to skip, `q` to stop. Progress saves after every keystroke.

4. Then score the judges:
   ```bash
   uv run python scripts/judge_agreement.py score
   ```

## The prompt

> You are helping validate an automated judge for a retrieval-augmented generation system.
>
> The attached document holds 20 answers produced by a documentation assistant over the
> FastAPI documentation. Each item gives the question, a reference answer a human wrote from
> the documentation, and the assistant's answer.
>
> Propose a verdict for each item. Your labels are a **proposal**: a human reviews every one
> of them and can override it, and the agreement statistic this feeds is computed against
> that person's decisions, not yours.
>
> Apply these rules exactly as written. A different reading of the rules would later surface
> as judge disagreement that is really rule confusion:
>
> - **correct** — conveys what the reference conveys.
> - **partial** — gets part of it, but omits something the reference treats as essential, or
>   adds a claim that is wrong.
> - **incorrect** — contradicts the reference, or does not answer the question.
> - Where the reference answer is **empty**, the documentation does not answer that question:
>   declining to answer is **correct**, and answering it anyway is **incorrect**.
>
> Judge substance only. An answer that is longer, more detailed, differently worded or
> differently formatted than the reference is still correct if it conveys the same thing.
> Citation markers like `[2]` are part of the system's output format — do not grade them.
>
> Where you are genuinely torn between two verdicts, choose the stricter one and say so in
> the reason. An over-strict proposal is cheap for a human to overturn; a lenient one gets
> waved through.
>
> Reply with a single YAML code block and nothing else. Use each `question_id` exactly as
> written in the document, and one of `correct`, `partial`, `incorrect` as the verdict:
>
> ```yaml
> lookup-017:
>   verdict: partial
>   reason: one short sentence
> multi_hop-005:
>   verdict: correct
>   reason: one short sentence
> ```

## What you may claim afterwards

| How the labels were made | Honest wording in the README |
|---|---|
| A person labelled all 20 | "judge–human agreement, n=20" |
| A model proposed, a person adjudicated every item | "judge–human agreement, n=20 (labels model-proposed, author-adjudicated)" |
| A model labelled, nobody adjudicated | "agreement with a stronger reference model" — **not** human agreement |

The third row still measures something real, but it does not answer the circularity
objection, and a reviewer who asks how the judge was validated will notice the difference.
