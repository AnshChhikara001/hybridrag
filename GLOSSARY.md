# Glossary

Every term below has one meaning in the code, the reports, the issues and the README. Source: spec #1.

| Term | Meaning |
|---|---|
| Golden set | All eval questions, stored in one file (`evals/golden_set.yaml`). Each question has an ID, a category, a Split and, when answerable, one or more Spans. |
| Span | The exact passage in the FastAPI docs that contains the answer to a question. |
| Split | Which part of the Golden set a question belongs to: Tuning set or Held-out set. |
| Tuning set | The 35 existing questions. Used only to choose settings. Never used for published numbers. |
| Held-out set | 52 new questions from real users, each with a source URL. The only source of published numbers. |
| Arm | One retrieval configuration under test: hybrid, dense only, or BM25 only. |
| Answer model | The LLM that writes answers: gpt-5-mini, in the eval and in the demo. |
| Judge | The LLM that grades an answer as correct, partial or incorrect. |
| Citation verifier | The LLM that checks whether a cited passage supports the claim it is attached to. |
| Plug-in model | A language model loaded from a private Python file named by an environment setting. Never committed. |
| Refusal gate | The check that refuses a question before any LLM call when the best dense similarity is below 0.30. Built for off-topic questions. |
| Demo mode | A setting that switches the API to its public configuration (limits on, uploads off). |
| Real question | A demo request that calls the Answer model. Cache hits and Refusal gate refusals are not real questions. |
| v0 reports | The eval reports made before spec #1, kept unchanged in `evals/reports/v0/`. |

## Question categories

| Category | Meaning | Spans |
|---|---|---|
| `lookup` | Answered by one passage. | exactly 1 |
| `multi_hop` | Needs passages from two or more documents. | 2 or more, across 2+ documents |
| `ambiguous` | Several passages each give an acceptable answer. | 2 or more |
| `no_answer` | In the FastAPI domain, but the docs do not answer it. | none |
| `off_topic` | Outside the domain, such as general Python or unrelated topics. | none |
