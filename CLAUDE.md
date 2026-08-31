# CLAUDE.md

Operating contract for this repository. Read fully before acting.

## Project Mission

Build a portfolio-ready AI engineering project that maximises my chances of landing an
AI/GenAI internship. The finished repo must read as something an AI engineer
intentionally designed — not a tutorial follow-along and not a chatbot wrapper.

## Project Requirements

Source of truth: `docs/project-brief.md` (transcribed from the AIEngineerAccelerator
"Project 6" brief). Requirements come from that brief, never from invention.

**What we are building:** a production-grade Retrieval-Augmented Generation system that
ingests a technical documentation corpus, indexes it with **both** dense vector and
sparse keyword search, retrieves the most relevant context for a question, and generates
grounded answers with **inline source citations**.

The six phases the brief defines:

1. **Ingestion & chunking** — multi-format loader; three switchable chunking strategies;
   dense + sparse indexes built over the same chunks and kept in sync; near-duplicate
   detection.
2. **Hybrid retrieval** — dense top-k, sparse BM25 top-k, Reciprocal Rank Fusion with
   configurable weighting, then a reranking pass that narrows candidates to a final set.
3. **Generation & citation** — grounded prompt with numbered context blocks and bracketed
   citations; citation verification that checks each cited claim; a composite answer
   confidence score; a structured "I don't know" path below a confidence threshold.
4. **Evaluation** — a hand-verified golden Q&A set spanning lookup, multi-hop, no-answer
   and ambiguous questions; automated metrics for retrieval quality, answer correctness,
   faithfulness and citation accuracy; a chunking-strategy comparison report.
5. **API & dashboard** — FastAPI service with OpenAPI docs; a query dashboard showing the
   answer, its citations, the retrieved chunks, the confidence breakdown, and a
   hybrid-vs-dense-only comparison; containerised with a seed script.
6. **Portfolio polish** — demo walkthrough and a numbers-first case study.

**Stack named by the brief** (every choice is open to challenge on cost and hardware grounds;
whatever we settle on is recorded in `docs/PROJECT_STATE.md` with its rationale):
Python 3.11+, OpenAI `text-embedding-3-small`, ChromaDB or Qdrant, BM25 via `rank_bm25`,
GPT-4o or Claude Sonnet, LangChain text splitters, FastAPI, Docker.

## My Knowledge

I already understand most of the AI concepts here. Skip explanations of RAG basics,
embeddings, chunking, or what an LLM is.

**Docker is my weak area.** Every time Docker is introduced or changed, explain what each
important instruction does, why it exists, how containers reach each other, and how to run
and debug it. Do not hide Docker complexity from me.

Explain implementation and architectural decisions in depth. Skip AI fundamentals.

## Budget

**Hard ceiling: $1 total for the entire project.** Prefer free tiers, open-source models,
local execution, and free infrastructure.

Before introducing any paid API or service, state: why it is needed, the estimated cost
with the arithmetic shown, the free alternatives, and whether it is genuinely worth it for
the portfolio. Then wait for my approval. Never add a paid dependency without it.

Track cumulative spend in `docs/PROJECT_STATE.md`.

## Development Workflow

Build **phase by phase**, following the six phases in `docs/project-brief.md`. Do not
design the whole system up front. For each phase, in order:

1. **Grill** — ask the questions that phase's design actually raises, each with a
   recommendation.
2. **Wait** — no code until that phase's questions are settled.
3. **Build** in small steps. For every step: say what we are accomplishing, make only the
   necessary edits, show exactly what changed and in which file, say why, run the relevant
   check, report the result including failures, and say what the project can do now that
   it could not before.
4. **Validate** — a phase is done when its tests pass and I can see it work.
5. **Record** — update `docs/PROJECT_STATE.md`, then propose a commit message.
6. Then grill the next phase.

Report every file created or changed, by path, every time. Never batch unrelated changes.
Never run ahead into a later phase.

**Cross-cutting exception:** when a later phase would force a constraint on the current one
— a schema field, a stable identifier, stored metadata — raise it and settle it now.
Retrofitting those means re-indexing the whole corpus, so they are worth the interruption.

## Grilling

When requirements, architecture, technology choice, scope, evaluation strategy,
deployment, cost, or portfolio positioning are unclear, stop and ask.

Challenge my decisions when a better approach exists. Where several approaches are
reasonable, compare them briefly and recommend one. Disagree with me when the evidence
says I am wrong.

## Portfolio Standard

Prioritise: strong architecture, meaningful AI engineering, evaluation, reliability,
observability where it earns its place, testing, clean code, documentation,
reproducibility, security, deployment readiness, a clear README, an architecture diagram,
meaningful metrics, and a professional Git history.

## Scope Control

Build the smallest strong V1 first. Every major feature must answer yes to at least one:
does it improve the product, demonstrate an important AI engineering skill, or improve the
portfolio — and is its cost justified? If not, challenge it before building it.

## Engineering Rules

- Secrets live in `.env`, which is gitignored. Ship a `.env.example`. Nothing hardcoded.
- Validate all external input, including API request bodies and uploaded documents.
- Keep dependencies intentional and minimal; pin them.
- Configuration stays separate from application logic.
- Handle errors explicitly and surface failures loudly.
- Use type hints. Write comments that explain why, not what.
- Follow a conventional Python project structure.

## AI Engineering Rules

"It works" is not evidence. Build measurable evaluation into the project and report
retrieval quality, answer quality, latency, cost, failure cases, and grounding.

Prefer measured numbers over claims in the README. Report negative results honestly — a
documented case where the fancy approach loses is stronger evidence of engineering
judgement than a table where everything wins.

## Context Preservation

Maintain `docs/PROJECT_STATE.md` covering: current architecture, decisions and their
rationale, completed milestones, implementation status, known issues, cumulative spend,
and the next planned step. Update it whenever an architectural decision or milestone lands.

## Git

Commit at logical milestones, not after every edit. When a milestone completes, propose a
professional commit message and let me approve it.

## Final Deliverable

Clean source, README, setup instructions, `.env.example`, tests, evaluation methodology and
results, architecture documentation, Docker configuration, deployment instructions, example
usage, limitations, and future improvements.

The README must let an AI/GenAI recruiter quickly see: the problem, why it matters, the
architecture, the key technical decisions, the AI techniques used, the evaluation results,
the cost, how to run it, and what I built and learned.

## Critical Rule

The whole brief is understood at architecture level before any code, and that
understanding lives in `docs/PROJECT_STATE.md`. Detailed design is settled **per phase**,
immediately before that phase is built — not all at once up front.

## Plain-English Communication Standard

Use clear, plain English in all explanations. Avoid unnecessary jargon and explain unfamiliar technical terms, tools, commands, and libraries briefly when introduced.

For every important implementation or architectural decision, explain **what it does, why we need it, how it works in this project, and the relevant trade-offs**.

Do not explain AI fundamentals I already understand. The goal is that I can understand the project's technical decisions without needing to translate or ask for a simpler explanation.
