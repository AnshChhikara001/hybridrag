# Project Brief — RAG Pipeline with Hybrid Search Over Internal Docs

Transcribed verbatim-in-substance from the three source screenshots
(AIEngineerAccelerator, "Project 6"). This file is the requirements source of truth;
the screenshots are the original artefact.

## What You're Building

A production-grade Retrieval-Augmented Generation system that ingests a company's
internal documentation, indexes it with both dense vector and sparse keyword search,
retrieves the most relevant context for any question, and generates grounded answers with
inline source citations.

## Why This Project Lands Interviews

> RAG is the single most requested skill in AI engineering job descriptions. But most
> candidates build a toy demo with a single PDF. You're building a system with hybrid
> retrieval, chunking strategy decisions, and citation verification — the production
> concerns that separate a real RAG engineer from someone who followed a LangChain
> quickstart.

## Tech Stack (as proposed by the brief)

| Component | Tool / Library | Why This Choice |
|---|---|---|
| Language | Python 3.11+ | Ecosystem standard |
| Embeddings | OpenAI `text-embedding-3-small` | Cost-effective, high quality |
| Vector Store | ChromaDB or Qdrant | File-based or containerized |
| Sparse Search | BM25 via `rank_bm25` | Keyword matching for exact terms |
| LLM | GPT-4o or Claude Sonnet | Strong grounding and citation |
| Chunking | LangChain text splitters | Configurable overlap and size |
| API | FastAPI | Async-native, production-grade |
| Containerization | Docker | Reproducible deployment |

## Phase 1: Ingestion and Chunking Pipeline (Day 1–3)

1. **Multi-format document loader** — accept markdown, text, HTML and PDF. Normalize
   everything into clean plaintext with metadata (source file, section heading, page
   number). Store raw documents alongside processed versions so re-indexing needs no
   re-upload.
2. **Configurable chunking** — three switchable strategies: fixed-size with overlap
   (baseline), recursive character splitting by section headers (structure-aware), and
   semantic chunking that splits on topic boundaries using embedding similarity. Track
   which strategy each chunk used.
3. **Generate and store embeddings** — embed every chunk. Store with metadata: source
   document, chunk index, section heading, chunking strategy, character count. Build the
   BM25 index in parallel over the same chunks. Both indexes must stay in sync.
4. **Deduplication** — before inserting a chunk, check for near-duplicates (cosine
   similarity > 0.95 against existing chunks). Flag and skip duplicates, so the retriever
   does not waste context-window slots on redundant content.

## Phase 2: Hybrid Retrieval Engine (Day 3–6)

1. **Dense retrieval** — query the vector store with the embedded question, return top-k
   chunks by cosine similarity. Start with k=10.
2. **Sparse retrieval** — run the same query through BM25 over the chunk corpus, return
   top-k by BM25 score. Catches exact keyword matches semantic search misses — critical
   for technical documentation with specific function names, config keys, or error codes.
3. **Fusion layer** — Reciprocal Rank Fusion to combine dense and sparse results into one
   ranked list. RRF scores by rank position across both lists. Weighting configurable
   (e.g. 0.7 dense / 0.3 sparse) so it can be tuned per use case.
4. **Reranker** — send the top 20 fused candidates through a cross-encoder reranker (a
   small model or LLM-as-judge) that scores each chunk's relevance to the actual question.
   Keep the top 5. This second pass dramatically improves precision and is a strong
   interview talking point.

## Phase 3: Generation and Citation Layer (Day 6–9)

1. **Grounded generation prompt** — a system prompt instructing the LLM to answer only
   from provided context, cite specific chunks using bracketed references ([1], [2]), and
   explicitly state when the context lacks enough information. Retrieved chunks are
   included as numbered context blocks.
2. **Citation verification** — after generation, parse the model's citations and verify
   each one. Does [1] actually support the claim attached to it? Send each citation-claim
   pair to an LLM-as-judge for verification. Flag unsupported citations. This is the
   quality layer most RAG systems skip entirely.
3. **Answer confidence scorer** — score each answer on retrieval confidence (how relevant
   were the top chunks?), citation coverage (what percentage of claims have verified
   citations?), and answer completeness (did the response address all parts of the
   question?). Return a composite confidence score alongside the answer.
4. **Graceful "I don't know"** — if retrieval confidence is below a threshold, do not
   hallucinate. Return a structured response saying what the system found, what it could
   not find, and which documents might be worth checking manually. More useful than a
   fabricated answer, and signals production maturity.

## Phase 4: Evaluation Framework (Day 9–11)

1. **Golden Q&A dataset** — 50+ question-answer pairs written by hand, each tied to
   specific sections of the corpus. Include straightforward lookups, multi-hop questions
   (answer requires combining two documents), questions with no answer in the corpus, and
   ambiguous questions.
2. **Automated eval metrics** — for each test case measure: answer correctness
   (LLM-as-judge against the golden answer), faithfulness (are all claims grounded in
   retrieved context?), retrieval relevance (were the right chunks retrieved?), and
   citation accuracy (do citations actually support claims?). Run the full suite on every
   pipeline change.
3. **Chunking strategy comparison** — run the same eval suite across all three chunking
   strategies. Generate a comparison report showing which strategy wins on which metrics.
   This data drives architecture decisions and gives concrete numbers for interviews.

## Phase 5: API and Dashboard (Day 11–13)

1. **FastAPI service** — `POST /v1/ask` accepts a question and returns the answer with
   citations, confidence scores and source metadata. `GET /v1/documents` lists indexed
   documents. `POST /v1/ingest` accepts new documents for indexing. Include OpenAPI docs.
2. **Query dashboard** — a Streamlit or React frontend to ask questions and see: the
   generated answer with clickable citations, the retrieved chunks ranked by relevance,
   confidence scores broken down by dimension, and a toggle to compare hybrid vs
   dense-only retrieval side by side.
3. **Containerize everything** — docker-compose with the API service, the vector store and
   the frontend. Include a seed script that indexes a sample documentation corpus so
   reviewers can spin it up and test immediately.

## Phase 6: Portfolio Polish (Day 13–14)

1. **Demo walkthrough** — under 4 minutes. Show ingesting documents, asking questions of
   varying difficulty, citation verification catching a hallucination, and the hybrid vs
   dense-only comparison.
2. **Case study** — framed as "I built a RAG system with hybrid search that achieves X%
   faithfulness and Y% citation accuracy on a 50-question eval suite." Lead with the
   numbers. Explain why hybrid beats dense-only for technical documentation. Show the
   chunking strategy comparison data.
