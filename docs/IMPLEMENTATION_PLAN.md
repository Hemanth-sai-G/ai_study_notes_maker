# Implementation Plan

## Phase 4 - Advanced RAG retrieval

**Objective:** Retrieve the best source evidence from the Phase 3 knowledge base without generating an answer.
**Why now:** Chunks/vectors already exist; retrieval must be correct and inspectable before an LLM is allowed to use it.
**Likely files:** new `backend/app/services/retrieval.py`, BM25 index service, new API route/models; `knowledge_base.py`; `frontend/src/App.tsx`; tests.
**Prerequisites:** At least one indexed document, MiniLM model cache, ChromaDB.
**Approach:** Define an evidence model; vector-query Chroma; build/rebuild in-memory or persisted BM25 from extracted chunks; normalize/fuse rankings; accept document/type filters; add conservative query normalization/rewriting with no LLM initially or a future local method; add a compact Cross-Encoder reranker only after baseline fusion is testable; cap/select context by relevance and token/word budget.
**Acceptance:** A query endpoint returns ranked chunks with source metadata, scores/diagnostics, filters, and no duplicate chunks.
**Verification:** Unit test chunk provenance, fusion ordering, filters and empty index; integration test on a known fixture corpus; build UI.
**Risks:** New model download/CPU cost; score scales differ; metadata schema must remain compatible.
**Not included:** Ollama response generation, chat memory, quiz/notes output.

## Phase 5 - Grounded chat and citations

**Objective:** Generate a concise answer with local Ollama, only from Phase 4 evidence, with citations.
**Why now:** Grounding can only be enforced after retrieval is observable and tested.
**Likely files:** new generation/chat services/routes/models and UI components.
**Prerequisites:** Phase 4 API contract; user confirms/installs Ollama and selected Qwen model.
**Approach:** Define strict evidence-only prompt; include source IDs as structured context; handle no-evidence response; generate citations from metadata in code, not model text; store session-only conversation memory initially.
**Acceptance:** Every answer either cites supplied sources or explicitly says evidence is insufficient.
**Verification:** Fixture questions, citation/page checks, no-evidence test, Ollama availability error test.
**Risks:** Model hallucination, latency/VRAM, prompt injection within uploads.
**Not included:** Authentication, translation, audio/video.

## Phase 6 - Study tools and exports

**Objective:** Add grounded notes, summaries, explanations, flashcards, quizzes, document comparison, and exports.
**Why now:** These outputs should reuse one proven evidence/generation pipeline.
**Approach:** Add typed output schemas and one endpoint/mode at a time; each output carries citations.
**Acceptance:** Every tool is source-grounded, usable in UI, and exportable where intended.
**Risks:** Scope expansion and duplicate prompt logic.
**Not included:** Evaluation dashboard/hardening.

## Phase 7 - Evaluation, hardening, and demonstration

**Objective:** Demonstrate quality and make the local app reproducible.
**Approach:** Create labelled retrieval fixtures; measure retrieval relevance/latency/citation coverage; add tests, errors, secure deletion/validation review, documentation, demo script, and packaging instructions.
**Acceptance:** Reproducible setup, objective results, stable demo, complete docs.
**Risks:** Evaluation requires representative academic documents and labels.

## Later / requires user decisions

- Translation of generated responses.
- Audio/video ingestion.
- Authentication and multi-user libraries.
- Deployment beyond localhost.
