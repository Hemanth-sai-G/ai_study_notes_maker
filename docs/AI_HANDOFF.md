# AI Handoff

You are continuing an existing project. Do not assume you should redesign anything.

Read [AGENTS.md](../AGENTS.md) first, then [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md), [CURRENT_STATE.md](CURRENT_STATE.md), [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), [DECISIONS.md](DECISIONS.md), [TODO.md](TODO.md), [TESTING.md](TESTING.md), [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md), and [HANDOFF_AUDIT.md](HANDOFF_AUDIT.md). Inspect actual source code and Git state before modifying files. Preserve existing architectural decisions and continue from the current state instead of restarting.

## Current Mission

Continue StudyMate AI from its completed local retrieval checkpoint toward grounded answer generation.

## Current State

The active app is React/Vite + FastAPI. Phase 2 stores local source files/extracted text and Phase 3 creates locally persistent MiniLM embeddings in ChromaDB. Phase 4 is complete: the backend provides semantic retrieval, local BM25, score fusion, conservative rewriting, metadata filters, deterministic reranking, deduplicated context selection, a word budget, provenance-rich evidence, and the React Workspace evidence-query UI. Ollama integration, chat, citations, and study-material generation are not yet implemented.

## Immediate Next Task

Phase 4 is committed on branch `main2` as `f4ab175`. Phase 5 is in progress with local `qwen2.5:3b`: the evidence-only generation service, endpoint, code-derived citations, tests, and UI are present. Start Ollama and download the model before live answer verification.

## Phase 4 Delivered

- `POST /api/v1/retrieval/query` returns ranked, provenance-rich local evidence from Chroma semantic retrieval plus local BM25 scoring, weighted fusion, deterministic reranking, and budgeted de-duplicated context selection.
- The retrieval contract supports document/type filters, exposes diagnostics, rejects whitespace-only questions, and returns explicit empty-index/no-evidence results.
- `frontend/src/App.tsx` provides an evidence-only Workspace search panel; no answer is generated in Phase 4.
- Ten deterministic fixture tests cover request validation, rewriting, BM25, hybrid ordering, filters, provenance, reranking, context budgets, and the empty-index case. One real-stack integration check is opt-in.

## Files to Inspect First

- `AGENTS.md`
- `backend/app/services/knowledge_base.py`
- `backend/app/services/chunking.py`
- `backend/app/api/documents.py`
- `backend/app/api/retrieval.py`
- `backend/app/models/documents.py`
- `backend/app/models/retrieval.py`
- `backend/app/services/retrieval.py`
- `backend/app/core/config.py`
- `frontend/src/App.tsx`
- `docs/IMPLEMENTATION_PLAN.md`
- `docs/DECISIONS.md`

## Phase 5 Files Expected to Change

- New generation/chat service(s), API route/model module(s), and focused tests.
- `backend/app/main.py` to register a generation route.
- `backend/requirements.txt` only if a justified local Ollama dependency is required.
- `frontend/src/App.tsx` and `styles.css` for grounded-answer UI.
- Project/handoff docs and new tests.

This is not blanket permission to modify these files; inspect and change only what the implemented design needs.

## Phase 5 Acceptance Criteria

- The generator uses only evidence returned by the Phase 4 retrieval service.
- Every answer has code-derived document/page-or-slide citations, or says that evidence is insufficient.
- Ollama-unavailable, no-evidence, and generation errors are explicit and user-safe.
- No source material, embedding, query, or answer leaves the device.
- Existing retrieval tests and frontend build still pass.

## Verification

Run the current verified commands in [TESTING.md](TESTING.md), then add/run focused generation and citation tests. Run `npm run build` in `frontend/` after UI changes.

## Required Phase Closeout

Do not call Phase 5 complete until `docs/PROJECT_GUIDE.md` is updated in the same change set. Follow the established format: completion log, files to know, flow diagram, explanation of models, verification, limitations, review talking points, and the exact next implementation step.

## Do Not Do

- Do not replace ChromaDB, MiniLM, React/FastAPI, or local storage without user approval.
- Do not implement study tools, translation, audio/video, or authentication during Phase 5.
- Do not expose raw source files or commit `backend/data/`.
- Do not silently re-enable network model downloads at runtime.
- Do not modify/delete the legacy Streamlit prototype as part of Phase 5.

## Open Questions

- The user must choose/confirm the final Ollama model before generation is implemented.
- Tesseract installation and OCR quality are unresolved.
