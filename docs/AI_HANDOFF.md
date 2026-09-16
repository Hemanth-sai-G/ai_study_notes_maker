# AI Handoff

You are continuing an existing project. Do not assume you should redesign anything.

Read [AGENTS.md](../AGENTS.md) first, then [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md), [CURRENT_STATE.md](CURRENT_STATE.md), [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), [DECISIONS.md](DECISIONS.md), [TODO.md](TODO.md), [TESTING.md](TESTING.md), [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md), and [HANDOFF_AUDIT.md](HANDOFF_AUDIT.md). Inspect actual source code and Git state before modifying files. Preserve existing architectural decisions and continue from the current state instead of restarting.

## Current Mission

Continue StudyMate AI from its completed local ingestion and semantic-index checkpoint toward Advanced RAG retrieval.

## Current State

The active app is React/Vite + FastAPI. Phase 2 stores local source files/extracted text and Phase 3 creates locally persistent MiniLM embeddings in ChromaDB. The Library UI can build a knowledge base. There is no retrieval API, BM25 index, reranker, Ollama integration, chat, citations, or study-material generation yet.

## Immediate Next Task

Implement **Phase 4: Advanced RAG retrieval**, beginning with a tested semantic-vector retrieval endpoint and evidence data contract. Then add BM25, score fusion, filters, query rewriting, reranking, and context optimization incrementally rather than in one untestable change.

## Implementation Approach

1. Inspect `knowledge_base.py`, `chunking.py`, `documents.py`, and the current Chroma metadata schema.
2. Define Pydantic request/response models for a query and evidence result. Preserve source metadata in the response.
3. Implement a semantic Chroma query service first; test known documents and an empty-index error.
4. Build BM25 from the same chunks/extracted text with a clear rebuild policy; test exact keyword matches.
5. Normalize/fuse the two rankings and deduplicate by chunk ID.
6. Add document/type metadata filtering.
7. Add reranking only after baseline retrieval is testable. Choose/download its local model only with user awareness.
8. Return compact context/evidence. Do not call an LLM in Phase 4.

## Files to Inspect First

- `AGENTS.md`
- `backend/app/services/knowledge_base.py`
- `backend/app/services/chunking.py`
- `backend/app/api/documents.py`
- `backend/app/models/documents.py`
- `backend/app/core/config.py`
- `frontend/src/App.tsx`
- `docs/IMPLEMENTATION_PLAN.md`
- `docs/DECISIONS.md`

## Files Expected to Change

- New retrieval service(s) and API route/model module(s).
- `backend/app/main.py` to register a new route.
- `backend/requirements.txt` only if a justified local dependency is required.
- `frontend/src/App.tsx` and `styles.css` for evidence-query UI, if included in the phase checkpoint.
- Project/handoff docs and new tests.

This is not blanket permission to modify these files; inspect and change only what the implemented design needs.

## Acceptance Criteria

- An indexed document can be queried semantically through FastAPI.
- Result evidence has text, score/rank, document identity/name, page/slide, section title, and chunk ID/index.
- Empty/no-text/error conditions are explicit and user-safe.
- BM25 and fused ranking are tested before reranking.
- No source material, embedding, or query is sent to cloud services.
- Existing indexing and frontend build still pass.

## Verification

Run the current verified commands in [TESTING.md](TESTING.md), then add/run focused retrieval tests. Test both exact-term and semantic-paraphrase queries with controlled fixture data. Run `npm run build` in `frontend/` after UI changes.

## Required Phase Closeout

Do not call Phase 4 complete until `docs/PROJECT_GUIDE.md` is updated in the same change set. Follow the format established for Phases 0-3: completion log, files to know, flow diagram, explanation of concepts/models, verification, limitations, roadmap status, and exact next implementation step.

## Do Not Do

- Do not replace ChromaDB, MiniLM, React/FastAPI, or local storage without user approval.
- Do not wire Ollama or generate answers during Phase 4.
- Do not expose raw source files or commit `backend/data/`.
- Do not silently re-enable network model downloads at runtime.
- Do not modify/delete the legacy Streamlit prototype as part of Phase 4.

## Open Questions

- Reranker model selection remains open; choose a compact local cross-encoder only after baseline retrieval works.
- The user must choose/confirm the final Ollama model before Phase 5.
- Tesseract installation and OCR quality are unresolved.
