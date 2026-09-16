# StudyMate AI - Agent Instructions

StudyMate AI is a local-first Advanced RAG learning assistant. The active product is the React frontend in `frontend/` and FastAPI backend in `backend/`; the root Streamlit files are a legacy prototype and must not be modified or removed without user approval.

Read `docs/AI_HANDOFF.md`, then the listed handoff documents, before coding. Inspect current code and `git status` before assuming documentation is current.

## Architecture rules

- Keep browser requests relative (`/api/...`) and use the Vite proxy to FastAPI.
- Keep study files, extracted text, embeddings, and generated content local. Never add cloud upload, analytics, or external inference without explicit approval.
- Preserve provenance metadata (`document_id`, name, type, chunk index, page/slide, section) through every retrieval/generation stage.
- Keep ingestion, chunking/indexing, retrieval, and generation as separate backend services.
- Do not merge Phase 4 retrieval or Phase 5 LLM work into an unrelated change.

## Conventions and verification

- Python: typed Pydantic API models in `backend/app/models/`; routes in `api/`; business logic in `services/`; configuration via `core/config.py`.
- React: functional components and hooks in `frontend/src/App.tsx`; custom CSS in `frontend/src/styles.css`; do not add a UI framework casually.
- Validate input on the FastAPI boundary and return `HTTPException` with a user-safe `detail` message.
- Do not commit files in `backend/data/`, model caches, `frontend/node_modules/`, builds, virtual environments, or personal uploaded material.
- Before completing work, run `python -m compileall -q backend/app`, `npm run build` from `frontend/`, and focused API/service verification appropriate to the changed feature. Document unrun tests or failures honestly.

## Special care / ask first

- Do not change the local-first architecture, model choice, storage format, supported file types, authentication scope, or deferred feature boundaries without asking the user.
- The embedding model is deliberately cache-only at runtime. Do not silently re-enable network downloads.
- Do not delete/re-index user data without clear user authorization; re-indexing a selected document is allowed only where the UI/API explicitly requests it.
- Avoid destructive Git operations. Make coherent commits only when the user asks.

## Definition of done

The requested behavior works, errors are handled, local data/provenance are preserved, focused checks and build pass, and the living handoff/project docs are updated when architecture, workflow, or phase status changes.
