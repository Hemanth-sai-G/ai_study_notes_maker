# Decision Record

## Decision: Local-first deployment

Status: Active
Date/phase: Initial planning / Phases 0-3
Decision: Run the app, document storage, vector database, embedding model, and future LLM locally.
Reason: User requested local development; this protects study-material privacy, avoids recurring API cost, and supports offline demonstration after model setup.
Alternatives considered: Cloud LLM/API and hosted database were discussed but not selected.
Consequences: Local dependencies/models must be installed; authentication and multi-user isolation are deferred.
Affected files/components: `backend/`, `frontend/vite.config.ts`, `backend/data/`, `docs/PROJECT_GUIDE.md`.

## Decision: React + FastAPI replaces the final Streamlit UI

Status: Active
Date/phase: Phase 0/1
Decision: Use React/Vite/TypeScript with custom CSS for the interface and FastAPI for Python services.
Reason: User wanted a good-looking non-basic UI; the split gives a polished frontend and modular backend.
Alternatives considered: Continue/extensively style the existing Streamlit prototype.
Consequences: Two local development processes run during development. Root Streamlit files are legacy.
Affected files/components: `frontend/`, `backend/app/`, root `app.py` and `utils/`.

## Decision: ChromaDB is the persistent vector store

Status: Active
Date/phase: Phase 0/3
Decision: Store semantic chunk vectors and metadata in local persistent ChromaDB.
Reason: It persists locally and supports metadata filtering required for Advanced RAG.
Alternatives considered: FAISS was considered during planning; ChromaDB was selected for persistence/filtering.
Consequences: `backend/data/chroma/` is runtime state and ignored.
Affected files/components: `backend/app/services/knowledge_base.py`, `backend/app/core/config.py`.

## Decision: all-MiniLM-L6-v2 embedding model

Status: Active
Date/phase: Phase 3
Decision: Use `sentence-transformers/all-MiniLM-L6-v2`, normalized 384-dimension embeddings.
Reason: Compact model suitable for the user's RTX 4050 6 GB VRAM / 16 GB RAM machine and sufficient for a first local semantic index.
Alternatives considered: Rationale for other specific embedding models was not recorded.
Consequences: One-time model setup download; subsequent runtime loading is cache-only.
Affected files/components: `backend/app/core/config.py`, `backend/app/services/knowledge_base.py`, `backend/requirements.txt`.

## Decision: Source metadata must accompany every chunk

Status: Active
Date/phase: Phases 2-3
Decision: Persist document identity, display name, type, chunk index, page/slide and section title with each Chroma record.
Reason: Future answers need deterministic citations rather than LLM-guessed source names.
Alternatives considered: None recorded.
Consequences: Future retrieval/generation code must preserve this metadata.
Affected files/components: `document_extractor.py`, `chunking.py`, `knowledge_base.py`.

## Decision: Phase-by-phase implementation and documentation

Status: Active
Date/phase: Initial planning
Decision: Build isolated, testable phases and update a beginner-friendly guide after each checkpoint.
Reason: User wants to pause/resume and learn enough to explain the project and hand it to teammates.
Alternatives considered: One large end-to-end implementation.
Consequences: Do not collapse later capabilities into current phases without user agreement.
Affected files/components: `docs/PROJECT_GUIDE.md`, all future work.

## Decision: Image OCR is optional local Tesseract

Status: Active
Date/phase: Phase 2
Decision: Validate/store images; use Tesseract only if its local executable exists.
Reason: No cloud OCR; avoid falsely claiming text was extracted.
Alternatives considered: Cloud OCR was rejected by the local-first constraint.
Consequences: Image documents may remain `ocr_unavailable` / `no_text` and cannot be indexed.
Affected files/components: `backend/app/services/document_extractor.py`.
