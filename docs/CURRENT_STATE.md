# Current State - 2026-09-16

## Snapshot

- Branch: `main1`.
- HEAD: `1ebf78b` - `phase-3 local semantic knowledge base dividing into chunks`.
- Working tree at the end of the handoff audit: the documentation package (`AGENTS.md` plus the new handoff files in `docs/`) is uncommitted. Application source behavior is unchanged by this handoff task.
- Current completed implementation: Phases 0-3 of the project roadmap.
- Current work in progress: no running implementation task. The next planned task is Phase 4, Advanced RAG retrieval.

## Completed

1. **Foundation/UI (Phase 1):** React/Vite/TypeScript interface, FastAPI health endpoint, localhost Vite proxy, responsive workspace and Library UI.
2. **Ingestion (Phase 2):** local PDF, DOCX, PPTX, TXT, PNG/JPG/JPEG/WEBP upload; 50 MB limit; UUID storage; extracted-text persistence; JSON catalogue; local Library display.
3. **Knowledge base (Phase 3):** paragraph-oriented chunks, `all-MiniLM-L6-v2` embeddings, ChromaDB persistence, source metadata, re-index endpoint and Library action.

## Current local data state

The live local data folders are ignored by Git. At the last verified Phase 3 run, two local documents were indexed into 15 chunks. This is machine-specific state, not portable repository content. ChromaDB and the downloaded MiniLM model must not be assumed to exist on a new machine.

## Recent meaningful changes

The latest commit added Phase 2 and Phase 3 application code, the React frontend, dependencies, and updated project guides. It also accidentally includes `tmp/phase3-pip-out.log` and `tmp/phase3-pip-err.log`, which are installation diagnostics and should be removed in a future documentation/cleanup commit after confirming they are not needed. They do not affect runtime behavior.

## Recently changed source areas

- `backend/app/api/documents.py`
- `backend/app/services/{document_extractor,document_repository,chunking,knowledge_base}.py`
- `backend/app/models/documents.py`
- `backend/app/core/config.py`
- `frontend/src/{App.tsx,styles.css}`
- `backend/requirements.txt`, `frontend/package*.json`

## Verification status

Passing checks recorded during implementation:

- Python compilation: `python -m compileall -q backend/app`.
- Focused upload/extraction checks for PDF, DOCX, PPTX, TXT, PNG plus unsupported-file rejection.
- Deterministic ChromaDB persistence check using a fake embedding model.
- Actual `all-MiniLM-L6-v2` cache-only embedding run and real local indexing: 2 documents / 15 chunks.
- `npm run build` in `frontend/`.

There is no committed test suite, test runner configuration, linter, or end-to-end test harness. These are test gaps, not known failing tests.

## Warnings and known runtime conditions

- PyMuPDF emits a `fitz` deprecation warning; the current code still uses `fitz` and works.
- FastAPI TestClient emits a Starlette/httpx deprecation warning in ad hoc checks.
- The Hugging Face cache can warn about Windows symlink support. The embedding model works without symlinks, using more disk space.
- Tesseract executable is not known to be installed. Image records can be saved but show `OCR needed` and are not indexable until OCR produces text.
- On a fresh machine, the first model download requires internet; normal runtime indexing deliberately uses `local_files_only=True`.

## Must not be lost

- `backend/data/` is intentionally local user data and excluded from Git.
- The JSON catalogue and ChromaDB collection are separate: catalogue tracks documents/status; ChromaDB tracks searchable chunks.
- Source provenance metadata is essential for later citations.
- The root Streamlit implementation remains legacy, while React/FastAPI is the active path.
