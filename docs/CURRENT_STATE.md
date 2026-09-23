# Current State - 2026-09-22

## Snapshot

- Branch: `main2`.
- HEAD: `f0694dd` - `phase-5 grounded local chat and citations`.
- Working tree contains the uncommitted Phase 6 implementation and closeout documentation.
- Current completed implementation: Phases 0-6. The exact next phase is Phase 7 evaluation, hardening, and demonstration preparation.

## Completed

1. **Foundation/UI (Phase 1):** React/Vite/TypeScript interface, FastAPI health endpoint, localhost Vite proxy, responsive workspace and Library UI.
2. **Ingestion (Phase 2):** local PDF, DOCX, PPTX, TXT, PNG/JPG/JPEG/WEBP upload; 50 MB limit; UUID storage; extracted-text persistence; JSON catalogue; local Library display.
3. **Knowledge base (Phase 3):** paragraph-oriented chunks, `all-MiniLM-L6-v2` embeddings, ChromaDB persistence, source metadata, re-index endpoint and Library action.
4. **Advanced retrieval (Phase 4):** semantic Chroma retrieval, local BM25, score fusion, filters, conservative rewriting, deterministic reranking, duplicate-aware word-budgeted context selection, provenance-rich evidence API, and Workspace evidence search.
5. **Grounded chat (Phase 5):** evidence-only Ollama prompt, code-derived citations, session-only in-memory context, safe unavailable-model handling, and Workspace answer display.
6. **Study tools (Phase 6):** selected-document notes, summaries, explanations, flashcards, quizzes, comparisons, verified sources, and local Markdown export.

## Current local data state

The live local data folders are ignored by Git. At the last verified Phase 3 run, two local documents were indexed into 15 chunks. This is machine-specific state, not portable repository content. ChromaDB and the downloaded MiniLM model must not be assumed to exist on a new machine.

## Recent meaningful changes

Phase 4 was committed as `f4ab175`, Phase 5 as `f0694dd`, and Phase 6 as `90c850f`. The current uncommitted Phase 6 enhancement adds JSON-validated activities, flip-card flashcards, scored quiz feedback with explanations, and an enforced two-document comparison flow. It preserves the existing Chroma metadata contract and does not call any cloud service.

## Recently changed source areas

- `backend/app/api/documents.py`
- `backend/app/services/{document_extractor,document_repository,chunking,knowledge_base}.py`
- `backend/app/models/documents.py`
- `backend/app/core/config.py`
- `backend/app/{api,models,services}/retrieval.py`
- `backend/tests/test_retrieval*.py`
- `backend/app/{api,models,services}/generation.py`
- `backend/app/api/chat.py`
- `backend/tests/test_generation.py`
- `frontend/src/{App.tsx,styles.css}`
- `backend/requirements.txt`, `frontend/package*.json`

## Verification status

Passing checks recorded during implementation:

- Python compilation: `python -m compileall -q backend/app`.
- Fifteen deterministic Phase 4/5 tests pass; the optional real FastAPI/ChromaDB integration check is skipped until `STUDYMATE_RUN_REAL_INTEGRATION=1` is set after indexing documents.
- Focused upload/extraction checks for PDF, DOCX, PPTX, TXT, PNG plus unsupported-file rejection.
- Deterministic ChromaDB persistence check using a fake embedding model.
- Actual `all-MiniLM-L6-v2` cache-only embedding run and real local indexing: 2 documents / 15 chunks.
- `npm run build` in `frontend/`.

There is a committed Python `unittest` suite for Phase 4 retrieval. Extraction/indexing tests, a frontend test runner, linting, and end-to-end automation remain test gaps.

## Warnings and known runtime conditions

- PyMuPDF emits a `fitz` deprecation warning; the current code still uses `fitz` and works.
- FastAPI TestClient emits a Starlette/httpx deprecation warning in ad hoc checks.
- The Hugging Face cache can warn about Windows symlink support. The embedding model works without symlinks, using more disk space.
- Tesseract executable is not known to be installed. Image records can be saved but show `OCR needed` and are not indexable until OCR produces text.
- On a fresh machine, the first model download requires internet; normal runtime indexing deliberately uses `local_files_only=True`.
- Phase 5 code is implemented and unit-tested, but the installed Ollama CLI could not connect to a running local service. Run `ollama serve` and `ollama pull qwen2.5:3b` before live answer verification. Study tools, translation, audio/video, and authentication remain deferred.

## Must not be lost

- `backend/data/` is intentionally local user data and excluded from Git.
- The JSON catalogue and ChromaDB collection are separate: catalogue tracks documents/status; ChromaDB tracks searchable chunks.
- Source provenance metadata is essential for later citations.
- The root Streamlit implementation remains legacy, while React/FastAPI is the active path.
