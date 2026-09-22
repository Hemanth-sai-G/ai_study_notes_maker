# StudyMate AI - Master Project Handoff

## A. Identity and status

**Name:** StudyMate AI (the proposal also describes it as an Advanced RAG-Powered Learning Assistant).
**Purpose:** Help students turn their own course materials into source-grounded learning support.
**Target users:** Students using local PDFs, Word documents, PowerPoint slides, text files, and eventually images/scanned notes.
**Current status:** Phases 0-4 are complete. Phase 5 implementation is in progress: the app has an evidence-only local Ollama generation layer and code-derived citations, pending live verification with the downloaded Qwen model.

## B. Technology stack

| Area | Actual choice |
| --- | --- |
| Languages | Python 3.11 virtual environment observed; TypeScript/JavaScript; CSS |
| Backend | FastAPI, Uvicorn, Pydantic Settings, python-multipart |
| Frontend/build | React, Vite, TypeScript, npm |
| Document processing | PyMuPDF (`fitz`), python-docx, python-pptx, Pillow, optional pytesseract/Tesseract |
| Semantic index | Sentence Transformers, `all-MiniLM-L6-v2`, ChromaDB |
| Data storage | Local filesystem, JSON document catalogue, persistent local ChromaDB |
| LLM/API | Local Ollama at `127.0.0.1:11434` using `qwen2.5:3b`; code is implemented but the local service/model needs setup for live verification. No cloud API is configured. |
| Authentication | None; explicitly deferred. |
| Tests | Python `unittest` fixture suite for Phase 4 retrieval, optional real-stack integration check, and `npm run build`. |

## C. Repository structure

```text
frontend/                 Active React/Vite UI
  src/App.tsx             Current single-screen UI and API calls
  src/styles.css          Custom styling
backend/
  app/main.py             FastAPI app/router/lifespan
  app/api/                HTTP endpoints (health, documents, retrieval)
  app/models/             Pydantic API models
  app/services/           Extraction, catalogue, chunking, vector indexing
  data/                   Local runtime data; excluded from Git
docs/                     Project guide plus this handoff package
app.py, utils/, config.py Legacy Streamlit prototype; not active final architecture
requirements.txt          Legacy Streamlit dependencies
backend/requirements.txt  Active backend dependencies
```

## D. Architecture and flow

The browser talks to Vite at `127.0.0.1:5173`. Vite proxies relative `/api` calls to FastAPI at `127.0.0.1:8000`. FastAPI does local file handling and persistence. There are no external runtime services today.

```text
React UI -> /api/v1/documents/upload -> FastAPI validation/extraction
  -> backend/data/uploads/<uuid>.<ext>
  -> backend/data/extracted/<uuid>.txt
  -> backend/data/documents.json

React UI -> /api/v1/documents/index -> chunker -> MiniLM embedding model
  -> ChromaDB backend/data/chroma/ (vectors + text + provenance metadata)

React UI -> /api/v1/retrieval/query -> local query rewriting + MiniLM/Chroma semantic search
  -> in-memory BM25 -> score fusion -> lexical reranking/context selection
  -> ranked evidence, context, and provenance diagnostics

React UI -> /api/v1/chat/answer -> Phase 4 retrieval evidence only -> local Ollama/Qwen
  -> code-derived citations from the same evidence metadata -> grounded answer
```

The document JSON record contains UUID storage name, original display name, type, size, page/slide count, extracted text pointer, extraction state, indexing state, chunk count, and messages. Chroma chunks contain `document_id`, document name/type, `chunk_index`, page/slide, section title, text, and an embedding.

## E. Actual feature implementation

| Feature | Exists / location | Limitations / tests |
| --- | --- | --- |
| Health/API connection | `api/health.py`, `main.py`, UI startup fetch | Health only; manually verified through TestClient and Vite proxy. |
| Upload/catalogue | `api/documents.py`, `document_repository.py` | One file per request, 50 MB, no deletion endpoint, JSON is not multi-user safe. Manually verified types/errors. |
| Extraction | `document_extractor.py` | PDF page markers; PPTX slide markers; DOCX/table text but no precise page map; image OCR depends on external Tesseract. |
| Semantic chunks | `chunking.py` | ~180-word target and 35-word overlap; heading heuristic is simple. Deterministic test verified page boundary handling. |
| Indexing | `knowledge_base.py`, `/documents/index` | MiniLM model must exist in local cache. Indexing replaces old chunks per document. Real run verified 15 chunks. |
| Library UI | `frontend/src/App.tsx` | No document deletion/details/re-index-per-item. Build verified. |
| Hybrid evidence retrieval | `api/retrieval.py`, `models/retrieval.py`, `services/retrieval.py` | Semantic + BM25 retrieval, filters, deterministic reranking, and budgeted evidence/context; no answer generation. |
| Grounded chat | `api/chat.py`, `models/generation.py`, `services/generation.py` | Uses retrieval before local Ollama, generates citations in code, and stores limited session context only in memory. Live model verification remains. |
| Study tools | Not implemented | Planned Phase 6. |

## F. Development conventions

- Routes are thin and call service objects; Pydantic models describe external data.
- Runtime configuration is centralized in `core/config.py`.
- User-visible API errors use FastAPI `HTTPException` with safe text.
- Python modules have concise module docstrings; comments explain non-obvious decisions.
- React currently keeps screen state with hooks in `App.tsx`; custom CSS avoids extra UI dependencies.
- Runtime data must remain ignored. No logging convention or database migration framework exists.

## G. Important architecture decisions

See [DECISIONS.md](DECISIONS.md). Do not casually change local-first operation, React/FastAPI split, ChromaDB persistence, cache-only embeddings, UUID storage, provenance metadata, or phase order.

## H. Constraints

- User hardware: 16 GB RAM, Ryzen 7 7000-series hexa-core, RTX 4050 6 GB VRAM.
- English ingestion/generation first; translation of completed answers later.
- Audio/video and authentication explicitly deferred.
- Local files must not leave the device.
- The first use of MiniLM requires a model download; later execution is offline from cache.

## I. Known issues / debt

See [TODO.md](TODO.md) and [CURRENT_STATE.md](CURRENT_STATE.md). The most immediate cleanup issue is tracked temporary pip logs. No confirmed Phase 3 functional defect was recorded.
