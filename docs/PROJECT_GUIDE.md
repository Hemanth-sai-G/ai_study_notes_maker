# StudyMate AI - Project Guide

> Living document. Update this guide whenever a feature, model, API, or folder is added or changed.

## 1. Project in one minute

StudyMate AI is a **local, source-grounded learning assistant**. A student uploads learning materials, such as PDFs, Word files, PowerPoint files, plain text, or images. The application turns their content into searchable pieces, finds the best pieces for a question, and gives only those pieces to a local AI model before it writes an answer.

This is called Retrieval-Augmented Generation (RAG). It differs from a normal chatbot because it first looks through the student's uploaded material. Answers show where the supporting information came from, including the document and page or slide when available.

The first release accepts and generates content in English. A later translation feature will translate completed responses without changing the original retrieval pipeline.

## 2. Goals and feature scope

### Core final-release features

- Upload PDF, DOCX, PPTX, TXT, and image/scanned-note files.
- Extract text and preserve metadata such as document name, page/slide, heading, and file type.
- OCR image-only or scanned documents locally where extraction finds no usable text.
- Query one or many uploaded documents at once.
- Generate structured notes, summaries, concept explanations, flashcards, quizzes, and document comparisons.
- Provide chat-style Q&A, answer citations, conversation memory, exports, and a retrieval/evaluation dashboard.
- Use semantic search, BM25 keyword search, query rewriting, reranking, metadata filtering, and context optimization.
- Run locally: no study document or generated answer leaves the computer.

### Deferred features

- Audio and video ingestion/transcription.
- Authentication and multi-user document isolation.
- Translating generated responses into other languages.

## 3. Technology choices

| Need | Choice | Plain-language reason |
| --- | --- | --- |
| Polished interface | React, Vite, TypeScript, custom CSS | Builds a responsive, professional web interface rather than a basic demo UI. Custom CSS keeps the first phase lightweight and transparent for the team to learn. |
| Application API | FastAPI | Python backend that is fast, typed, and easy for the React UI to call. |
| Local answer model | Ollama, Qwen 2.5 3B default | Runs on the developer's computer; Qwen 2.5 3B fits the available 6 GB RTX 4050 VRAM reliably. |
| Optional answer model | Qwen 2.5 7B (quantized) | Often gives stronger answers but may respond more slowly on this hardware. |
| Embeddings | Sentence Transformers | Converts text and questions into numerical meaning-vectors for semantic search. |
| Vector database | ChromaDB | Keeps vectors and document metadata locally, persistently, and supports filtering. |
| Keyword retrieval | BM25 | Finds exact terms, formulas, names, and acronyms that semantic search can miss. |
| Reranking | Compact Cross-Encoder | Re-scores candidate passages so only the best evidence reaches the LLM. |
| OCR | Local OCR engine (selected during ingestion milestone) | Reads text from images and scanned/handwritten notes where feasible. |

## 4. What is Advanced RAG?

**Chunking** breaks a long document into meaningful sections. Instead of asking an AI model to read an entire textbook at once, the app searches these smaller sections.

**Embeddings** are lists of numbers representing the meaning of text. Similar meanings have nearby vectors, so a question can find relevant content even without matching the exact words.

**BM25** is classic keyword search. It complements embeddings when a question contains an exact technical term.

**Hybrid retrieval** combines semantic/vector search and BM25 so the system has both conceptual and literal search strengths.

**Query rewriting** makes an ambiguous student question more explicit before search. It never replaces the user's original question in the displayed conversation.

**Reranking** takes the initial candidates and uses a more precise model to put the strongest evidence first.

**Context optimization** selects only the small set of high-quality passages that fit the LLM's context window. This reduces irrelevant content and hallucination risk.

**Grounded generation** tells the LLM to answer only from retrieved evidence. If the evidence is insufficient, it must say so.

## 5. End-to-end flow

```text
Upload document
  -> extract text and metadata
  -> OCR if needed
  -> clean and semantic-chunk text
  -> create embeddings
  -> save chunks + metadata in ChromaDB

Ask a question
  -> optionally rewrite query
  -> retrieve with vector search and BM25
  -> merge and rerank results
  -> filter and compress context
  -> prompt local Ollama LLM with evidence
  -> return answer + citations to React UI
```

## 6. Repository map

| Path | Responsibility |
| --- | --- |
| `frontend/` | React user interface. |
| `backend/app/main.py` | FastAPI application entry point and route registration. |
| `backend/app/api/` | HTTP endpoints called by the UI. |
| `backend/app/services/` | Business logic for ingestion, retrieval, generation, and study modes. |
| `backend/app/models/` | Pydantic request/response data models. |
| `backend/app/core/` | Configuration and shared application utilities. |
| `backend/data/` | Local runtime storage for uploads, vector data, and generated exports. Never commit personal study material. |
| `docs/` | Living technical explanation, setup instructions, architecture, and review preparation. |
| `app.py`, `utils/` | Legacy Streamlit prototype retained during migration; not the final application entry point. |

## 7. Development log

### Phase 0 - Architecture foundation (complete)

- Confirmed local-first design, English-first ingestion/generation, and deferred translation/authentication/audio/video.
- Selected React + FastAPI + Ollama + ChromaDB architecture.
- Selected Qwen 2.5 3B as the reliable default for RTX 4050 6 GB hardware; Qwen 2.5 7B remains optional.
- Created the project documentation and initial backend/API scaffold.
- Installed the small FastAPI dependency set in the existing project virtual environment.
- Verified the backend health endpoint with an automated local test.

### Phase 1 - Project shell and polished UI (complete)

- Created the standalone React + Vite + TypeScript frontend in `frontend/`.
- Built a responsive StudyMate AI workspace interface with navigation, a local-privacy status card, study-mode entry points, a knowledge-base empty state, and an upload-ready modal.
- Connected the UI to `GET /api/v1/health` through Vite's local development proxy. The footer and sidebar display whether FastAPI is connected.
- Kept upload actions visibly marked as a Phase 2 capability; no file is silently uploaded or claimed to be indexed yet.
- Added build scripts and ignored generated frontend/runtime data from Git.

### Phase 1: files to know

| File | What it does |
| --- | --- |
| `frontend/src/App.tsx` | React screen layout and small temporary UI state (active navigation, backend status, upload dialog). |
| `frontend/src/styles.css` | The visual design, responsive behavior, cards, sidebar, and dialog. |
| `frontend/vite.config.ts` | Starts the local frontend on port 5173 and forwards `/api` calls to FastAPI on port 8000. |
| `frontend/package.json` | Lists the JavaScript tools and commands used to run or build the UI. |
| `backend/app/api/health.py` | The first backend endpoint; proves the UI can reach FastAPI. |

### Phase 2 - Local document ingestion (complete)

- Added a real local upload flow to the React UI and a Library page showing uploaded documents, extraction state, size, page/slide count, and extracted character count.
- Added `POST /api/v1/documents/upload` and `GET /api/v1/documents` to FastAPI.
- Files are validated by supported extension and size (50 MB maximum), then stored using a generated UUID filename under `backend/data/uploads/`. The original filename is stored only as display metadata.
- Extracted text is stored in `backend/data/extracted/<uuid>.txt`; document metadata is saved in `backend/data/documents.json`. This makes Phase 3 reproducible: it chunks already-extracted local text rather than parsing every source file again. The JSON file is a deliberately simple catalogue before ChromaDB holds the searchable chunks.
- Implemented local extraction: PyMuPDF for PDF pages, python-docx for paragraphs/tables, python-pptx for slide text, and safe decoding fallbacks for TXT.
- Implemented image validation and OCR integration points. Image OCR is marked **OCR needed** until the local Tesseract executable is installed; a file is never sent to a cloud OCR service.
- Verified API ingestion for PDF, DOCX, PPTX, TXT, and PNG; verified library listing and unsupported-file rejection. Verified the React production build.

### Phase 2: files to know

| File | What it does |
| --- | --- |
| `backend/app/api/documents.py` | Defines upload and library-list API endpoints plus file-size/type validation. |
| `backend/app/services/document_extractor.py` | Chooses the correct local parser and returns extracted text plus page/slide metadata. |
| `backend/app/services/document_repository.py` | Reads/writes the local JSON document catalogue atomically. |
| `backend/app/models/documents.py` | Defines the predictable document data returned by the backend. |
| `frontend/src/App.tsx` | Upload form, upload feedback, and document library interface. |

### Phase 2 flow

```text
Student selects a supported file
  -> React sends multipart file data to the local FastAPI endpoint
  -> API validates extension, empty files, and 50 MB maximum size
  -> parser extracts text locally and records page/slide information
  -> original file is stored under backend/data/uploads/<uuid>.<extension>
  -> extracted text is stored under backend/data/extracted/<uuid>.txt when text is available
  -> metadata is added to backend/data/documents.json
  -> UI refreshes the Library with extraction status
```

### Enable local OCR (optional before Phase 3)

The Python `pytesseract` adapter is installed, but it calls a separate local program named **Tesseract OCR**. Install that program and ensure its `tesseract` command is available in PowerShell's PATH. After restarting the backend, new image uploads will be OCRed locally. Existing images can be re-uploaded once OCR is available. This works best for clear typed scans; handwritten notes may still need a later specialist OCR improvement.

### Phase 3 - Local semantic knowledge base (complete)

- Added semantic chunking that splits extracted text at paragraph boundaries, preserves section-like headings, keeps a small overlap between adjacent chunks, and never carries text across a page/slide boundary.
- New PDF extractions retain `[Page n]` source markers so future citation metadata can identify the originating page. PPTX extraction already retains slide markers.
- Added `all-MiniLM-L6-v2`, a compact Sentence Transformer embedding model. It turns each chunk into a 384-number meaning vector. The model downloads once, then is loaded cache-only for local/offline indexing.
- Added persistent ChromaDB in `backend/data/chroma/`. Every vector has its original text plus document ID/name, file type, chunk number, page/slide, and section-title metadata.
- Added `POST /api/v1/documents/index`, which reads Phase 2 extracted text, builds/rebuilds vectors, and records the result in the document catalogue.
- Added a **Build knowledge base** control in the Library UI. It displays a document's indexing state and saved chunk count.
- Verified deterministic Chroma persistence with a test embedding model. Verified the actual MiniLM model and indexed the current local library: 2 documents, 15 semantic chunks.

### Phase 3: files to know

| File | What it does |
| --- | --- |
| `backend/app/services/chunking.py` | Splits long extracted text into overlap-aware, metadata-preserving chunks. |
| `backend/app/services/knowledge_base.py` | Loads the local embedding model, writes vectors/metadata to persistent ChromaDB, and replaces a document's old chunks during re-indexing. |
| `backend/app/api/documents.py` | Adds the `POST /api/v1/documents/index` endpoint and saves indexing results in the catalogue. |
| `backend/app/models/documents.py` | Adds `indexing_status`, `chunk_count`, and the index API data models. |
| `frontend/src/App.tsx` | Adds the Library's knowledge-base button, progress message, and search-readiness states. |

### Phase 3 flow

```text
Extracted UTF-8 text from Phase 2
  -> semantic chunker preserves paragraphs, headings, and page/slide markers
  -> all-MiniLM-L6-v2 turns every chunk into a 384-dimension meaning vector
  -> ChromaDB stores: vector + text + document/page/section metadata
  -> document catalogue records indexed status and chunk count
  -> Phase 4 can query this local knowledge base
```

### Explain it in a review

**Why chunks?** LLMs and searches work better with focused passages than whole textbooks. Chunking makes retrieval fast and limits irrelevant context.

**Why overlap?** A concept can begin near one chunk's end and finish at the next chunk's beginning. Repeating a small number of words prevents that idea from being cut in half.

**Why embeddings?** An embedding represents meaning numerically. A question about “how a vector store works” can find a passage discussing “saving semantic representations” even when it uses different words.

**Why ChromaDB?** It persists vectors and source metadata locally and supports similarity search plus metadata filters in the next retrieval phase.

**Does Phase 3 answer questions already?** No. It prepares the searchable knowledge base. Phase 4 retrieves evidence using semantic search plus BM25, then reranks it before Phase 5 sends it to Ollama.

### Phase 4 - Advanced retrieval (complete)

- Added `POST /api/v1/retrieval/query`. It returns local evidence only; it never calls an LLM or creates an answer.
- Semantic search embeds the question with the existing cache-only `all-MiniLM-L6-v2` model and queries the persistent ChromaDB chunk collection.
- A local BM25-style keyword scorer evaluates the same filtered chunks. Deterministic weighted fusion combines semantic and keyword results by stable chunk ID.
- The service normalizes a question conservatively, supports document-ID and file-type filters (including their logical AND), reranks with transparent lexical signals, suppresses near-duplicate passages, and respects the requested context word budget.
- Each evidence item retains text, rank, score, document ID/name, file type, page or slide, section title, chunk ID, and chunk index. Diagnostics expose the original and rewritten query and counts at each retrieval stage.
- The Workspace has an evidence-query panel with document/type filters, ranked source cards, provenance, safe errors, and expandable diagnostics. It clearly labels this as retrieval rather than answer generation.

### Phase 4: files to know

| File | What it does |
| --- | --- |
| `backend/app/models/retrieval.py` | Validates retrieval requests and defines the provenance-rich response contract. |
| `backend/app/api/retrieval.py` | Exposes the safe local `POST /api/v1/retrieval/query` endpoint. |
| `backend/app/services/retrieval.py` | Runs query rewriting, semantic search, BM25 scoring, fusion, reranking, de-duplication, and context selection. |
| `backend/tests/test_retrieval.py` | Deterministic tests for ranking, filters, provenance, budgets, and invalid/empty inputs. |
| `backend/tests/test_retrieval_integration.py` | Optional check against the real local FastAPI/ChromaDB stack after indexing documents. |
| `frontend/src/App.tsx` | Provides the Workspace evidence-search interaction and displays retrieved sources. |

### Phase 4 flow

```text
Question and optional filters
  -> FastAPI validates the input
  -> local query normalization (no LLM)
  -> MiniLM embeds the search query from local cache
  -> ChromaDB returns semantic candidates with provenance
  -> local BM25 scores the same filtered chunks
  -> weighted fusion and lexical reranking
  -> duplicate-aware, word-budgeted context selection
  -> UI receives labelled evidence and diagnostics
```

### Explain it in a review

**Why hybrid retrieval?** Semantic vectors find related meanings, while BM25 gives exact technical terms a strong signal. Combining both handles more question styles than either method alone.

**Why keep provenance at every stage?** Phase 5 can make citations from code-owned document/page/slide metadata instead of trusting an LLM to invent a source.

**Why is the reranker not another AI model?** The initial local reranker is deterministic and inspectable: it rewards term coverage, exact phrases, and heading matches. This keeps Phase 4 lightweight and testable; a cross-encoder is a later optional quality experiment, not a hidden dependency.

**What does Phase 4 produce?** Ranked source passages and compact context, not a conversational answer. Phase 5 is the first phase that may send this evidence to the locally installed Ollama model.

### Phase 4 verification and limitations

`python -m compileall -q backend/app` and the focused `unittest` suite pass: ten deterministic retrieval tests pass and one real-stack test is intentionally skipped unless `STUDYMATE_RUN_REAL_INTEGRATION=1` is set after indexing local data. The frontend production build passes with the current Windows dependencies.

BM25 is calculated in memory from the Chroma-filtered chunks at query time, so it is simple and local but may need a persisted index for very large libraries. The real integration check requires an indexed local document and the MiniLM cache. Phase 4 does not include Ollama, answers, citations, chat history, authentication, translation, audio, or video.

### Phase 5 - Grounded local chat and citations (complete)

- Added `POST /api/v1/chat/answer`. It always uses the Phase 4 retrieval service first; Ollama never reads ChromaDB or uploaded files directly.
- The selected local model is `qwen2.5:3b`, requested through local Ollama at `http://127.0.0.1:11434`. No cloud API or Python Ollama dependency was added.
- The model receives labelled evidence passages only. The prompt rejects outside knowledge and instructions embedded in source text, and provides a fixed insufficient-evidence response.
- Citations are constructed in Python from retrieved metadata. The model cannot invent document names, pages/slides, sections, or chunk IDs.
- Conversation context is a four-turn, in-memory session history. It is never saved and disappears when the backend stops.
- The Workspace can display a grounded answer and its verified sources. The existing retrieval inspector remains available for review.
- Local Qwen model verification passed before Phase 6: `qwen2.5:3b` answered a local prompt and produced grounded flashcards from an indexed Library document.

### Phase 5: files to know

| File | What it does |
| --- | --- |
| `backend/app/models/generation.py` | Validates chat requests and defines grounded answer/citation responses. |
| `backend/app/services/generation.py` | Retrieves evidence, creates the evidence-only prompt, calls local Ollama, and creates citations. |
| `backend/app/api/chat.py` | Provides the local answer API and safe Ollama error messages. |
| `backend/tests/test_generation.py` | Tests grounding, citations, no-evidence behavior, session memory, and unavailable Ollama. |

### Phase 5 local model setup

Ollama is running locally and `qwen2.5:3b` is installed. On a new machine, install Ollama, run `ollama serve`, and run `ollama pull qwen2.5:3b` once before using grounded answers. All study material and inference remain on the device.

### Phase 6 - Study tools and exports (complete)

- Workspace cards now open **Study tools**, not the upload dialog. Students choose one or more documents already indexed in their local Library.
- Added local, grounded generators for notes, summaries, explanations, flashcards, quizzes, and document comparisons through `POST /api/v1/study/generate`.
- The selected document IDs become retrieval filters before Ollama is called. Every generated study material returns code-derived source references from the same evidence metadata.
- The Study Tools screen has a mode selector, multi-document picker, optional topic field, generated material display, verified source list, and a local Markdown download that includes the sources.
- Flashcards are now interactive: one question card is shown at a time, clicking it flips to its answer, and Previous/Next controls reset the card cleanly.
- Quizzes are structured multiple-choice activities: each question has four selectable options; after submission, the score and a generated explanation appear for every question. Incorrect or unanswered questions reveal the correct choice, while correctly selected choices are simply marked correct.
- Flashcard and quiz prompts request JSON-mode output from local Qwen. The backend validates question/answer pairs, exactly four quiz options, correct-option indexes, and explanations before returning the activity to the UI.
- Document comparison now explicitly requires two or more indexed Library documents in both the interface and API. A live comparison of two indexed proposal documents succeeded with eight verified citations.
- The Workspace now exposes summary and concept-explanation cards alongside notes, quizzes, flashcards, and comparisons; each opens Study Tools with its matching mode already selected.
- A live flashcard test succeeded with the installed local Qwen model and an indexed proposal PDF: it returned eight verified citations. No study data left the device.

### Phase 6: files to know

| File | What it does |
| --- | --- |
| `backend/app/models/study.py` | Validates study mode, selected Library documents, and optional topic. |
| `backend/app/services/study_tools.py` | Retrieves evidence from the selected documents and prompts local Qwen with mode-specific instructions. |
| `backend/app/api/study_tools.py` | Provides the local study-material generation endpoint. |
| `backend/tests/test_study_tools.py` | Tests selected-document filtering, citation preservation, and no-evidence behavior. |
| `frontend/src/App.tsx` | Opens Study Tools from the Workspace, lets students select existing documents, provides flip-card flashcards and scored quiz feedback, and exports results as Markdown. |

### Phase 6 flow

```text
Workspace card -> Study Tools screen -> choose indexed Library documents
  -> retrieval filter restricts evidence to those documents
  -> local Qwen creates the selected study format
  -> code adds verified sources -> optional Markdown download stays local
```

### Explain it in a review

**How do flashcards avoid using unrelated documents?** The frontend sends selected document IDs, and the backend applies them as ChromaDB retrieval filters before the model receives any text.

**Why generate all study modes through one service?** Notes, quizzes, flashcards, and comparisons share the same tested evidence/citation pipeline; only the output instruction changes. This avoids duplicating RAG logic.

### Phase 6 verification and limitations

The full Python suite passes: eighteen deterministic tests plus one live local flashcard generation check. The frontend production build passes. Markdown export is browser-local; PDF/DOCX exports, user-editable saved collections, and a retrieval/evaluation dashboard are deferred to Phase 7 or future enhancement.

## 8. Phased build roadmap

Each phase is intentionally self-contained. We will stop at the end of a phase, update this guide, and resume from the next unchecked phase when development time is available.

| Phase | Deliverable | Working checkpoint |
| --- | --- | --- |
| 0 | Architecture and documentation foundation | FastAPI health endpoint responds locally. |
| 1 | Project shell and polished UI | Complete - React interface starts locally and communicates with the API. |
| 2 | Document ingestion | Complete - uploads are stored locally and PDF, DOCX, PPTX, and TXT text/metadata are extracted. Images are validated and OCR-ready. |
| 3 | Local knowledge base | Complete - semantic chunks, local MiniLM embeddings, and persistent ChromaDB indexing work for uploaded documents. |
| 4 | Advanced retrieval | Complete - backend retrieval pipeline, provenance-rich API, opt-in real-stack test, and React evidence-query UI. |
| 5 | Grounded chat and citations | Complete - local Qwen evidence-only answers, code-derived citations, session-only memory, UI, and live local-model verification. |
| 6 | Study tools | Complete - selected-document notes, summaries, explanations, flashcards, quizzes, comparisons, verified sources, and local Markdown exports. |
| 7 | Evaluation and hardening | Evaluation dashboard, tests, error handling, performance tuning, local security controls, documentation, and demo preparation. |
| Later | Deferred features | Audio/video ingestion, authentication/multi-user access, and response translation. |

### How to resume safely

1. Open this guide and find the first incomplete phase in the table.
2. Read the most recent entry in the development log.
3. Run the phase's documented setup/test command.
4. Implement only that phase, verify its checkpoint, and update this guide before proceeding.

## 9. Review and interview talking points

1. **Why RAG instead of a normal chatbot?** The answer is grounded in student-provided documents, reducing unsupported claims and enabling citations.
2. **Why hybrid search?** Semantic vectors capture meaning; BM25 protects exact-match retrieval. Combining them increases recall.
3. **Why rerank after retrieval?** Fast retrieval gets plausible candidates; the Cross-Encoder makes a slower, more accurate relevance decision on only those candidates.
4. **Why run locally?** It preserves academic-document privacy, avoids recurring API costs, and makes the capstone demonstrable without internet.
5. **Does RAG eliminate hallucinations?** No. It reduces the risk. The app also uses an evidence-only prompt, source citations, and an insufficient-evidence response path.

## 10. Running Phase 1 locally

Open two terminals in the project root.

```powershell
# Terminal 1: start the API
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000

# Terminal 2: start the frontend
Set-Location frontend
npm run dev
```

Open `http://127.0.0.1:5173`. The status indicator should say **Local system ready**. If it says **Backend unavailable**, start Terminal 1 or check port 8000.

To prepare a production frontend bundle, run `npm run build` in `frontend/`. Vite writes the generated files to `frontend/dist/`; FastAPI deployment integration is intentionally deferred until the final hardening phase.

## 11. Next implementation step

Phase 7: add evaluation, hardening, broader automated coverage, secure deletion controls, demonstration preparation, and reproducible local setup guidance.
