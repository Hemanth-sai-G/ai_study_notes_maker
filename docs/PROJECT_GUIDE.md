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
| Polished interface | React, Vite, Tailwind CSS | Builds a responsive, professional web interface rather than a basic demo UI. |
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

## 8. Phased build roadmap

Each phase is intentionally self-contained. We will stop at the end of a phase, update this guide, and resume from the next unchecked phase when development time is available.

| Phase | Deliverable | Working checkpoint |
| --- | --- | --- |
| 0 | Architecture and documentation foundation | FastAPI health endpoint responds locally. |
| 1 | Project shell and polished UI | React interface starts locally and communicates with the API. |
| 2 | Document ingestion | PDF, DOCX, PPTX, TXT, and image uploads are validated, stored locally, and text/metadata are extracted. |
| 3 | Local knowledge base | Semantic chunking, embeddings, and persistent ChromaDB indexing work for uploaded documents. |
| 4 | Advanced retrieval | BM25 + semantic hybrid search, metadata filters, query rewriting, reranking, and context selection return ranked evidence. |
| 5 | Grounded chat and citations | Ollama generates source-grounded answers with page/slide citations and conversation memory. |
| 6 | Study tools | Notes, summaries, explanations, quizzes, flashcards, comparisons, and exports use the same evidence pipeline. |
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

## 10. Next implementation step

Phase 1: create the React/Vite/Tailwind project shell and connect its UI to the existing FastAPI health endpoint. Update this guide after the implementation and tests are complete.
