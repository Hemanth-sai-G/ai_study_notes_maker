# StudyMate AI Architecture

## System boundary

The browser interface runs locally at a Vite development address. It sends only localhost requests to the FastAPI backend. The backend invokes Ollama and stores documents/indexes on the same machine.

```text
React UI <-> FastAPI API <-> Ingestion / Retrieval / Generation services
                                 |          |             |
                              local files  ChromaDB      Ollama
```

## Service boundaries

- **Ingestion service:** validates files, extracts/OCRs text, creates chunks, embeds them, and writes metadata-rich records.
- **Retrieval service:** runs vector and BM25 searches, fuses their candidates, filters metadata, and reranks them.
- **Generation service:** builds a strict grounded prompt, calls Ollama, parses citations, and reports insufficient context safely.
- **Study-material service:** uses the same retrieved evidence to make notes, flashcards, quizzes, explanations, summaries, and comparisons.
- **Evaluation service:** records retrieval rankings, relevance labels, latency, citation coverage, and answer-faithfulness checks.

## Phase 1 UI shell

The React frontend is deliberately a separate application in `frontend/`. Vite serves the interface at `http://127.0.0.1:5173` during development. Requests beginning with `/api` are proxied to FastAPI at `http://127.0.0.1:8000`, so browser code uses relative URLs and the backend can remain local.

`src/App.tsx` owns the Phase 1 workspace layout, backend status request, navigation state, and a temporary upload dialog. The dialog is intentionally transparent about its state: it becomes a real upload action in Phase 2, rather than falsely implying that files are already processed. `src/styles.css` provides the responsive visual system; it uses no server-hosted application assets.

## Citation metadata

Every stored chunk will carry `document_id`, `document_name`, `file_type`, `page_or_slide`, `section_title`, `chunk_index`, and `text`. Citations will be built from this metadata, not guessed by the LLM.

## Phase 2 ingestion implementation

`POST /api/v1/documents/upload` accepts one supported local file at a time. The file is size-limited to 50 MB, given a generated ID, and saved under `backend/data/uploads/`; the user-provided name is not used as a filesystem path. Extracted text is saved as UTF-8 under `backend/data/extracted/` using the same generated ID. `GET /api/v1/documents` returns display-safe metadata for the Library interface.

The document catalogue lives in `backend/data/documents.json` during Phase 2. This is not the future vector database: it is an audit-friendly record of what was ingested, how much text was extracted, and whether OCR is still required. ChromaDB is introduced only in Phase 3.

Text extraction stays local: PyMuPDF handles PDF page text, python-docx handles Word paragraphs/tables, python-pptx handles slide shape text, and TXT supports UTF-8/UTF-16/Latin-1 fallback decoding. Image files are verified with Pillow; their text is read by Tesseract only when its local executable is installed.

## Phase 3 local semantic index

`semantic_chunks()` uses paragraphs as the first boundary because paragraph breaks are more meaningful than arbitrary character counts. A target chunk contains about 180 words and repeats up to 35 ending words in the next chunk. It recognizes `[Page n]` and `[Slide n]` markers and resets overlap at those source boundaries so one citation does not accidentally point at two pages.

`KnowledgeBase` opens a persistent ChromaDB collection at `backend/data/chroma/`. It embeds each chunk with `sentence-transformers/all-MiniLM-L6-v2`, which produces a 384-dimension normalized vector. Chroma stores that vector beside the original chunk and provenance metadata. On re-indexing, prior chunks from the same document are removed and replaced, avoiding duplicate evidence.

The model is initially downloaded once from Hugging Face during setup. The runtime loader uses `local_files_only=True` afterward, so semantic indexing works offline and source documents never leave the computer.

## Security posture for local release

- Validate upload extensions, MIME types, and file size.
- Generate server-side IDs rather than trusting filenames or paths.
- Keep uploaded files outside frontend-served directories.
- Bind development services to localhost by default.
- Defer login/multi-user permissions until deployment requires them.
