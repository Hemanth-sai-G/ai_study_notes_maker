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

## Citation metadata

Every stored chunk will carry `document_id`, `document_name`, `file_type`, `page_or_slide`, `section_title`, `chunk_index`, and `text`. Citations will be built from this metadata, not guessed by the LLM.

## Security posture for local release

- Validate upload extensions, MIME types, and file size.
- Generate server-side IDs rather than trusting filenames or paths.
- Keep uploaded files outside frontend-served directories.
- Bind development services to localhost by default.
- Defer login/multi-user permissions until deployment requires them.
