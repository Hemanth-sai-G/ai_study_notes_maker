# Unfinished Work

## Critical

- **Implement Phase 4 Advanced RAG retrieval.** Current state: Chroma stores vectors but no query endpoint exists. Relevant: `knowledge_base.py`, new retrieval services/routes, `frontend/src/App.tsx`. Dependencies: indexed text and MiniLM cache. Next step: implement vector query, BM25, score fusion, filters, query rewriting, reranking, and context selection. Acceptance: one API returns ranked, metadata-rich evidence for a query without LLM generation.

## High Priority

- **Implement Phase 5 grounded chat/citations.** Current state: Ollama is planned but not installed/integrated. Relevant: new generation service/API/UI. Dependencies: Phase 4 evidence contract and Ollama/Qwen local model. Acceptance: answers only from retrieved evidence and show deterministic source citations.
- **Add automated tests.** Current state: only ad hoc commands were run. Relevant: new `backend/tests/`, frontend test configuration if chosen. Dependencies: none. Acceptance: reproducible tests cover extraction, chunking, indexing, and retrieval edge cases.
- **Remove tracked temporary pip logs.** Current state: `tmp/phase3-pip-*.log` are in HEAD. Relevant: `tmp/`. Dependencies: user approval for cleanup commit. Acceptance: logs removed and `tmp/` ignored or otherwise handled.

## Medium Priority

- **Study tools / exports (Phase 6).** Notes, summaries, explanations, quizzes, flashcards, comparison, and export are planned but absent from the new stack. Dependency: Phase 5 grounded generation.
- **Evaluation dashboard.** Retrieval metrics, latency, citation coverage and faithfulness are planned for Phase 7. Dependency: retrieval/generation APIs and a labelled evaluation corpus.
- **Improve document extraction.** DOCX and PPTX do not currently preserve fine-grained page/slide/heading provenance as completely as PDF; scanned PDFs do not automatically fall back to OCR. Dependency: OCR strategy and source mapping design.

## Low Priority

- **Response translation.** User requested English ingestion/generation first, then translation of generated answers. Dependency: stable Phase 5 response contract.
- **Audio/video ingestion.** Explicitly deferred by the user. Dependency: transcription model/tool decision.

## Technical Debt

- Replace deprecated `fitz` import with the current PyMuPDF import only after verifying behavior.
- Add model/version metadata and an explicit re-index strategy for embedding model or chunk-config changes.
- Avoid broad `except Exception` in indexing once errors have stable classes/logging.
- Consider moving the large one-file React screen into components when Phase 4/5 expands it; do not refactor solely for style.

## Known Bugs

- No confirmed functional bug in the Phase 3 checkpoint.
- Tesseract missing means image OCR cannot produce text on the current environment; this is an environment dependency, not a parser defect.

## Future Improvements

- Local authentication/multi-user separation, secure document deletion controls, deployment packaging, and a production process manager.
- Better handwritten-note OCR and structured extraction for tables/images.

## Questions Requiring User Decision

- Select the final Ollama model (current plan: Qwen 2.5 3B default; 7B optional but slower).
- Approve any future authentication/multi-user scope.
- Decide exact translation provider/model and supported target languages.
- Decide whether the temporary pip logs should be removed in the next commit (recommended).
