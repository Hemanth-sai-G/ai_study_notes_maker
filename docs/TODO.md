# Unfinished Work

## Critical

- **Implement Phase 7 evaluation and hardening.** Current state: Phases 0-6 work locally. Next: build a labelled evaluation set, report retrieval/citation/latency metrics, add broader automated tests, improve errors, and prepare a repeatable demo.

## High Priority

- **Add automated tests.** Current state: only ad hoc commands were run. Relevant: new `backend/tests/`, frontend test configuration if chosen. Dependencies: none. Acceptance: reproducible tests cover extraction, chunking, indexing, and retrieval edge cases.
- **Remove tracked temporary pip logs.** Current state: `tmp/phase3-pip-*.log` are in HEAD. Relevant: `tmp/`. Dependencies: user approval for cleanup commit. Acceptance: logs removed and `tmp/` ignored or otherwise handled.

## Medium Priority

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
