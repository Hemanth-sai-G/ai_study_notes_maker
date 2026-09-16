# Context Recovery

This is recovered implementation context, not a conversation transcript.

## User requirements and decisions

- The project is a capstone Advanced RAG learning assistant built from the existing `ai_study_notes_maker` repository.
- User wants a final, locally runnable app and a professional UI, not basic Streamlit.
- Hardware disclosed: 16 GB RAM, Ryzen 7 7000-series hexa-core CPU, RTX 4050 with 6 GB VRAM.
- English is the initial ingestion/generation language. Translation should be added later to generated responses, not to the initial RAG pipeline.
- Core inputs now: PDF, DOCX, PPTX, TXT, images/scanned notes. Audio/video are deferred.
- Authentication/security and multi-user account handling are deferred; the app is single-user local for now.
- User explicitly wants work divided into resumable phases and a living beginner-friendly project guide so they can explain it in reviews/interviews and hand it to teammates.

## Architecture reasoning already settled

- React/FastAPI was chosen instead of extending Streamlit to satisfy the requested UI quality and modularity.
- ChromaDB was chosen over FAISS for persistent local collections and future metadata filtering.
- MiniLM was chosen as a lightweight semantic embedding model. It is not the future answer LLM.
- Proposed generation model is Qwen 2.5 3B through Ollama (7B optional/likely slower), but Ollama integration has not begun.
- RAG stages deliberately remain separated: ingestion -> index -> retrieval -> generation -> study tools -> evaluation. This makes review explanations and failure diagnosis simpler.

## Previous implementation bugs and resolutions

- **Vite port 5173 conflict:** an old Node/Vite process held the port. Stopping that process resolved startup; no code change was needed.
- **Phase 2 temporary test directory error:** upload code assumed lifespan-created directories. The route/repository now create needed directories when writing, so direct tests work too.
- **Repository annotation bug:** `DocumentRepository.list` shadowed the built-in `list` in a later type annotation. Added `from __future__ import annotations`.
- **Chunk page boundary bug:** the overlap buffer originally carried prior-page text into the next page. It now resets on `[Page n]`/`[Slide n]` markers.
- **Embedding offline retries:** SentenceTransformer tried to contact Hugging Face on every construction. Runtime loader now uses `local_files_only=True`; setup must download/cache the model once.

## Intentional postponements

- BM25/hybrid retrieval, query rewriting, reranking, metadata filters, and context compression: Phase 4.
- Ollama Q&A, source-cited answers, conversation memory: Phase 5.
- Notes, chapter summaries, quizzes, flashcards, explanations, comparison, exports: Phase 6.
- Evaluation dashboard, performance/citation metrics, broad tests, demo hardening: Phase 7.
- Audio/video, authentication, translation: later only.

## Do not revisit unless requirements change

- Local-first privacy constraint.
- UUID-based source storage and ignored runtime-data directories.
- Requirement for source-backed citations derived from provenance metadata, not model guesses.
- Phase-by-phase workflow and documentation updates.
