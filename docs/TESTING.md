# Testing and Verification

## Environment

- Windows PowerShell was used.
- Project virtual environment: `.venv` (Python 3.11 was observed from warnings).
- Node/npm are required for `frontend/`.
- Tesseract executable is optional; without it images remain OCR-needed.
- MiniLM must be downloaded once before cache-only Phase 3 indexing can run.

## Install dependencies

Verified backend command:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Verified frontend command:

```powershell
Set-Location frontend
npm install
```

## Run locally

Terminal 1 from repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
Set-Location frontend
npm run dev
```

Open `http://127.0.0.1:5173`. The UI should show local backend connection status.

## Verified checks

```powershell
.\.venv\Scripts\python.exe -m compileall -q backend\app
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests -v
Set-Location frontend
npm run build
```

`npm run build` performs TypeScript checking and a Vite production build. The last known build passed. The Phase 4/5 fixtures run without user documents or model downloads: fifteen deterministic tests pass and one real-stack retrieval test is skipped by default.

After indexing at least one local document, run the optional real-stack retrieval check separately:

```powershell
$env:STUDYMATE_RUN_REAL_INTEGRATION = "1"
.\.venv\Scripts\python.exe -m unittest backend.tests.test_retrieval_integration -v
```

This check uses the local ChromaDB collection and cache-only MiniLM model. It must not be treated as portable because `backend/data/` and model caches are intentionally excluded from Git.

## Phase 5 local Ollama setup

The grounded-chat service expects local Ollama at `http://127.0.0.1:11434` with `qwen2.5:3b` installed. Before live answer testing, run:

```powershell
ollama serve
ollama pull qwen2.5:3b
```

The automated generation fixtures use a fake generator, so they run while Ollama is stopped. A real chat check also requires indexed local documents and the local model.

Focused TestClient scripts were executed manually for PDF/DOCX/PPTX/TXT/PNG ingestion, extension rejection, extracted-text persistence, deterministic Chroma persistence, actual MiniLM encoding, and real local index creation. They are not saved as test files; recreate these as formal tests before making major retrieval changes.

## Not established

- No `pytest`, frontend unit test runner, linter, formatter, E2E framework, database migration command, or CI configuration exists.
- No known failing automated tests, because none are present.
- Do not claim an E2E test command exists until one is added and verified.
