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
Set-Location frontend
npm run build
```

`npm run build` performs TypeScript checking and a Vite production build. The last known build passed.

Focused TestClient scripts were executed manually for PDF/DOCX/PPTX/TXT/PNG ingestion, extension rejection, extracted-text persistence, deterministic Chroma persistence, actual MiniLM encoding, and real local index creation. They are not saved as test files; recreate these as formal tests before making major retrieval changes.

## Not established

- No `pytest`, frontend unit test runner, linter, formatter, E2E framework, database migration command, or CI configuration exists.
- No known failing automated tests, because none are present.
- Do not claim an E2E test command exists until one is added and verified.
