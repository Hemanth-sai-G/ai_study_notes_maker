# Development Workflow

1. Read `AGENTS.md`, `AI_HANDOFF.md`, `CURRENT_STATE.md`, and the current phase in `IMPLEMENTATION_PLAN.md`.
2. Run `git status`, inspect relevant implementation files, and compare docs to actual code before planning changes.
3. Trace the request/data flow through route -> Pydantic model -> service -> local storage/UI. Preserve existing provenance fields.
4. Keep the next change within one phase. Do not implement generation while retrieval is untested, or add cloud functionality to solve a local setup problem.
5. Make small, focused edits that follow the existing location pattern: APIs in `api/`, data contracts in `models/`, business logic in `services/`, settings in `core/`, UI behavior in `App.tsx` and CSS in `styles.css`.
6. Add a reusable test when changing behavior. For current code, first add an appropriate test harness rather than relying forever on ad hoc scripts.
7. Verify compilation, focused behavior, and frontend build. Inspect a live browser UI when layout/user interaction changes.
8. Update `PROJECT_GUIDE.md` and handoff docs when a phase, architecture, decision, run command, or known limitation changes.
9. Refactor only when the current feature requires it or a tested seam is clearly needed. Do not redesign the active stack.
10. Stop and ask the user before changing scope: cloud services, model family, authentication, audio/video, translation, destructive local-data operations, or major dependency/platform shifts.

Never automatically commit, delete user documents, expose `backend/data/`, add secrets, or treat legacy Streamlit files as the active product without user direction.
