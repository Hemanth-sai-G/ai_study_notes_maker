from fastapi import APIRouter

router = APIRouter(tags=["system"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Small endpoint used by the UI and tests to verify the API is alive."""
    return {"status": "ok", "service": "studymate-api"}
