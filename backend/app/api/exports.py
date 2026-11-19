"""Local PDF export endpoint for generated study material."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.models.exports import StudyExportRequest
from app.services.exporter import build_study_pdf

router = APIRouter(prefix="/exports", tags=["exports"])


@router.post("/pdf", response_class=Response)
def export_pdf(request: StudyExportRequest) -> Response:
    """Stream a generated PDF without writing it to disk."""
    try:
        pdf = build_study_pdf(request)
    except Exception as error:
        raise HTTPException(status_code=500, detail="The local PDF export could not be created.") from error
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="studymate-study-material.pdf"'},
    )
