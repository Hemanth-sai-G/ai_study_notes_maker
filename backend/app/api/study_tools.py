"""HTTP endpoint for local Phase 6 study tools."""

from fastapi import APIRouter, HTTPException

from app.models.study import StudyRequest, StudyResponse
from app.services.generation import OllamaGenerationError, OllamaUnavailableError
from app.services.study_tools import StudyToolsService

router = APIRouter(prefix="/study", tags=["study tools"])
study_tools_service = StudyToolsService()


@router.post("/generate", response_model=StudyResponse)
def generate_study_material(request: StudyRequest) -> StudyResponse:
    try:
        return study_tools_service.generate(request.mode, request.document_ids, request.topic)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except OllamaUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except OllamaGenerationError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="Local study-material generation is unavailable. Check the retrieval and Ollama setup.") from error
