"""Grounded local chat endpoint backed by Phase 4 evidence retrieval."""

from fastapi import APIRouter, HTTPException

from app.models.generation import ChatRequest, ChatResponse
from app.services.generation import ChatService, OllamaGenerationError, OllamaUnavailableError

router = APIRouter(prefix="/chat", tags=["chat"])
chat_service = ChatService()


@router.post("/answer", response_model=ChatResponse)
def answer_question(request: ChatRequest) -> ChatResponse:
    """Generate a local answer only after successful evidence retrieval."""
    try:
        return chat_service.answer(
            query=request.query,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            context_word_budget=request.context_word_budget,
            filters=request.filters,
            session_id=request.session_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except OllamaUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except OllamaGenerationError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="Local answer generation is unavailable. Check the retrieval and Ollama setup.") from error
