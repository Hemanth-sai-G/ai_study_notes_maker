"""HTTP endpoint for local evidence retrieval."""

from fastapi import APIRouter, HTTPException

from app.models.retrieval import RetrievalRequest, RetrievalResponse
from app.services.retrieval import RetrievalService

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/query", response_model=RetrievalResponse)
def query_retrieval(request: RetrievalRequest) -> RetrievalResponse:
    """Return ranked local evidence; this endpoint never generates an answer."""
    try:
        return RetrievalService().retrieve(
            query=request.query,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            context_word_budget=request.context_word_budget,
            filters=request.filters,
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="Local retrieval is unavailable. Check the knowledge-base setup.") from error
