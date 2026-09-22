"""Pydantic contracts for locally grounded chat and code-derived citations."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.models.retrieval import RetrievalFilters


class Citation(BaseModel):
    """A source reference constructed from retrieved evidence, never guessed by the LLM."""

    evidence_id: str
    document_id: str
    document_name: str
    file_type: str
    page_or_slide: int | None = None
    section_title: str | None = None
    chunk_id: str
    chunk_index: int


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    session_id: str | None = Field(default=None, max_length=100)
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=100)
    context_word_budget: int = Field(default=1_200, ge=50, le=8_000)
    filters: RetrievalFilters | None = None

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Question must contain text.")
        return normalized

    @field_validator("session_id")
    @classmethod
    def session_id_must_contain_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class ChatResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    status: Literal["ok", "insufficient_evidence"]
    message: str | None = None
    model: str | None = None
    evidence_count: int
