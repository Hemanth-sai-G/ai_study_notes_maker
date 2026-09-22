"""Pydantic contracts for local evidence retrieval."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RetrievalFilters(BaseModel):
    document_ids: list[str] | None = None
    file_types: list[str] | None = None


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=100)
    context_word_budget: int = Field(default=1_200, ge=50, le=8_000)
    filters: RetrievalFilters | None = None

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        """Reject whitespace-only questions before they reach the embedder."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Query must contain text.")
        return normalized


class EvidenceItem(BaseModel):
    evidence_id: str
    text: str
    score: float
    rank: int
    document_id: str
    document_name: str
    file_type: str
    page_or_slide: int | None = None
    section_title: str | None = None
    chunk_id: str
    chunk_index: int


class RetrievalDiagnostics(BaseModel):
    original_query: str
    rewritten_query: str
    semantic_candidates: int
    keyword_candidates: int
    fused_candidates: int
    reranked_candidates: int
    returned_candidates: int
    context_word_count: int
    filters_applied: bool


class RetrievalResponse(BaseModel):
    query: str
    evidence: list[EvidenceItem]
    context: str
    diagnostics: RetrievalDiagnostics
    status: Literal["ok", "no_evidence"]
    message: str | None = None


class RetrievalErrorResponse(BaseModel):
    detail: str
