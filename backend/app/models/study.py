"""Contracts for Phase 6 source-grounded study materials."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.models.generation import Citation

StudyMode = Literal["notes", "summary", "explanation", "flashcards", "quiz", "comparison"]


class StudyRequest(BaseModel):
    mode: StudyMode
    document_ids: list[str] = Field(min_length=1, max_length=5)
    topic: str | None = Field(default=None, max_length=1_000)

    @field_validator("document_ids")
    @classmethod
    def document_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        if not cleaned:
            raise ValueError("Choose at least one indexed document.")
        return cleaned

    @field_validator("topic")
    @classmethod
    def normalize_topic(cls, value: str | None) -> str | None:
        return value.strip() if value and value.strip() else None


class StudyResponse(BaseModel):
    mode: StudyMode
    title: str
    content: str
    citations: list[Citation]
    status: Literal["ok", "insufficient_evidence"]
    message: str | None = None
    model: str | None = None
    evidence_count: int
