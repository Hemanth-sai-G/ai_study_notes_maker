"""Validated contracts for local generated-study-material exports."""

from pydantic import BaseModel, Field, field_validator

from app.models.generation import Citation


class StudyExportRequest(BaseModel):
    """The already-generated study material rendered as a local PDF."""

    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=80_000)
    citations: list[Citation] = Field(default_factory=list, max_length=40)

    @field_validator("title", "content")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Export content must contain text.")
        return cleaned
