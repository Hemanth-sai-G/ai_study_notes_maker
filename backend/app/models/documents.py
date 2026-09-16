from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ExtractionStatus = Literal["ready", "ocr_unavailable"]
IndexStatus = Literal["not_indexed", "indexed", "no_text", "indexing_error"]


class DocumentRecord(BaseModel):
    """Persistent local record for one uploaded study resource."""

    id: str
    original_name: str
    stored_name: str
    extracted_text_name: str | None = None
    file_type: str
    size_bytes: int
    page_or_slide_count: int | None = None
    character_count: int
    extraction_status: ExtractionStatus
    extraction_message: str | None = None
    indexing_status: IndexStatus = "not_indexed"
    chunk_count: int = 0
    indexing_message: str | None = None
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]
    total: int


class UploadResponse(BaseModel):
    document: DocumentRecord
    message: str = Field(description="Human-readable upload result.")


class IndexRequest(BaseModel):
    document_ids: list[str] | None = None


class IndexResponse(BaseModel):
    indexed_documents: int
    indexed_chunks: int
    message: str
