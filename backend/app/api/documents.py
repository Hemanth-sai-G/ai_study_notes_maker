from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.models.documents import DocumentListResponse, DocumentRecord, IndexRequest, IndexResponse, UploadResponse
from app.services.document_extractor import ExtractionError, extract_text
from app.services.knowledge_base import KnowledgeBase
from app.services.document_repository import DocumentRepository

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_SUFFIXES = {".pdf", ".docx", ".pptx", ".txt", ".png", ".jpg", ".jpeg", ".webp"}


def repository() -> DocumentRepository:
    return DocumentRepository(settings.documents_index_path)


@router.get("", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    documents = sorted(repository().list(), key=lambda item: item.created_at, reverse=True)
    return DocumentListResponse(documents=documents, total=len(documents))


@router.post("/index", response_model=IndexResponse)
def index_documents(request: IndexRequest = IndexRequest()) -> IndexResponse:
    """Create local embeddings for extracted documents and persist them in ChromaDB."""
    records = repository().list()
    if request.document_ids is not None:
        wanted = set(request.document_ids)
        records = [record for record in records if record.id in wanted]
    if not records:
        raise HTTPException(status_code=404, detail="No matching documents are available to index.")

    knowledge_base = KnowledgeBase()
    indexed_documents = 0
    indexed_chunks = 0
    for record in records:
        if not record.extracted_text_name:
            record.indexing_status = "no_text"
            record.indexing_message = "No extracted text is available. Enable OCR or upload a text-based document."
            repository().replace(record)
            continue
        try:
            chunk_count = knowledge_base.index_document(record)
            if chunk_count:
                record.indexing_status = "indexed"
                record.chunk_count = chunk_count
                record.indexing_message = None
                indexed_documents += 1
                indexed_chunks += chunk_count
            else:
                record.indexing_status = "no_text"
                record.indexing_message = "No readable text was found in this document."
            repository().replace(record)
        except Exception as error:
            record.indexing_status = "indexing_error"
            record.indexing_message = f"Indexing did not finish: {error}"
            repository().replace(record)
            raise HTTPException(status_code=503, detail=record.indexing_message) from error
    return IndexResponse(
        indexed_documents=indexed_documents,
        indexed_chunks=indexed_chunks,
        message=f"Indexed {indexed_chunks} chunks from {indexed_documents} document(s) locally.",
    )


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Validate, persist, and extract one study document on the local machine."""
    original_name = Path(file.filename or "untitled").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Supported files: PDF, DOCX, PPTX, TXT, PNG, JPG, JPEG, and WEBP.")

    file_bytes = await file.read()
    maximum_size = settings.max_upload_size_mb * 1024 * 1024
    if not file_bytes:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    if len(file_bytes) > maximum_size:
        raise HTTPException(status_code=413, detail=f"Files must be smaller than {settings.max_upload_size_mb} MB.")

    try:
        extracted = extract_text(file_bytes, suffix)
    except ExtractionError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    document_id = str(uuid4())
    stored_name = f"{document_id}{suffix}"
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.uploads_dir / stored_name
    destination.write_bytes(file_bytes)
    extracted_text_name = None
    if extracted.text:
        settings.extracted_text_dir.mkdir(parents=True, exist_ok=True)
        extracted_text_name = f"{document_id}.txt"
        (settings.extracted_text_dir / extracted_text_name).write_text(extracted.text, encoding="utf-8")

    record = DocumentRecord(
        id=document_id,
        original_name=original_name,
        stored_name=stored_name,
        extracted_text_name=extracted_text_name,
        file_type=suffix[1:].upper(),
        size_bytes=len(file_bytes),
        page_or_slide_count=extracted.page_or_slide_count,
        character_count=len(extracted.text),
        extraction_status=extracted.status,
        extraction_message=extracted.message,
        created_at=datetime.now(timezone.utc),
    )
    repository().add(record)
    message = "Document saved and text extracted locally." if extracted.status == "ready" else record.extraction_message or "Document saved locally."
    return UploadResponse(document=record, message=message)
