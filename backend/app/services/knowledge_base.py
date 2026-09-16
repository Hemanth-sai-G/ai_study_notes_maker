"""Persistent ChromaDB indexing built from the Phase 2 extracted-text files."""

from pathlib import Path
from typing import Protocol

import chromadb

from app.core.config import settings
from app.models.documents import DocumentRecord
from app.services.chunking import semantic_chunks


class Embedder(Protocol):
    def encode(self, sentences: list[str], **kwargs: object) -> object: ...


class KnowledgeBase:
    collection_name = "study_chunks"

    def __init__(self, data_dir: Path | None = None, embedder: Embedder | None = None) -> None:
        self.data_dir = data_dir or settings.data_dir
        self._embedder = embedder
        self._client: chromadb.PersistentClient | None = None

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            chroma_dir = self.data_dir / "chroma"
            chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(chroma_dir))
        return self._client

    @property
    def collection(self):
        return self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            # The model is downloaded once during setup, then deliberately loaded from the local cache.
            # This prevents document indexing from depending on an internet connection.
            self._embedder = SentenceTransformer(settings.embedding_model_name, local_files_only=True)
        return self._embedder

    def index_document(self, document: DocumentRecord) -> int:
        if not document.extracted_text_name:
            return 0
        path = self.data_dir / "extracted" / document.extracted_text_name
        if not path.exists():
            raise FileNotFoundError("The extracted-text file is missing. Re-upload this document.")
        chunks = semantic_chunks(path.read_text(encoding="utf-8"), settings.chunk_size_words, settings.chunk_overlap_words)
        if not chunks:
            return 0
        ids = [f"{document.id}:{chunk.index}" for chunk in chunks]
        self.collection.delete(where={"document_id": document.id})
        embeddings = self.embedder.encode([chunk.text for chunk in chunks], normalize_embeddings=True, show_progress_bar=False)
        self.collection.upsert(
            ids=ids,
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[{
                "document_id": document.id,
                "document_name": document.original_name,
                "file_type": document.file_type,
                "chunk_index": chunk.index,
                "page_or_slide": chunk.page_or_slide or 0,
                "section_title": chunk.section_title or "",
            } for chunk in chunks],
        )
        return len(chunks)

    def count(self) -> int:
        return self.collection.count()
