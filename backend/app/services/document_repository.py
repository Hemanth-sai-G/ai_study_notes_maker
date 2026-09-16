"""Small local JSON catalogue for Phase 2 document metadata."""

from __future__ import annotations

import json
from pathlib import Path

from app.models.documents import DocumentRecord


class DocumentRepository:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path

    def list(self) -> list[DocumentRecord]:
        if not self.index_path.exists():
            return []
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        return [DocumentRecord.model_validate(item) for item in data]

    def add(self, record: DocumentRecord) -> None:
        documents = self.list()
        documents.append(record)
        self._write(documents)

    def replace(self, updated: DocumentRecord) -> None:
        documents = self.list()
        found = False
        for index, record in enumerate(documents):
            if record.id == updated.id:
                documents[index] = updated
                found = True
                break
        if not found:
            raise KeyError(f"Document {updated.id} was not found.")
        self._write(documents)

    def _write(self, documents: list[DocumentRecord]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.index_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps([item.model_dump(mode="json") for item in documents], indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.index_path)
