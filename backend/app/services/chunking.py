"""Meaning-preserving text chunking for the local knowledge base."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    text: str
    index: int
    page_or_slide: int | None
    section_title: str | None


_LOCATION = re.compile(r"^\[(Page|Slide)\s+(\d+)]$", re.IGNORECASE)


def semantic_chunks(text: str, chunk_size_words: int, overlap_words: int) -> list[TextChunk]:
    """Split on paragraphs first, retaining nearby context and source markers."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[TextChunk] = []
    buffer: list[str] = []
    word_count = 0
    page_or_slide: int | None = None
    section_title: str | None = None

    def flush() -> None:
        nonlocal buffer, word_count
        if not buffer:
            return
        body = "\n\n".join(buffer).strip()
        if body:
            chunks.append(TextChunk(body, len(chunks), page_or_slide, section_title))
        overlap = " ".join(body.split()[-overlap_words:]) if overlap_words else ""
        buffer = [overlap] if overlap else []
        word_count = len(overlap.split())

    for paragraph in paragraphs:
        marker = _LOCATION.match(paragraph)
        if marker:
            flush()
            # Do not carry text from the previous page/slide into the next source location.
            buffer = []
            word_count = 0
            page_or_slide = int(marker.group(2))
            continue
        if _looks_like_heading(paragraph):
            section_title = paragraph[:160]
        paragraph_words = paragraph.split()
        if word_count and word_count + len(paragraph_words) > chunk_size_words:
            flush()
        if len(paragraph_words) > chunk_size_words:
            for start in range(0, len(paragraph_words), max(1, chunk_size_words - overlap_words)):
                if buffer:
                    flush()
                buffer = [" ".join(paragraph_words[start : start + chunk_size_words])]
                word_count = len(buffer[0].split())
                flush()
            continue
        buffer.append(paragraph)
        word_count += len(paragraph_words)
    flush()
    return chunks


def _looks_like_heading(paragraph: str) -> bool:
    words = paragraph.split()
    return 1 <= len(words) <= 12 and (paragraph.isupper() or (not paragraph.endswith((".", "?", "!")) and len(words) <= 8))
