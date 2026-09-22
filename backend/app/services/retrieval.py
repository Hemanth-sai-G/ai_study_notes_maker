"""Hybrid retrieval, deterministic reranking, and context selection over local Chroma data."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.models.retrieval import EvidenceItem, RetrievalDiagnostics, RetrievalFilters, RetrievalResponse

if TYPE_CHECKING:
    from app.services.knowledge_base import KnowledgeBase

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_QUESTION_WORDS = {"a", "an", "are", "can", "does", "how", "is", "of", "the", "to", "what", "when", "where", "which", "who", "why"}


@dataclass(frozen=True)
class _Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


class RetrievalService:
    """Query local semantic and keyword indexes while preserving provenance."""

    def __init__(self, knowledge_base: "KnowledgeBase | None" = None) -> None:
        if knowledge_base is None:
            from app.services.knowledge_base import KnowledgeBase

            knowledge_base = KnowledgeBase()
        self.knowledge_base = knowledge_base

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: RetrievalFilters | None,
        candidate_k: int = 20,
        context_word_budget: int = 1_200,
    ) -> RetrievalResponse:
        collection = self.knowledge_base.collection
        if collection.count() == 0:
            raise ValueError("The knowledge base is empty. Index at least one document before searching.")

        where = self._where_clause(filters)
        rewritten_query = rewrite_query(query)
        chunks = self._chunks(collection.get(where=where, include=["documents", "metadatas"]))
        if not chunks:
            return self._empty_response(query, rewritten_query, where)

        candidate_limit = min(candidate_k, len(chunks))
        semantic = self._semantic_scores(collection, rewritten_query, candidate_limit, where)
        keyword = self.bm25_scores(rewritten_query, chunks)
        fused = self._fuse(semantic, keyword, candidate_limit)
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        fused_candidates = [(by_id[chunk_id], score) for chunk_id, score in fused if chunk_id in by_id]
        reranked = self._rerank(rewritten_query, fused_candidates)
        selected = self._select_context(reranked, top_k, context_word_budget)
        evidence = [
            self._evidence(chunk, score, rank)
            for rank, (chunk, score) in enumerate(selected, start=1)
        ]
        return RetrievalResponse(
            query=query,
            evidence=evidence,
            context="\n\n".join(f"[{item.evidence_id}] {item.text}" for item in evidence),
            status="ok" if evidence else "no_evidence",
            message=None if evidence else "No evidence matched the query and filters.",
            diagnostics=RetrievalDiagnostics(
                original_query=query,
                rewritten_query=rewritten_query,
                semantic_candidates=len(semantic),
                keyword_candidates=len([score for score in keyword.values() if score > 0]),
                fused_candidates=len(fused),
                reranked_candidates=len(reranked),
                returned_candidates=len(evidence),
                context_word_count=sum(len(item.text.split()) for item in evidence),
                filters_applied=bool(where),
            ),
        )

    def _semantic_scores(self, collection: Any, query: str, limit: int, where: dict[str, Any] | None) -> dict[str, float]:
        embedding = self.knowledge_base.embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        result = collection.query(query_embeddings=embedding, n_results=limit, where=where, include=["distances"])
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return {chunk_id: max(0.0, 1.0 - float(distance)) for chunk_id, distance in zip(ids, distances)}

    @staticmethod
    def bm25_scores(query: str, chunks: list[_Chunk]) -> dict[str, float]:
        """Return normalized BM25 scores for exact-term retrieval over local chunk text."""
        if not chunks:
            return {}
        query_terms = _tokens(query)
        documents = {chunk.chunk_id: _tokens(chunk.text) for chunk in chunks}
        average_length = sum(len(words) for words in documents.values()) / len(documents)
        document_frequency = {term: sum(term in words for words in documents.values()) for term in set(query_terms)}
        scores: dict[str, float] = {}
        for chunk_id, words in documents.items():
            score = 0.0
            for term in set(query_terms):
                frequency = words.count(term)
                if frequency == 0:
                    continue
                idf = math.log(1 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * len(words) / max(1, average_length))
                score += idf * frequency * 2.2 / denominator
            scores[chunk_id] = score
        maximum = max(scores.values(), default=0.0)
        return {chunk_id: score / maximum if maximum else 0.0 for chunk_id, score in scores.items()}

    @staticmethod
    def _fuse(semantic: dict[str, float], keyword: dict[str, float], candidate_k: int) -> list[tuple[str, float]]:
        semantic_ranked = sorted(semantic.items(), key=lambda item: (-item[1], item[0]))[:candidate_k]
        keyword_ranked = sorted(keyword.items(), key=lambda item: (-item[1], item[0]))[:candidate_k]
        semantic_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(semantic_ranked, start=1)}
        keyword_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(keyword_ranked, start=1)}
        all_ids = set(semantic_rank) | {chunk_id for chunk_id, score in keyword_ranked if score > 0}
        fused = {
            chunk_id: 0.6 * semantic.get(chunk_id, 0.0) + 0.4 * keyword.get(chunk_id, 0.0)
            + 0.15 / semantic_rank.get(chunk_id, candidate_k + 1)
            + 0.10 / keyword_rank.get(chunk_id, candidate_k + 1)
            for chunk_id in all_ids
        }
        return sorted(fused.items(), key=lambda item: (-item[1], item[0]))

    @staticmethod
    def _rerank(query: str, candidates: list[tuple[_Chunk, float]]) -> list[tuple[_Chunk, float]]:
        """Apply transparent lexical reranking without a downloaded model."""
        query_terms = set(_tokens(query))
        query_phrase = " ".join(_tokens(query))
        rescored: list[tuple[_Chunk, float]] = []
        for chunk, fused_score in candidates:
            text_lower = " ".join(_tokens(chunk.text))
            text_terms = set(_tokens(chunk.text))
            coverage = len(query_terms & text_terms) / max(1, len(query_terms))
            phrase_bonus = 0.08 if query_phrase and query_phrase in text_lower else 0.0
            section_bonus = 0.04 if query_terms & set(_tokens(str(chunk.metadata.get("section_title", "")))) else 0.0
            rescored.append((chunk, fused_score + 0.12 * coverage + phrase_bonus + section_bonus))
        return sorted(rescored, key=lambda item: (-item[1], item[0].chunk_id))

    @staticmethod
    def _select_context(ranked: list[tuple[_Chunk, float]], top_k: int, budget: int) -> list[tuple[_Chunk, float]]:
        selected: list[tuple[_Chunk, float]] = []
        selected_terms: list[set[str]] = []
        used_words = 0
        for chunk, score in ranked:
            words = len(chunk.text.split())
            if words > budget or used_words + words > budget:
                continue
            terms = set(_tokens(chunk.text))
            if any(_jaccard(terms, existing) >= 0.82 for existing in selected_terms):
                continue
            selected.append((chunk, score))
            selected_terms.append(terms)
            used_words += words
            if len(selected) >= top_k:
                break
        return selected

    @staticmethod
    def _chunks(result: dict[str, Any]) -> list[_Chunk]:
        return [_Chunk(chunk_id, text, metadata or {}) for chunk_id, text, metadata in zip(result.get("ids", []), result.get("documents", []), result.get("metadatas", []))]

    @staticmethod
    def _where_clause(filters: RetrievalFilters | None) -> dict[str, Any] | None:
        if not filters:
            return None
        clauses: list[dict[str, Any]] = []
        if filters.document_ids:
            clauses.append({"document_id": {"$in": filters.document_ids}})
        if filters.file_types:
            clauses.append({"file_type": {"$in": [value.upper().lstrip(".") for value in filters.file_types]}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    @staticmethod
    def _empty_response(query: str, rewritten_query: str, where: dict[str, Any] | None) -> RetrievalResponse:
        return RetrievalResponse(
            query=query, evidence=[], context="", status="no_evidence", message="No evidence matched the query and filters.",
            diagnostics=RetrievalDiagnostics(original_query=query, rewritten_query=rewritten_query, semantic_candidates=0, keyword_candidates=0, fused_candidates=0, reranked_candidates=0, returned_candidates=0, context_word_count=0, filters_applied=bool(where)),
        )

    @staticmethod
    def _evidence(chunk: _Chunk, score: float, rank: int) -> EvidenceItem:
        metadata = chunk.metadata
        page = metadata.get("page_or_slide") or None
        return EvidenceItem(evidence_id=chunk.chunk_id, text=chunk.text, score=round(score, 6), rank=rank, document_id=str(metadata.get("document_id", "")), document_name=str(metadata.get("document_name", "")), file_type=str(metadata.get("file_type", "")), page_or_slide=int(page) if page is not None else None, section_title=str(metadata.get("section_title")) or None, chunk_id=chunk.chunk_id, chunk_index=int(metadata.get("chunk_index", 0)))


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def rewrite_query(query: str) -> str:
    """Normalize a question locally without inventing facts or calling an LLM."""
    normalized = " ".join(query.strip().split())
    terms = [term for term in _tokens(normalized) if term not in _QUESTION_WORDS]
    return " ".join(terms) or normalized


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


__all__ = ["RetrievalService", "rewrite_query"]
