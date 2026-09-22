"""Focused hybrid retrieval tests with a deterministic Chroma-shaped fixture."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.retrieval import RetrievalFilters, RetrievalRequest
from app.services.retrieval import RetrievalService, rewrite_query


class FakeEmbedder:
    def encode(self, sentences, **_kwargs):
        return SimpleNamespace(tolist=lambda: [[1.0, 0.0] for _ in sentences])


class FakeCollection:
    def __init__(self):
        self.rows = [
            ("doc-a:0", "Photosynthesis converts light energy into chemical energy in plants.", {"document_id": "doc-a", "document_name": "Biology.txt", "file_type": "TXT", "chunk_index": 0, "page_or_slide": 2, "section_title": "Photosynthesis"}),
            ("doc-b:0", "Mitochondria produce ATP through cellular respiration.", {"document_id": "doc-b", "document_name": "Cells.pdf", "file_type": "PDF", "chunk_index": 0, "page_or_slide": 4, "section_title": "Cellular respiration"}),
            ("doc-c:0", "The Krebs cycle is a sequence of chemical reactions in respiration.", {"document_id": "doc-c", "document_name": "Chemistry.pptx", "file_type": "PPTX", "chunk_index": 0, "page_or_slide": 7, "section_title": "Krebs cycle"}),
        ]

    def _filtered(self, where):
        if not where:
            return self.rows
        if "$and" in where:
            return [row for row in self.rows if all(self._matches(row, clause) for clause in where["$and"])]
        return [row for row in self.rows if self._matches(row, where)]

    @staticmethod
    def _matches(row, where):
        if "document_id" in where:
            allowed = set(where["document_id"]["$in"])
            return row[2]["document_id"] in allowed
        if "file_type" in where:
            allowed = set(where["file_type"]["$in"])
            return row[2]["file_type"] in allowed
        return True

    def count(self):
        return len(self.rows)

    def get(self, where, include):
        rows = self._filtered(where)
        return {"ids": [row[0] for row in rows], "documents": [row[1] for row in rows], "metadatas": [row[2] for row in rows]}

    def query(self, query_embeddings, n_results, where, include):
        rows = self._filtered(where)[:n_results]
        return {"ids": [[row[0] for row in rows]], "distances": [[0.1 + index * 0.2 for index, _ in enumerate(rows)]]}


class FakeKnowledgeBase:
    def __init__(self):
        self.collection = FakeCollection()
        self.embedder = FakeEmbedder()


class RetrievalTests(unittest.TestCase):
    def test_query_rewrite_is_conservative_and_local(self):
        self.assertEqual(rewrite_query("  What is   Krebs-cycle? "), "krebs cycle")
        self.assertEqual(rewrite_query("ATP"), "atp")

    def test_bm25_exact_term_ranks_matching_chunk(self):
        chunks = RetrievalService._chunks(FakeKnowledgeBase().collection.get(None, None))
        scores = RetrievalService.bm25_scores("Krebs", chunks)
        self.assertGreater(scores["doc-c:0"], scores["doc-a:0"])
        self.assertGreater(scores["doc-c:0"], scores["doc-b:0"])

    def test_hybrid_response_preserves_provenance_and_deduplicates(self):
        response = RetrievalService(FakeKnowledgeBase()).retrieve("Krebs cycle", 3, None, 3)
        self.assertEqual(response.status, "ok")
        self.assertEqual(response.evidence[0].document_id, "doc-c")
        self.assertEqual(len({item.chunk_id for item in response.evidence}), len(response.evidence))
        self.assertGreater(response.diagnostics.keyword_candidates, 0)
        self.assertEqual(response.diagnostics.fused_candidates, 3)
        self.assertIn("[doc-c:0]", response.context)

    def test_reranker_rewards_query_term_coverage(self):
        chunks = RetrievalService._chunks(FakeKnowledgeBase().collection.get(None, None))
        ranked = RetrievalService._rerank("Krebs cycle", [(chunks[0], 0.8), (chunks[2], 0.7)])
        self.assertEqual(ranked[0][0].chunk_id, "doc-c:0")

    def test_context_budget_limits_selected_evidence(self):
        response = RetrievalService(FakeKnowledgeBase()).retrieve("respiration", 5, None, 5, 6)
        self.assertLessEqual(response.diagnostics.context_word_count, 6)
        self.assertEqual(response.diagnostics.context_word_count, sum(len(item.text.split()) for item in response.evidence))

    def test_context_budget_never_includes_an_oversized_first_chunk(self):
        chunks = RetrievalService._chunks(FakeKnowledgeBase().collection.get(None, None))
        selected = RetrievalService._select_context([(chunks[0], 1.0)], top_k=1, budget=3)
        self.assertEqual(selected, [])

    def test_whitespace_only_query_is_rejected_at_the_api_boundary(self):
        with self.assertRaises(ValidationError):
            RetrievalRequest(query="   ")

    def test_document_filter_is_applied_to_both_retrievers(self):
        response = RetrievalService(FakeKnowledgeBase()).retrieve("ATP", 5, RetrievalFilters(document_ids=["doc-a"]), 5)
        self.assertEqual(len(response.evidence), 1)
        self.assertEqual(response.evidence[0].document_id, "doc-a")
        self.assertTrue(response.diagnostics.filters_applied)

    def test_file_type_and_combined_filters_are_applied(self):
        response = RetrievalService(FakeKnowledgeBase()).retrieve("respiration", 5, RetrievalFilters(file_types=[".pptx"]), 5)
        self.assertEqual([item.document_id for item in response.evidence], ["doc-c"])
        response = RetrievalService(FakeKnowledgeBase()).retrieve("respiration", 5, RetrievalFilters(document_ids=["doc-c"], file_types=["PPTX"]), 5)
        self.assertEqual([item.document_id for item in response.evidence], ["doc-c"])

    def test_empty_index_is_explicit(self):
        knowledge_base = FakeKnowledgeBase()
        knowledge_base.collection.rows = []
        with self.assertRaisesRegex(ValueError, "knowledge base is empty"):
            RetrievalService(knowledge_base).retrieve("anything", 5, None)


if __name__ == "__main__":
    unittest.main()
