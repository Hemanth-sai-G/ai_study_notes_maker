"""Focused tests for source-grounded Phase 6 study material generation."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.retrieval import EvidenceItem, RetrievalDiagnostics, RetrievalResponse
from app.services.study_tools import StudyToolsService


class FakeRetriever:
    def __init__(self, has_evidence: bool = True) -> None:
        self.has_evidence = has_evidence
        self.filters = None

    def retrieve(self, **kwargs):
        self.filters = kwargs["filters"]
        evidence = [EvidenceItem(evidence_id="doc-1:0", text="Mitosis produces two identical cells.", score=1, rank=1, document_id="doc-1", document_name="Cells.pdf", file_type="PDF", page_or_slide=4, section_title="Mitosis", chunk_id="doc-1:0", chunk_index=0)] if self.has_evidence else []
        return RetrievalResponse(query=kwargs["query"], evidence=evidence, context="", status="ok" if evidence else "no_evidence", diagnostics=RetrievalDiagnostics(original_query="", rewritten_query="", semantic_candidates=1, keyword_candidates=1, fused_candidates=1, reranked_candidates=1, returned_candidates=len(evidence), context_word_count=6, filters_applied=True))


class FakeGenerator:
    model = "qwen2.5:3b"
    def __init__(self): self.prompts = []
    def generate(self, prompt): self.prompts.append(prompt); return "Q: What does mitosis produce?\nA: Two identical cells."


class StudyToolsTests(unittest.TestCase):
    def test_flashcards_use_selected_documents_and_attach_sources(self):
        retriever, generator = FakeRetriever(), FakeGenerator()
        response = StudyToolsService(retriever, generator).generate("flashcards", ["doc-1"], "mitosis")
        self.assertEqual(response.status, "ok")
        self.assertEqual(retriever.filters.document_ids, ["doc-1"])
        self.assertIn("`Q:` and `A:`", generator.prompts[0])
        self.assertEqual(response.citations[0].page_or_slide, 4)

    def test_no_evidence_does_not_call_the_generator(self):
        generator = FakeGenerator()
        response = StudyToolsService(FakeRetriever(False), generator).generate("quiz", ["doc-1"])
        self.assertEqual(response.status, "insufficient_evidence")
        self.assertEqual(generator.prompts, [])


if __name__ == "__main__":
    unittest.main()
