"""Focused tests for source-grounded Phase 6 study material generation."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.retrieval import EvidenceItem, RetrievalDiagnostics, RetrievalResponse
from app.models.study import StudyRequest
from app.services.study_tools import StudyActivityFormatError, StudyToolsService


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
    def generate(self, prompt, json_mode=False):
        self.prompts.append(prompt)
        if "quiz_questions" in prompt:
            return '{"quiz_questions":[{"question":"What does mitosis produce?","options":["Two identical cells","Four cells","A protein","No cells"],"correct_option":0,"explanation":"The source says mitosis produces two identical cells."}]}'
        return '{"flashcards":[{"question":"What does mitosis produce?","answer":"Two identical cells."}]}'


class StudyToolsTests(unittest.TestCase):
    def test_flashcards_use_selected_documents_and_attach_sources(self):
        retriever, generator = FakeRetriever(), FakeGenerator()
        response = StudyToolsService(retriever, generator).generate("flashcards", ["doc-1"], "mitosis")
        self.assertEqual(response.status, "ok")
        self.assertEqual(retriever.filters.document_ids, ["doc-1"])
        self.assertIn("flashcards", generator.prompts[0])
        self.assertEqual(response.citations[0].page_or_slide, 4)
        self.assertEqual(response.flashcards[0].answer, "Two identical cells.")

    def test_quiz_has_options_answer_index_and_explanation(self):
        response = StudyToolsService(FakeRetriever(), FakeGenerator()).generate("quiz", ["doc-1"], "mitosis")
        question = response.quiz_questions[0]
        self.assertEqual(len(question.options), 4)
        self.assertEqual(question.correct_option, 0)
        self.assertIn("source says", question.explanation)

    def test_quiz_accepts_common_model_key_and_answer_variations(self):
        content = '''Here is the activity:
        {"questions":[{"prompt":"What does mitosis produce?","choices":{"A":"A. Two identical cells","B":"B. Four cells","C":"C. A protein","D":"D. No cells"},"correct_answer":"A","rationale":"The source says mitosis produces two identical cells."}]}'''
        _, questions = StudyToolsService._structured_content("quiz", content)
        self.assertEqual(questions[0].options[0], "Two identical cells")
        self.assertEqual(questions[0].correct_option, 0)
        self.assertIn("source says", questions[0].explanation)

    def test_flashcards_accept_front_and_back_variations(self):
        cards, _ = StudyToolsService._structured_content("flashcards", '{"cards":[{"front":"What does mitosis produce?","back":"Two identical cells."}]}')
        self.assertEqual(cards[0].question, "What does mitosis produce?")

    def test_invalid_activity_has_a_model_format_error(self):
        with self.assertRaises(StudyActivityFormatError):
            StudyToolsService._structured_content("quiz", '{"questions":[{"question":"Incomplete"}]}')

    def test_no_evidence_does_not_call_the_generator(self):
        generator = FakeGenerator()
        response = StudyToolsService(FakeRetriever(False), generator).generate("quiz", ["doc-1"])
        self.assertEqual(response.status, "insufficient_evidence")
        self.assertEqual(generator.prompts, [])

    def test_comparison_requires_two_documents_at_the_api_boundary(self):
        with self.assertRaisesRegex(ValueError, "two or more"):
            StudyRequest(mode="comparison", document_ids=["doc-1"])


if __name__ == "__main__":
    unittest.main()
