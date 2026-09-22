"""Focused tests for evidence-only local generation and code-derived citations."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.retrieval import EvidenceItem, RetrievalDiagnostics, RetrievalResponse
from app.services.generation import ChatService, OllamaUnavailableError


def retrieval(status: str = "ok") -> RetrievalResponse:
    evidence = [] if status == "no_evidence" else [
        EvidenceItem(
            evidence_id="doc-a:0", text="Plants convert light energy into chemical energy.", score=0.9, rank=1,
            document_id="doc-a", document_name="Biology.pdf", file_type="PDF", page_or_slide=2,
            section_title="Photosynthesis", chunk_id="doc-a:0", chunk_index=0,
        )
    ]
    return RetrievalResponse(
        query="What do plants convert?", evidence=evidence, context="", status=status,
        message="No evidence matched." if not evidence else None,
        diagnostics=RetrievalDiagnostics(
            original_query="What do plants convert?", rewritten_query="plants convert", semantic_candidates=1,
            keyword_candidates=1, fused_candidates=1, reranked_candidates=1, returned_candidates=len(evidence),
            context_word_count=7, filters_applied=False,
        ),
    )


class FakeRetriever:
    def __init__(self, response: RetrievalResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs: object) -> RetrievalResponse:
        self.calls.append(kwargs)
        return self.response


class FakeGenerator:
    model = "qwen2.5:3b"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Plants convert light energy into chemical energy."


class UnavailableGenerator:
    def generate(self, _prompt: str) -> str:
        raise OllamaUnavailableError("Ollama is not available locally.")


class GenerationTests(unittest.TestCase):
    def test_answer_uses_retrieval_evidence_and_attaches_code_derived_citations(self):
        retriever = FakeRetriever(retrieval())
        generator = FakeGenerator()
        response = ChatService(retriever, generator).answer("What do plants convert?", 5, 20, 1_200, None)

        self.assertEqual(response.status, "ok")
        self.assertEqual(response.answer, "Plants convert light energy into chemical energy.")
        self.assertEqual(response.citations[0].document_name, "Biology.pdf")
        self.assertEqual(response.citations[0].page_or_slide, 2)
        self.assertIn("SOURCE [doc-a:0]", generator.prompts[0])
        self.assertIn("only from the SOURCE passages", generator.prompts[0])
        self.assertEqual(retriever.calls[0]["query"], "What do plants convert?")

    def test_no_evidence_does_not_call_the_generator(self):
        retriever = FakeRetriever(retrieval("no_evidence"))
        generator = FakeGenerator()
        response = ChatService(retriever, generator).answer("What do plants convert?", 5, 20, 1_200, None)

        self.assertEqual(response.status, "insufficient_evidence")
        self.assertEqual(response.citations, [])
        self.assertEqual(generator.prompts, [])

    def test_session_history_is_memory_only_and_limited_in_prompt(self):
        retriever = FakeRetriever(retrieval())
        generator = FakeGenerator()
        service = ChatService(retriever, generator)
        service.answer("First question", 5, 20, 1_200, None, session_id="local-session")
        service.answer("Second question", 5, 20, 1_200, None, session_id="local-session")

        self.assertIn("- First question", generator.prompts[1])
        self.assertNotIn("- Second question", generator.prompts[1])

    def test_ollama_unavailable_error_is_preserved_for_the_api(self):
        service = ChatService(FakeRetriever(retrieval()), UnavailableGenerator())
        with self.assertRaisesRegex(OllamaUnavailableError, "not available"):
            service.answer("Question", 5, 20, 1_200, None)

    def test_chat_endpoint_returns_a_safe_ollama_setup_error(self):
        from fastapi.testclient import TestClient

        from app.api.chat import chat_service
        from app.main import app

        with patch.object(chat_service, "answer", side_effect=OllamaUnavailableError("Ollama is not available locally.")):
            with TestClient(app) as client:
                response = client.post("/api/v1/chat/answer", json={"query": "Question"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Ollama is not available locally.")


if __name__ == "__main__":
    unittest.main()
