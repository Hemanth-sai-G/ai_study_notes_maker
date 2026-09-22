"""Opt-in integration check for the real local FastAPI and ChromaDB stack."""

import os
import unittest


@unittest.skipUnless(
    os.getenv("STUDYMATE_RUN_REAL_INTEGRATION") == "1",
    "Set STUDYMATE_RUN_REAL_INTEGRATION=1 after indexing local documents to run this check.",
)
class RetrievalIntegrationTests(unittest.TestCase):
    def test_real_api_returns_typed_evidence_or_explicit_empty_index(self):
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/retrieval/query",
                json={"query": "What is the main concept?", "top_k": 3},
            )
        self.assertIn(response.status_code, {200, 409})
        if response.status_code == 200:
            payload = response.json()
            self.assertIn(payload["status"], {"ok", "no_evidence"})
            self.assertIn("diagnostics", payload)
            for item in payload["evidence"]:
                self.assertIn("document_id", item)
                self.assertIn("chunk_id", item)


if __name__ == "__main__":
    unittest.main()
