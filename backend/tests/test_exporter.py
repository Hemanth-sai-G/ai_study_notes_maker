"""Tests for browser-requested local PDF study exports."""

import sys
import unittest
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.exports import StudyExportRequest
from app.models.generation import Citation
from app.services.exporter import build_study_pdf


class StudyPdfExportTests(unittest.TestCase):
    def test_pdf_contains_study_content_and_verified_source(self):
        export = StudyExportRequest(
            title="Practice quiz",
            content="1. What is retrieval?\nA. Finding relevant source passages\nAnswer: A",
            citations=[
                Citation(
                    evidence_id="doc-1:0",
                    document_id="doc-1",
                    document_name="Biology.pdf",
                    file_type="PDF",
                    page_or_slide=2,
                    section_title="Retrieval",
                    chunk_id="doc-1:0",
                    chunk_index=0,
                )
            ],
        )
        pdf = build_study_pdf(export)
        document = fitz.open(stream=pdf, filetype="pdf")
        text = "\n".join(page.get_text() for page in document)

        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertIn("Practice quiz", text)
        self.assertIn("Finding relevant source passages", text)
        self.assertIn("Biology.pdf", text)


if __name__ == "__main__":
    unittest.main()
