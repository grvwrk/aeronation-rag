"""Unit tests for the Cleaner stage in the ingestion pipeline.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from errors import CleanerError, PipelineError
from ingestion.shared_processing.cleaner import (
    clean_document,
    clean_documents,
    clean_text,
    normalize_hyphenated_line_breaks,
    normalize_line_endings,
    normalize_whitespace,
    remove_control_characters,
)
from models import RawDoc, StageCounts


class CleanerUnitTests(unittest.TestCase):
    # 1. Normal text remains essentially unchanged.
    def test_normal_text_remains_essentially_unchanged(self):
        text = "Aerodynamics is the study of the motion of air."
        doc = RawDoc(text=text, source_id="aero.txt")
        cleaned = clean_document(doc)
        self.assertIsNotNone(cleaned)
        self.assertEqual(cleaned.text, text)

    # 2. Windows and Mac line endings are normalized.
    def test_line_endings_normalized(self):
        text_crlf = "Line 1\r\nLine 2\r\n\r\nLine 3"
        text_cr = "Line 1\rLine 2\r\rLine 3"
        expected = "Line 1 Line 2\n\nLine 3"

        cleaned_crlf = clean_text(text_crlf)
        cleaned_cr = clean_text(text_cr)

        self.assertEqual(cleaned_crlf, expected)
        self.assertEqual(cleaned_cr, expected)
        self.assertEqual(normalize_line_endings("A\r\nB\rC"), "A\nB\nC")

    # 3. Excessive spaces and horizontal whitespace are normalized.
    def test_excessive_spaces_normalized(self):
        text = "Hello        world!   This  is   a    test."
        self.assertEqual(clean_text(text), "Hello world! This is a test.")

    # 4. Excessive blank lines are normalized to double newline.
    def test_excessive_blank_lines_normalized(self):
        text = "Paragraph 1\n\n\n\n\n\nParagraph 2\n\n\nParagraph 3"
        expected = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        self.assertEqual(clean_text(text), expected)

    # 5. Leading and trailing whitespace is removed.
    def test_leading_trailing_whitespace_removed(self):
        text = "   \n\t  Important document content.   \n\n  "
        self.assertEqual(clean_text(text), "Important document content.")

    # 6. Whitespace-only documents are discarded (clean_document returns None).
    def test_whitespace_only_document_discarded(self):
        doc = RawDoc(text="   \n\t\r\n   ", source_id="empty.txt")
        self.assertIsNone(clean_document(doc))

    # 7. Empty documents (0-length) are discarded.
    def test_empty_document_discarded(self):
        doc = RawDoc(text="", source_id="blank.pdf")
        self.assertIsNone(clean_document(doc))

    # 8. Paragraph boundaries are preserved.
    def test_paragraph_boundaries_preserved(self):
        text = (
            "This is paragraph one with multiple lines\n"
            "that belong together.\n\n"
            "This is paragraph two with separate\n"
            "information and facts."
        )
        expected = (
            "This is paragraph one with multiple lines that belong together.\n\n"
            "This is paragraph two with separate information and facts."
        )
        self.assertEqual(clean_text(text), expected)

    # 9. Unicode characters are preserved.
    def test_unicode_characters_preserved(self):
        text = "Café au lait with façade and résumé in Zürich, España: naïve garçon."
        doc = RawDoc(text=text, source_id="unicode.txt")
        cleaned = clean_document(doc)
        self.assertIsNotNone(cleaned)
        self.assertEqual(cleaned.text, text)

    # 10. Mathematical symbols, Greek letters, superscripts/subscripts are preserved.
    def test_mathematical_and_scientific_symbols_preserved(self):
        text = (
            "Navier-Stokes: ∂u/∂t + (u·∇)u = -1/ρ ∇p + ν ∇²u + g\n\n"
            "Lift equation: L = ½ ρ v² S C_L\n\n"
            "Gas Law: PV = nRT, where ΔT = 25°C ± 0.5°C, α = 0.05, β ≈ 1.2, π ≈ 3.14159, ∑ ∫ √x ≤ ∞"
        )
        doc = RawDoc(text=text, source_id="physics.pdf")
        cleaned = clean_document(doc)
        self.assertIsNotNone(cleaned)
        # Verify all special symbols remain in the cleaned text
        for symbol in ["∂", "∇", "ρ", "ν", "²", "½", "Δ", "°", "±", "α", "β", "≈", "π", "∑", "∫", "√", "≤", "∞"]:
            self.assertIn(symbol, cleaned.text)

    # 11. Punctuation is preserved.
    def test_punctuation_preserved(self):
        text = "“Special quotes,” dashes—em and –en, brackets [1], colons: semicolons; ellipses… and URLs (https://example.com/api?v=1)."
        doc = RawDoc(text=text, source_id="punct.txt")
        cleaned = clean_document(doc)
        self.assertIsNotNone(cleaned)
        self.assertEqual(cleaned.text, text)

    # 12. Page metadata is preserved.
    def test_metadata_preserved(self):
        doc = RawDoc(
            text="Mach number is the ratio of flow velocity past a boundary to the local speed of sound.",
            source_id="fluid_dynamics.pdf",
            metadata={
                "file_name": "fluid_dynamics.pdf",
                "source_id": "fluid_dynamics-a1b2c3d4",
                "page_num": 42,
                "page_count": 300,
                "source_format": "pdf",
                "extraction_status": "ok",
                "is_empty": False,
                "custom_field": "aerospace",
            },
        )
        cleaned = clean_document(doc)
        self.assertIsNotNone(cleaned)
        self.assertEqual(cleaned.metadata["page_num"], 42)
        self.assertEqual(cleaned.metadata["page_count"], 300)
        self.assertEqual(cleaned.metadata["source_format"], "pdf")
        self.assertEqual(cleaned.metadata["extraction_status"], "ok")
        self.assertEqual(cleaned.metadata["custom_field"], "aerospace")
        # Ensure modifying cleaned metadata does not mutate original doc
        cleaned.metadata["page_num"] = 99
        self.assertEqual(doc.metadata["page_num"], 42)

    # 13. source_id is preserved.
    def test_source_id_preserved(self):
        doc = RawDoc(text="Sample text content.", source_id="unique-source-id-12345")
        cleaned = clean_document(doc)
        self.assertIsNotNone(cleaned)
        self.assertEqual(cleaned.source_id, "unique-source-id-12345")

    # 14. Obvious PDF line-break hyphenation is handled correctly.
    def test_pdf_line_break_hyphenation(self):
        text = "This is a sen-\ntence split across lines with infor-\nmation."
        expected = "This is a sentence split across lines with information."
        self.assertEqual(clean_text(text), expected)

    # 15. Real hyphenated words are not blindly destroyed.
    def test_real_hyphenated_words_preserved(self):
        text = "This is a state-of-the-art\nmodel with a high-level\ndescription."
        expected = "This is a state-of-the-art model with a high-level description."
        self.assertEqual(clean_text(text), expected)

    # 16. Cleaning is deterministic.
    def test_determinism(self):
        doc = RawDoc(
            text="  Complex text with sen-\ntence breaks,   excessive    spaces\r\nand α + β = 90°.\n\nParagraph 2.  ",
            source_id="det.pdf",
            metadata={"page_num": 1},
        )
        cleaned_first = clean_document(doc)
        for _ in range(5):
            cleaned_subsequent = clean_document(doc)
            self.assertEqual(cleaned_first.text, cleaned_subsequent.text)
            self.assertEqual(cleaned_first.source_id, cleaned_subsequent.source_id)
            self.assertEqual(cleaned_first.metadata, cleaned_subsequent.metadata)

    # 17. Multiple RawDocs are cleaned independently and empty ones filtered.
    def test_clean_documents_batch_and_stage_counts(self):
        docs = [
            RawDoc(text="Valid document 1 text.", source_id="doc1.txt", metadata={"page_num": 1}),
            RawDoc(text="   \n\t  ", source_id="doc2.txt", metadata={"page_num": 2}),
            RawDoc(text="", source_id="doc3.txt", metadata={"page_num": 3}),
            RawDoc(text="Valid document 4 text with sen-\ntence split.", source_id="doc4.txt", metadata={"page_num": 4}),
        ]
        counts = StageCounts(files_seen=4, docs_loaded=4)
        cleaned_docs = clean_documents(docs, counts=counts)

        self.assertEqual(len(cleaned_docs), 2)
        self.assertEqual(cleaned_docs[0].source_id, "doc1.txt")
        self.assertEqual(cleaned_docs[0].text, "Valid document 1 text.")
        self.assertEqual(cleaned_docs[1].source_id, "doc4.txt")
        self.assertEqual(cleaned_docs[1].text, "Valid document 4 text with sentence split.")

        # Check telemetry
        self.assertEqual(counts.docs_cleaned, 2)
        self.assertEqual(counts.docs_discarded, 2)

    # 18. Error handling follows the existing error contract.
    def test_error_handling_strict_and_non_strict(self):
        invalid_doc = RawDoc(text=12345, source_id="bad_doc.txt", metadata={"page_num": 7})  # type: ignore

        # strict=True raises CleanerError (which inherits from PipelineError)
        with self.assertRaises(CleanerError) as ctx:
            clean_documents([invalid_doc], strict=True)
        self.assertIsInstance(ctx.exception, PipelineError)
        self.assertEqual(ctx.exception.failure.source, "bad_doc.txt")
        self.assertEqual(ctx.exception.failure.page, 7)

        # strict=False logs, increments discarded count, and does not crash
        counts = StageCounts()
        results = clean_documents([invalid_doc], counts=counts, strict=False)
        self.assertEqual(results, [])
        self.assertEqual(counts.docs_discarded, 1)
        self.assertEqual(counts.docs_cleaned, 0)

    # 19. Unwanted control characters are removed.
    def test_control_character_removal(self):
        text_with_control_chars = "Header\x00\x07\x08 Text \x0bwith\x0c control\x1b characters\x7f."
        cleaned = clean_text(text_with_control_chars)
        self.assertEqual(cleaned, "Header Text with control characters.")
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\x07", cleaned)
        self.assertNotIn("\x08", cleaned)
        self.assertNotIn("\x0b", cleaned)
        self.assertNotIn("\x0c", cleaned)
        self.assertNotIn("\x1b", cleaned)
        self.assertNotIn("\x7f", cleaned)

    # 20. URLs and citations are preserved.

    def test_urls_and_citations_preserved(self):
        text = "Reference [1] (Einstein et al., 1905) available at https://arxiv.org/abs/physics/0501234?query=relativity#section2."
        doc = RawDoc(text=text, source_id="citations.txt")
        cleaned = clean_document(doc)
        self.assertIsNotNone(cleaned)
        self.assertEqual(cleaned.text, text)

    # 21. Empty documents list handling.
    def test_clean_empty_documents_list(self):
        counts = StageCounts()
        results = clean_documents([], counts=counts)
        self.assertEqual(results, [])
        self.assertEqual(counts.docs_cleaned, 0)
        self.assertEqual(counts.docs_discarded, 0)

    # 22. clean_document raises CleanerError when input is not a RawDoc.
    def test_clean_document_invalid_type_raises_cleaner_error(self):
        with self.assertRaises(CleanerError) as ctx:
            clean_document("not a rawdoc")  # type: ignore
        self.assertIn("Expected RawDoc instance", str(ctx.exception))

    # 23. Pipeline integration with loader-like RawDoc dictionaries.
    def test_loader_contract_compatibility(self):
        loader_doc = RawDoc(
            text="First line of paragraph 1.\r\nSecond line of paragraph 1.\r\n\r\nParagraph 2 with sen-\ntence split.",
            source_id="textbook-7a8b9c0d",
            metadata={
                "file_name": "textbook.pdf",
                "source_id": "textbook-7a8b9c0d",
                "page_num": 12,
                "page_count": 150,
                "source_format": "pdf",
                "extraction_status": "ok",
                "is_empty": False,
            },
        )
        cleaned = clean_document(loader_doc)
        self.assertIsNotNone(cleaned)
        self.assertEqual(
            cleaned.text,
            "First line of paragraph 1. Second line of paragraph 1.\n\nParagraph 2 with sentence split.",
        )
        self.assertEqual(cleaned.source_id, "textbook-7a8b9c0d")
        self.assertEqual(cleaned.metadata["page_num"], 12)
        self.assertEqual(cleaned.metadata["page_count"], 150)
        self.assertEqual(cleaned.metadata["source_format"], "pdf")


if __name__ == "__main__":
    unittest.main()
