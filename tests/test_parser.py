"""Unit tests for src/parser.py.

Tests PyMuPDFParser, PaddleOCRParser, and ResumeParser against
the synthetic PDF at tests/sample_resumes/sample_cv.pdf.
No LLM calls — pymupdf4llm runs entirely locally.
"""

from pathlib import Path

import pytest

from src.core.models import Resume
from src.core.parser_engine import PaddleOCRParser, PyMuPDFParser, ResumeParser

SAMPLE_PDF = "tests/sample_resumes/sample_cv.pdf"


def test_pymupdf_parser_is_available() -> None:
    """Verify pymupdf4llm is installed and PyMuPDFParser reports available."""
    parser = PyMuPDFParser()
    assert parser.is_available() is True


def test_pymupdf_parser_extracts_text() -> None:
    """Verify PyMuPDFParser extracts meaningful text from a sample PDF."""
    parser = PyMuPDFParser()
    result = parser.extract_text(SAMPLE_PDF)

    assert isinstance(result, str)
    assert len(result) > 0
    assert len(result.split()) >= 50


def test_resume_parser_parse_returns_resume() -> None:
    """Verify ResumeParser.parse returns a Resume with filename and text."""
    rp = ResumeParser(parser=PyMuPDFParser())
    resume = rp.parse(SAMPLE_PDF)

    assert isinstance(resume, Resume)
    assert resume.filename == "sample_cv.pdf"
    assert len(resume.raw_text) > 0


def test_resume_parser_parse_all_finds_pdf() -> None:
    """Verify parse_all discovers the sample PDF in the test directory."""
    rp = ResumeParser(parser=PyMuPDFParser())
    result = rp.parse_all("tests/sample_resumes/")

    assert len(result) == 1
    assert result[0].filename == "sample_cv.pdf"


def test_resume_parser_empty_folder_returns_empty_list(tmp_path: Path) -> None:
    """Verify parse_all returns an empty list for a folder with no PDFs."""
    rp = ResumeParser(parser=PyMuPDFParser())
    result = rp.parse_all(str(tmp_path))

    assert result == []


def test_paddle_parser_not_available() -> None:
    """Verify PaddleOCRParser is unavailable and raises NotImplementedError."""
    parser = PaddleOCRParser()
    assert parser.is_available() is False

    with pytest.raises(NotImplementedError):
        parser.extract_text("any_path.pdf")
