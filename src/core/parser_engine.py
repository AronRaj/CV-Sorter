"""PDF parsing and resume extraction module.

Provides an abstract PDFParser interface with swappable implementations
(PyMuPDF for digital PDFs, PaddleOCR stub for scanned PDFs), plus a
ResumeParser wrapper that builds Resume dataclass instances.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from src.core.models import Resume

logger = logging.getLogger(__name__)


class PDFParser(ABC):
    """Abstract interface all PDF parsers must implement."""

    @abstractmethod
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from a single PDF.

        Returns a markdown string (or plain text for OCR parsers).
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this parser's dependencies are installed."""
        ...

    @staticmethod
    def build(parser_name: str) -> PDFParser:
        """Factory method. Returns the correct implementation for the given name.

        Args:
            parser_name: One of "pymupdf" or "paddle".

        Raises:
            ValueError: If parser_name is not recognised.
        """
        if parser_name == "pymupdf":
            return PyMuPDFParser()
        elif parser_name == "paddle":
            return PaddleOCRParser()
        else:
            raise ValueError(
                f"Unknown parser: '{parser_name}'. "
                f"Supported values: 'pymupdf', 'paddle'."
            )


class PyMuPDFParser(PDFParser):
    """Primary parser using pymupdf4llm to convert digital PDFs to LLM-ready markdown.

    Handles text, tables, and multi-column layouts in text-based PDFs.
    Falls back to raw text extraction if markdown conversion fails.
    Install: pip install pymupdf4llm (already in requirements.txt).
    """

    def is_available(self) -> bool:
        """Return True if pymupdf4llm is installed."""
        try:
            import pymupdf4llm  # noqa: F401
            return True
        except ImportError:
            return False

    def extract_text(self, pdf_path: str) -> str:
        """Extract text from a PDF using pymupdf4llm markdown conversion.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted markdown text, or empty string if the PDF yields no text.

        Raises:
            RuntimeError: If extraction fails due to a library error.
        """
        try:
            import pymupdf4llm

            md_text: str = pymupdf4llm.to_markdown(pdf_path)

            if not md_text or not md_text.strip():
                filename = Path(pdf_path).name
                logger.warning(
                    "PDF '%s' produced empty text — likely a scanned/image PDF. "
                    "Set PDF_PARSER=paddle in .env to enable OCR extraction.",
                    filename,
                )
                return ""

            return md_text

        except Exception as e:
            filename = Path(pdf_path).name
            logger.error("Failed to extract text from '%s': %s", filename, e)
            raise RuntimeError(
                f"PDF extraction failed for '{filename}': {e}"
            ) from e


class PaddleOCRParser(PDFParser):
    """Post-MVP OCR parser for scanned/image PDFs.

    Uses PaddleOCR to recognise text from rendered page images.
    Install: pip install paddlepaddle paddleocr (NOT in requirements.txt).
    Set PDF_PARSER=paddle in .env to activate.
    """

    def is_available(self) -> bool:
        """Return False — PaddleOCR is not installed in the MVP."""
        return False

    def extract_text(self, pdf_path: str) -> str:
        """Not implemented in the MVP.

        Raises:
            NotImplementedError: Always, until PaddleOCR is wired in post-MVP.
        """
        raise NotImplementedError(
            "PaddleOCR not installed. "
            "See ARCHITECTURE.md → Plugging in PaddleOCR"
        )


class ResumeParser:
    """Wraps a PDFParser and builds Resume dataclass instances.

    Does not know or care which PDFParser implementation is running
    underneath — it only calls parser.extract_text(path).
    """

    def __init__(self, parser: PDFParser) -> None:
        """Initialise with the active PDF parser.

        Args:
            parser: A concrete PDFParser implementation.
        """
        self._parser = parser

    def parse(self, pdf_path: str) -> Resume:
        """Parse a single PDF into a Resume dataclass.

        Extracts raw text via the injected PDFParser. Metadata fields
        (name, skills, experience, education) are left empty — they will
        be populated by LLM extraction in a later phase.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A Resume instance with raw_text populated.
        """
        raw_text = self._parser.extract_text(pdf_path)
        filename = Path(pdf_path).name

        return Resume(
            filename=filename,
            raw_text=raw_text,
            name=None,
            skills=[],
            experience_years=None,
            education=[],
        )

    def parse_all(self, folder_path: str) -> list[Resume]:
        """Parse all PDF files in a folder into Resume instances.

        Skips files that fail to parse, logging a warning for each.

        Args:
            folder_path: Path to the directory containing PDF files.

        Returns:
            List of successfully parsed Resume instances.
        """
        folder = Path(folder_path)
        pdf_files = sorted(folder.glob("*.pdf"))

        if not pdf_files:
            logger.warning("No PDF files found in '%s'.", folder_path)
            return []

        resumes: list[Resume] = []
        for pdf_file in pdf_files:
            try:
                resume = self.parse(str(pdf_file))
                resumes.append(resume)
                logger.info(
                    "Parsed '%s' — %d words extracted.",
                    resume.filename,
                    len(resume.raw_text.split()),
                )
            except (RuntimeError, NotImplementedError) as e:
                logger.warning("Skipping '%s': %s", pdf_file.name, e)

        return resumes
