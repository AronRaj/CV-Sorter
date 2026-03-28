"""PDF parsing and resume extraction module.

Provides an abstract PDFParser interface with the PyMuPDF implementation
for digital PDFs, plus a ResumeParser wrapper that builds Resume
dataclass instances from parsed PDF text.

The pipeline intentionally separates "get text from PDF" from "interpret
that text": this module only produces raw markdown/plain text; structured
fields on Resume are filled later by LLM-based extraction so scoring and
reporting stay decoupled from the PDF library choice.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from models import Resume

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract PDF parser and factory
# ---------------------------------------------------------------------------
class PDFParser(ABC):
    """Abstract interface all PDF parsers must implement.

    Callers depend on this type so new backends (e.g. OCR) can be swapped in
    without changing ResumeParser or downstream agents—only configuration
    and the factory need to know the concrete class.
    """

    @abstractmethod
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from a single PDF file.

        Implementations should return content suitable for downstream LLM
        consumption where possible (e.g. markdown preserves headings and tables
        better than a single flattened string).

        Args:
            pdf_path: Filesystem path to the PDF (string form keeps the
                interface simple for callers that already use str paths).

        Returns:
            Extracted text, typically markdown from pymupdf4llm or plain text
            from OCR-oriented parsers. Empty string is allowed when the file
            yields no extractable text (e.g. image-only PDF without OCR).

        Side effects:
            None required by the contract; concrete classes may log warnings
            or errors when extraction fails or produces empty output.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Report whether this parser's optional dependencies are importable.

        Used at startup or in health checks so the app can fail fast or fall
        back to another parser instead of crashing on first use.

        Returns:
            True if the implementation can run (dependencies present), False
            otherwise.

        Side effects:
            None.
        """
        ...

    @staticmethod
    def build(parser_name: str) -> PDFParser:
        """Construct the PDFParser implementation registered for ``parser_name``.

        Centralizes parser selection so environment/config only stores a
        string key; adding a new backend means extending this method and
        documenting the new name, without scattering ``if`` chains elsewhere.

        Args:
            parser_name: Identifier for the desired backend. Currently only
                ``"pymupdf"`` is implemented in this module.

        Returns:
            A new instance of the matching concrete ``PDFParser``.

        Raises:
            ValueError: If ``parser_name`` is not a supported key.

        Side effects:
            Instantiates the chosen parser class (no global state).
        """
        if parser_name == "pymupdf":
            return PyMuPDFParser()
        else:
            raise ValueError(
                f"Unknown parser: '{parser_name}'. "
                f"Supported values: 'pymupdf'."
            )


# ---------------------------------------------------------------------------
# PyMuPDF / pymupdf4llm implementation
# ---------------------------------------------------------------------------
class PyMuPDFParser(PDFParser):
    """Primary parser using pymupdf4llm to convert digital PDFs to LLM-ready markdown.

    Handles text, tables, and multi-column layouts in text-based PDFs. Library
    errors surface as ``RuntimeError``; empty output (e.g. scanned PDFs with
    no text layer) yields an empty string with a log hint to use OCR.
    Install: pip install pymupdf4llm (already in requirements.txt).

    Lazy-imports pymupdf4llm inside methods so importing this module does not
    require the dependency until a code path actually parses a PDF—useful for
    tests and optional installs.
    """

    def is_available(self) -> bool:
        """Return True if ``pymupdf4llm`` can be imported.

        Returns:
            True when the package is installed; False if ``ImportError`` occurs.

        Side effects:
            None (import attempt is discarded after the check).
        """
        try:
            import pymupdf4llm  # noqa: F401
            return True
        except ImportError:
            return False

    def extract_text(self, pdf_path: str) -> str:
        """Extract text from a PDF using pymupdf4llm's markdown conversion.

        Empty markdown after a "successful" call usually means the PDF has no
        embedded text layer (scanned document). We return ``""`` rather than
        raising so batch flows can skip or flag the file; callers that need
        hard failure can check the return value.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Markdown string on success, or empty string when extraction yields
            no non-whitespace content.

        Raises:
            RuntimeError: When the library raises an unexpected exception
                (wrapped with context and original cause preserved).

        Side effects:
            Logs a warning for empty extraction (operational hint to enable
            OCR). Logs an error and re-raises as ``RuntimeError`` on failure.
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


class ResumeParser:
    """Wraps a PDFParser and builds Resume dataclass instances.

    Does not know or care which PDFParser implementation is running
    underneath — it only calls parser.extract_text(path).

    This class is the boundary between file I/O / PDF tooling and the domain
    ``Resume`` model: it always attaches ``filename`` and ``raw_text`` and
    leaves structured slots empty for the LLM pipeline to fill consistently.
    """

    def __init__(self, parser: PDFParser) -> None:
        """Store the parser used for every ``parse`` / ``parse_all`` call.

        Dependency injection keeps this class testable (inject a stub parser)
        and aligned with the strategy pattern used by ``PDFParser.build``.

        Args:
            parser: A concrete ``PDFParser`` implementation.

        Returns:
            None.

        Side effects:
            Stores ``parser`` on ``self._parser``.
        """
        self._parser = parser

    def parse(self, pdf_path: str) -> Resume:
        """Parse a single PDF into a ``Resume`` with raw text only.

        Structured fields are intentionally ``None`` or empty lists here so a
        single downstream extraction step owns name/skills/education parsing;
        that avoids duplicating heuristics between PDF parsers and keeps
        resumes comparable regardless of PDF backend.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A ``Resume`` instance with ``filename``, ``raw_text`` set and other
            fields left for later population.

        Side effects:
            Invokes ``self._parser.extract_text`` (may log inside the parser).
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
        """Parse every ``*.pdf`` in a directory into ``Resume`` instances.

        Uses a sorted glob so runs are deterministic across OSes and easier
        to debug (same order in logs and shortlists). Individual failures do
        not abort the whole batch: recruiters still get partial results when
        one corrupt file exists.

        Args:
            folder_path: Path to the directory containing PDF files.

        Returns:
            List of successfully parsed ``Resume`` instances (may be empty if
            no PDFs exist or all parses fail).

        Side effects:
            Logs warning when the folder has no PDFs, info per successful
            parse (word count as a quick quality signal), and warning when a
            file is skipped.

        Note:
            Only ``RuntimeError`` and ``NotImplementedError`` from ``parse``
            are caught so unexpected bugs (e.g. ``KeyboardInterrupt``) still
            propagate; those two cover parser failures and unfinished backends.
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
