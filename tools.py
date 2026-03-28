"""LangChain @tool wrappers around core CV-Sorter capabilities.

This module exposes parsing, job-description extraction, scoring, and
shortlist pre-filtering as LangChain tools so agents (or an executor) can
invoke them by name with string inputs/outputs. Each wrapper delegates to
engines or LLM factories; the tools stay thin so business logic remains
testable and reusable outside LangChain.

Imports of ``config``, ``parser_engine``, and ``model_factory`` are deferred
until each tool runs so importing this module does not pull in optional or
heavy dependencies until a tool is actually called.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Prompt templates live next to the project root (sibling of this file) so
# agents and the Streamlit app share one canonical copy of wording.
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


# ---------------------------------------------------------------------------
# Resume parsing
# ---------------------------------------------------------------------------


@tool
def parse_resume_tool(pdf_path: str) -> str:
    """Extract plain text from a PDF resume using the configured PDF backend.

    Builds a ``ResumeParser`` around the project's ``PDFParser`` (PyMuPDF,
    pdfplumber, etc., per ``load_config()``) and returns the raw extracted
    string that downstream scoring and JD tools expect as markdown-ish text.

    Args:
        pdf_path: Absolute or relative filesystem path to the candidate's
            ``.pdf`` file. Must be readable by the process running the tool.

    Returns:
        The resume body as a single string (``Resume.raw_text``), formatted
        as markdown-style text for downstream LLM prompts without re-reading
        the file.

    Raises:
        RuntimeError: Wraps any failure (missing file, corrupt PDF, parser
            misconfiguration) with context so agent logs show which path failed.
            Chains the original exception via ``from exc`` for tracebacks.

    Note:
        Side effects: reads the PDF from disk; may log at INFO when invoked.
        Does not persist output; callers own caching or file writes.
    """
    logger.info("parse_resume_tool called with: %s", pdf_path)
    try:
        # Lazy import keeps startup light and avoids import cycles with
        # packages that might eventually import tools.
        from config import load_config
        from parser_engine import PDFParser, ResumeParser

        config = load_config()
        # Parser implementation is swappable via config so the same tool
        # works across environments without code changes here.
        parser = PDFParser.build(config.pdf_parser)
        rp = ResumeParser(parser=parser)
        resume = rp.parse(pdf_path)
        return resume.raw_text
    except Exception as exc:
        # Agents only see tool-level errors; re-raise with a stable prefix
        # so supervisors can pattern-match or log failures consistently.
        raise RuntimeError(f"parse_resume_tool failed on '{pdf_path}': {exc}") from exc


# ---------------------------------------------------------------------------
# Job description structuring (LLM)
# ---------------------------------------------------------------------------


@tool
def extract_jd_tool(jd_text: str) -> str:
    """Turn unstructured job-description prose into a structured JSON string.

    Uses the Ollama-backed JD model (see ``get_ollama_jd_model``) and the
    ``extract_jd.txt`` prompt so the LLM emits fields the scorer expects:
    job title, required vs nice-to-have skills, experience floor, and
    responsibilities. Callers typically ``json.loads`` the result before
    passing slices into ``score_resume_tool``.

    Args:
        jd_text: Full raw text of the posting (paste from web or file).
            Empty or very short input may yield weak structure; validation
            is the caller's responsibility.

    Returns:
        Model output as a string. Conventionally valid JSON matching the
        prompt's schema, but still treated as opaque text here—parse
        defensively downstream.

    Raises:
        RuntimeError: Prompt read failure, model errors, or network issues
            talking to Ollama, wrapped for uniform agent error handling.

    Note:
        Side effects: INFO log with character count; network I/O to the
        local Ollama endpoint when the model is invoked.
    """
    logger.info("extract_jd_tool called (%d chars)", len(jd_text))
    try:
        from langchain_core.messages import HumanMessage

        from model_factory import get_ollama_jd_model

        template = (PROMPTS_DIR / "extract_jd.txt").read_text()
        # Single-shot user message: template embeds the JD so the model
        # cannot confuse system vs user boundaries across turns.
        prompt = template.format(raw_jd_text=jd_text)

        model = get_ollama_jd_model()
        response = model.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as exc:
        raise RuntimeError(f"extract_jd_tool failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Full scoring (Claude + structured JD)
# ---------------------------------------------------------------------------


@tool
def score_resume_tool(resume_text: str, jd_json: str) -> str:
    """Produce a detailed fit score for one resume against a structured JD.

    Merges JD fields into ``score_candidate.txt`` and invokes the configured
    Claude model so scoring can use a stronger model than the lightweight
    Ollama steps used for extraction or shortlisting. Output is intended
    for ranking UI and narrative explanations to recruiters.

    Args:
        resume_text: Parsed resume text (e.g. from ``parse_resume_tool``).
        jd_json: JSON string of the structured job description, typically
            from ``extract_jd_tool``. Keys use ``.get`` with defaults so
            partially filled JDs still produce a prompt instead of failing
            on KeyError.

    Returns:
        Model response string—by convention JSON with overall_score,
        fit_label, explanation, and per-requirement scores. Parsing is
        left to the caller.

    Raises:
        RuntimeError: Includes ``json.JSONDecodeError`` if ``jd_json`` is
            not valid JSON, or any Claude/API error during ``invoke``.

    Note:
        Side effects: loads app config for API keys and model id; network
        call to the Claude provider; INFO logging with resume length only.
    """
    logger.info("score_resume_tool called (%d chars resume)", len(resume_text))
    try:
        from langchain_core.messages import HumanMessage

        from model_factory import get_claude_model
        from config import load_config

        config = load_config()
        jd = json.loads(jd_json)

        template = (PROMPTS_DIR / "score_candidate.txt").read_text()
        # Join list fields into comma-separated strings so the prompt stays
        # a flat block of text the scorer model can scan in one pass.
        prompt = template.format(
            job_title=jd.get("job_title", ""),
            required_skills=", ".join(jd.get("required_skills", [])),
            nice_to_have_skills=", ".join(jd.get("nice_to_have_skills", [])),
            responsibilities=", ".join(jd.get("responsibilities", [])),
            resume_text=resume_text,
        )

        model = get_claude_model(config)
        response = model.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as exc:
        raise RuntimeError(f"score_resume_tool failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Cheap pre-filter before full scoring
# ---------------------------------------------------------------------------


@tool
def quick_scan_tool(resume_text: str, jd_summary: str) -> str:
    """Gate expensive scoring: decide PROCEED vs SKIP from a short JD summary.

    Uses a dedicated Ollama shortlist model and ``quick_scan.txt`` so many
    resumes can be triaged locally with lower latency/cost than Claude full
    scores. The summary should distill must-haves; the model returns JSON
    with decision, confidence, and reason for auditability.

    Args:
        resume_text: Same shape as for full scoring—parsed resume body.
        jd_summary: Condensed JD (title + key asks), not necessarily the
            full structured JSON; keeps the pre-scan prompt small and fast.

    Returns:
        Model output string—expected JSON with decision, confidence (0–100),
        and reason. Callers parse JSON to branch the pipeline.

    Raises:
        RuntimeError: Template or model failures, wrapped like other tools.

    Note:
        Side effects: INFO log; local Ollama inference. Does not read files
        or mutate persistent state.
    """
    logger.info("quick_scan_tool called (%d chars resume)", len(resume_text))
    try:
        from langchain_core.messages import HumanMessage

        from model_factory import get_ollama_shortlist_model

        template = (PROMPTS_DIR / "quick_scan.txt").read_text()
        prompt = template.format(
            jd_summary=jd_summary,
            resume_text=resume_text,
        )

        model = get_ollama_shortlist_model()
        response = model.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as exc:
        raise RuntimeError(f"quick_scan_tool failed: {exc}") from exc
