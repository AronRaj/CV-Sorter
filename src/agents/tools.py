"""LangChain @tool wrappers around existing src/ modules.

Each tool is a thin adapter — the real logic lives in src/.
No src/ module is modified or monkey-patched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@tool
def parse_resume_tool(pdf_path: str) -> str:
    """Parse a PDF resume file and return its text content as markdown.

    Input is the full file path to the PDF.
    """
    logger.info("parse_resume_tool called with: %s", pdf_path)
    try:
        from src.core.config import load_config
        from src.core.parser_engine import PDFParser, ResumeParser

        config = load_config()
        parser = PDFParser.build(config.pdf_parser)
        rp = ResumeParser(parser=parser)
        resume = rp.parse(pdf_path)
        return resume.raw_text
    except Exception as exc:
        raise RuntimeError(f"parse_resume_tool failed on '{pdf_path}': {exc}") from exc


@tool
def extract_jd_tool(jd_text: str) -> str:
    """Extract structured fields from a job description text.

    Input is the raw text of the job description.
    Returns a JSON string with job_title, required_skills,
    nice_to_have_skills, min_experience_years, and responsibilities.
    """
    logger.info("extract_jd_tool called (%d chars)", len(jd_text))
    try:
        from langchain_core.messages import HumanMessage

        from src.agents.model_factory import get_ollama_jd_model

        template = (PROMPTS_DIR / "extract_jd.txt").read_text()
        prompt = template.format(raw_jd_text=jd_text)

        model = get_ollama_jd_model()
        response = model.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as exc:
        raise RuntimeError(f"extract_jd_tool failed: {exc}") from exc


@tool
def score_resume_tool(resume_text: str, jd_json: str) -> str:
    """Score a candidate resume against a job description.

    resume_text is the parsed resume markdown.
    jd_json is the JSON string of the structured job description.
    Returns a JSON string with overall_score, fit_label, explanation,
    and requirement_scores.
    """
    logger.info("score_resume_tool called (%d chars resume)", len(resume_text))
    try:
        from langchain_core.messages import HumanMessage

        from src.agents.model_factory import get_claude_model
        from src.core.config import load_config

        config = load_config()
        jd = json.loads(jd_json)

        template = (PROMPTS_DIR / "score_candidate.txt").read_text()
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


@tool
def quick_scan_tool(resume_text: str, jd_summary: str) -> str:
    """Quickly scan a resume against a job description summary to decide if it is worth deep-scoring.

    Returns a JSON string with: decision (PROCEED or SKIP),
    confidence (0-100), reason (string).
    """
    logger.info("quick_scan_tool called (%d chars resume)", len(resume_text))
    try:
        from langchain_core.messages import HumanMessage

        from src.agents.model_factory import get_ollama_shortlist_model

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
