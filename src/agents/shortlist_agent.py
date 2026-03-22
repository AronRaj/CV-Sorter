"""Shortlist agent — fast first-pass filter using a local LLM.

Uses Ollama llama3.1:8b via ChatOllama for fast local inference.
No API key required. Requires Ollama running on localhost:11434.

Screens all parsed resumes against the job description and decides
which candidates are worth the cost of deep-scoring with a cloud
API model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.core.models import JobDescription, Resume

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class ShortlistAgent:
    """Fast-pass filter agent using a local LLM.

    Decides which resumes are worth deep-scoring.
    Calls the quick-scan prompt directly via the injected model
    rather than going through the @tool wrapper, keeping the
    agent self-contained and testable.
    """

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model
        self._template = (PROMPTS_DIR / "quick_scan.txt").read_text()

    def run(
        self,
        resumes: list[Resume],
        jd: JobDescription,
    ) -> tuple[list[Resume], list[str]]:
        """Screen all resumes and split into shortlisted vs skipped.

        Args:
            resumes: All parsed Resume objects.
            jd:      The structured JobDescription.

        Returns:
            A tuple of (shortlisted, skipped) where shortlisted is
            the list of Resume objects to deep-score and skipped is
            a list of human-readable strings like
            "filename.pdf — reason text".
        """
        jd_summary = self._build_jd_summary(jd)
        shortlisted: list[Resume] = []
        skipped: list[str] = []

        for resume in resumes:
            try:
                result = self._scan_one(
                    resume_text=resume.raw_text,
                    jd_summary=jd_summary,
                )
                decision = result.get("decision", "PROCEED").upper()
                reason = result.get("reason", "no reason given")

                if decision == "SKIP":
                    skipped.append(f"{resume.filename} — {reason}")
                    logger.info(
                        "[Shortlist]   %s → SKIP    (%s)", resume.filename, reason
                    )
                else:
                    shortlisted.append(resume)
                    logger.info(
                        "[Shortlist]   %s → PROCEED (%s)", resume.filename, reason
                    )

            except Exception:
                logger.warning(
                    "[Shortlist]   %s → PROCEED (tool error, fail-safe)",
                    resume.filename,
                    exc_info=True,
                )
                shortlisted.append(resume)

        return shortlisted, skipped

    def _scan_one(self, resume_text: str, jd_summary: str) -> dict:
        """Run the quick-scan prompt for a single resume and parse the JSON response."""
        prompt = self._template.format(
            jd_summary=jd_summary,
            resume_text=resume_text,
        )
        response = self._model.invoke([HumanMessage(content=prompt)])
        return json.loads(response.content)

    def _build_jd_summary(self, jd: JobDescription) -> str:
        """Build a short JD summary string for the quick scan prompt."""
        skills = ", ".join(jd.required_skills[:5])
        return (
            f"Role: {jd.job_title}. "
            f"Required skills: {skills}. "
            f"Min experience: {jd.min_experience_years or 'not specified'} years."
        )
