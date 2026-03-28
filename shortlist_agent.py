"""Shortlist agent — fast first-pass filter using a local LLM.

**Role in the multi-agent pipeline**

This agent sits **between parsing and scoring**. The supervisor passes every
`Resume` plus a structured `JobDescription`. For each resume, the agent asks a
local model (Ollama llama3.1:8b via `ChatOllama`) for a JSON decision:
PROCEED (send to Claude for deep scoring) or SKIP (drop from the expensive
stage). That trade-off exists because scoring N resumes on a cloud API is
costly and slow; shortlisting is deliberately "good enough" gatekeeping.

**Why local / no API key**

Batch size is often large; running hundreds of cheap local calls dominates
running dozens of API calls. No Anthropic key is required for this stage,
which also makes the pipeline usable in air-gapped or budget-constrained demos.

**Implementation note**

The agent calls the model **directly** with `HumanMessage` and the quick-scan
template instead of wrapping the call in a LangChain `@tool`. That keeps the
agent a small, testable unit (inject a fake `BaseChatModel`) and avoids
executor overhead for a single deterministic prompt per resume.

**Operational requirements**

Ollama must be listening (default localhost:11434) with the configured model
pulled; otherwise `invoke` fails and the per-resume fail-safe below applies.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from models import JobDescription, Resume

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class ShortlistAgent:
    """Fast-pass filter: split resumes into shortlisted vs skipped with reasons.

    Consumes the same domain types as downstream agents (`Resume`, `JobDescription`)
    so the supervisor can pass references through without conversion. Output is
    a tuple of concrete lists: full `Resume` objects for scoring, and human-readable
    skip strings for logging or UI.
    """

    def __init__(self, model: BaseChatModel) -> None:
        """Attach the chat model and load the quick-scan prompt template from disk.

        Args:
            model: Typically `ChatOllama` from `get_ollama_shortlist_model()`;
                any `BaseChatModel` with `invoke` works for tests.

        Side effects:
            Reads ``prompts/quick_scan.txt`` at construction time (fail-fast if
            missing or unreadable).
        """
        self._model = model
        self._template = (PROMPTS_DIR / "quick_scan.txt").read_text()

    def run(
        self,
        resumes: list[Resume],
        jd: JobDescription,
    ) -> tuple[list[Resume], list[str]]:
        """Screen every resume; collect PROCEED vs SKIP with skip explanations.

        Iteration is sequential by design: local Ollama is often single-GPU;
        parallelizing here could saturate VRAM or the daemon. Each iteration is
        independent, so failures on one file do not abort the batch.

        Args:
            resumes: All parsed resumes from the folder (same order as parsing).
            jd: Structured job description used to build a compact JD summary
                for the prompt (not the full raw JD text — keeps context small).

        Returns:
            A tuple ``(shortlisted, skipped)``:
            - ``shortlisted``: `Resume` instances to pass to `ScorerAgent`.
            - ``skipped``: Strings like ``"file.pdf — reason"`` for audit trails.

        Side effects:
            Logs one line per resume (PROCEED/SKIP) and warnings on tool/model
            errors. Does not write files.

        **Fail-safe:** On any exception during `_scan_one`, the resume is treated
        as PROCEED. Rationale: false negatives (dropping a good candidate) are
        worse for recruiters than an extra API score; transient Ollama glitches
        should not empty the pipeline.
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
                # Default to PROCEED if the model omits the key — conservative
                # for the same reason as the exception path: avoid silent drops.
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
        """Invoke the quick-scan prompt for one resume and parse JSON from the reply.

        The model is instructed to return JSON; we `json.loads` the entire
        `content` string. If the model wraps prose around JSON or returns
        invalid JSON, the caller's `run()` exception handler promotes PROCEED.

        Args:
            resume_text: Full extracted text of the PDF resume.
            jd_summary: Compact string built from `JobDescription` fields.

        Returns:
            Parsed dict expected to contain at least ``decision`` and ``reason``.

        Raises:
            json.JSONDecodeError, TypeError, or other errors if output is not
            parseable as JSON — propagated to `run()` for fail-safe handling.

        Side effects:
            One synchronous LLM `invoke` (network/local inference).
        """
        prompt = self._template.format(
            jd_summary=jd_summary,
            resume_text=resume_text,
        )
        response = self._model.invoke([HumanMessage(content=prompt)])
        return json.loads(response.content)

    def _build_jd_summary(self, jd: JobDescription) -> str:
        """Compress the JD into a short line for the quick-scan context window.

        Full JD + full resume can exceed small local context limits or dilute
        attention. We cap displayed required skills to five so the model focuses
        on headline fit; deep scoring later sees the full structured JD.

        Args:
            jd: Parsed job description.

        Returns:
            A single human-readable summary string embedded in the prompt.

        Side effects:
            None.
        """
        skills = ", ".join(jd.required_skills[:5])
        return (
            f"Role: {jd.job_title}. "
            f"Required skills: {skills}. "
            f"Min experience: {jd.min_experience_years or 'not specified'} years."
        )
