"""Resume scoring engine.

Orchestrates LLM-based evaluation of a parsed resume against a structured job
description. The model is instructed via ``prompts/score_candidate.txt`` to
return a JSON object; this module normalises that output into a
``CandidateScore`` (including per-requirement breakdowns).

Because production LLMs sometimes emit markdown-wrapped or malformed JSON, the
engine strips common wrappers and performs **one** automatic retry with a
stricter instruction before surfacing a hard failure. That trade-off balances
resilience against unbounded cost or latency from repeated calls.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from models import CandidateScore, JobDescription, RequirementScore, Resume

logger = logging.getLogger(__name__)

# Resolved relative to the process working directory (same convention as callers).
PROMPT_TEMPLATE_PATH = Path("prompts/score_candidate.txt")


class Scorer:
    """Scores a candidate resume against a job description using a chat model.

    The scoring contract is defined by the external prompt template: the model
    must return JSON with at least ``overall_score`` and optionally
    ``requirement_scores``, ``fit_label``, and narrative fields. This class does
    not implement rubrics itself—it **delegates** judgement to the LLM and only
    validates structure enough to build typed domain objects.

    Side effects: **network I/O** to the configured provider on ``score`` (and
    again on retry), plus logging at INFO/WARNING levels.
    """

    def __init__(self, model: BaseChatModel) -> None:
        """Attach a LangChain chat model used for all subsequent ``score`` calls.

        Args:
            model: Any ``BaseChatModel`` implementation (Claude, OpenAI, Gemini,
                etc.). Must be configured with API keys and options by the caller.

        Returns:
            None

        Side effects:
            Stores a reference to ``model`` on this instance; does not invoke the
            model until ``score`` is called.
        """
        self._model = model

    def score(self, resume: Resume, jd: JobDescription) -> CandidateScore:
        """Score one resume against one job description.

        Builds the user message from the JD fields and raw resume text, invokes
        the model once, then parses the reply. If ``json.loads`` fails after
        fence-stripping (common when the model adds prose or broken JSON), a
        **single** retry is attempted with explicit JSON-only instructions so
        downstream pipelines still get a structured result when possible.

        Args:
            resume: Parsed resume including ``raw_text`` and ``filename`` (used
                in logs and error messages).
            jd: Structured job description whose lists are formatted as bullet
                lines in the prompt.

        Returns:
            A ``CandidateScore`` linking the resume to scores and evidence.

        Raises:
            RuntimeError: If JSON parsing or schema checks fail after the retry
                path, or if the retry itself raises.

        Side effects:
            Logs at INFO when scoring starts; at WARNING if the first parse
            fails. Performs one LLM invocation, or two if retrying.
        """
        template = self._read_prompt_template()
        # Bullet lists keep the prompt readable for the model and mirror how JDs are authored.
        prompt = template.format(
            job_title=jd.job_title,
            required_skills="\n".join(f"- {s}" for s in jd.required_skills),
            nice_to_have_skills="\n".join(f"- {s}" for s in jd.nice_to_have_skills),
            responsibilities="\n".join(f"- {r}" for r in jd.responsibilities),
            resume_text=resume.raw_text,
        )

        logger.info("Scoring '%s'...", resume.filename)
        raw_response = self._model.invoke([HumanMessage(content=prompt)]).content

        try:
            return self._parse_response(raw=raw_response, resume=resume)
        except json.JSONDecodeError:
            # Do not retry on RuntimeError (e.g. missing keys)—those imply wrong shape, not just syntax.
            logger.warning(
                "First parse attempt failed for %s, retrying...",
                resume.filename,
            )
            return self._retry_score(
                resume=resume,
                jd=jd,
                failed_response=raw_response,
            )

    def _parse_response(self, raw: str, resume: Resume) -> CandidateScore:
        """Turn the model's string output into a ``CandidateScore``.

        Expects JSON (optionally wrapped in markdown fences). Missing optional
        fields are back-filled so callers always get a consistent object: fit
        label can be inferred from the numeric score, and explanation can fall
        back to a short default or legacy ``summary`` key from older prompts.

        Args:
            raw: Untrimmed model output string.
            resume: The resume being scored; used for error context and stored
                on the returned ``CandidateScore``.

        Returns:
            A fully constructed ``CandidateScore`` including
            ``requirement_scores`` (possibly empty).

        Raises:
            json.JSONDecodeError: If content is not valid JSON after fence removal.
            RuntimeError: If ``overall_score`` is absent (cannot rank without it).

        Side effects:
            None beyond Python object allocation.
        """
        cleaned = self._strip_code_fences(raw)
        data = json.loads(cleaned)

        # ``overall_score`` is the only mandatory field; everything else degrades gracefully.
        if "overall_score" not in data:
            raise RuntimeError(
                f"Missing 'overall_score' in LLM response for '{resume.filename}'. "
                f"Response preview: {raw[:200]}"
            )

        score = int(data["overall_score"])

        # Per-item defaults avoid KeyError when the model omits keys on sparse rows.
        requirement_scores = [
            RequirementScore(
                requirement=item.get("requirement", "Unknown"),
                score=int(item.get("score", 0)),
                evidence=item.get("evidence", "No evidence provided"),
            )
            for item in data.get("requirement_scores", [])
        ]

        return CandidateScore(
            resume=resume,
            overall_score=score,
            fit_label=data.get("fit_label", self._derive_fit_label(score)),
            explanation=data.get(
                "explanation",
                data.get("summary", f"Scored {score}/100 for {resume.filename}"),
            ),
            requirement_scores=requirement_scores,
        )

    @staticmethod
    def _derive_fit_label(score: int) -> str:
        """Map a 0–100 overall score to a coarse fit band when the model omits ``fit_label``.

        Thresholds are product choices for reporting (not statistical): they
        give hiring stakeholders a quick verbal bucket aligned with the numeric
        score.

        Args:
            score: Integer overall score from the model.

        Returns:
            One of: ``Strong match``, ``Good match``, ``Partial match``,
            ``Weak match``.

        Raises:
            None

        Side effects:
            None
        """
        if score >= 80:
            return "Strong match"
        elif score >= 60:
            return "Good match"
        elif score >= 40:
            return "Partial match"
        return "Weak match"

    def _retry_score(
        self,
        resume: Resume,
        jd: JobDescription,
        failed_response: str,
    ) -> CandidateScore:
        """Second scoring attempt after JSON decode failure on the first reply.

        Prepends a short diagnostic prefix (truncated prior output) so the
        model can self-correct, then repeats the **full** original task prompt.
        We deliberately resend the entire JD and resume rather than only asking
        for JSON repair, because partial re-prompts sometimes drop required
        context and produce inconsistent scores.

        Args:
            resume: Resume under evaluation.
            jd: Job description (same as the first attempt).
            failed_response: Raw first response, used only in the retry prefix
                for debugging/model guidance.

        Returns:
            ``CandidateScore`` from parsing the retry response.

        Raises:
            RuntimeError: If the retry still fails JSON parsing or validation,
                wrapping the underlying exception for a single clear message.

        Side effects:
            One additional LLM invocation; INFO log line.
        """
        template = self._read_prompt_template()
        original_prompt = template.format(
            job_title=jd.job_title,
            required_skills="\n".join(f"- {s}" for s in jd.required_skills),
            nice_to_have_skills="\n".join(f"- {s}" for s in jd.nice_to_have_skills),
            responsibilities="\n".join(f"- {r}" for r in jd.responsibilities),
            resume_text=resume.raw_text,
        )

        # Truncate the failed blob to keep the retry prompt bounded for token limits.
        retry_prompt = (
            "Your previous response could not be parsed as JSON.\n"
            f"Previous response (first 300 chars): {failed_response[:300]}\n\n"
            "You MUST respond with ONLY a valid JSON object.\n"
            "No explanation, no markdown, no code fences. Just the raw JSON.\n\n"
            f"{original_prompt}"
        )

        logger.info("Retry scoring '%s' with stricter prompt...", resume.filename)
        raw_response = self._model.invoke([HumanMessage(content=retry_prompt)]).content

        try:
            return self._parse_response(raw=raw_response, resume=resume)
        except (json.JSONDecodeError, RuntimeError) as e:
            raise RuntimeError(
                f"Scorer failed twice for {resume.filename}. "
                f"Check prompts/score_candidate.txt and the LLM response."
            ) from e

    def _read_prompt_template(self) -> str:
        """Load the scoring prompt template from disk.

        The path is module-level and relative to the current working directory,
        so CLI and app entrypoints should start from the project root (or set
        cwd accordingly).

        Args:
            None

        Returns:
            The template string containing ``str.format`` placeholders for JD
            and resume fields.

        Raises:
            FileNotFoundError: If ``prompts/score_candidate.txt`` is missing.

        Side effects:
            Disk read only.
        """
        if not PROMPT_TEMPLATE_PATH.exists():
            raise FileNotFoundError(
                f"Scoring prompt template not found at '{PROMPT_TEMPLATE_PATH}'. "
                f"Create this file before running the scorer."
            )
        return PROMPT_TEMPLATE_PATH.read_text()

    def _strip_code_fences(self, text: str) -> str:
        """Remove common markdown code fences so ``json.loads`` can run on bare JSON.

        Models frequently wrap JSON in ``` or ```json blocks despite instructions
        not to; stripping avoids failing the entire scoring pipeline for that
        formatting habit.

        Args:
            text: Raw assistant message content.

        Returns:
            Inner JSON text with leading/trailing fence lines removed when
            present.

        Raises:
            None

        Side effects:
            None
        """
        stripped = text.strip()
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
        return stripped.strip()
