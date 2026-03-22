"""Scorer agent — deep-scores shortlisted candidates with self-evaluation.

Uses Claude Sonnet (claude-sonnet-4-5) via the Anthropic API.
Requires ANTHROPIC_API_KEY in .env.

Produces detailed per-requirement scores, then reviews its own
evidence quality and retries once if any high score lacks specific
evidence.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.core.models import CandidateScore, JobDescription, RequirementScore, Resume
from src.core.scorer_engine import Scorer as CoreScorer

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class ScorerAgent:
    """Deep-scoring agent with self-evaluation loop.

    Uses an API model for quality scoring.  After each score, the agent
    reviews its own evidence via prompts/self_eval.txt and re-scores
    once if the evidence is weak or generic.
    """

    MAX_RETRIES: int = 1
    MAX_JSON_PARSE_ATTEMPTS: int = 3

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model
        self._score_template = (PROMPTS_DIR / "score_candidate.txt").read_text()
        self._eval_template = (PROMPTS_DIR / "self_eval.txt").read_text()

    def run(
        self,
        resumes: list[Resume],
        jd: JobDescription,
    ) -> list[CandidateScore]:
        """Score each resume with self-evaluation, return results sorted by score DESC.

        Args:
            resumes: Shortlisted Resume objects to deep-score.
            jd:      The structured JobDescription.

        Returns:
            List of CandidateScore sorted by overall_score descending.
        """
        jd_dict = self._jd_to_dict(jd)
        results: list[CandidateScore] = []

        for resume in resumes:
            score = self._score_one(resume=resume, jd=jd, jd_dict=jd_dict)
            results.append(score)

        results.sort(key=lambda cs: cs.overall_score, reverse=True)
        return results

    def _score_one(
        self,
        resume: Resume,
        jd: JobDescription,
        jd_dict: dict,
    ) -> CandidateScore:
        """Score a single resume, self-evaluate, and retry if needed."""
        raw = self._call_score(resume=resume, jd=jd)
        candidate: CandidateScore | None = None
        for attempt in range(self.MAX_JSON_PARSE_ATTEMPTS):
            try:
                candidate = self._parse_score_response(raw=raw, resume=resume)
                break
            except (RuntimeError, KeyError) as exc:
                if attempt + 1 >= self.MAX_JSON_PARSE_ATTEMPTS:
                    logger.warning(
                        "[Scorer]      %s — agent parse exhausted (%s); "
                        "falling back to core Scorer",
                        resume.filename,
                        exc,
                    )
                    try:
                        candidate = CoreScorer(model=self._model).score(
                            resume=resume,
                            jd=jd,
                        )
                    except (RuntimeError, KeyError) as fallback_exc:
                        logger.warning(
                            "[Scorer]      %s — core Scorer fallback also "
                            "failed (%s); using lenient parse",
                            resume.filename,
                            fallback_exc,
                        )
                        candidate = self._lenient_parse(raw=raw, resume=resume)
                    break
                logger.warning(
                    "[Scorer]      %s — JSON parse failed (attempt %d/%d): %s",
                    resume.filename,
                    attempt + 1,
                    self.MAX_JSON_PARSE_ATTEMPTS,
                    exc,
                )
                raw = self._retry_score_after_bad_json(
                    resume=resume,
                    jd=jd,
                    failed_response=raw,
                )
        if candidate is None:
            raise RuntimeError(
                f"Scorer failed to produce a score for '{resume.filename}'"
            )

        quality = self._self_evaluate(candidate)

        retries = 0
        while quality == "NEEDS_RETRY" and retries < self.MAX_RETRIES:
            retries += 1
            logger.info(
                "[Scorer]      %s — evidence weak, retrying...",
                resume.filename,
            )
            raw = self._call_score(
                resume=resume,
                jd=jd,
                stricter=True,
            )
            candidate = self._parse_score_response(raw=raw, resume=resume)
            quality = self._self_evaluate(candidate)

        eval_note = "evidence OK" if quality == "GOOD" else f"retried {retries}x"
        logger.info(
            "[Scorer]      %s → scored %d/100  (self-eval: %s)",
            resume.filename,
            candidate.overall_score,
            eval_note,
        )
        return candidate

    def _call_score(
        self,
        resume: Resume,
        jd: JobDescription,
        stricter: bool = False,
    ) -> str:
        """Fill the scoring prompt and invoke the model."""
        prompt = self._score_template.format(
            job_title=jd.job_title,
            required_skills="\n".join(f"- {s}" for s in jd.required_skills),
            nice_to_have_skills="\n".join(f"- {s}" for s in jd.nice_to_have_skills),
            responsibilities="\n".join(f"- {r}" for r in jd.responsibilities),
            resume_text=resume.raw_text,
        )
        if stricter:
            prompt += (
                "\n\nIMPORTANT: For every requirement score, the evidence field "
                "MUST cite a specific section, skill, project, or metric from the "
                "resume. Do NOT use generic phrases like 'candidate has experience'. "
                "Quote or closely paraphrase the resume."
            )
        response = self._model.invoke([HumanMessage(content=prompt)])
        return response.content

    def _retry_score_after_bad_json(
        self,
        resume: Resume,
        jd: JobDescription,
        failed_response: str,
    ) -> str:
        """Ask the model again after a JSON parse failure.

        Args:
            resume: The resume being scored.
            jd: Job description.
            failed_response: The previous model output that did not parse.

        Returns:
            Raw string response from the model.

        Raises:
            RuntimeError: If the retry response still cannot be parsed.
        """
        base_prompt = self._score_template.format(
            job_title=jd.job_title,
            required_skills="\n".join(f"- {s}" for s in jd.required_skills),
            nice_to_have_skills="\n".join(f"- {s}" for s in jd.nice_to_have_skills),
            responsibilities="\n".join(f"- {r}" for r in jd.responsibilities),
            resume_text=resume.raw_text,
        )
        repair_prompt = (
            "Your previous response was not valid JSON and could not be parsed.\n"
            f"Broken output (first 400 chars): {failed_response[:400]!r}\n\n"
            "Reply with ONLY a single valid JSON object. No markdown fences, "
            "no commentary before or after. Escape any double quotes inside "
            "string values.\n\n"
            f"{base_prompt}"
        )
        response = self._model.invoke([HumanMessage(content=repair_prompt)])
        return response.content

    def _self_evaluate(self, result: CandidateScore) -> str:
        """Ask the model to review the scoring quality.

        Args:
            result: The CandidateScore to evaluate.

        Returns:
            "GOOD" or "NEEDS_RETRY".  On any error, returns "GOOD"
            (fail-safe — self-eval must never block the pipeline).
        """
        try:
            req_text = "\n".join(
                f"- {rs.requirement}: {rs.score}/100 — {rs.evidence}"
                for rs in result.requirement_scores
            )
            prompt = self._eval_template.format(
                overall_score=result.overall_score,
                fit_label=result.fit_label,
                requirement_scores_text=req_text,
            )
            response = self._model.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)

            quality = data.get("quality", "GOOD")
            weak = data.get("weak_requirements", [])
            if weak:
                logger.info("  [Self-eval] Weak requirements: %s", weak)
            return quality

        except Exception:
            logger.warning(
                "  [Self-eval] Parse error for %s, assuming GOOD (fail-safe)",
                result.resume.filename,
                exc_info=True,
            )
            return "GOOD"

    def _parse_score_response(self, raw: str, resume: Resume) -> CandidateScore:
        """Parse the LLM's JSON response into a CandidateScore dataclass.

        Args:
            raw:    Raw string response from the model.
            resume: The Resume being scored.

        Returns:
            A populated CandidateScore.

        Raises:
            RuntimeError: If the response cannot be parsed.
        """
        cleaned = self._strip_code_fences(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Scorer JSON parse failed for '{resume.filename}': {exc}"
            ) from exc

        if "overall_score" not in data:
            raise RuntimeError(
                f"Missing 'overall_score' in LLM response for '{resume.filename}'"
            )

        req_scores = [
            RequirementScore(
                requirement=item.get("requirement", "Unknown"),
                score=int(item.get("score", 0)),
                evidence=item.get("evidence", "No evidence provided"),
            )
            for item in data.get("requirement_scores", [])
        ]

        score = int(data["overall_score"])
        fit_label = data.get("fit_label", self._derive_fit_label(score))
        explanation = data.get(
            "explanation",
            data.get("summary", f"Scored {score}/100 for {resume.filename}"),
        )

        return CandidateScore(
            resume=resume,
            overall_score=score,
            fit_label=fit_label,
            explanation=explanation,
            requirement_scores=req_scores,
        )

    def _strip_code_fences(self, text: str) -> str:
        """Remove markdown code fences from LLM output."""
        stripped = text.strip()
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
        return stripped.strip()

    @staticmethod
    def _derive_fit_label(score: int) -> str:
        """Derive fit label from a 0-100 score when the LLM omits it."""
        if score >= 80:
            return "Strong match"
        elif score >= 60:
            return "Good match"
        elif score >= 40:
            return "Partial match"
        return "Weak match"

    def _lenient_parse(self, raw: str, resume: Resume) -> CandidateScore:
        """Best-effort parse when strict parsing and all retries have failed.

        Extracts whatever fields are present and fills in defaults for the rest.
        Only raises if there is no parseable JSON at all.
        """
        cleaned = self._strip_code_fences(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return CandidateScore(
                resume=resume,
                overall_score=0,
                fit_label="Weak match",
                explanation="Could not parse scorer response.",
                requirement_scores=[],
            )

        score = int(data.get("overall_score", 0))
        return CandidateScore(
            resume=resume,
            overall_score=score,
            fit_label=data.get("fit_label", self._derive_fit_label(score)),
            explanation=data.get(
                "explanation",
                data.get("summary", f"Scored {score}/100 for {resume.filename}"),
            ),
            requirement_scores=[
                RequirementScore(
                    requirement=item.get("requirement", "Unknown"),
                    score=int(item.get("score", 0)),
                    evidence=item.get("evidence", "No evidence provided"),
                )
                for item in data.get("requirement_scores", [])
            ],
        )

    def _jd_to_dict(self, jd: JobDescription) -> dict:
        """Convert a JobDescription to a plain dict for JSON serialisation."""
        return {
            "job_title": jd.job_title,
            "required_skills": jd.required_skills,
            "nice_to_have_skills": jd.nice_to_have_skills,
            "min_experience_years": jd.min_experience_years,
            "responsibilities": jd.responsibilities,
        }
