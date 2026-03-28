"""Scorer agent — deep-scores shortlisted candidates with self-evaluation.

**Role in the multi-agent pipeline**

`Supervisor` passes only **shortlisted** `Resume` objects plus `JobDescription`.
This agent produces a `CandidateScore` per resume: overall score, fit label,
narrative explanation, and per-requirement scores with evidence strings. Results
are sorted descending by `overall_score` before they reach `ReportAgent` and
`OutputWriter`.

**Why Claude (API) here**

Scoring is not a binary filter: it requires mapping JD requirements to resume
claims, citing evidence, and staying coherent across many fields. That matches
a larger frontier model with long context better than the local shortlist model,
at acceptable cost because the shortlist already shrank N.

**Retry and fallback strategy (defense in depth)**

1. **JSON parse loop (`MAX_JSON_PARSE_ATTEMPTS`):** Models sometimes emit
   markdown fences or broken JSON. We re-prompt with a repair instruction
   including a snippet of the bad output before giving up.
2. **CoreScorer fallback:** If strict parsing still fails, delegate to
   `scorer_engine.Scorer` — same model, alternate parsing/scoring path — to
   avoid losing a candidate entirely.
3. **Lenient parse:** Last resort: extract whatever JSON is possible or emit a
   zero-score placeholder so the pipeline completes and humans see a failure
   mode in the output rather than a crash.
4. **Self-evaluation (`self_eval.txt`):** After a successful parse, a second
   pass judges whether high scores are backed by specific evidence. If the
   verdict is NEEDS_RETRY, we re-run scoring once with stricter instructions
   appended to the prompt (`MAX_RETRIES`).

**Fail-safe on self-eval**

If self-eval parsing fails, we assume GOOD so a flaky eval step never blocks
delivery — scoring quality may be imperfect but the run finishes.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from models import CandidateScore, JobDescription, RequirementScore, Resume
from scorer_engine import Scorer as CoreScorer

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class ScorerAgent:
    """LLM-backed scorer with JSON repair, engine fallback, and evidence self-check.

    Wraps prompt templates under ``prompts/`` and normalizes all model outputs
    into `CandidateScore`. The public `run` API is batch-oriented; per-resume
    logic lives in `_score_one`.
    """

    # Single retry after self-eval: balances quality vs latency/API cost.
    MAX_RETRIES: int = 1
    # Multiple parse attempts before escalating to CoreScorer / lenient parse.
    MAX_JSON_PARSE_ATTEMPTS: int = 3

    def __init__(self, model: BaseChatModel) -> None:
        """Load scoring and self-evaluation templates; store the shared model.

        Args:
            model: Typically Claude from `get_claude_model(config)`; reused for
                scoring, JSON repair, self-eval, and `CoreScorer` fallback.

        Side effects:
            Reads ``score_candidate.txt`` and ``self_eval.txt`` at init.
        """
        self._model = model
        self._score_template = (PROMPTS_DIR / "score_candidate.txt").read_text()
        self._eval_template = (PROMPTS_DIR / "self_eval.txt").read_text()

    def run(
        self,
        resumes: list[Resume],
        jd: JobDescription,
    ) -> list[CandidateScore]:
        """Score each shortlisted resume and return results sorted by overall score.

        Sorting here means `ReportAgent` and JSON consumers get a canonical
        ranking without re-sorting; relative order within ties is stable enough
        for recruiter review.

        Args:
            resumes: Shortlisted resumes (may be full set if supervisor fell back).
            jd: Full structured job description for requirement-level scoring.

        Returns:
            List of `CandidateScore`, sorted by ``overall_score`` descending.

        Side effects:
            Multiple LLM calls per resume (score + optional repairs + self-eval
            + optional strict re-score). Logs per-file outcomes.
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
        """Produce one `CandidateScore` with parse retries and self-eval loop.

        `jd_dict` is precomputed in `run()` for potential future use (e.g.
        logging or passing to tools); scoring prompts currently rebuild from `jd`.

        Args:
            resume: The resume being scored.
            jd: Job description (full prompt content).
            jd_dict: Serializable view of `jd` (currently unused inside this
                method but kept for API symmetry / extension).

        Returns:
            Final `CandidateScore` after any JSON repair, fallback, or strict
            re-score triggered by self-eval.

        Raises:
            RuntimeError: Only if no candidate object could be produced at all
                (should be unreachable if lenient parse always returns).

        Side effects:
            Logging for parse attempts, fallbacks, and self-eval retries.
        """
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
                        # Same model, different orchestration — sometimes yields
                        # valid structure when prompt/response shape diverged.
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
        """Format the scoring prompt and return the model's raw string output.

        When ``stricter`` is True, append hard requirements that evidence must
        quote concrete resume content — used after self-eval flags generic
        justifications for high scores.

        Args:
            resume: Resume whose `raw_text` is embedded in the prompt.
            jd: Supplies title, skills, responsibilities lists for the rubric.
            stricter: If True, append additional anti-generic-evidence rules.

        Returns:
            Unparsed model content (expected to be JSON, possibly fenced).

        Side effects:
            One `invoke` on `self._model`.
        """
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
        """Re-ask for valid JSON after a parse failure, showing a truncated error context.

        Including only the first ~400 characters keeps the repair prompt within
        token limits while still giving the model a hint about delimiter issues.

        Args:
            resume: Resume being scored.
            jd: Job description (full scoring context is re-sent).
            failed_response: Prior model output that `json.loads` rejected.

        Returns:
            New raw string from the model.

        Side effects:
            One additional LLM call. Parsing and escalation (more attempts,
            `CoreScorer`, lenient parse) are handled by `_score_one`, not here.
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
        """Second-pass LLM check: is the evidence concrete enough?

        Args:
            result: Parsed `CandidateScore` including requirement rows.

        Returns:
            ``"GOOD"`` if evidence passes or evaluation fails (fail-open), or
            ``"NEEDS_RETRY"`` to trigger one stricter re-score.

        Side effects:
            One `invoke`; may log weak requirement ids from parsed JSON.

        **Fail-safe:** Any exception returns ``"GOOD"`` so eval bugs never stall
        the batch — product choice: deliver scores over blocking on meta-judge.
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
        """Strictly parse scorer JSON into `CandidateScore` or raise.

        Used for the happy path and after strict retries; raises trigger the
        outer repair/fallback machinery in `_score_one`.

        Args:
            raw: Raw model output.
            resume: Source resume (stored on the result for traceability).

        Returns:
            Populated `CandidateScore`.

        Raises:
            RuntimeError: Missing JSON, invalid JSON, or missing `overall_score`.

        Side effects:
            None.
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
        """Remove optional ``` / ```json wrappers so `json.loads` can run.

        Models frequently wrap JSON in markdown even when asked not to; this
        normalizes before strict parsing.

        Args:
            text: Raw LLM string.

        Returns:
            Stripped text suitable for `json.loads` when valid.

        Side effects:
            None.
        """
        stripped = text.strip()
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
        return stripped.strip()

    @staticmethod
    def _derive_fit_label(score: int) -> str:
        """Map numeric overall score to a recruiter-facing bucket if LLM omitted it.

        Keeps downstream report templates and JSON consumers stable when the
        model returns scores but forgets `fit_label`.

        Args:
            score: Integer 0–100.

        Returns:
            One of: Strong / Good / Partial / Weak match (by band).

        Side effects:
            None.
        """
        if score >= 80:
            return "Strong match"
        elif score >= 60:
            return "Good match"
        elif score >= 40:
            return "Partial match"
        return "Weak match"

    def _lenient_parse(self, raw: str, resume: Resume) -> CandidateScore:
        """Best-effort recovery: never raise; prefer degraded score over crash.

        If JSON is totally invalid, return a zeroed placeholder with an
        explanation string so `ReportAgent` still lists the candidate and humans
        see the parse failure surfaced in text.

        Args:
            raw: Last known model output (possibly broken).
            resume: Resume row for the result object.

        Returns:
            Always a `CandidateScore` (possibly with overall_score 0).

        Side effects:
            None.
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
        """Serialize `JobDescription` to a plain dict for logging or export hooks.

        Args:
            jd: Parsed JD.

        Returns:
            Dict with string/list fields mirroring the dataclass.

        Side effects:
            None.
        """
        return {
            "job_title": jd.job_title,
            "required_skills": jd.required_skills,
            "nice_to_have_skills": jd.nice_to_have_skills,
            "min_experience_years": jd.min_experience_years,
            "responsibilities": jd.responsibilities,
        }
