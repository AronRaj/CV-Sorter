"""Resume scoring engine.

Takes a Resume and a JobDescription, sends them to the LLM via
prompts/score_candidate.txt, and parses the JSON response into
a CandidateScore dataclass. Retries once with a stricter prompt
if the first JSON parse fails.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.llm_client import LLMClient
from src.models import CandidateScore, JobDescription, RequirementScore, Resume

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = Path("prompts/score_candidate.txt")


class Scorer:
    """Scores a candidate resume against a job description via LLM.

    Uses prompts/score_candidate.txt as the prompt template and parses
    the structured JSON response into a CandidateScore dataclass.
    Implements one automatic retry on JSON parse failure.
    """

    def __init__(self, client: LLMClient) -> None:
        """Initialise with an LLM client.

        Args:
            client: Any concrete LLMClient implementation.
        """
        self._client = client

    def score(self, resume: Resume, jd: JobDescription) -> CandidateScore:
        """Score a single resume against a job description.

        Fills the prompt template, calls the LLM, and parses the JSON
        response. On parse failure, retries once with a stricter prompt.

        Args:
            resume: The parsed Resume to evaluate.
            jd: The structured JobDescription to score against.

        Returns:
            A populated CandidateScore dataclass.

        Raises:
            RuntimeError: If scoring fails after both attempts.
        """
        template = self._read_prompt_template()
        prompt = template.format(
            job_title=jd.job_title,
            required_skills="\n".join(f"- {s}" for s in jd.required_skills),
            nice_to_have_skills="\n".join(f"- {s}" for s in jd.nice_to_have_skills),
            responsibilities="\n".join(f"- {r}" for r in jd.responsibilities),
            resume_text=resume.raw_text,
        )

        logger.info("Scoring '%s'...", resume.filename)
        raw_response = self._client.complete(prompt)

        try:
            return self._parse_response(raw=raw_response, resume=resume)
        except json.JSONDecodeError:
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
        """Parse the LLM's raw JSON response into a CandidateScore.

        Args:
            raw: The raw string response from the LLM.
            resume: The Resume object being scored.

        Returns:
            A populated CandidateScore dataclass.

        Raises:
            json.JSONDecodeError: If the response is not valid JSON.
            RuntimeError: If required keys are missing from the JSON.
        """
        cleaned = self._strip_code_fences(raw)
        data = json.loads(cleaned)

        try:
            requirement_scores = [
                RequirementScore(
                    requirement=item["requirement"],
                    score=item["score"],
                    evidence=item["evidence"],
                )
                for item in data["requirement_scores"]
            ]

            return CandidateScore(
                resume=resume,
                overall_score=int(data["overall_score"]),
                fit_label=data["fit_label"],
                explanation=data["explanation"],
                requirement_scores=requirement_scores,
            )
        except KeyError as e:
            raise RuntimeError(
                f"Missing key {e} in LLM response for '{resume.filename}'. "
                f"Response preview: {raw[:200]}"
            ) from e

    def _retry_score(
        self,
        resume: Resume,
        jd: JobDescription,
        failed_response: str,
    ) -> CandidateScore:
        """Retry scoring with a stricter prompt after a parse failure.

        Args:
            resume: The Resume being scored.
            jd: The JobDescription to score against.
            failed_response: The raw LLM response that failed to parse.

        Returns:
            A populated CandidateScore from the retry attempt.

        Raises:
            RuntimeError: If the retry also fails.
        """
        template = self._read_prompt_template()
        original_prompt = template.format(
            job_title=jd.job_title,
            required_skills="\n".join(f"- {s}" for s in jd.required_skills),
            nice_to_have_skills="\n".join(f"- {s}" for s in jd.nice_to_have_skills),
            responsibilities="\n".join(f"- {r}" for r in jd.responsibilities),
            resume_text=resume.raw_text,
        )

        retry_prompt = (
            "Your previous response could not be parsed as JSON.\n"
            f"Previous response (first 300 chars): {failed_response[:300]}\n\n"
            "You MUST respond with ONLY a valid JSON object.\n"
            "No explanation, no markdown, no code fences. Just the raw JSON.\n\n"
            f"{original_prompt}"
        )

        logger.info("Retry scoring '%s' with stricter prompt...", resume.filename)
        raw_response = self._client.complete(retry_prompt)

        try:
            return self._parse_response(raw=raw_response, resume=resume)
        except (json.JSONDecodeError, RuntimeError) as e:
            raise RuntimeError(
                f"Scorer failed twice for {resume.filename}. "
                f"Check prompts/score_candidate.txt and the LLM response."
            ) from e

    def _read_prompt_template(self) -> str:
        """Read the scoring prompt template from disk.

        Returns:
            The template string with placeholders.

        Raises:
            FileNotFoundError: If prompts/score_candidate.txt is missing.
        """
        if not PROMPT_TEMPLATE_PATH.exists():
            raise FileNotFoundError(
                f"Scoring prompt template not found at '{PROMPT_TEMPLATE_PATH}'. "
                f"Create this file before running the scorer."
            )
        return PROMPT_TEMPLATE_PATH.read_text()

    def _strip_code_fences(self, text: str) -> str:
        """Remove markdown code fences from LLM output.

        Args:
            text: Raw LLM response string.

        Returns:
            The text with any leading/trailing code fences removed.
        """
        stripped = text.strip()
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
        return stripped.strip()
