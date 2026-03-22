"""Job description parser using LLM extraction.

Reads a plain-text job description file, sends it to the LLM via a
prompt template, and parses the structured JSON response into a
JobDescription dataclass.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.core.models import JobDescription

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = Path("src/prompts/extract_jd.txt")


class JDParser:
    """Extracts structured job description data from a text file via LLM."""

    def __init__(self, model: BaseChatModel) -> None:
        """Initialise with a LangChain chat model.

        Args:
            model: Any BaseChatModel implementation (Claude, OpenAI, Gemini, etc.).
        """
        self._model = model

    def parse(self, jd_path: str) -> JobDescription:
        """Parse a job description text file into a JobDescription dataclass.

        Reads the raw text, fills the prompt template, calls the LLM,
        and maps the JSON response to a JobDescription.

        Args:
            jd_path: Path to the .txt job description file.

        Returns:
            A populated JobDescription dataclass.

        Raises:
            RuntimeError: If the LLM does not return valid JSON.
            FileNotFoundError: If jd_path or the prompt template is missing.
        """
        raw_text = Path(jd_path).read_text()
        template = PROMPT_TEMPLATE_PATH.read_text()
        prompt = template.format(raw_jd_text=raw_text)

        logger.info("Sending job description to LLM for extraction...")
        response = self._model.invoke([HumanMessage(content=prompt)]).content
        cleaned = self._strip_code_fences(response)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(
                "LLM returned invalid JSON for JD extraction. "
                "Raw response:\n%s",
                response,
            )
            raise RuntimeError(
                "JD extraction failed: LLM did not return valid JSON"
            ) from e

        return JobDescription(
            raw_text=raw_text,
            job_title=data.get("job_title", "Unknown"),
            required_skills=data.get("required_skills", []),
            nice_to_have_skills=data.get("nice_to_have_skills", []),
            min_experience_years=data.get("min_experience_years"),
            responsibilities=data.get("responsibilities", []),
        )

    def _strip_code_fences(self, text: str) -> str:
        """Remove markdown code fences from LLM output.

        LLMs sometimes wrap JSON responses in ```json ... ``` markers
        even when instructed not to. This strips those fences so
        json.loads() can parse the content cleanly.

        Args:
            text: Raw LLM response string.

        Returns:
            The text with any leading/trailing code fences removed.
        """
        stripped = text.strip()
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
        return stripped.strip()
