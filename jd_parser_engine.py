"""Job description parser using LLM extraction.

Reads a plain-text job description file, sends it to the LLM via a
prompt template, and parses the structured JSON response into a
JobDescription dataclass.

The JD is treated as unstructured text on disk; structuring it once here
lets scorers and agents compare candidates against a single canonical
schema (skills, experience floor, responsibilities) instead of re-prompting
with the full JD on every resume.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from models import JobDescription

logger = logging.getLogger(__name__)

# Resolved relative to the process working directory (typically project root
# when the app is launched from there); keeps the prompt editable without
# embedding a huge string in code.
PROMPT_TEMPLATE_PATH = Path("prompts/extract_jd.txt")


class JDParser:
    """Extracts structured job description data from a text file via LLM.

    The model is injected so the same parsing logic works across providers
    (OpenAI, Anthropic, etc.) and tests can substitute a fake model that
    returns fixed JSON.
    """

    def __init__(self, model: BaseChatModel) -> None:
        """Attach the LangChain chat model used for extraction.

        Args:
            model: Any ``BaseChatModel`` implementation; must accept
                ``HumanMessage`` and expose ``.content`` on the response.

        Returns:
            None.

        Side effects:
            Stores the model on ``self._model`` for subsequent ``parse`` calls.
        """
        self._model = model

    def parse(self, jd_path: str) -> JobDescription:
        """Parse a job description text file into a ``JobDescription``.

        Orchestrates: load JD text, render the extraction prompt, invoke the
        LLM once, normalize fenced JSON if needed, then map keys into the
        dataclass with safe defaults so minor schema drift from the model
        does not crash the pipeline.

        Args:
            jd_path: Path to the ``.txt`` job description file.

        Returns:
            A ``JobDescription`` with ``raw_text`` preserved and structured
            fields filled from parsed JSON.

        Raises:
            RuntimeError: If the model output is not valid JSON after fence
                stripping (wraps ``json.JSONDecodeError`` for a clear domain
                error message).
            FileNotFoundError: Propagated if ``jd_path`` or the prompt template
                file cannot be read (Python's default for missing paths).

        Side effects:
            Performs synchronous LLM I/O; logs info before the call and error
            with raw response body on JSON parse failure (aids debugging bad
            model output without changing control flow).
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

        # .get with defaults tolerates missing keys or partial JSON so we
        # still produce a usable JobDescription when the model omits optional
        # sections; lists default to [] to avoid None checks downstream.
        return JobDescription(
            raw_text=raw_text,
            job_title=data.get("job_title", "Unknown"),
            required_skills=data.get("required_skills", []),
            nice_to_have_skills=data.get("nice_to_have_skills", []),
            min_experience_years=data.get("min_experience_years"),
            responsibilities=data.get("responsibilities", []),
        )

    def _strip_code_fences(self, text: str) -> str:
        """Remove markdown code fences from LLM output so ``json.loads`` succeeds.

        Models often wrap JSON in ```json ... ``` despite instructions not to;
        stripping those markers is cheaper and more reliable than asking for
        a second completion. The regex allows an optional ``json`` language
        tag after the opening fence to match common model habits.

        Args:
            text: Raw string from ``response.content`` (may include fences).

        Returns:
            Inner content with leading/trailing fence lines removed and outer
            whitespace trimmed.

        Side effects:
            None.

        Note:
            This is intentionally conservative (only leading opening fence
            and trailing closing fence) to avoid deleting legitimate ``` that
            might appear inside string values in rare cases.
        """
        stripped = text.strip()
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
        return stripped.strip()
