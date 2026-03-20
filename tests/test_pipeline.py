"""Integration-lite tests for src/pipeline.py.

Tests pipeline error handling without making real LLM calls.
Uses a mock LLMClient to avoid API calls while exercising the
pipeline's coordination logic.
"""

import json
from pathlib import Path

import pytest

from src.config import load_config
from src.models import JobDescription
from src.pipeline import Pipeline


VALID_JD_JSON = json.dumps({
    "job_title": "Test Engineer",
    "required_skills": ["Python"],
    "nice_to_have_skills": [],
    "min_experience_years": 1,
    "responsibilities": ["Write code"],
})


class _MockLLMClient:
    """Returns valid JD JSON for any prompt. Avoids real API calls."""

    def complete(self, prompt: str) -> str:
        return VALID_JD_JSON


def _mock_build(provider: str, config: object) -> _MockLLMClient:
    """Factory replacement that returns a mock client for any provider."""
    return _MockLLMClient()


def test_empty_resumes_folder_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Pipeline.run raises RuntimeError when the resumes folder is empty."""
    monkeypatch.setattr("src.config.load_dotenv", lambda: None)
    monkeypatch.setenv("DEFAULT_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setattr("src.pipeline.LLMClient.build", _mock_build)

    config = load_config()
    pipeline = Pipeline(config=config)

    with pytest.raises(RuntimeError, match="No PDF files"):
        pipeline.run(
            jd_path="job_description.txt",
            resumes_folder=str(tmp_path),
            provider="groq",
            output_path=str(tmp_path / "output.json"),
        )


def test_invalid_provider_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Pipeline.run raises ValueError for an unrecognised provider name."""
    monkeypatch.setattr("src.config.load_dotenv", lambda: None)
    monkeypatch.setenv("DEFAULT_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    config = load_config()
    pipeline = Pipeline(config=config)

    with pytest.raises(ValueError, match="badprovider"):
        pipeline.run(
            jd_path="job_description.txt",
            resumes_folder=str(tmp_path),
            provider="badprovider",
            output_path=str(tmp_path / "output.json"),
        )
