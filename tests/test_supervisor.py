"""Unit tests for src/agents/supervisor.py.

Tests edge cases: empty resumes folder and missing API key.
Uses unittest.mock.patch to avoid constructing real LLM models.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.supervisor import Supervisor
from src.core.config import load_config


def test_empty_resumes_folder_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Supervisor.run() raises RuntimeError when no PDFs are found."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kw: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("PDF_PARSER", "pymupdf")
    monkeypatch.setenv("OUTPUT_PATH", "src/results/ranked_output.json")

    config = load_config()

    with patch("src.agents.supervisor.get_claude_model"), \
         patch("src.agents.supervisor.get_ollama_shortlist_model"), \
         patch("src.agents.supervisor.get_ollama_jd_model"):
        supervisor = Supervisor(config=config)

    with pytest.raises(RuntimeError) as exc_info:
        supervisor.run(
            jd_path="job_description.txt",
            resumes_folder=str(tmp_path),
        )
    assert "No PDF files" in str(exc_info.value)


def test_missing_api_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supervisor.__init__ raises ValueError if ANTHROPIC_API_KEY is missing."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kw: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    config = load_config()

    with pytest.raises(ValueError) as exc_info:
        Supervisor(config=config)
    assert "ANTHROPIC_API_KEY" in str(exc_info.value)
