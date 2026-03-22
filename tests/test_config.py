"""Unit tests for src/core/config.py.

Uses monkeypatch to set environment variables in isolation —
no real .env file is read during these tests.
"""

import pytest

from src.core.config import OLLAMA_JD_MODEL, OLLAMA_SHORTLIST_MODEL, load_config


def test_load_config_with_all_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify load_config reads all env vars and populates Config correctly."""
    monkeypatch.setattr("src.core.config.load_dotenv", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")
    monkeypatch.setenv("OUTPUT_PATH", "results/test_output.json")
    monkeypatch.setenv("PDF_PARSER", "pymupdf")

    config = load_config()

    assert config.anthropic_api_key == "sk-ant-test123"
    assert config.output_path == "results/test_output.json"
    assert config.pdf_parser == "pymupdf"


def test_load_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify load_config applies correct defaults when optional vars are unset."""
    monkeypatch.setattr("src.core.config.load_dotenv", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.delenv("OUTPUT_PATH", raising=False)
    monkeypatch.delenv("PDF_PARSER", raising=False)

    config = load_config()

    assert config.output_path == "src/results/ranked_output.json"
    assert config.pdf_parser == "pymupdf"
    assert config.verbose is False


def test_ollama_model_constants() -> None:
    """Verify Ollama model constants are set to the expected fixed values."""
    assert OLLAMA_SHORTLIST_MODEL == "llama3.1:latest"
    assert OLLAMA_JD_MODEL == "gemma2:latest"


def test_load_config_langsmith_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify LangSmith fields default to safe values."""
    monkeypatch.setattr("src.core.config.load_dotenv", lambda: None)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    config = load_config()

    assert config.langchain_tracing_v2 == "false"
    assert config.langchain_api_key is None
