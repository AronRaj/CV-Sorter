"""Unit tests for src/config.py.

Uses monkeypatch to set environment variables in isolation —
no real .env file is read during these tests.
"""

import pytest

from src.config import load_config


def test_load_config_with_all_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify load_config reads all env vars and populates Config correctly."""
    monkeypatch.setattr("src.config.load_dotenv", lambda: None)
    monkeypatch.setenv("DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test456")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test789")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-testgroq")
    monkeypatch.setenv("GROQ_MODEL", "llama-test")
    monkeypatch.setenv("OUTPUT_PATH", "results/test_output.json")
    monkeypatch.setenv("PDF_PARSER", "pymupdf")

    config = load_config()

    assert config.default_provider == "openai"
    assert config.anthropic_api_key == "sk-ant-test123"
    assert config.openai_api_key == "sk-test456"
    assert config.gemini_api_key == "AIza-test789"
    assert config.groq_api_key == "gsk-testgroq"
    assert config.groq_model == "llama-test"
    assert config.output_path == "results/test_output.json"
    assert config.pdf_parser == "pymupdf"


def test_load_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify load_config applies correct defaults when optional vars are unset."""
    monkeypatch.setattr("src.config.load_dotenv", lambda: None)
    monkeypatch.setenv("DEFAULT_PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.delenv("OUTPUT_PATH", raising=False)
    monkeypatch.delenv("PDF_PARSER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    config = load_config()

    assert config.output_path == "results/ranked_output.json"
    assert config.pdf_parser == "pymupdf"
    assert config.verbose is False
    assert config.openai_api_key is None
    assert config.gemini_api_key is None
    assert config.groq_api_key is None
    assert config.groq_model == "llama-3.3-70b-versatile"


def test_missing_default_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify load_config raises ValueError when DEFAULT_PROVIDER is missing."""
    monkeypatch.setattr("src.config.load_dotenv", lambda: None)
    monkeypatch.delenv("DEFAULT_PROVIDER", raising=False)

    with pytest.raises(ValueError, match="DEFAULT_PROVIDER"):
        load_config()
