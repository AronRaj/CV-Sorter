"""Configuration loader for the CV Sorter project.

Reads .env and exposes a typed Config dataclass.
All other modules import Config from here — no module ever calls
os.getenv() directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

OLLAMA_SHORTLIST_MODEL: str = "llama3.1:latest"
OLLAMA_JD_MODEL: str = "gemma2:latest"


@dataclass
class Config:
    """Typed configuration object populated from environment variables."""

    # PDF parsing
    pdf_parser: str

    # Output
    output_path: str
    verbose: bool

    # API keys
    anthropic_api_key: str | None

    # LangSmith tracing (optional)
    langchain_tracing_v2: str
    langchain_api_key: str | None
    langchain_project: str


def load_config(verbose: bool = False) -> Config:
    """Load configuration from .env and return a populated Config instance.

    Environment variables read:
        ANTHROPIC_API_KEY    — Anthropic / Claude API key (required)
        OUTPUT_PATH          — path for ranked results JSON (default: src/results/ranked_output.json)
        PDF_PARSER           — PDF parser backend: "pymupdf" or "paddle" (default: pymupdf)
        LANGCHAIN_TRACING_V2 — "true" to enable LangSmith tracing (default: "false")
        LANGCHAIN_API_KEY    — LangSmith API key (optional)
        LANGCHAIN_PROJECT    — LangSmith project name (default: cv-sorter-agents)

    Ollama model names are fixed constants (OLLAMA_SHORTLIST_MODEL,
    OLLAMA_JD_MODEL) — not configurable via env vars.

    Args:
        verbose: Override for verbose mode (typically set via CLI --verbose flag).
    """
    load_dotenv()

    def _read_key(env_var: str) -> str | None:
        value = os.getenv(env_var, "")
        if not value or value.startswith("your_"):
            return None
        return value

    return Config(
        pdf_parser=os.getenv("PDF_PARSER", "pymupdf"),
        output_path=os.getenv("OUTPUT_PATH", "src/results/ranked_output.json"),
        verbose=verbose,
        anthropic_api_key=_read_key("ANTHROPIC_API_KEY"),
        langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "false"),
        langchain_api_key=_read_key("LANGCHAIN_API_KEY"),
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "cv-sorter-agents"),
    )


def _mask_key(key: str | None) -> str:
    """Show only the first 8 characters of an API key, followed by '...'."""
    if key is None:
        return "(not set)"
    return key[:8] + "..."


if __name__ == "__main__":
    config = load_config()
    print("=== Config loaded ===")
    print(f"  Anthropic key   : {_mask_key(config.anthropic_api_key)}")
    print(f"  Shortlist model : {OLLAMA_SHORTLIST_MODEL}")
    print(f"  JD model        : {OLLAMA_JD_MODEL}")
    print(f"  Output path     : {config.output_path}")
    print(f"  PDF parser      : {config.pdf_parser}")
    print(f"  Verbose         : {config.verbose}")
