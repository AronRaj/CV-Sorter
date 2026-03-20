"""Configuration loader for the CV Sorter project.

Reads .env and exposes a typed Config dataclass.
All other modules import Config from here — no module ever calls
os.getenv() directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Config:
    """Typed configuration object populated from environment variables."""

    default_provider: str
    anthropic_api_key: str | None
    openai_api_key: str | None
    gemini_api_key: str | None
    groq_api_key: str | None
    groq_model: str
    output_path: str
    pdf_parser: str
    verbose: bool


def load_config(verbose: bool = False) -> Config:
    """Load configuration from .env and return a populated Config instance.

    Environment variables read:
        DEFAULT_PROVIDER   — required, which LLM provider to use
        ANTHROPIC_API_KEY  — Anthropic / Claude API key
        OPENAI_API_KEY     — OpenAI API key
        GEMINI_API_KEY     — Google Gemini API key
        GROQ_API_KEY       — Groq API key (uses OpenAI-compatible endpoint)
        GROQ_MODEL         — Groq model name (default: llama-3.3-70b-versatile)
        OUTPUT_PATH        — path for ranked results JSON (default: results/ranked_output.json)
        PDF_PARSER         — PDF parser backend: "pymupdf" or "paddle" (default: pymupdf)

    Args:
        verbose: Override for verbose mode (typically set via CLI --verbose flag).

    Raises:
        ValueError: If DEFAULT_PROVIDER is not set in the environment.
    """
    load_dotenv()

    default_provider = os.getenv("DEFAULT_PROVIDER")
    if not default_provider:
        raise ValueError(
            "DEFAULT_PROVIDER is not set. "
            "Add DEFAULT_PROVIDER=claude (or openai, gemini, groq) to your .env file."
        )

    def _read_key(env_var: str) -> str | None:
        value = os.getenv(env_var, "")
        if not value or value.startswith("your_"):
            return None
        return value

    return Config(
        default_provider=default_provider,
        anthropic_api_key=_read_key("ANTHROPIC_API_KEY"),
        openai_api_key=_read_key("OPENAI_API_KEY"),
        gemini_api_key=_read_key("GEMINI_API_KEY"),
        groq_api_key=_read_key("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        output_path=os.getenv("OUTPUT_PATH", "results/ranked_output.json"),
        pdf_parser=os.getenv("PDF_PARSER", "pymupdf"),
        verbose=verbose,
    )


def _mask_key(key: str | None) -> str:
    """Show only the first 8 characters of an API key, followed by '...'."""
    if key is None:
        return "(not set)"
    return key[:8] + "..."


if __name__ == "__main__":
    config = load_config()
    print("=== Config loaded ===")
    print(f"  Provider        : {config.default_provider}")
    print(f"  Anthropic key   : {_mask_key(config.anthropic_api_key)}")
    print(f"  OpenAI key      : {_mask_key(config.openai_api_key)}")
    print(f"  Gemini key      : {_mask_key(config.gemini_api_key)}")
    print(f"  Groq key        : {_mask_key(config.groq_api_key)}")
    print(f"  Groq model      : {config.groq_model}")
    print(f"  Output path     : {config.output_path}")
    print(f"  PDF parser      : {config.pdf_parser}")
    print(f"  Verbose         : {config.verbose}")
