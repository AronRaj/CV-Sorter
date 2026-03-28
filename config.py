"""Configuration loader for the CV Sorter project.

This module is the **single source of truth** for environment-driven settings.
Downstream code imports ``Config`` from here instead of calling ``os.getenv``
directly, so reviewers can see every tunable knob in one place and the app
stays consistent (e.g. one default output path, one parser choice).

Ollama model identifiers for local inference are fixed **constants** below
rather than env vars so deployments do not accidentally point at the wrong
local tags; change the constants in code if you need different models.

Reads ``.env`` via ``python-dotenv`` when ``load_config`` runs; that only
affects the current process environment for lookups after the call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# --- Local LLM defaults (not overridable via .env) ---
# Keeping these as code constants avoids misconfiguration when .env is copied
# from a template: reviewers and operators see exactly which Ollama tags the
# pipeline expects for shortlisting vs. JD extraction.
OLLAMA_SHORTLIST_MODEL: str = "llama3.1:latest"
OLLAMA_JD_MODEL: str = "gemma2:latest"


@dataclass
class Config:
    """Typed configuration object populated from environment variables.

    Instances are produced exclusively by :func:`load_config` so every field
    reflects the same parsing rules (defaults, placeholder stripping, etc.).

    Attributes:
        pdf_parser: Backend used to extract text from PDF resumes
            (e.g. ``"pymupdf"``). Chosen per deployment capabilities.
        output_path: Filesystem path where ranked results JSON is written.
            Central default keeps CLI and agents aligned on one artifact location.
        verbose: When True, downstream modules may emit richer logs; can be
            forced by the caller (e.g. CLI ``--verbose``) independent of .env.
        anthropic_api_key: Secret for Claude/Anthropic APIs, or None if missing
            or placeholder. None signals callers to fail fast or skip cloud paths.
        langchain_tracing_v2: String flag LangChain reads (typically ``"true"``
            or ``"false"``) to toggle LangSmith tracing — stored as str to match
            ecosystem conventions, not a Python bool.
        langchain_api_key: LangSmith API key if tracing is enabled; None when
            unset or placeholder so we do not send junk credentials.
        langchain_project: Project name in LangSmith for grouping traces.
    """

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
    """Load configuration from ``.env`` and the process environment.

    Calls :func:`dotenv.load_dotenv` first so variables defined in a project
    ``.env`` file are visible to subsequent ``os.getenv`` calls without
    requiring the shell to export them manually.

    Environment variables read:

        * ``ANTHROPIC_API_KEY`` — Anthropic / Claude API key; empty or
          placeholder values become ``None`` (see :func:`_read_key` logic).
        * ``OUTPUT_PATH`` — path for ranked results JSON
          (default: ``results/ranked_output.json``).
        * ``PDF_PARSER`` — PDF parser backend (default: ``pymupdf``).
        * ``LANGCHAIN_TRACING_V2`` — ``"true"`` to enable LangSmith tracing
          (default: ``"false"``).
        * ``LANGCHAIN_API_KEY`` — LangSmith API key (optional; placeholder stripped).
        * ``LANGCHAIN_PROJECT`` — LangSmith project name
          (default: ``cv-sorter-agents``).

    Ollama model names are **not** read here; they are module constants
    ``OLLAMA_SHORTLIST_MODEL`` and ``OLLAMA_JD_MODEL``.

    Args:
        verbose: Explicit verbose flag, usually from CLI. Passed through to
            ``Config`` so command-line intent wins over any future .env key
            for the same behavior (callers currently set this from flags).

    Returns:
        A fully populated :class:`Config` instance.

    Side effects:
        ``load_dotenv()`` may set environment variables from the ``.env`` file
        for the current process.
    """
    # Merge .env into os.environ for this process so defaults and keys resolve
    # the same way in development (file-based) and production (often real env).
    load_dotenv()

    def _read_key(env_var: str) -> str | None:
        """Normalize a secret-like environment variable to a usable value or None.

        Template repos often ship ``your_*`` placeholders; treating those as
        unset prevents accidental use of non-keys and makes "optional" keys
        (e.g. LangSmith) genuinely optional without crashing on garbage strings.

        Args:
            env_var: Name of the environment variable to read.

        Returns:
            The stripped non-empty string if the value looks like a real secret,
            otherwise ``None`` (missing, empty, or placeholder prefix ``your_``).
        """
        value = os.getenv(env_var, "")
        if not value or value.startswith("your_"):
            return None
        return value

    return Config(
        pdf_parser=os.getenv("PDF_PARSER", "pymupdf"),
        output_path=os.getenv("OUTPUT_PATH", "results/ranked_output.json"),
        verbose=verbose,
        anthropic_api_key=_read_key("ANTHROPIC_API_KEY"),
        langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "false"),
        langchain_api_key=_read_key("LANGCHAIN_API_KEY"),
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "cv-sorter-agents"),
    )


def _mask_key(key: str | None) -> str:
    """Render an API key for human-readable logging without exposing the full secret.

    Used by the ``__main__`` smoke print so operators can confirm a key is
    loaded while minimizing leak risk in shared terminals or screenshots.

    Args:
        key: Raw API key string, or ``None`` if unset.

    Returns:
        ``"(not set)"`` when ``key`` is ``None``; otherwise the first eight
        characters followed by ``"..."`` (no validation of minimum length —
        short values still truncate safely for display only).

    Side effects:
        None.
    """
    if key is None:
        return "(not set)"
    return key[:8] + "..."


# --- Manual sanity check when run as a script ---
# Allows ``python config.py`` to verify .env loading without starting the full app.
if __name__ == "__main__":
    config = load_config()
    print("=== Config loaded ===")
    print(f"  Anthropic key   : {_mask_key(config.anthropic_api_key)}")
    print(f"  Shortlist model : {OLLAMA_SHORTLIST_MODEL}")
    print(f"  JD model        : {OLLAMA_JD_MODEL}")
    print(f"  Output path     : {config.output_path}")
    print(f"  PDF parser      : {config.pdf_parser}")
    print(f"  Verbose         : {config.verbose}")
