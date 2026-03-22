"""Model factory for the CV Sorter agent pipeline.

Design decisions:
  - Shortlist agent uses Ollama llama3.1:8b (local, zero API cost).
    Task: binary PROCEED/SKIP decision. Speed > quality.
  - JD extraction uses Ollama gemma2:9b (local, zero API cost).
    Task: structured field extraction. No reasoning required.
  - Scorer agent uses Claude Sonnet (Anthropic API).
    Task: evidence-based scoring requiring long-document reasoning.
  - Report agent uses Claude Sonnet (Anthropic API).
    Task: cross-candidate synthesis requiring professional prose quality.

These are fixed architectural decisions documented in ARCHITECTURE.md.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.core.config import Config, OLLAMA_JD_MODEL, OLLAMA_SHORTLIST_MODEL

CLAUDE_MODEL: str = "claude-sonnet-4-5"
TEMPERATURE: float = 0.0


def get_claude_model(config: Config) -> BaseChatModel:
    """Return a Claude Sonnet model for tasks requiring reasoning quality.

    Used by: ScorerAgent, ReportAgent.

    Args:
        config: A populated Config instance.

    Returns:
        A BaseChatModel backed by Claude Sonnet.

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set.
    """
    if not config.anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set in .env. "
            "This key is required for the scorer and report agents."
        )
    return ChatAnthropic(
        model=CLAUDE_MODEL,
        api_key=config.anthropic_api_key,
        temperature=TEMPERATURE,
        max_tokens=16_384,
    )


def get_ollama_shortlist_model() -> BaseChatModel:
    """Return a local Llama 3.1 8B model for the shortlist agent.

    Used by: ShortlistAgent.
    Requires Ollama running locally on port 11434.
    Free — no API key needed.

    Returns:
        A BaseChatModel backed by Ollama llama3.1:8b.
    """
    return ChatOllama(
        model=OLLAMA_SHORTLIST_MODEL,
        temperature=TEMPERATURE,
    )


def get_ollama_jd_model() -> BaseChatModel:
    """Return a local Gemma 2 9B model for JD extraction.

    Used by: JDParser inside Supervisor.
    Requires Ollama running locally on port 11434.
    Free — no API key needed.

    Returns:
        A BaseChatModel backed by Ollama gemma2:9b.
    """
    return ChatOllama(
        model=OLLAMA_JD_MODEL,
        temperature=TEMPERATURE,
    )
