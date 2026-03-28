"""Model factory for the CV Sorter agent pipeline.

Centralizes **which LLM backs which stage** so the rest of the codebase depends
on `BaseChatModel` interfaces, not vendor-specific constructors. That keeps
agents testable with mocks and documents architectural trade-offs in one place.

**Design decisions (why each model)**

- **Shortlist — Ollama llama3.1:8b:** Many cheap, fast decisions (PROCEED/SKIP).
  Quality bar is lower than scoring; speed and zero marginal cost dominate.
- **JD extraction — Ollama gemma2:9b:** Template-filling / structured extraction
  from the JD file; does not need the same narrative depth as scoring.
- **Scorer — Claude Sonnet (API):** Evidence-based rubric scoring over long
  resumes benefits from a strong reasoning model and large context.
- **Report — Claude Sonnet (API):** Compares all candidates in one pass; weak
  prose here undermines the whole deliverable, so it matches the scorer tier.

These are **fixed** product decisions (not toggled per run via Config) to keep
evaluation and demos reproducible. See project docs (e.g. ARCHITECTURE.md) for
the full rationale.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from config import Config, OLLAMA_JD_MODEL, OLLAMA_SHORTLIST_MODEL

# API model id for Anthropic; bump intentionally when upgrading capabilities.
CLAUDE_MODEL: str = "claude-sonnet-4-5"
# Zero temperature: scoring and reporting should be stable and auditable, not
# creative sampling from run to run.
TEMPERATURE: float = 0.0


def get_claude_model(config: Config) -> BaseChatModel:
    """Construct the shared Claude chat model for high-stakes reasoning tasks.

    Used by `ScorerAgent` and `ReportAgent`. Both stages need consistent
    calibration (tone, strictness, JSON discipline), so they intentionally share
    one model configuration rather than divergent hyperparameters.

    Args:
        config: Application configuration; must expose ``anthropic_api_key``
            for authenticated API calls.

    Returns:
        A LangChain `BaseChatModel` backed by Claude Sonnet with deterministic
        sampling (`temperature=0`) and a generous `max_tokens` ceiling so long
        scoring JSON and reports are not truncated mid-generation.

    Raises:
        ValueError: If ``ANTHROPIC_API_KEY`` is unset — fail fast before any
            scorer/report call rather than opaque HTTP errors deep in the run.
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
        # Large cap: requirement-level evidence blocks can grow quickly; better
        # to allow headroom than lose JSON to truncation.
        max_tokens=16_384,
    )


def get_ollama_shortlist_model() -> BaseChatModel:
    """Construct the local Ollama chat model used by the shortlist agent.

    Shortlisting is I/O bound over many resumes; a small local model keeps
    latency acceptable and avoids billing. The exact tag comes from config
    constants so ops can retarget weights without touching agent code.

    Args:
        None.

    Returns:
        A LangChain `ChatOllama` instance for ``OLLAMA_SHORTLIST_MODEL``
        (expected: llama3.1 8B class), temperature 0 for repeatable filters.

    Raises:
        Connection errors at invoke time if Ollama is not running — not raised
        here; callers should handle operational failures in the agent layer.

    Side effects:
        None at construction time; network I/O happens on first `invoke`.
    """
    return ChatOllama(
        model=OLLAMA_SHORTLIST_MODEL,
        temperature=TEMPERATURE,
    )


def get_ollama_jd_model() -> BaseChatModel:
    """Construct the local Ollama chat model used by `JDParser` in the supervisor.

    JD parsing is a separate model from shortlist so each prompt family can use
    a weight that fits structured extraction vs. quick binary screening without
    conflating two different prompt shapes on one endpoint.

    Args:
        None.

    Returns:
        A LangChain `ChatOllama` for ``OLLAMA_JD_MODEL`` (expected: Gemma2 9B
        class), temperature 0 for stable field extraction.

    Raises:
        Same operational caveats as `get_ollama_shortlist_model` — connectivity
        is validated when `invoke` runs, not in the factory.

    Side effects:
        None at construction time.
    """
    return ChatOllama(
        model=OLLAMA_JD_MODEL,
        temperature=TEMPERATURE,
    )
