"""LLM client abstraction layer.

Provides a single complete() interface that all LLM providers implement.
Callers never need to know which provider is running underneath — they
call client.complete(prompt) and get a string back.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from src.config import Config

logger = logging.getLogger(__name__)

VALID_PROVIDERS = ["claude", "openai", "gemini", "groq"]


class LLMClient(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the text response.

        Args:
            prompt: The full prompt string to send.

        Returns:
            The LLM's text response.
        """
        ...

    @staticmethod
    def build(provider: str, config: Config) -> LLMClient:
        """Factory method. Returns the correct LLMClient for the given provider name.

        Args:
            provider: One of "claude", "openai", "gemini", or "groq".
            config: The application Config instance.

        Raises:
            ValueError: If provider is not recognised.
        """
        if provider == "claude":
            return ClaudeClient(config)
        elif provider == "openai":
            return OpenAIClient(config)
        elif provider == "gemini":
            return GeminiClient(config)
        elif provider == "groq":
            return GroqClient(config)
        else:
            raise ValueError(
                f"Unknown provider: '{provider}'. "
                f"Valid options: {', '.join(VALID_PROVIDERS)}"
            )


class ClaudeClient(LLMClient):
    """LLM client using the Anthropic Claude API."""

    def __init__(self, config: Config) -> None:
        """Initialise the Claude client.

        Args:
            config: Application config with anthropic_api_key set.

        Raises:
            ValueError: If anthropic_api_key is not configured.
        """
        if config.anthropic_api_key is None:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file to use the Claude provider."
            )
        import anthropic

        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def complete(self, prompt: str) -> str:
        """Send a prompt to Claude and return the text response.

        Uses claude-sonnet-4-5 with max_tokens=1024.

        Args:
            prompt: The full prompt string to send.

        Returns:
            The text content from Claude's response.

        Raises:
            RuntimeError: If the Anthropic API returns an error.
        """
        import anthropic

        try:
            message = self._client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except anthropic.APIError as e:
            raise RuntimeError(
                f"Claude API call failed: {e}"
            ) from e


class OpenAIClient(LLMClient):
    """LLM client using the OpenAI API (gpt-4o)."""

    def __init__(self, config: Config) -> None:
        """Initialise the OpenAI client.

        Args:
            config: Application config with openai_api_key set.

        Raises:
            ValueError: If openai_api_key is not configured.
        """
        if config.openai_api_key is None:
            raise ValueError(
                "OPENAI_API_KEY is not set in .env"
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=config.openai_api_key)

    def complete(self, prompt: str) -> str:
        """Send a prompt to OpenAI and return the text response.

        Uses gpt-4o with max_tokens=1024.

        Args:
            prompt: The full prompt string to send.

        Returns:
            The text content from OpenAI's response.

        Raises:
            RuntimeError: If the OpenAI API returns an error.
        """
        from openai import OpenAIError

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except OpenAIError as e:
            raise RuntimeError(
                f"OpenAI API call failed: {e}"
            ) from e


class GeminiClient(LLMClient):
    """LLM client using the Google Gemini API (gemini-2.0-flash)."""

    def __init__(self, config: Config) -> None:
        """Initialise the Gemini client.

        Args:
            config: Application config with gemini_api_key set.

        Raises:
            ValueError: If gemini_api_key is not configured.
        """
        if config.gemini_api_key is None:
            raise ValueError(
                "GEMINI_API_KEY is not set in .env"
            )
        import google.generativeai as genai

        genai.configure(api_key=config.gemini_api_key)
        self._model = genai.GenerativeModel("gemini-2.0-flash")

    def complete(self, prompt: str) -> str:
        """Send a prompt to Gemini and return the text response.

        Uses gemini-2.0-flash.

        Args:
            prompt: The full prompt string to send.

        Returns:
            The text content from Gemini's response.

        Raises:
            RuntimeError: If the Gemini API returns an error.
        """
        try:
            response = self._model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(
                f"Gemini API error: {e}"
            ) from e


class GroqClient(LLMClient):
    """LLM client using Groq's OpenAI-compatible API.

    Uses the openai SDK with a custom base_url pointing to Groq's endpoint.
    This avoids adding a separate groq dependency.
    """

    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, config: Config) -> None:
        """Initialise the Groq client via the OpenAI SDK.

        Args:
            config: Application config with groq_api_key and groq_model set.

        Raises:
            ValueError: If groq_api_key is not configured.
        """
        if config.groq_api_key is None:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file to use the Groq provider."
            )
        from openai import OpenAI

        self._client = OpenAI(
            api_key=config.groq_api_key,
            base_url=self.GROQ_BASE_URL,
        )
        self._model = config.groq_model

    def complete(self, prompt: str) -> str:
        """Send a prompt to Groq and return the text response.

        Args:
            prompt: The full prompt string to send.

        Returns:
            The text content from Groq's response.

        Raises:
            RuntimeError: If the Groq API returns an error.
        """
        from openai import APIError

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except APIError as e:
            raise RuntimeError(
                f"Groq API call failed: {e}"
            ) from e
