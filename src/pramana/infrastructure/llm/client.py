"""LLM client implementations for Tier 2 judge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pramana.config.settings import PramanaSettings

LLMProvider = Literal["openai", "anthropic"]


class LLMClientError(RuntimeError):
    """Raised when LLM client initialization or calls fail."""


@dataclass(frozen=True)
class LLMClientConfig:
    """Configuration for LLM clients."""

    provider: LLMProvider
    model: str
    api_key: str
    max_tokens: int
    timeout_seconds: float


class OpenAILLMClient:
    """OpenAI LLM client."""

    def __init__(self, config: LLMClientConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LLMClientError(
                "OpenAI client not installed. Install with 'pramana[llm-judge]'."
            ) from exc

        self._config = config
        self._client = OpenAI(api_key=config.api_key, timeout=config.timeout_seconds)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=self._config.max_tokens,
            )
        except Exception as exc:  # pragma: no cover - networked call
            raise LLMClientError(f"OpenAI request failed: {exc}") from exc

        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise LLMClientError("OpenAI response contained no content.")
        return content.strip()


class AnthropicLLMClient:
    """Anthropic LLM client."""

    def __init__(self, config: LLMClientConfig) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LLMClientError(
                "Anthropic client not installed. Install with 'pramana[llm-judge]'."
            ) from exc

        self._config = config
        self._client = Anthropic(api_key=config.api_key, timeout=config.timeout_seconds)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        try:
            response = self._client.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # pragma: no cover - networked call
            raise LLMClientError(f"Anthropic request failed: {exc}") from exc

        if not response.content:
            raise LLMClientError("Anthropic response contained no content.")

        first = response.content[0]
        text = getattr(first, "text", None)
        if not text:
            raise LLMClientError("Anthropic response contained no text.")
        return text.strip()


def create_llm_client(settings: PramanaSettings) -> OpenAILLMClient | AnthropicLLMClient:
    """Create an LLM client based on settings."""

    provider_raw = settings.llm_provider.strip().lower()
    if provider_raw not in ("openai", "anthropic"):
        raise LLMClientError(
            f"Unsupported LLM provider '{settings.llm_provider}'. Use 'openai' or 'anthropic'."
        )

    provider: LLMProvider = provider_raw  # type: ignore[assignment]

    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMClientError("OPENAI_API_KEY is required for OpenAI LLM judge.")
        config = LLMClientConfig(
            provider=provider,
            model=settings.openai_judge_model,
            api_key=settings.openai_api_key,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        return OpenAILLMClient(config)

    if not settings.anthropic_api_key:
        raise LLMClientError("ANTHROPIC_API_KEY is required for Anthropic LLM judge.")
    config = LLMClientConfig(
        provider=provider,
        model=settings.anthropic_judge_model,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.llm_max_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    return AnthropicLLMClient(config)
