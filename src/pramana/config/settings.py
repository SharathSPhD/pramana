"""Pramana settings using Pydantic Settings."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PramanaSettings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    Uses PRAMANA_ prefix for environment variables. Also supports
    WANDB_PROJECT, HF_TOKEN, and OPENAI_API_KEY without prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="PRAMANA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Allow reading non-prefixed env vars for specific fields
        env_ignore_empty=True,
    )

    data_dir: Path = Field(
        default=Path("/workspace/pramana/data"),
        description="Directory for training data",
    )
    models_dir: Path = Field(
        default=Path("/workspace/pramana/models"),
        description="Directory for model checkpoints",
    )
    wandb_project: str = Field(
        default="pramana",
        validation_alias="WANDB_PROJECT",
        description="Weights & Biases project name",
    )
    hf_token: str | None = Field(
        default=None,
        validation_alias="HF_TOKEN",
        description="HuggingFace Hub authentication token",
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key for LLM judge",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
        description="Anthropic API key for LLM judge",
    )
    llm_provider: str = Field(
        default="openai",
        description="LLM provider for judge (openai or anthropic)",
    )
    openai_judge_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for LLM judge scoring",
    )
    anthropic_judge_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Anthropic model for LLM judge scoring",
    )
    llm_max_tokens: int = Field(
        default=512,
        ge=1,
        description="Maximum tokens for LLM judge responses",
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        ge=1,
        description="Timeout in seconds for LLM judge requests",
    )

    @field_validator("data_dir", "models_dir", mode="before")
    @classmethod
    def expand_path(cls, value: str | Path) -> Path:
        """Expand user home directory in paths."""
        if isinstance(value, str) and value.startswith("~"):
            return Path(value).expanduser()
        return Path(value)
