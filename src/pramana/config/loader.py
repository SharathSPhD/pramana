"""YAML-based configuration loading for training stages."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = Field(description="Model name or HuggingFace model ID")
    revision: str = Field(default="main", description="Model revision/branch")


class LoRAConfig(BaseModel):
    """LoRA (Low-Rank Adaptation) configuration."""

    rank: int = Field(gt=0, description="LoRA rank")
    alpha: int | None = Field(
        default=None, description="LoRA alpha (defaults to rank)"
    )
    target_modules: list[str] = Field(
        default_factory=lambda: ["all-linear"],
        description="Target modules for LoRA adaptation",
    )

    @model_validator(mode="after")
    def set_alpha_default(self) -> "LoRAConfig":
        """Set alpha to rank if not provided."""
        if self.alpha is None:
            self.alpha = self.rank
        return self


class TrainingParams(BaseModel):
    """Training hyperparameters."""

    learning_rate: float = Field(gt=0, description="Learning rate")
    batch_size: int = Field(gt=0, description="Batch size")
    gradient_accumulation_steps: int = Field(
        default=1, ge=1, description="Gradient accumulation steps"
    )
    epochs: int = Field(ge=0, description="Number of training epochs")
    max_steps: int | None = Field(
        default=None, description="Maximum training steps (overrides epochs)"
    )
    warmup_steps: int | None = Field(
        default=None, ge=0, description="Warmup steps (calculated from ratio if not provided)"
    )
    warmup_ratio: float | None = Field(
        default=None, ge=0, le=1, description="Warmup ratio (fraction of total steps)"
    )
    weight_decay: float = Field(
        default=0.0, ge=0, description="Weight decay"
    )
    max_grad_norm: float = Field(
        default=1.0, gt=0, description="Maximum gradient norm for clipping"
    )

    @model_validator(mode="after")
    def validate_warmup(self) -> "TrainingParams":
        """Ensure at least one warmup parameter is set if warmup_ratio is provided."""
        if self.warmup_ratio is not None and self.warmup_steps is None:
            # warmup_steps will be calculated during training based on total steps
            pass
        elif self.warmup_steps is None and self.warmup_ratio is None:
            # Default to 0 warmup steps if neither is provided
            self.warmup_steps = 0
        return self


class DataConfig(BaseModel):
    """Data configuration."""

    train_path: Path = Field(description="Path to training data")
    eval_path: Path | None = Field(
        default=None, description="Path to evaluation data"
    )
    max_length: int = Field(
        default=2048, gt=0, description="Maximum sequence length"
    )


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    tier1_threshold: float = Field(
        default=0.95, ge=0, le=1, description="Tier 1 evaluation threshold"
    )
    tier2_threshold: float = Field(
        default=0.85, ge=0, le=1, description="Tier 2 evaluation threshold"
    )


class StageConfig(BaseModel):
    """Complete stage configuration."""

    stage: int = Field(ge=0, le=4, description="Stage number (0-4)")
    model: ModelConfig = Field(description="Model configuration")
    lora: LoRAConfig = Field(description="LoRA configuration")
    training: TrainingParams = Field(description="Training parameters")
    data: DataConfig = Field(description="Data configuration")
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig, description="Evaluation configuration"
    )


class StageConfigLoader:
    """Loads stage configuration from YAML files with inheritance."""

    @staticmethod
    def _deep_merge(
        base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = StageConfigLoader._deep_merge(
                    result[key], value
                )
            else:
                result[key] = value
        return result

    @staticmethod
    def load(stage: int, config_dir: Path) -> StageConfig:
        """Load stage configuration with inheritance from base.yaml.

        Args:
            stage: Stage number (0-4)
            config_dir: Directory containing YAML config files

        Returns:
            StageConfig instance

        Raises:
            FileNotFoundError: If base.yaml not found
            ValueError: If configuration is invalid
        """
        config_dir = Path(config_dir)
        base_file = config_dir / "base.yaml"

        if not base_file.exists():
            raise FileNotFoundError(
                f"Base configuration file not found: {base_file}"
            )

        # Load base configuration
        try:
            with base_file.open(encoding="utf-8") as f:
                base_config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {base_file}: {e}") from e

        # Load stage-specific configuration if it exists
        stage_file = config_dir / f"stage_{stage}.yaml"
        stage_config: dict[str, Any] = {}
        if stage_file.exists():
            try:
                with stage_file.open(encoding="utf-8") as f:
                    stage_config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Invalid YAML in {stage_file}: {e}"
                ) from e

        # Merge configurations (stage overrides base)
        merged_config = StageConfigLoader._deep_merge(base_config, stage_config)

        # Ensure stage number is set
        merged_config["stage"] = stage

        # Validate and create StageConfig
        try:
            return StageConfig.model_validate(merged_config)
        except ValidationError as e:
            raise ValueError(
                f"Invalid configuration for stage {stage}: {e}"
            ) from e

