"""Tests for StageConfig models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from pramana.config.loader import (
    DataConfig,
    EvaluationConfig,
    LoRAConfig,
    ModelConfig,
    StageConfig,
    TrainingParams,
)


class TestModelConfig:
    """Test ModelConfig model."""

    def test_valid_model_config(self) -> None:
        """Test creating valid ModelConfig."""
        config = ModelConfig(name="meta-llama/Llama-3.2-3B", revision="main")
        assert config.name == "meta-llama/Llama-3.2-3B"
        assert config.revision == "main"

    def test_model_config_default_revision(self) -> None:
        """Test ModelConfig with default revision."""
        config = ModelConfig(name="meta-llama/Llama-3.2-3B")
        assert config.revision == "main"

    def test_model_config_missing_name(self) -> None:
        """Test ModelConfig requires name."""
        with pytest.raises(ValidationError):
            ModelConfig()


class TestLoRAConfig:
    """Test LoRAConfig model."""

    def test_valid_lora_config(self) -> None:
        """Test creating valid LoRAConfig."""
        config = LoRAConfig(
            rank=64,
            alpha=128,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        assert config.rank == 64
        assert config.alpha == 128
        assert config.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]

    def test_lora_config_defaults(self) -> None:
        """Test LoRAConfig with default values."""
        config = LoRAConfig(rank=64)
        assert config.rank == 64
        assert config.alpha == 64  # Defaults to rank
        assert config.target_modules == ["all-linear"]  # Default

    def test_lora_config_missing_rank(self) -> None:
        """Test LoRAConfig requires rank."""
        with pytest.raises(ValidationError):
            LoRAConfig()


class TestTrainingParams:
    """Test TrainingParams model."""

    def test_valid_training_params(self) -> None:
        """Test creating valid TrainingParams."""
        params = TrainingParams(
            learning_rate=2e-5,
            batch_size=4,
            gradient_accumulation_steps=4,
            epochs=10,
            max_steps=None,
            warmup_steps=100,
            weight_decay=0.01,
            max_grad_norm=1.0,
        )
        assert params.learning_rate == 2e-5
        assert params.batch_size == 4
        assert params.gradient_accumulation_steps == 4
        assert params.epochs == 10
        assert params.max_steps is None
        assert params.warmup_steps == 100
        assert params.weight_decay == 0.01
        assert params.max_grad_norm == 1.0

    def test_training_params_defaults(self) -> None:
        """Test TrainingParams with default values."""
        params = TrainingParams(learning_rate=2e-5, batch_size=4, epochs=10)
        assert params.gradient_accumulation_steps == 1
        assert params.max_steps is None
        assert params.warmup_steps == 0
        assert params.weight_decay == 0.0
        assert params.max_grad_norm == 1.0

    def test_training_params_validation(self) -> None:
        """Test TrainingParams validation."""
        # Negative learning rate should fail
        with pytest.raises(ValidationError):
            TrainingParams(learning_rate=-1e-5, batch_size=4, epochs=10)

        # Zero batch size should fail
        with pytest.raises(ValidationError):
            TrainingParams(learning_rate=2e-5, batch_size=0, epochs=10)

        # Negative epochs should fail
        with pytest.raises(ValidationError):
            TrainingParams(learning_rate=2e-5, batch_size=4, epochs=-1)


class TestDataConfig:
    """Test DataConfig model."""

    def test_valid_data_config(self) -> None:
        """Test creating valid DataConfig."""
        config = DataConfig(
            train_path=Path("/data/train.jsonl"),
            eval_path=Path("/data/eval.jsonl"),
            max_length=4096,
        )
        assert config.train_path == Path("/data/train.jsonl")
        assert config.eval_path == Path("/data/eval.jsonl")
        assert config.max_length == 4096

    def test_data_config_defaults(self) -> None:
        """Test DataConfig with default values."""
        config = DataConfig(train_path=Path("/data/train.jsonl"))
        assert config.eval_path is None
        assert config.max_length == 2048  # Default

    def test_data_config_validation(self) -> None:
        """Test DataConfig validation."""
        # Negative max_length should fail
        with pytest.raises(ValidationError):
            DataConfig(
                train_path=Path("/data/train.jsonl"), max_length=-1
            )

        # Zero max_length should fail
        with pytest.raises(ValidationError):
            DataConfig(
                train_path=Path("/data/train.jsonl"), max_length=0
            )


class TestEvaluationConfig:
    """Test EvaluationConfig model."""

    def test_valid_evaluation_config(self) -> None:
        """Test creating valid EvaluationConfig."""
        config = EvaluationConfig(
            tier1_threshold=0.9,
            tier2_threshold=0.8,
        )
        assert config.tier1_threshold == 0.9
        assert config.tier2_threshold == 0.8

    def test_evaluation_config_defaults(self) -> None:
        """Test EvaluationConfig with default values."""
        config = EvaluationConfig()
        assert config.tier1_threshold == 0.95
        assert config.tier2_threshold == 0.85

    def test_evaluation_config_validation(self) -> None:
        """Test EvaluationConfig validation."""
        # Threshold > 1.0 should fail
        with pytest.raises(ValidationError):
            EvaluationConfig(tier1_threshold=1.5)

        # Threshold < 0.0 should fail
        with pytest.raises(ValidationError):
            EvaluationConfig(tier1_threshold=-0.1)


class TestStageConfig:
    """Test StageConfig model."""

    def test_valid_stage_config(self) -> None:
        """Test creating valid StageConfig."""
        config = StageConfig(
            stage=0,
            model=ModelConfig(name="meta-llama/Llama-3.2-3B"),
            lora=LoRAConfig(rank=32),
            training=TrainingParams(
                learning_rate=2e-5, batch_size=4, epochs=10
            ),
            data=DataConfig(train_path=Path("/data/train.jsonl")),
            evaluation=EvaluationConfig(),
        )
        assert config.stage == 0
        assert config.model.name == "meta-llama/Llama-3.2-3B"
        assert config.lora.rank == 32
        assert config.training.learning_rate == 2e-5
        assert config.data.train_path == Path("/data/train.jsonl")

    def test_stage_config_validation(self) -> None:
        """Test StageConfig validation."""
        # Stage < 0 should fail
        with pytest.raises(ValidationError):
            StageConfig(
                stage=-1,
                model=ModelConfig(name="test"),
                lora=LoRAConfig(rank=32),
                training=TrainingParams(
                    learning_rate=2e-5, batch_size=4, epochs=10
                ),
                data=DataConfig(train_path=Path("/data/train.jsonl")),
                evaluation=EvaluationConfig(),
            )

        # Stage > 4 should fail
        with pytest.raises(ValidationError):
            StageConfig(
                stage=5,
                model=ModelConfig(name="test"),
                lora=LoRAConfig(rank=32),
                training=TrainingParams(
                    learning_rate=2e-5, batch_size=4, epochs=10
                ),
                data=DataConfig(train_path=Path("/data/train.jsonl")),
                evaluation=EvaluationConfig(),
            )

    def test_stage_config_all_stages(self) -> None:
        """Test StageConfig for all valid stages."""
        for stage in range(5):
            config = StageConfig(
                stage=stage,
                model=ModelConfig(name="test"),
                lora=LoRAConfig(rank=32),
                training=TrainingParams(
                    learning_rate=2e-5, batch_size=4, epochs=10
                ),
                data=DataConfig(train_path=Path("/data/train.jsonl")),
                evaluation=EvaluationConfig(),
            )
            assert config.stage == stage
