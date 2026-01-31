"""Tests for BaseTrainer with Template Method pattern."""

from abc import ABC
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest

from pramana.config.loader import (
    DataConfig,
    EvaluationConfig,
    LoRAConfig,
    ModelConfig,
    StageConfig,
    TrainingParams,
)
from pramana.application.training.base import BaseTrainer, TrainingResult


class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_training_result_creation(self) -> None:
        """Test creating TrainingResult with all fields."""
        result = TrainingResult(
            final_loss=0.5,
            best_checkpoint_path=Path("/models/checkpoint-100"),
            metrics={"accuracy": 0.95, "f1": 0.92},
            training_time_seconds=3600.0,
        )
        assert result.final_loss == 0.5
        assert result.best_checkpoint_path == Path("/models/checkpoint-100")
        assert result.metrics == {"accuracy": 0.95, "f1": 0.92}
        assert result.training_time_seconds == 3600.0

    def test_training_result_minimal(self) -> None:
        """Test TrainingResult with minimal fields."""
        result = TrainingResult(
            final_loss=1.0,
            best_checkpoint_path=Path("/models/checkpoint"),
            metrics={},
            training_time_seconds=0.0,
        )
        assert result.final_loss == 1.0
        assert result.metrics == {}


class ConcreteTrainer(BaseTrainer):
    """Concrete implementation for testing."""

    def __init__(self) -> None:
        """Initialize concrete trainer."""
        self.setup_called = False
        self.prepare_data_called = False
        self.train_called = False
        self.cleanup_called = False
        self.setup_config: StageConfig | None = None

    def _setup(self, config: StageConfig) -> None:
        """Setup implementation."""
        self.setup_called = True
        self.setup_config = config

    def _prepare_data(self) -> None:
        """Prepare data implementation."""
        self.prepare_data_called = True

    def _train(self) -> TrainingResult:
        """Train implementation."""
        self.train_called = True
        return TrainingResult(
            final_loss=0.5,
            best_checkpoint_path=Path("/models/checkpoint"),
            metrics={"loss": 0.5},
            training_time_seconds=100.0,
        )


class TestBaseTrainer:
    """Test BaseTrainer template method pattern."""

    def test_train_template_method_workflow(self) -> None:
        """Test that train() calls all methods in correct order."""
        trainer = ConcreteTrainer()
        config = StageConfig(
            stage=0,
            model=ModelConfig(name="test-model"),
            lora=LoRAConfig(rank=32),
            training=TrainingParams(
                learning_rate=2e-5, batch_size=4, epochs=10
            ),
            data=DataConfig(train_path=Path("/data/train.jsonl")),
            evaluation=EvaluationConfig(),
        )

        result = trainer.train(config)

        # Verify all methods were called
        assert trainer.setup_called is True
        assert trainer.prepare_data_called is True
        assert trainer.train_called is True
        assert trainer.setup_config == config

        # Verify result
        assert isinstance(result, TrainingResult)
        assert result.final_loss == 0.5
        assert result.best_checkpoint_path == Path("/models/checkpoint")

    def test_train_calls_cleanup(self) -> None:
        """Test that cleanup is called after training."""
        trainer = ConcreteTrainer()

        # Override cleanup to track calls
        cleanup_called = False

        def custom_cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        trainer._cleanup = custom_cleanup  # type: ignore[assignment]

        config = StageConfig(
            stage=0,
            model=ModelConfig(name="test-model"),
            lora=LoRAConfig(rank=32),
            training=TrainingParams(
                learning_rate=2e-5, batch_size=4, epochs=10
            ),
            data=DataConfig(train_path=Path("/data/train.jsonl")),
            evaluation=EvaluationConfig(),
        )

        trainer.train(config)

        assert cleanup_called is True

    def test_abstract_methods_must_be_implemented(self) -> None:
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # This should fail because abstract methods aren't implemented
            BaseTrainer()  # type: ignore[abstract]

    def test_cleanup_is_optional(self) -> None:
        """Test that cleanup hook is optional and doesn't raise."""
        trainer = ConcreteTrainer()
        config = StageConfig(
            stage=0,
            model=ModelConfig(name="test-model"),
            lora=LoRAConfig(rank=32),
            training=TrainingParams(
                learning_rate=2e-5, batch_size=4, epochs=10
            ),
            data=DataConfig(train_path=Path("/data/train.jsonl")),
            evaluation=EvaluationConfig(),
        )

        # Should not raise even if cleanup is not overridden
        result = trainer.train(config)
        assert isinstance(result, TrainingResult)

    def test_train_preserves_exceptions(self) -> None:
        """Test that exceptions from _train are propagated."""
        class FailingTrainer(BaseTrainer):
            def _setup(self, config: StageConfig) -> None:
                pass

            def _prepare_data(self) -> None:
                pass

            def _train(self) -> TrainingResult:
                raise ValueError("Training failed")

        trainer = FailingTrainer()
        config = StageConfig(
            stage=0,
            model=ModelConfig(name="test-model"),
            lora=LoRAConfig(rank=32),
            training=TrainingParams(
                learning_rate=2e-5, batch_size=4, epochs=10
            ),
            data=DataConfig(train_path=Path("/data/train.jsonl")),
            evaluation=EvaluationConfig(),
        )

        with pytest.raises(ValueError, match="Training failed"):
            trainer.train(config)
