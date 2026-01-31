"""Tests for BaseTrainer."""

import pytest

from pramana.application.training.base import BaseTrainer, TrainingResult


class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_training_result_creation(self) -> None:
        """Test creating TrainingResult."""
        from pathlib import Path

        result = TrainingResult(
            final_loss=0.5,
            best_checkpoint_path=Path("/tmp/checkpoint"),
            training_time_seconds=100.0,
            metrics={"accuracy": 0.9},
        )
        assert result.final_loss == 0.5
        assert result.training_time_seconds == 100.0
        assert result.metrics == {"accuracy": 0.9}

    def test_training_result_default_metrics(self) -> None:
        """Test TrainingResult with default metrics."""
        from pathlib import Path

        result = TrainingResult(
            final_loss=0.5,
            best_checkpoint_path=Path("/tmp/checkpoint"),
            training_time_seconds=100.0,
            metrics={},
        )
        assert result.metrics == {}


class TestBaseTrainer:
    """Test BaseTrainer abstract class."""

    def test_base_trainer_is_abstract(self) -> None:
        """Test BaseTrainer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTrainer()  # type: ignore[misc]
