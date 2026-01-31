"""Tests for SupervisedFineTuningTrainer."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from datasets import Dataset

from pramana.application.training.sft import SupervisedFineTuningTrainer
from pramana.config.loader import (
    DataConfig,
    LoRAConfig,
    ModelConfig,
    StageConfig,
    TrainingParams,
)
from pramana.infrastructure.ml.unsloth_adapter import UnslothAdapter
from pramana.infrastructure.storage.checkpoint_repository import (
    CheckpointRepository,
)


@pytest.fixture
def mock_adapter() -> UnslothAdapter:
    """Fixture for UnslothAdapter with mocked methods."""
    adapter = UnslothAdapter()
    adapter.load_model = Mock(return_value=(Mock(), Mock()))  # type: ignore[method-assign]
    adapter.apply_lora = Mock(return_value=Mock())  # type: ignore[method-assign]
    return adapter


@pytest.fixture
def mock_checkpoint_repo(tmp_path: Path) -> CheckpointRepository:
    """Fixture for CheckpointRepository."""
    return CheckpointRepository(tmp_path / "checkpoints")


@pytest.fixture
def stage_config() -> StageConfig:
    """Fixture for StageConfig."""
    return StageConfig(
        stage=0,
        model=ModelConfig(name="meta-llama/Llama-3.2-3B"),
        lora=LoRAConfig(rank=32, alpha=32),
        training=TrainingParams(
            learning_rate=2e-5,
            batch_size=4,
            epochs=10,
            gradient_accumulation_steps=4,
        ),
        data=DataConfig(train_path=Path("/data/train.jsonl")),
    )


class TestSupervisedFineTuningTrainer:
    """Test SupervisedFineTuningTrainer."""

    def test_initialization(
        self,
        mock_adapter: UnslothAdapter,
        mock_checkpoint_repo: CheckpointRepository,
    ) -> None:
        """Test trainer initialization."""
        trainer = SupervisedFineTuningTrainer(
            adapter=mock_adapter,
            checkpoint_repo=mock_checkpoint_repo,
        )

        assert trainer.adapter == mock_adapter
        assert trainer.checkpoint_repo == mock_checkpoint_repo
        assert not hasattr(trainer, "model") or trainer.model is None
        assert not hasattr(trainer, "tokenizer") or trainer.tokenizer is None

    def test_setup_loads_model_and_applies_lora(
        self,
        mock_adapter: UnslothAdapter,
        mock_checkpoint_repo: CheckpointRepository,
        stage_config: StageConfig,
    ) -> None:
        """Test _setup loads model and applies LoRA."""
        trainer = SupervisedFineTuningTrainer(
            adapter=mock_adapter,
            checkpoint_repo=mock_checkpoint_repo,
        )

        trainer._setup(stage_config)

        # Verify adapter methods were called
        mock_adapter.load_model.assert_called_once()  # type: ignore[attr-defined]
        mock_adapter.apply_lora.assert_called_once()  # type: ignore[attr-defined]

        # Verify model and tokenizer are set
        assert trainer.model is not None
        assert trainer.tokenizer is not None
