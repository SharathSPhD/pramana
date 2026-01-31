"""Tests for CheckpointRepository."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pytest

from pramana.infrastructure.storage.checkpoint_repository import (
    CheckpointMetadata,
    CheckpointRepository,
)


class TestCheckpointMetadata:
    """Test CheckpointMetadata."""

    def test_metadata_creation(self) -> None:
        """Test creating CheckpointMetadata."""
        metadata = CheckpointMetadata(
            checkpoint_id="test-001",
            stage=0,
            epoch=1,
            step=100,
            metrics={"loss": 0.5},
        )
        assert metadata.checkpoint_id == "test-001"
        assert metadata.stage == 0
        assert metadata.epoch == 1
        assert metadata.step == 100
        assert metadata.metrics == {"loss": 0.5}


class TestCheckpointRepository:
    """Test CheckpointRepository."""

    def test_initialization(self) -> None:
        """Test repository initialization."""
        with TemporaryDirectory() as tmpdir:
            repo = CheckpointRepository(Path(tmpdir))
            assert repo.base_dir == Path(tmpdir)
            assert repo.base_dir.exists()

    def test_save_checkpoint(self) -> None:
        """Test saving checkpoint."""
        with TemporaryDirectory() as tmpdir:
            repo = CheckpointRepository(Path(tmpdir))

            mock_model = Mock()
            mock_model.save_pretrained = Mock()
            mock_tokenizer = Mock()
            mock_tokenizer.save_pretrained = Mock()

            metadata = CheckpointMetadata(
                checkpoint_id="test-001",
                stage=0,
            )

            checkpoint_path = repo.save(
                checkpoint_id="test-001",
                model=mock_model,
                tokenizer=mock_tokenizer,
                metadata=metadata,
            )

            assert checkpoint_path.exists()
            assert (checkpoint_path / "metadata.json").exists()
            mock_model.save_pretrained.assert_called_once()
            mock_tokenizer.save_pretrained.assert_called_once()
