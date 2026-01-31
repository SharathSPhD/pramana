"""Tests for CheckpointRepository with metadata tracking."""

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from pramana.infrastructure.storage.checkpoint import (
    CheckpointMetadata,
    CheckpointRepository,
)


class TestCheckpointMetadata:
    """Test CheckpointMetadata dataclass."""

    def test_checkpoint_metadata_creation(self) -> None:
        """Test creating CheckpointMetadata with all fields."""
        metadata = CheckpointMetadata(
            checkpoint_id="test-id-123",
            stage=0,
            step=100,
            created_at=datetime(2025, 1, 31, 12, 0, 0),
            metrics={"loss": 0.5, "accuracy": 0.95},
            git_commit="abc123def",
            data_version="v1.0.0",
            path=Path("/checkpoints/stage_0/test-id-123"),
        )
        assert metadata.checkpoint_id == "test-id-123"
        assert metadata.stage == 0
        assert metadata.step == 100
        assert metadata.metrics == {"loss": 0.5, "accuracy": 0.95}
        assert metadata.git_commit == "abc123def"
        assert metadata.data_version == "v1.0.0"
        assert metadata.path == Path("/checkpoints/stage_0/test-id-123")

    def test_checkpoint_metadata_optional_fields(self) -> None:
        """Test CheckpointMetadata with optional fields as None."""
        metadata = CheckpointMetadata(
            checkpoint_id="test-id-456",
            stage=1,
            step=200,
            created_at=datetime(2025, 1, 31, 12, 0, 0),
            metrics={},
            git_commit=None,
            data_version=None,
            path=Path("/checkpoints/stage_1/test-id-456"),
        )
        assert metadata.git_commit is None
        assert metadata.data_version is None

    def test_checkpoint_metadata_serialization(self, tmp_path: Path) -> None:
        """Test that CheckpointMetadata can be serialized to JSON."""
        metadata = CheckpointMetadata(
            checkpoint_id="test-id-789",
            stage=2,
            step=300,
            created_at=datetime(2025, 1, 31, 12, 0, 0),
            metrics={"loss": 0.3, "f1": 0.92},
            git_commit="def456ghi",
            data_version="v2.0.0",
            path=tmp_path / "checkpoint",
        )
        # Convert to dict for JSON serialization
        metadata_dict = {
            "checkpoint_id": metadata.checkpoint_id,
            "stage": metadata.stage,
            "step": metadata.step,
            "created_at": metadata.created_at.isoformat(),
            "metrics": metadata.metrics,
            "git_commit": metadata.git_commit,
            "data_version": metadata.data_version,
            "path": str(metadata.path),
        }
        json_str = json.dumps(metadata_dict)
        assert "test-id-789" in json_str
        assert "def456ghi" in json_str


class TestCheckpointRepository:
    """Test CheckpointRepository save/load/list operations."""

    def test_repository_initialization(self, tmp_path: Path) -> None:
        """Test CheckpointRepository initialization."""
        repo = CheckpointRepository(base_path=tmp_path)
        assert repo.base_path == tmp_path
        assert tmp_path.exists()

    def test_save_checkpoint_creates_directory(self, tmp_path: Path) -> None:
        """Test that save creates checkpoint directory structure."""
        repo = CheckpointRepository(base_path=tmp_path)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("pramana.infrastructure.storage.checkpoint.subprocess.run") as mock_git:
            mock_git.return_value = MagicMock(stdout="abc123\n", returncode=0)
            metadata = repo.save(
                model=mock_model,
                tokenizer=mock_tokenizer,
                stage=0,
                step=100,
                metrics={"loss": 0.5},
            )

        # Check that checkpoint directory was created
        checkpoint_dir = tmp_path / f"stage_{metadata.stage}" / metadata.checkpoint_id
        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()

    def test_save_checkpoint_generates_uuid(self, tmp_path: Path) -> None:
        """Test that save generates UUID checkpoint IDs."""
        repo = CheckpointRepository(base_path=tmp_path)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("pramana.infrastructure.storage.checkpoint.subprocess.run") as mock_git:
            mock_git.return_value = MagicMock(stdout="abc123\n", returncode=0)
            metadata = repo.save(
                model=mock_model,
                tokenizer=mock_tokenizer,
                stage=0,
                step=100,
                metrics={"loss": 0.5},
            )

        # Verify checkpoint_id is a valid UUID
        try:
            UUID(metadata.checkpoint_id)
        except ValueError:
            pytest.fail(f"checkpoint_id {metadata.checkpoint_id} is not a valid UUID")

    def test_save_checkpoint_captures_git_commit(self, tmp_path: Path) -> None:
        """Test that save captures git commit hash when in git repo."""
        repo = CheckpointRepository(base_path=tmp_path)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("pramana.infrastructure.storage.checkpoint.subprocess.run") as mock_git:
            mock_git.return_value = MagicMock(stdout="abc123def456\n", returncode=0)
            metadata = repo.save(
                model=mock_model,
                tokenizer=mock_tokenizer,
                stage=0,
                step=100,
                metrics={"loss": 0.5},
            )

        assert metadata.git_commit == "abc123def456"  # strip() removes newline

    def test_save_checkpoint_handles_no_git_repo(self, tmp_path: Path) -> None:
        """Test that save handles case when not in git repo."""
        repo = CheckpointRepository(base_path=tmp_path)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("pramana.infrastructure.storage.checkpoint.subprocess.run") as mock_git:
            mock_git.return_value = MagicMock(returncode=128)  # Git error
            metadata = repo.save(
                model=mock_model,
                tokenizer=mock_tokenizer,
                stage=0,
                step=100,
                metrics={"loss": 0.5},
            )

        assert metadata.git_commit is None

    def test_save_checkpoint_stores_metadata_json(self, tmp_path: Path) -> None:
        """Test that save stores metadata.json file."""
        repo = CheckpointRepository(base_path=tmp_path)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("pramana.infrastructure.storage.checkpoint.subprocess.run") as mock_git:
            mock_git.return_value = MagicMock(stdout="abc123\n", returncode=0)
            metadata = repo.save(
                model=mock_model,
                tokenizer=mock_tokenizer,
                stage=0,
                step=100,
                metrics={"loss": 0.5, "accuracy": 0.95},
                data_version="v1.0.0",
            )

        # Check metadata.json exists
        metadata_file = metadata.path / "metadata.json"
        assert metadata_file.exists()

        # Verify metadata.json content
        with open(metadata_file) as f:
            stored_metadata = json.load(f)

        assert stored_metadata["checkpoint_id"] == metadata.checkpoint_id
        assert stored_metadata["stage"] == 0
        assert stored_metadata["step"] == 100
        assert stored_metadata["metrics"] == {"loss": 0.5, "accuracy": 0.95}
        assert stored_metadata["data_version"] == "v1.0.0"
        assert stored_metadata["git_commit"] == "abc123"

    def test_save_checkpoint_saves_model_and_tokenizer(
        self, tmp_path: Path
    ) -> None:
        """Test that save calls model.save_pretrained and tokenizer.save_pretrained."""
        repo = CheckpointRepository(base_path=tmp_path)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("pramana.infrastructure.storage.checkpoint.subprocess.run") as mock_git:
            mock_git.return_value = MagicMock(stdout="abc123\n", returncode=0)
            metadata = repo.save(
                model=mock_model,
                tokenizer=mock_tokenizer,
                stage=0,
                step=100,
                metrics={"loss": 0.5},
            )

        # Verify save_pretrained was called with correct path
        mock_model.save_pretrained.assert_called_once_with(str(metadata.path))
        mock_tokenizer.save_pretrained.assert_called_once_with(str(metadata.path))

    def test_load_latest_returns_checkpoint(self, tmp_path: Path) -> None:
        """Test that load_latest loads the most recent checkpoint."""
        repo = CheckpointRepository(base_path=tmp_path)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Create two checkpoints
        with patch("pramana.infrastructure.storage.checkpoint.subprocess.run") as mock_git:
            mock_git.return_value = MagicMock(stdout="abc123\n", returncode=0)

            # Save first checkpoint
            metadata1 = repo.save(
                model=mock_model,
                tokenizer=mock_tokenizer,
                stage=0,
                step=100,
                metrics={"loss": 0.5},
            )

            # Save second checkpoint (later step)
            metadata2 = repo.save(
                model=mock_model,
                tokenizer=mock_tokenizer,
                stage=0,
                step=200,
                metrics={"loss": 0.3},
            )

        # Mock model/tokenizer loading
        mock_loaded_model = MagicMock()
        mock_loaded_tokenizer = MagicMock()

        # Create mock transformers module
        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained = MagicMock(
            return_value=mock_loaded_model
        )
        mock_transformers.AutoTokenizer.from_pretrained = MagicMock(
            return_value=mock_loaded_tokenizer
        )

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            model, tokenizer, loaded_metadata = repo.load_latest(stage=0)

        # Should load the latest checkpoint (step 200)
        assert loaded_metadata.checkpoint_id == metadata2.checkpoint_id
        assert loaded_metadata.step == 200
        assert model == mock_loaded_model
        assert tokenizer == mock_loaded_tokenizer

    def test_load_latest_raises_when_no_checkpoints(self, tmp_path: Path) -> None:
        """Test that load_latest raises when no checkpoints exist."""
        repo = CheckpointRepository(base_path=tmp_path)

        with pytest.raises(ValueError, match="No checkpoints found"):
            repo.load_latest(stage=0)

    def test_list_checkpoints_returns_all_for_stage(self, tmp_path: Path) -> None:
        """Test that list_checkpoints returns all checkpoints for a stage."""
        repo = CheckpointRepository(base_path=tmp_path)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("pramana.infrastructure.storage.checkpoint.subprocess.run") as mock_git:
            mock_git.return_value = MagicMock(stdout="abc123\n", returncode=0)

            # Create checkpoints for stage 0
            metadata1 = repo.save(
                model=mock_model,
                tokenizer=mock_tokenizer,
                stage=0,
                step=100,
                metrics={"loss": 0.5},
            )
            metadata2 = repo.save(
                model=mock_model,
                tokenizer=mock_tokenizer,
                stage=0,
                step=200,
                metrics={"loss": 0.3},
            )

            # Create checkpoint for stage 1
            metadata3 = repo.save(
                model=mock_model,
                tokenizer=mock_tokenizer,
                stage=1,
                step=50,
                metrics={"loss": 0.7},
            )

        checkpoints = repo.list_checkpoints(stage=0)
        assert len(checkpoints) == 2
        checkpoint_ids = {c.checkpoint_id for c in checkpoints}
        assert metadata1.checkpoint_id in checkpoint_ids
        assert metadata2.checkpoint_id in checkpoint_ids
        assert metadata3.checkpoint_id not in checkpoint_ids

    def test_list_checkpoints_returns_all_when_stage_none(
        self, tmp_path: Path
    ) -> None:
        """Test that list_checkpoints returns all checkpoints when stage is None."""
        repo = CheckpointRepository(base_path=tmp_path)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("pramana.infrastructure.storage.checkpoint.subprocess.run") as mock_git:
            mock_git.return_value = MagicMock(stdout="abc123\n", returncode=0)

            # Create checkpoints for different stages
            metadata1 = repo.save(
                model=mock_model,
                tokenizer=mock_tokenizer,
                stage=0,
                step=100,
                metrics={"loss": 0.5},
            )
            metadata2 = repo.save(
                model=mock_model,
                tokenizer=mock_tokenizer,
                stage=1,
                step=200,
                metrics={"loss": 0.3},
            )

        checkpoints = repo.list_checkpoints(stage=None)
        assert len(checkpoints) == 2
        checkpoint_ids = {c.checkpoint_id for c in checkpoints}
        assert metadata1.checkpoint_id in checkpoint_ids
        assert metadata2.checkpoint_id in checkpoint_ids

    def test_list_checkpoints_returns_empty_when_none_exist(
        self, tmp_path: Path
    ) -> None:
        """Test that list_checkpoints returns empty list when no checkpoints exist."""
        repo = CheckpointRepository(base_path=tmp_path)
        checkpoints = repo.list_checkpoints(stage=0)
        assert checkpoints == []

    def test_list_checkpoints_sorted_by_step(self, tmp_path: Path) -> None:
        """Test that list_checkpoints returns checkpoints sorted by step."""
        repo = CheckpointRepository(base_path=tmp_path)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("pramana.infrastructure.storage.checkpoint.subprocess.run") as mock_git:
            mock_git.return_value = MagicMock(stdout="abc123\n", returncode=0)

            # Create checkpoints in non-sequential order
            repo.save(
                model=mock_model,
                tokenizer=mock_tokenizer,
                stage=0,
                step=300,
                metrics={"loss": 0.3},
            )
            repo.save(
                model=mock_model,
                tokenizer=mock_tokenizer,
                stage=0,
                step=100,
                metrics={"loss": 0.5},
            )
            repo.save(
                model=mock_model,
                tokenizer=mock_tokenizer,
                stage=0,
                step=200,
                metrics={"loss": 0.4},
            )

        checkpoints = repo.list_checkpoints(stage=0)
        assert len(checkpoints) == 3
        # Should be sorted by step (descending, latest first)
        steps = [c.step for c in checkpoints]
        assert steps == [300, 200, 100]
