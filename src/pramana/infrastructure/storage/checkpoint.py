"""Checkpoint repository for model checkpoint persistence with metadata tracking."""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class CheckpointMetadata:
    """Metadata for a model checkpoint."""

    checkpoint_id: str
    stage: int
    step: int
    created_at: datetime
    metrics: dict[str, float]
    git_commit: str | None
    data_version: str | None
    path: Path

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "stage": self.stage,
            "step": self.step,
            "created_at": self.created_at.isoformat(),
            "metrics": self.metrics,
            "git_commit": self.git_commit,
            "data_version": self.data_version,
            "path": str(self.path),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        """Create CheckpointMetadata from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            stage=data["stage"],
            step=data["step"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metrics=data["metrics"],
            git_commit=data.get("git_commit"),
            data_version=data.get("data_version"),
            path=Path(data["path"]),
        )


class CheckpointRepository:
    """Repository for model checkpoints with metadata tracking."""

    def __init__(self, base_path: Path) -> None:
        """Initialize checkpoint repository.

        Args:
            base_path: Base directory for storing checkpoints.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash if in a git repository.

        Returns:
            Git commit hash as string, or None if not in git repo or error.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return None

    def save(
        self,
        model: Any,
        tokenizer: Any,
        stage: int,
        step: int,
        metrics: dict[str, float],
        data_version: str | None = None,
    ) -> CheckpointMetadata:
        """Save checkpoint with metadata.

        Args:
            model: Model to save (must have save_pretrained method).
            tokenizer: Tokenizer to save (must have save_pretrained method).
            stage: Training stage number.
            step: Training step number.
            metrics: Dictionary of metrics to store.
            data_version: Optional data version string.

        Returns:
            CheckpointMetadata for the saved checkpoint.
        """
        # Generate unique checkpoint ID
        checkpoint_id = str(uuid4())

        # Create checkpoint directory
        checkpoint_dir = self.base_path / f"stage_{stage}" / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        model.save_pretrained(str(checkpoint_dir))
        tokenizer.save_pretrained(str(checkpoint_dir))

        # Capture git commit
        git_commit = self._get_git_commit()

        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            stage=stage,
            step=step,
            created_at=datetime.now(),
            metrics=metrics,
            git_commit=git_commit,
            data_version=data_version,
            path=checkpoint_dir,
        )

        # Save metadata JSON
        metadata_file = checkpoint_dir / "metadata.json"
        with metadata_file.open("w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        return metadata

    def load_latest(self, stage: int) -> tuple[Any, Any, CheckpointMetadata]:
        """Load the latest checkpoint for a stage.

        Args:
            stage: Training stage number.

        Returns:
            Tuple of (model, tokenizer, metadata) for the latest checkpoint.

        Raises:
            ValueError: If no checkpoints found for the stage.
            ImportError: If transformers is not installed.
        """
        checkpoints = self.list_checkpoints(stage=stage)
        if not checkpoints:
            raise ValueError(f"No checkpoints found for stage {stage}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers package is required for loading checkpoints. "
                "Install with: pip install transformers"
            ) from e

        # Get latest checkpoint (sorted by step descending)
        latest = checkpoints[0]

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(str(latest.path))
        tokenizer = AutoTokenizer.from_pretrained(str(latest.path))

        return model, tokenizer, latest

    def list_checkpoints(self, stage: int | None = None) -> list[CheckpointMetadata]:
        """List available checkpoints.

        Args:
            stage: Optional stage number to filter by. If None, returns all stages.

        Returns:
            List of CheckpointMetadata, sorted by step descending (latest first).
        """
        checkpoints: list[CheckpointMetadata] = []

        if stage is not None:
            # List checkpoints for specific stage
            stage_dir = self.base_path / f"stage_{stage}"
            if stage_dir.exists():
                checkpoints.extend(self._load_checkpoints_from_dir(stage_dir))
        else:
            # List checkpoints from all stages
            for stage_dir in self.base_path.iterdir():
                if stage_dir.is_dir() and stage_dir.name.startswith("stage_"):
                    checkpoints.extend(self._load_checkpoints_from_dir(stage_dir))

        # Sort by step descending (latest first)
        checkpoints.sort(key=lambda c: c.step, reverse=True)

        return checkpoints

    def _load_checkpoints_from_dir(self, stage_dir: Path) -> list[CheckpointMetadata]:
        """Load checkpoint metadata from a stage directory.

        Args:
            stage_dir: Directory containing checkpoint subdirectories.

        Returns:
            List of CheckpointMetadata objects.
        """
        checkpoints: list[CheckpointMetadata] = []

        for checkpoint_dir in stage_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue

            metadata_file = checkpoint_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with metadata_file.open() as f:
                    metadata_dict = json.load(f)
                metadata = CheckpointMetadata.from_dict(metadata_dict)
                checkpoints.append(metadata)
            except (json.JSONDecodeError, KeyError, ValueError):
                # Skip invalid metadata files
                continue

        return checkpoints
