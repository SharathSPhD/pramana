"""Checkpoint repository for saving and loading model checkpoints."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel


class CheckpointMetadata(BaseModel):
    """Metadata for a checkpoint."""

    checkpoint_id: str
    stage: int
    epoch: int | None = None
    step: int | None = None
    metrics: dict[str, Any] | None = None


class CheckpointRepository:
    """Repository for managing model checkpoints."""

    def __init__(self, base_dir: Path) -> None:
        """Initialize checkpoint repository.

        Args:
            base_dir: Base directory for checkpoints
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        checkpoint_id: str,
        model: Any,
        tokenizer: Any,
        metadata: CheckpointMetadata | None = None,
    ) -> Path:
        """Save checkpoint.

        Args:
            checkpoint_id: Unique checkpoint identifier
            model: Model to save
            tokenizer: Tokenizer to save
            metadata: Optional checkpoint metadata

        Returns:
            Path to saved checkpoint directory
        """
        checkpoint_dir = self.base_dir / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        model_path = checkpoint_dir / "model"
        tokenizer_path = checkpoint_dir / "tokenizer"

        # Use model's save_pretrained if available
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(model_path))
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(tokenizer_path))

        # Save metadata if provided
        if metadata:
            metadata_path = checkpoint_dir / "metadata.json"
            metadata_path.write_text(metadata.model_dump_json(indent=2))

        return checkpoint_dir
