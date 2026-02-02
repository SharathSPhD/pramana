"""Supervised Fine-Tuning trainer for Stages 0-2."""

from typing import Any

from datasets import Dataset, load_dataset
from transformers import TrainingArguments

from pramana.application.training.base import BaseTrainer, TrainingResult
from pramana.application.training.callbacks import NyayaMetricsCallback
from pramana.config.loader import StageConfig
from pramana.infrastructure.ml.unsloth_adapter import UnslothAdapter
from pramana.infrastructure.storage.checkpoint_repository import (
    CheckpointMetadata,
    CheckpointRepository,
)

try:
    from trl import SFTTrainer
except ImportError:
    SFTTrainer = None  # type: ignore[assignment]

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


class SupervisedFineTuningTrainer(BaseTrainer):
    """SFT trainer for Stages 0-2."""

    def __init__(
        self,
        adapter: UnslothAdapter,
        checkpoint_repo: CheckpointRepository,
    ) -> None:
        """Initialize SFT trainer.

        Args:
            adapter: Unsloth adapter for model operations
            checkpoint_repo: Checkpoint repository for saving models
        """
        super().__init__()
        self.config: StageConfig | None = None
        self.adapter = adapter
        self.checkpoint_repo = checkpoint_repo
        self.train_dataset: Dataset | None = None
        self.eval_dataset: Dataset | None = None
        self.trainer: Any = None

    def _setup(self, config: StageConfig) -> None:
        """Load model, apply LoRA, setup tokenizer.

        Args:
            config: Stage configuration
        """
        self.config = config
        # Load model
        model, tokenizer = self.adapter.load_model(
            model_name=config.model.name,
            quantization="4bit",  # Always use 4-bit for efficiency
        )

        # Apply LoRA
        model = self.adapter.apply_lora(model, config.lora)

        self.model = model
        self.tokenizer = tokenizer

    def _prepare_data(self) -> None:
        """Load training data, create DataLoader."""
        if self.config is None:
            raise ValueError("Config must be set via _setup() before _prepare_data()")

        config = self.config

        # Load training dataset
        if config.data.train_path.suffix == ".jsonl":
            train_dataset = load_dataset(
                "json",
                data_files=str(config.data.train_path),
                split="train",
            )
        else:
            # Assume it's a directory with dataset files
            train_dataset = load_dataset(str(config.data.train_path), split="train")

        # Format dataset with prompt template
        train_dataset = train_dataset.map(
            self._format_nyaya_example,
            remove_columns=[
                col
                for col in train_dataset.column_names
                if col not in ["text"]
            ],
        )

        self.train_dataset = train_dataset

        # Load evaluation dataset if provided
        if config.data.eval_path:
            if config.data.eval_path.suffix == ".jsonl":
                eval_dataset = load_dataset(
                    "json",
                    data_files=str(config.data.eval_path),
                    split="train",
                )
            else:
                eval_dataset = load_dataset(
                    str(config.data.eval_path), split="train"
                )

            eval_dataset = eval_dataset.map(
                self._format_nyaya_example,
                remove_columns=[
                    col
                    for col in eval_dataset.column_names
                    if col not in ["text"]
                ],
            )

            self.eval_dataset = eval_dataset
        else:
            self.eval_dataset = None

    def _format_nyaya_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Format example with Nyaya prompt template.

        Args:
            example: Raw example dictionary

        Returns:
            Formatted example with 'text' field
        """
        # Extract problem and solution
        problem = example.get("problem", "")
        solution = example.get("solution", "")

        # Format as instruction-response pair
        # This is a simple template - can be enhanced based on actual data format
        text = f"### Instruction:\n{problem}\n\n### Response:\n{solution}"

        return {"text": text}

    def _build_validation_prompt(self) -> str | None:
        """Build a prompt for generating validation samples."""
        if self.eval_dataset is None or len(self.eval_dataset) == 0:
            return None
        sample_text = self.eval_dataset[0].get("text", "")
        if not sample_text:
            return None
        if "### Instruction:" in sample_text:
            instruction = sample_text.split("### Instruction:")[1].split(
                "### Response:"
            )[0]
            instruction = instruction.strip()
        else:
            instruction = sample_text.strip()
        return f"### Instruction:\n{instruction}\n\n### Response:\n"

    def _train(self) -> TrainingResult:
        """Run SFT training loop via HuggingFace Trainer.

        Returns:
            Training result with metrics
        """
        if self.config is None:
            raise ValueError("Config must be set via _setup() before _train()")

        config = self.config

        if SFTTrainer is None:
            raise ImportError(
                "trl is not installed. Install with: pip install trl"
            )

        # Setup W&B if configured
        wandb_project = getattr(config, "wandb_project", None)
        if wandb_project and wandb is not None:
            wandb.init(
                project=wandb_project,
                name=f"stage_{config.stage}_sft",
                config=config.model_dump(),
            )

        # Prepare warmup parameter (HuggingFace supports both warmup_steps and warmup_ratio)
        warmup_kwargs: dict[str, Any] = {}
        if config.training.warmup_ratio is not None:
            warmup_kwargs["warmup_ratio"] = config.training.warmup_ratio
        elif config.training.warmup_steps is not None:
            warmup_kwargs["warmup_steps"] = config.training.warmup_steps
        else:
            warmup_kwargs["warmup_steps"] = 0

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=str(self.checkpoint_repo.base_dir / "training_output"),
            num_train_epochs=config.training.epochs,
            per_device_train_batch_size=config.training.batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            max_grad_norm=config.training.max_grad_norm,
            logging_steps=10,
            save_steps=100,
            eval_steps=100 if self.eval_dataset else None,
            save_strategy="steps",
            eval_strategy="steps" if self.eval_dataset else "no",
            load_best_model_at_end=bool(self.eval_dataset),
            report_to="wandb" if wandb_project and wandb else None,
            fp16=False,  # Use bf16 if supported
            bf16=False,  # Disable bf16 for CPU compatibility in tests
            optim="adamw_torch",
            **warmup_kwargs,
        )

        # Create SFT trainer
        callbacks = []
        validation_prompt = self._build_validation_prompt()
        if validation_prompt:
            callbacks.append(
                NyayaMetricsCallback(
                    tokenizer=self.tokenizer,
                    prompt=validation_prompt,
                    max_new_tokens=min(512, config.data.max_length),
                )
            )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            dataset_text_field="text",
            max_seq_length=config.data.max_length,
            args=training_args,
            callbacks=callbacks,
        )

        # Train
        train_output = self.trainer.train()

        # Save final checkpoint
        checkpoint_id = f"stage_{config.stage}_final"
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            stage=config.stage,
            metrics={
                "train_loss": train_output.training_loss,
                "train_runtime": train_output.metrics.get("train_runtime", 0.0),
            },
        )

        self.checkpoint_repo.save(
            checkpoint_id=checkpoint_id,
            model=self.model,
            tokenizer=self.tokenizer,
            metadata=metadata,
        )

        # Return training result
        return TrainingResult(
            final_loss=train_output.training_loss,
            best_checkpoint_path=self.checkpoint_repo.base_dir / checkpoint_id,
            metrics=train_output.metrics,
            training_time_seconds=train_output.metrics.get("train_runtime", 0.0),
        )
