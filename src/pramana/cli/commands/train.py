"""Training command for Pramana CLI."""

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from pramana.application.training.sft import SupervisedFineTuningTrainer
from pramana.config.loader import StageConfigLoader
from pramana.infrastructure.ml.unsloth_adapter import UnslothAdapter
from pramana.infrastructure.storage.checkpoint_repository import CheckpointRepository

console = Console()


def train_model(
    stage: int,
    config_path: Path | None = None,
    resume: Path | None = None,
) -> None:
    """Train a model for the specified stage.

    Args:
        stage: Training stage (0-4)
        config_path: Path to config YAML file
        resume: Path to checkpoint to resume from
    """
    if stage < 0 or stage > 4:
        console.print(f"[red]Error: Stage must be between 0 and 4, got {stage}[/red]")
        raise typer.Exit(1)

    # Determine config directory
    if config_path:
        config_dir = config_path.parent if config_path.is_file() else config_path
    else:
        # Default to configs/ directory in project root
        config_dir = Path(__file__).parent.parent.parent.parent.parent / "configs"

    if not config_dir.exists():
        console.print(f"[red]Error: Config directory not found: {config_dir}[/red]")
        raise typer.Exit(1)

    # Load configuration
    try:
        console.print(f"[cyan]Loading configuration for stage {stage}...[/cyan]")
        config = StageConfigLoader.load(stage, config_dir)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print(f"[red]Error: Invalid configuration: {e}[/red]")
        raise typer.Exit(1) from e

    # Initialize dependencies
    console.print(f"[cyan]Initializing trainer for stage {stage}...[/cyan]")
    adapter = UnslothAdapter()

    # Set up checkpoint repository (use config output directory or default)
    checkpoint_dir = Path("checkpoints") / f"stage_{stage}"
    checkpoint_repo = CheckpointRepository(checkpoint_dir)

    trainer = SupervisedFineTuningTrainer(
        adapter=adapter,
        checkpoint_repo=checkpoint_repo,
    )

    # Resume from checkpoint if provided
    if resume:
        if not resume.exists():
            console.print(f"[red]Error: Checkpoint not found: {resume}[/red]")
            raise typer.Exit(1)
        console.print(f"[cyan]Resuming from checkpoint: {resume}[/cyan]")
        # TODO: Implement checkpoint loading
        # trainer.load_checkpoint(resume)

    # Start training
    console.print(f"[green]Starting training for stage {stage}...[/green]")
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Training...", total=None)
            result = trainer.train(config)
            progress.update(task, completed=True)

        console.print("[green]Training completed successfully![/green]")
        console.print(f"Final loss: {result.final_loss:.4f}")
        console.print(f"Checkpoint: {result.best_checkpoint_path}")
    except Exception as e:
        console.print(f"[red]Error during training: {e}[/red]")
        raise typer.Exit(1) from e


def train(
    stage: int = typer.Option(..., "--stage", "-s", help="Training stage (0-4)"),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to config YAML file"
    ),
    resume: Path | None = typer.Option(
        None, "--resume", "-r", help="Path to checkpoint to resume from"
    ),
) -> None:
    """Train a Pramana model for the specified stage.

    Examples:
        pramana train --stage 0
        pramana train --stage 1 --config configs/stage_one.yaml
        pramana train --stage 2 --resume checkpoints/checkpoint-100
    """
    train_model(stage=stage, config_path=config, resume=resume)
