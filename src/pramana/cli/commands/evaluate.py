"""Evaluation command for Pramana CLI."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pramana.application.evaluation.handlers import (
    Tier1StructuralHandler,
)
from pramana.application.evaluation.pipeline import EvaluationPipeline

console = Console()


def evaluate_model(
    model_path: Path,
    data_path: Path,
    tier: str = "all",
) -> None:
    """Evaluate a model on evaluation data.

    Args:
        model_path: Path to model directory or checkpoint
        data_path: Path to evaluation data JSON file
        tier: Which tier to run (1, 2, 3, or 'all')
    """
    if not model_path.exists():
        console.print(f"[red]Error: Model path not found: {model_path}[/red]")
        raise typer.Exit(1)

    if not data_path.exists():
        console.print(f"[red]Error: Data path not found: {data_path}[/red]")
        raise typer.Exit(1)

    # Parse tier option
    tiers_to_run: list[int] = []
    if tier.lower() == "all":
        tiers_to_run = [1, 2, 3]
    else:
        try:
            tier_num = int(tier)
            if tier_num < 1 or tier_num > 3:
                console.print(
                    f"[red]Error: Tier must be 1, 2, 3, or 'all', got {tier}[/red]"
                )
                raise typer.Exit(1)
            tiers_to_run = [tier_num]
        except ValueError:
            console.print(
                f"[red]Error: Invalid tier value: {tier}. Must be 1, 2, 3, or 'all'[/red]"
            )
            raise typer.Exit(1) from None

    # Build evaluation pipeline with requested tiers
    handlers = []
    if 1 in tiers_to_run:
        handlers.append(Tier1StructuralHandler())
    if 2 in tiers_to_run:
        # TODO: Initialize LLM client for Tier 2
        # For now, skip Tier 2 if LLM client is not available
        console.print("[yellow]Warning: Tier 2 (LLM judge) requires LLM client configuration[/yellow]")
        console.print("[yellow]Skipping Tier 2 evaluation[/yellow]")
        # handlers.append(Tier2LLMJudgeHandler(llm_client=...))
    if 3 in tiers_to_run:
        # TODO: Add Tier3ManualHandler when implemented
        console.print("[yellow]Warning: Tier 3 (manual review) not yet implemented[/yellow]")

    if not handlers:
        console.print("[red]Error: No valid evaluation handlers configured[/red]")
        raise typer.Exit(1)

    # Initialize evaluation pipeline (will be used in evaluation loop)
    _pipeline = EvaluationPipeline(handlers=handlers)

    # Load evaluation data
    console.print(f"[cyan]Loading evaluation data from {data_path}...[/cyan]")
    try:
        import json
        with data_path.open(encoding="utf-8") as f:
            eval_data = json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading evaluation data: {e}[/red]")
        raise typer.Exit(1) from e

    # TODO: Load model
    console.print(f"[cyan]Loading model from {model_path}...[/cyan]")
    # model = load_model(model_path)

    # Run evaluation
    console.print(f"[green]Running evaluation on {len(eval_data)} examples...[/green]")
    # TODO: Implement actual evaluation loop
    # results = []
    # for example_data in eval_data:
    #     # Convert example_data to NyayaExample
    #     # Generate model output
    #     # example = parse_example(example_data)
    #     # output = model.generate(example.problem)
    #     # result = pipeline.evaluate(example, output)
    #     # results.append(result)

    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Tier", style="cyan")
    table.add_column("Passed", style="green")
    table.add_column("Failed", style="red")
    table.add_column("Total", style="blue")

    # TODO: Aggregate results by tier
    # For now, show placeholder
    console.print("\n[yellow]Evaluation pipeline initialized. Full evaluation not yet implemented.[/yellow]")
    console.print(f"[cyan]Pipeline configured with {len(handlers)} handler(s)[/cyan]")
    console.print(table)


def evaluate(
    model_path: Path = typer.Option(
        ..., "--model-path", "-m", help="Path to model directory or checkpoint"
    ),
    data_path: Path = typer.Option(
        ..., "--data-path", "-d", help="Path to evaluation data JSON file"
    ),
    tier: str = typer.Option(
        "all", "--tier", "-t", help="Which tier to run (1, 2, 3, or 'all')"
    ),
) -> None:
    """Evaluate a Pramana model on evaluation data.

    Examples:
        pramana evaluate --model-path models/stage0 --data-path data/eval.json
        pramana evaluate --model-path models/stage1 --data-path data/eval.json --tier 1
        pramana evaluate --model-path models/stage2 --data-path data/eval.json --tier all
    """
    evaluate_model(model_path=model_path, data_path=data_path, tier=tier)
