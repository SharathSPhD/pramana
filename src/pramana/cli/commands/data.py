"""Data management commands for Pramana CLI."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pramana.application.data.parser import MarkdownParser

app = typer.Typer(name="data", help="Data management utilities")
console = Console()


@app.command()
def parse(
    input_dir: Path = typer.Option(..., "--input-dir", "-i", help="Input directory with markdown files"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory for JSON files"),
) -> None:
    """Parse markdown files to JSON format.

    Examples:
        pramana data parse --input-dir data/seed_examples --output-dir data/json
    """
    if not input_dir.exists():
        console.print(f"[red]Error: Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)

    if not input_dir.is_dir():
        console.print(f"[red]Error: Input path is not a directory: {input_dir}[/red]")
        raise typer.Exit(1)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all markdown files
    markdown_files = list(input_dir.rglob("*.md"))
    if not markdown_files:
        console.print(f"[yellow]Warning: No markdown files found in {input_dir}[/yellow]")
        raise typer.Exit(0)

    console.print(f"[cyan]Parsing {len(markdown_files)} markdown files...[/cyan]")

    parser = MarkdownParser()
    parsed_count = 0
    error_count = 0

    for md_file in markdown_files:
        try:
            content = md_file.read_text(encoding="utf-8")
            example = parser.parse(content)

            # Convert to JSON-serializable dict
            example_dict = example.model_dump(mode="json")

            # Write JSON file
            relative_path = md_file.relative_to(input_dir)
            json_path = output_dir / relative_path.with_suffix(".json")
            json_path.parent.mkdir(parents=True, exist_ok=True)

            with json_path.open("w", encoding="utf-8") as f:
                json.dump(example_dict, f, indent=2, ensure_ascii=False, default=str)

            parsed_count += 1

        except Exception as e:
            console.print(f"[red]Error parsing {md_file}: {e}[/red]")
            error_count += 1

    console.print(f"\n[green]Parsed {parsed_count} files successfully[/green]")
    if error_count > 0:
        console.print(f"[yellow]{error_count} files had errors[/yellow]")


@app.command()
def stats(
    data_path: Path = typer.Option(..., "--data-path", "-d", help="Path to JSON data file or directory"),
) -> None:
    """Show dataset statistics.

    Examples:
        pramana data stats --data-path data/train.json
        pramana data stats --data-path data/
    """
    if not data_path.exists():
        console.print(f"[red]Error: Data path not found: {data_path}[/red]")
        raise typer.Exit(1)

    # Collect all JSON files
    json_files: list[Path] = []
    if data_path.is_file():
        if data_path.suffix == ".json":
            json_files = [data_path]
        else:
            console.print(f"[red]Error: File must be a JSON file: {data_path}[/red]")
            raise typer.Exit(1)
    else:
        json_files = list(data_path.rglob("*.json"))

    if not json_files:
        console.print(f"[yellow]Warning: No JSON files found in {data_path}[/yellow]")
        raise typer.Exit(0)

    console.print(f"[cyan]Analyzing {len(json_files)} JSON files...[/cyan]")

    examples = []
    for json_file in json_files:
        try:
            with json_file.open(encoding="utf-8") as f:
                data = json.load(f)
                # Handle both single objects and arrays
                if isinstance(data, list):
                    examples.extend(data)
                else:
                    examples.append(data)
        except Exception as e:
            console.print(f"[yellow]Warning: Error reading {json_file}: {e}[/yellow]")

    if not examples:
        console.print("[yellow]No valid examples found[/yellow]")
        raise typer.Exit(0)

    # Calculate statistics
    total_examples = len(examples)
    problem_types: dict[str, int] = {}
    difficulties: dict[str, int] = {}
    stages: dict[int, int] = {}
    validated_count = 0
    z3_verifiable_count = 0

    for example in examples:
        # Problem types
        problem_type = example.get("problem_type", "unknown")
        problem_types[problem_type] = problem_types.get(problem_type, 0) + 1

        # Difficulties
        difficulty = example.get("difficulty", "unknown")
        difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        # Stages
        metadata = example.get("metadata", {})
        stage = metadata.get("stage", 0)
        stages[stage] = stages.get(stage, 0) + 1

        # Validation flags
        if metadata.get("validated", False):
            validated_count += 1
        if metadata.get("z3_verifiable", False):
            z3_verifiable_count += 1

    # Display statistics
    table = Table(title="Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Examples", str(total_examples))
    table.add_row("Validated", f"{validated_count} ({validated_count/total_examples*100:.1f}%)")
    table.add_row("Z3 Verifiable", f"{z3_verifiable_count} ({z3_verifiable_count/total_examples*100:.1f}%)")

    console.print("\n")
    console.print(table)

    # Problem types table
    if problem_types:
        type_table = Table(title="Problem Types")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", style="green")
        type_table.add_column("Percentage", style="blue")

        for ptype, count in sorted(problem_types.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_examples * 100
            type_table.add_row(ptype, str(count), f"{percentage:.1f}%")

        console.print("\n")
        console.print(type_table)

    # Stages table
    if stages:
        stage_table = Table(title="Stages")
        stage_table.add_column("Stage", style="cyan")
        stage_table.add_column("Count", style="green")
        stage_table.add_column("Percentage", style="blue")

        for stage, count in sorted(stages.items()):
            percentage = count / total_examples * 100
            stage_table.add_row(str(stage), str(count), f"{percentage:.1f}%")

        console.print("\n")
        console.print(stage_table)


@app.command()
def split(
    data_path: Path = typer.Option(..., "--data-path", "-d", help="Path to JSON data file"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory for splits"),
    train_ratio: float = typer.Option(0.8, "--train-ratio", "-r", help="Ratio for training set (0.0-1.0)"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for shuffling"),
) -> None:
    """Create train/eval splits from dataset.

    Examples:
        pramana data split --data-path data/all.json --output-dir data/splits
        pramana data split --data-path data/all.json --output-dir data/splits --train-ratio 0.9
    """
    if not data_path.exists():
        console.print(f"[red]Error: Data file not found: {data_path}[/red]")
        raise typer.Exit(1)

    if not 0.0 < train_ratio < 1.0:
        console.print(f"[red]Error: Train ratio must be between 0.0 and 1.0, got {train_ratio}[/red]")
        raise typer.Exit(1)

    # Load data
    console.print(f"[cyan]Loading data from {data_path}...[/cyan]")
    try:
        with data_path.open(encoding="utf-8") as f:
            data = json.load(f)

            # Handle both single objects and arrays
            examples = data if isinstance(data, list) else [data]

    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        raise typer.Exit(1) from e

    # Shuffle with seed
    import random
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    # Split
    split_idx = int(len(shuffled) * train_ratio)
    train_data = shuffled[:split_idx]
    eval_data = shuffled[split_idx:]

    console.print(f"[cyan]Split: {len(train_data)} train, {len(eval_data)} eval[/cyan]")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write splits
    train_path = output_dir / "train.json"
    eval_path = output_dir / "eval.json"

    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False, default=str)

    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False, default=str)

    console.print(f"[green]Splits written to {output_dir}[/green]")
    console.print(f"  Train: {train_path}")
    console.print(f"  Eval: {eval_path}")
