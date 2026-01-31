"""Validation command for Pramana CLI."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pramana.application.data.parser import MarkdownParser, ParseError
from pramana.application.data.parser import ValidationError as ParserValidationError
from pramana.domain.validators.structure import NyayaStructureValidator

console = Console()


def validate_file(file_path: Path, strict: bool = False) -> int:
    """Validate a single Nyaya example file.

    Args:
        file_path: Path to markdown file to validate
        strict: If True, fail on any validation error

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        return 1

    parser = MarkdownParser()
    validator = NyayaStructureValidator()

    try:
        # Parse markdown
        content = file_path.read_text(encoding="utf-8")
        example = parser.parse(content)

        # Validate structure
        result = validator.validate(example)

        if result.is_valid:
            console.print(f"[green]✓[/green] {file_path.name} is valid")
            return 0
        else:
            console.print(f"[red]✗[/red] {file_path.name} has validation errors:")
            for error in result.errors:
                console.print(f"  [red]- {error.phase}: {error.message}[/red]")
            if strict:
                return 1
            return 0

    except ParseError as e:
        console.print(f"[red]✗[/red] {file_path.name} parse error: {e}[/red]")
        return 1
    except ParserValidationError as e:
        console.print(f"[red]✗[/red] {file_path.name} validation error: {e}[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]✗[/red] {file_path.name} unexpected error: {e}[/red]")
        return 1


def validate_directory(dir_path: Path, strict: bool = False) -> int:
    """Validate all markdown files in a directory.

    Args:
        dir_path: Path to directory containing markdown files
        strict: If True, fail on any validation error

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    if not dir_path.exists():
        console.print(f"[red]Error: Directory not found: {dir_path}[/red]")
        return 1

    if not dir_path.is_dir():
        console.print(f"[red]Error: Path is not a directory: {dir_path}[/red]")
        return 1

    # Find all markdown files
    markdown_files = list(dir_path.glob("*.md"))
    if not markdown_files:
        console.print(f"[yellow]Warning: No markdown files found in {dir_path}[/yellow]")
        return 0

    console.print(f"[cyan]Validating {len(markdown_files)} files...[/cyan]\n")

    parser = MarkdownParser()
    validator = NyayaStructureValidator()

    results: list[tuple[Path, bool, list[str]]] = []

    for file_path in markdown_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            example = parser.parse(content)
            result = validator.validate(example)

            errors = [f"{e.phase}: {e.message}" for e in result.errors]
            results.append((file_path, result.is_valid, errors))

        except (ParseError, ParserValidationError) as e:
            results.append((file_path, False, [str(e)]))
        except Exception as e:
            results.append((file_path, False, [f"Unexpected error: {e}"]))

    # Display results
    table = Table(title="Validation Results")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Errors", style="red")

    all_valid = True
    for file_path, is_valid, errors in results:
        status = "[green]✓ Valid[/green]" if is_valid else "[red]✗ Invalid[/red]"
        error_str = "\n".join(errors) if errors else ""
        table.add_row(file_path.name, status, error_str)
        if not is_valid:
            all_valid = False

    console.print("\n")
    console.print(table)

    if strict and not all_valid:
        console.print("\n[red]Validation failed (strict mode)[/red]")
        return 1

    if all_valid:
        console.print("\n[green]All files are valid![/green]")
    else:
        console.print("\n[yellow]Some files have validation errors[/yellow]")

    return 0 if all_valid else 1


def validate(
    file: Path | None = typer.Option(None, "--file", "-f", help="Path to single example file"),
    dir: Path | None = typer.Option(None, "--dir", "-d", help="Path to directory of examples"),
    strict: bool = typer.Option(False, "--strict", help="Fail on any validation error"),
) -> None:
    """Validate Nyaya example files.

    Examples:
        pramana validate --file examples/example1.md
        pramana validate --dir examples/ --strict
    """
    if file and dir:
        console.print("[red]Error: Cannot specify both --file and --dir[/red]")
        raise typer.Exit(1)

    if not file and not dir:
        console.print("[red]Error: Must specify either --file or --dir[/red]")
        raise typer.Exit(1)

    exit_code = 0
    if file:
        exit_code = validate_file(file, strict=strict)
    elif dir:
        exit_code = validate_directory(dir, strict=strict)

    raise typer.Exit(exit_code)
