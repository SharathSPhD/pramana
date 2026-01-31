"""Main CLI application for Pramana."""

import typer

from pramana.cli.commands import data, evaluate, train, validate

app = typer.Typer(
    name="pramana",
    help="Pramana: Epistemic reasoning engine based on Navya-Nyaya logic",
    add_completion=False,
)

# Register commands directly (not as subcommand groups)
app.command()(train.train)
app.command()(evaluate.evaluate)
app.command()(validate.validate)

# Register data subcommands
app.add_typer(data.app, name="data")


def main() -> None:
    """Entry point for CLI application."""
    app()


if __name__ == "__main__":
    main()
