"""Command-line interface for Pramana."""

import click


@click.group()
def main() -> None:
    """Pramana: Epistemic reasoning engine based on Navya-Nyaya logic."""
    pass


__all__ = ["main"]
