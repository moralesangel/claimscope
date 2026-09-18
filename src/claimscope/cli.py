"""Command line interface for ClaimScope."""

from __future__ import annotations

import typer
from rich.console import Console

from claimscope import __version__
from claimscope.config import get_settings

app = typer.Typer(
    name="claimscope",
    help="Test ML paper claims at reduced scale.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the ClaimScope version."""
    console.print(f"claimscope {__version__}")


@app.command()
def config() -> None:
    """Show the resolved configuration, with secrets redacted."""
    settings = get_settings()
    dumped = settings.model_dump()
    dumped["anthropic_api_key"] = "set" if settings.anthropic_api_key else "unset"
    for key, value in dumped.items():
        console.print(f"{key}: {value}")


if __name__ == "__main__":
    app()
