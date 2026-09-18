"""Command line interface for ClaimScope."""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.table import Table

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


def _report_failure(exc: Exception, verbose: bool) -> None:
    """Turn a library traceback into an actionable one-liner.

    The underlying errors surface as long stack traces from anthropic and
    langchain; what the user needs is the cause and the fix.
    """
    message = str(exc)
    hints = (
        (
            "credit balance is too low",
            "Add credits in the Anthropic console under Plans & Billing.",
        ),
        ("authentication_error", "Check ANTHROPIC_API_KEY in your .env file."),
        ("invalid x-api-key", "Check ANTHROPIC_API_KEY in your .env file."),
        ("ANTHROPIC_API_KEY is not set", "Copy .env.example to .env and fill in your API key."),
        ("rate_limit", "Rate limited by the API. Wait a moment and retry."),
    )

    for needle, hint in hints:
        if needle.lower() in message.lower():
            console.print(f"[red]Error:[/red] {hint}")
            if verbose:
                console.print_exception()
            return

    console.print(f"[red]Error:[/red] {message}")
    if verbose:
        console.print_exception()
    else:
        console.print("[dim]Re-run with --verbose for the full traceback.[/dim]")


@app.command()
def analyze(
    paper_id: str = typer.Argument(help="arXiv id, e.g. 1512.03385"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show progress logs."),
) -> None:
    """Ingest a paper and extract its claims."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from claimscope.graph import build_graph

    settings = get_settings()
    try:
        with console.status(f"Analyzing {paper_id}..."):
            result = build_graph(settings).invoke({"paper_id": paper_id})
    except Exception as exc:  # the CLI reports failures, it does not recover from them
        _report_failure(exc, verbose)
        raise typer.Exit(1) from exc

    console.print(f"\n[bold]{result.get('paper_title', paper_id)}[/bold]")
    if repo := result.get("repo_url"):
        console.print(f"Repository: {repo}")

    claims = result.get("claims", [])
    if not claims:
        console.print("[yellow]No claims extracted.[/yellow]")
        return

    table = Table(title=f"{len(claims)} claims", show_lines=True)
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("type", style="magenta")
    table.add_column("claim")
    table.add_column("source", style="dim")
    for claim in claims:
        table.add_row(claim.id, claim.claim_type, claim.text, claim.source_location)
    console.print(table)

    out = settings.runs_dir / paper_id / "claims.json"
    console.print(f"\nWrote [green]{out}[/green]")


if __name__ == "__main__":
    app()
