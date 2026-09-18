"""Command line interface for ClaimScope."""

from __future__ import annotations

import logging
from typing import Any

import typer
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from rich.console import Console
from rich.table import Table

from claimscope import __version__
from claimscope.config import get_settings
from claimscope.review_ui import collect_decisions
from claimscope.session import (
    checkpoint_path,
    checkpointer,
    list_threads,
    new_thread_id,
    thread_config,
)
from claimscope.tracing import traced_run

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
    for key_field in ("anthropic_api_key", "gemini_api_key"):
        dumped[key_field] = "set" if dumped.get(key_field) else "unset"
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
        (
            "perdayperproject",
            "Daily request quota exhausted for this model. It resets tomorrow, "
            "or switch provider with CLAIMSCOPE_PROVIDER.",
        ),
        ("authentication_error", "Check the API key for your provider in .env."),
        ("invalid x-api-key", "Check the API key for your provider in .env."),
        ("api_key is not set", "Copy .env.example to .env and fill in your API key."),
        ("rate_limit", "Rate limited by the API. Wait a moment and retry."),
        ("resource_exhausted", "Quota exhausted. Check your provider's rate limits."),
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


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _show_claims(result: dict[str, Any]) -> None:
    """Render the extracted claims and what triage decided about them."""
    claims = result.get("claims", [])
    if not claims:
        console.print("[yellow]No claims extracted.[/yellow]")
        return

    selected = set(result.get("selected_claim_ids", []))
    table = Table(title=f"{len(claims)} claims", show_lines=True)
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("type", style="magenta")
    table.add_column("claim")
    table.add_column("source", style="dim")
    table.add_column("triage")

    for claim in claims:
        if claim.id in selected:
            verdict = "[bold green]selected[/bold green]"
        elif claim.testable:
            verdict = "[green]testable[/green]"
        elif claim.testable is False:
            verdict = "[dim]not testable[/dim]"
        else:
            verdict = ""
        table.add_row(claim.id, claim.claim_type, claim.text, claim.source_location, verdict)

    console.print(table)

    for claim in claims:
        if claim.testable is False and claim.triage_reason:
            console.print(f"[dim]{claim.id}: {claim.triage_reason}[/dim]")


def _drain_interrupts(graph: Any, result: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Answer every review interrupt until the graph runs to completion."""
    while interrupts := result.get("__interrupt__"):
        decisions = collect_decisions(console, interrupts[0].value)
        result = dict(graph.invoke(Command(resume=decisions), config))
    return result


def _report_outcome(result: dict[str, Any], thread_id: str, settings: Any) -> None:
    """Summarise where the run ended up."""
    console.print(f"\n[bold]{result.get('paper_title', '')}[/bold]")
    if repo := result.get("repo_url"):
        console.print(f"Repository: {repo}")

    _show_claims(result)

    approved = result.get("approved_plan_ids", [])
    if approved:
        console.print(f"\n[green]Approved plans:[/green] {', '.join(approved)}")
        out_dir = settings.runs_dir / result.get("paper_id", "")
        console.print(f"Wrote [green]{out_dir}[/green]")
    elif result.get("selected_claim_ids"):
        console.print("\n[yellow]No plans approved.[/yellow]")

    console.print(f"[dim]Thread: {thread_id}[/dim]")


@app.command()
def analyze(
    paper_id: str = typer.Argument(help="arXiv id, e.g. 1512.03385"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show progress logs."),
) -> None:
    """Analyze a paper: extract claims, triage them, and plan experiments."""
    _setup_logging(verbose)

    from claimscope.graph import build_graph

    settings = get_settings()
    thread_id = new_thread_id(paper_id)

    try:
        with (
            checkpointer(settings) as saver,
            traced_run(settings, thread_config(thread_id)) as config,
        ):
            graph = build_graph(settings, checkpointer=saver)
            with console.status(f"Analyzing {paper_id}..."):
                started = graph.invoke({"paper_id": paper_id}, config)

            result = _drain_interrupts(graph, dict(started), config)
            _report_outcome(result, thread_id, settings)
    except Exception as exc:  # the CLI reports failures, it does not recover from them
        _report_failure(exc, verbose)
        console.print(f"[dim]Resume with: claimscope resume {thread_id}[/dim]")
        raise typer.Exit(1) from exc


@app.command()
def resume(
    thread_id: str = typer.Argument(help="Thread id printed when the run was interrupted."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show progress logs."),
) -> None:
    """Resume an interrupted run from its checkpoint."""
    _setup_logging(verbose)

    from claimscope.graph import build_graph

    settings = get_settings()

    try:
        with checkpointer(settings) as saver:
            graph = build_graph(settings, checkpointer=saver)
            config = thread_config(thread_id)

            snapshot = graph.get_state(config)
            if not snapshot.created_at:
                console.print(f"[red]Error:[/red] no checkpoint found for thread {thread_id}")
                raise typer.Exit(1)

            pending = snapshot.tasks[0].interrupts if snapshot.tasks else ()
            if not pending:
                console.print("[yellow]This thread is not waiting for input.[/yellow]")
                _report_outcome(dict(snapshot.values), thread_id, settings)
                return

            decisions = collect_decisions(console, pending[0].value)
            resumed = dict(graph.invoke(Command(resume=decisions), config))
            _report_outcome(_drain_interrupts(graph, resumed, config), thread_id, settings)
    except typer.Exit:
        raise
    except Exception as exc:  # the CLI reports failures, it does not recover from them
        _report_failure(exc, verbose)
        raise typer.Exit(1) from exc


@app.command()
def threads() -> None:
    """List runs that have a saved checkpoint."""
    settings = get_settings()
    path = checkpoint_path(settings)
    if not path.exists():
        console.print("No runs yet.")
        return

    with checkpointer(settings) as saver:
        from claimscope.graph import build_graph

        graph = build_graph(settings, checkpointer=saver)
        ids = list_threads(saver)
        if not ids:
            console.print("No runs yet.")
            return

        table = Table(title=f"{len(ids)} run(s)")
        table.add_column("thread", style="cyan", no_wrap=True)
        table.add_column("paper", style="dim")
        table.add_column("status")

        for thread_id in ids:
            snapshot = graph.get_state(thread_config(thread_id))
            waiting = bool(snapshot.tasks and snapshot.tasks[0].interrupts)
            status = "[yellow]awaiting review[/yellow]" if waiting else "done"
            table.add_row(thread_id, str(snapshot.values.get("paper_id", "")), status)

        console.print(table)


if __name__ == "__main__":
    app()
