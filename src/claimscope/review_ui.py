"""Terminal rendering and prompting for the plan review interrupt."""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class NoReviewerError(RuntimeError):
    """The run needs a human decision and there is nobody to ask."""


def render_plan(console: Console, request: dict[str, Any]) -> None:
    """Show one reduction plan in full, so the user can judge it."""
    plan = request["plan"]

    console.print()
    console.print(
        Panel(
            request.get("claim_text", ""),
            title=f"[bold]{request['claim_id']}[/bold]",
            border_style="cyan",
        )
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    table.add_row("Original", plan.get("original_setup", ""))
    table.add_row("Reduced", plan.get("reduced_setup", ""))
    table.add_row("Seeds", str(plan.get("seeds", "")))
    table.add_row("Estimated", f"{plan.get('estimated_minutes', 0):.0f} min")
    table.add_row("Code", plan.get("code_source", ""))
    console.print(table)

    for heading, key in (("Changes", "changes"), ("Preserved", "preserved")):
        items = plan.get(key) or []
        if items:
            console.print(f"\n[bold]{heading}[/bold]")
            for item in items:
                console.print(f"  • {item}")

    if rationale := plan.get("why_claim_should_transfer"):
        console.print(f"\n[bold]Why it should transfer[/bold]\n  {rationale}")


def _approve(claim_id: str) -> dict[str, Any]:
    return {"claim_id": claim_id, "action": "approve", "feedback": ""}


def prompt_for_decision(console: Console, request: dict[str, Any]) -> dict[str, Any]:
    """Ask the user to approve or reject one plan.

    With no terminal to read from -- a redirected stdin, a script, CI -- there
    is nobody to ask, so the run stops rather than looping on an exhausted
    stream. Use ``--yes`` to approve without being asked.
    """
    render_plan(console, request)
    console.print()

    try:
        approved = typer.confirm("Approve this plan?", default=True)
    except (EOFError, typer.Abort) as exc:
        raise NoReviewerError(
            "No input available to review the plan. Re-run with --yes to approve "
            "automatically, or resume the run from a terminal."
        ) from exc

    if approved:
        return _approve(request["claim_id"])

    feedback = ""
    while not feedback.strip():
        try:
            feedback = typer.prompt("What should change?")
        except (EOFError, typer.Abort) as exc:
            raise NoReviewerError("No input available to give feedback on the plan.") from exc
    return {"claim_id": request["claim_id"], "action": "reject", "feedback": feedback}


def collect_decisions(
    console: Console, payload: dict[str, Any], assume_yes: bool = False
) -> list[dict[str, Any]]:
    """Walk the user through every plan awaiting review.

    ``assume_yes`` approves every plan without asking, for unattended runs. That
    means generated code runs without a human having read the plan, so the
    caller is responsible for deciding that is acceptable.
    """
    requests = payload.get("requests", [])
    console.print(f"\n[bold yellow]{len(requests)} plan(s) awaiting review[/bold yellow]")

    if assume_yes:
        for request in requests:
            render_plan(console, request)
        console.print("\n[yellow]Approving automatically (--yes).[/yellow]")
        return [_approve(request["claim_id"]) for request in requests]

    return [prompt_for_decision(console, request) for request in requests]
