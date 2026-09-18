"""Terminal rendering and prompting for the plan review interrupt."""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


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


def prompt_for_decision(console: Console, request: dict[str, Any]) -> dict[str, Any]:
    """Ask the user to approve or reject one plan."""
    render_plan(console, request)
    console.print()

    approved = typer.confirm("Approve this plan?", default=True)
    if approved:
        return {"claim_id": request["claim_id"], "action": "approve", "feedback": ""}

    feedback = ""
    while not feedback.strip():
        feedback = typer.prompt("What should change?")
    return {"claim_id": request["claim_id"], "action": "reject", "feedback": feedback}


def collect_decisions(console: Console, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Walk the user through every plan awaiting review."""
    requests = payload.get("requests", [])
    console.print(
        f"\n[bold yellow]{len(requests)} plan(s) awaiting review[/bold yellow]",
    )
    return [prompt_for_decision(console, request) for request in requests]
