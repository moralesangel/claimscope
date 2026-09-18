"""Run the agent over annotated papers and report how well it did.

Evaluation runs unattended, so the review interrupt is auto-approved: the point
is to measure the agent's own judgement, and a human approving every plan would
measure the human instead. That does mean generated code runs without review, so
the sandbox is not optional here -- without one, execution is skipped rather
than run unguarded.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TypeVar, cast

import typer
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from claimscope.config import Settings, get_settings
from claimscope.graph import build_graph
from claimscope.llm import StructuredLLM
from claimscope.session import thread_config
from claimscope.state import GraphState
from eval.metrics import score_paper
from eval.schemas import CostMetrics, EvalReport, PaperAnnotation, PaperResult

logger = logging.getLogger(__name__)

app = typer.Typer(help="Evaluate ClaimScope against annotated papers.")
console = Console()

ANNOTATIONS_DIR = Path(__file__).parent / "annotations"


def load_annotations(directory: Path) -> list[PaperAnnotation]:
    """Read every annotation file, newest-looking order aside."""
    annotations: list[PaperAnnotation] = []
    for path in sorted(directory.glob("*.json")):
        try:
            annotations.append(
                PaperAnnotation.model_validate_json(path.read_text(encoding="utf-8"))
            )
        except ValueError as exc:
            logger.warning("skipping malformed annotation %s: %s", path.name, exc)
    return annotations


T = TypeVar("T", bound=BaseModel)


class _CountingLLM:
    """Wraps the real client to count calls, for the cost metric."""

    def __init__(self, inner: StructuredLLM) -> None:
        self._inner = inner
        self.calls = 0

    def invoke_structured(self, prompt: str, schema: type[T]) -> T:
        self.calls += 1
        return self._inner.invoke_structured(prompt, schema)


def _sandbox_available(settings: Settings) -> bool:
    """Whether experiments can actually be run here."""
    try:
        from claimscope.sandbox.docker_runner import build_runner

        return build_runner(settings).available()
    except Exception:  # a missing docker package is the same as no sandbox
        logger.debug("sandbox unavailable", exc_info=True)
        return False


def evaluate_paper(
    annotation: PaperAnnotation,
    settings: Settings,
    sandbox_available: bool,
) -> PaperResult:
    """Run the agent over one paper and score the outcome."""
    from claimscope.llm import ProviderStructuredLLM

    llm = _CountingLLM(ProviderStructuredLLM(settings))
    graph = build_graph(settings, llm, InMemorySaver())
    config = thread_config(f"eval-{annotation.paper_id}")

    started = time.monotonic()
    state: GraphState = {}
    errors: list[str] = []

    try:
        # invoke() is typed as returning a plain dict, but it is the graph state.
        result = cast(GraphState, graph.invoke({"paper_id": annotation.paper_id}, config))

        # Auto-approve every plan: a human in the loop would be measuring the
        # human. Stop before execution when there is no sandbox to run in.
        while interrupts := result.get("__interrupt__"):
            requests = interrupts[0].value["requests"]  # type: ignore[index]
            if not sandbox_available:
                errors.append("no sandbox: stopped before executing generated code")
                break
            decisions = [
                {"claim_id": request["claim_id"], "action": "approve"} for request in requests
            ]
            result = cast(GraphState, graph.invoke(Command(resume=decisions), config))

        state = result
    except Exception as exc:  # one bad paper must not end the evaluation
        logger.exception("evaluating %s failed", annotation.paper_id)
        errors.append(f"run failed: {exc}")
        state = cast(GraphState, graph.get_state(config).values)

    cost = CostMetrics(
        wall_clock_s=time.monotonic() - started,
        compute_minutes=state.get("budget_minutes_used") or 0.0,
        llm_calls=llm.calls,
    )

    scored = score_paper(annotation, state, cost, sandbox_available)
    scored.errors.extend(errors)
    return scored


def render_report(report: EvalReport) -> Table:
    """The per-paper table (PLAN.md section 9)."""
    table = Table(title=f"ClaimScope evaluation: {len(report.results)} paper(s)")
    table.add_column("paper", style="cyan", no_wrap=True)
    table.add_column("P", justify="right")
    table.add_column("R", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("type acc", justify="right")
    table.add_column("testable acc", justify="right")
    table.add_column("ran", justify="right")
    table.add_column("verdict", justify="right")
    table.add_column("calls", justify="right")
    table.add_column("time", justify="right")

    for result in report.results:
        table.add_row(
            result.paper_id,
            f"{result.extraction.precision:.2f}",
            f"{result.extraction.recall:.2f}",
            f"{result.extraction.f1:.2f}",
            f"{result.classification.type_accuracy:.2f}",
            f"{result.classification.testable_accuracy:.2f}",
            f"{result.execution.completion_rate:.2f}" if result.execution.measured else "—",
            f"{result.verdicts.agreement:.2f}" if result.verdicts.measured else "—",
            str(result.cost.llm_calls),
            f"{result.cost.wall_clock_s:.0f}s",
        )

    overall = report.extraction
    table.add_section()
    table.add_row(
        "[bold]overall[/bold]",
        f"[bold]{overall.precision:.2f}[/bold]",
        f"[bold]{overall.recall:.2f}[/bold]",
        f"[bold]{overall.f1:.2f}[/bold]",
        f"[bold]{report.classification.type_accuracy:.2f}[/bold]",
        f"[bold]{report.classification.testable_accuracy:.2f}[/bold]",
        f"[bold]{report.execution.completion_rate:.2f}[/bold]"
        if report.execution.measured
        else "—",
        f"[bold]{report.verdicts.agreement:.2f}[/bold]" if report.verdicts.measured else "—",
        f"[bold]{sum(r.cost.llm_calls for r in report.results)}[/bold]",
        f"[bold]{sum(r.cost.wall_clock_s for r in report.results):.0f}s[/bold]",
    )
    return table


@app.command()
def run(
    annotations_dir: Path = typer.Option(ANNOTATIONS_DIR, help="Directory of annotation files."),
    output: Path | None = typer.Option(None, help="Write the full report as JSON here."),
    paper: str | None = typer.Option(None, help="Evaluate only this paper id."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate the agent over every annotated paper."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    annotations = load_annotations(annotations_dir)
    if paper:
        annotations = [a for a in annotations if a.paper_id == paper]
    if not annotations:
        console.print(f"[yellow]No annotations found in {annotations_dir}[/yellow]")
        raise typer.Exit(1)

    settings = get_settings()
    sandbox = _sandbox_available(settings)
    if not sandbox:
        console.print(
            "[yellow]No sandbox available: execution and verdict metrics will not be "
            "measured, and generated code will not be run.[/yellow]"
        )

    report = EvalReport()
    for annotation in annotations:
        console.print(f"Evaluating [cyan]{annotation.paper_id}[/cyan]...")
        report.results.append(evaluate_paper(annotation, settings, sandbox))

    console.print()
    console.print(render_report(report))

    for result in report.results:
        for error in result.errors:
            console.print(f"[dim]{result.paper_id}: {error}[/dim]")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_report_json(report), encoding="utf-8")
        console.print(f"\nWrote [green]{output}[/green]")


def _report_json(report: EvalReport) -> str:
    """Serialise the report with the derived rates included, for later analysis."""
    payload = {
        "papers": [
            {
                **result.model_dump(mode="json"),
                "derived": {
                    "precision": result.extraction.precision,
                    "recall": result.extraction.recall,
                    "f1": result.extraction.f1,
                    "type_accuracy": result.classification.type_accuracy,
                    "testable_accuracy": result.classification.testable_accuracy,
                },
            }
            for result in report.results
        ],
        "overall": {
            "precision": report.extraction.precision,
            "recall": report.extraction.recall,
            "f1": report.extraction.f1,
            "type_accuracy": report.classification.type_accuracy,
            "testable_accuracy": report.classification.testable_accuracy,
            "execution_measured": report.execution.measured,
            "completion_rate": report.execution.completion_rate,
            "mean_debug_attempts": report.execution.mean_debug_attempts,
            "verdicts_measured": report.verdicts.measured,
            "verdict_agreement": report.verdicts.agreement,
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    app()
