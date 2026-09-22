"""Write the Markdown report (PLAN.md section 6).

The report must never overstate what a reduced-scale run shows. A negative
result here does not refute the paper, and section "Limitations" says so in
every report, unconditionally.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from claimscope.config import Settings, get_settings
from claimscope.integrity import MARKER as INTEGRITY_MARKER
from claimscope.schemas import Claim, ClaimVerdict, ReductionPlan, RunResult
from claimscope.state import GraphState

logger = logging.getLogger(__name__)

REPORT_FILENAME = "report.md"

_VERDICT_LABELS = {
    "consistent_at_reduced_scale": "Consistent at reduced scale",
    "not_consistent_at_reduced_scale": "Not consistent at reduced scale",
    "inconclusive": "Inconclusive",
    "not_testable": "Not testable at reduced scale",
}

LIMITATIONS = """\
## Limitations

Read these results narrowly.

- **A negative result here does not refute the paper.** These experiments run at a fraction of the
  original scale. An effect that disappears when the model, the data, or the training budget shrinks
  may still be real at full scale; that is precisely what reduced-scale testing cannot tell us.
- **A positive result is not a reproduction.** It says the direction of the claim survived
  shrinking, not that the paper's numbers were reproduced.
- **Seed counts are small.** With a handful of seeds per arm, statistical power is low. An
  inconclusive verdict usually means there was not enough evidence either way, not that the effect
  is absent.
- **Absolute claims are never testable this way** and are recorded without being run.
- The reduced experiments were written from the paper's description, so they may differ from the
  authors' implementation in ways that matter.
"""


def _format_verdict_row(claim: Claim, verdict: ClaimVerdict) -> str:
    label = _VERDICT_LABELS.get(verdict.verdict, verdict.verdict)
    effect = "—" if verdict.effect_estimate is None else f"{verdict.effect_estimate:+.4g}"
    if verdict.ci_low is None or verdict.ci_high is None:
        interval = "—"
    else:
        interval = f"[{verdict.ci_low:+.3g}, {verdict.ci_high:+.3g}]"
    return f"| `{claim.id}` | {claim.claim_type} | {label} | {effect} | {interval} |"


def _plan_section(plan: ReductionPlan) -> str:
    lines = [
        f"**Original setup.** {plan.original_setup}",
        "",
        f"**Reduced setup.** {plan.reduced_setup}",
        "",
        f"**Why the claim should transfer.** {plan.why_claim_should_transfer}",
        "",
        f"**Seeds per arm:** {plan.seeds} · **Estimated cost:** {plan.estimated_minutes:.0f} min "
        f"· **Code source:** {plan.code_source}",
    ]
    if plan.changes:
        lines += ["", "**Changes from the paper**", ""]
        lines += [f"- {change}" for change in plan.changes]
    if plan.preserved:
        lines += ["", "**Deliberately preserved**", ""]
        lines += [f"- {item}" for item in plan.preserved]
    return "\n".join(lines)


def _results_table(results: list[RunResult]) -> str:
    lines = ["| Arm | Seed | Metric | Runtime (s) |", "|---|---|---|---|"]
    for result in sorted(results, key=lambda r: (r.arm, r.seed)):
        lines.append(
            f"| {result.arm} | {result.seed} | {result.metric_value:.6g} | {result.runtime_s:.1f} |"
        )
    return "\n".join(lines)


def _is_integrity_warning(error: str) -> bool:
    """Whether this error says the script departed from its plan.

    These are worth more prominence than the rest of the errors list: a repo
    that failed to clone costs a claim, while a substituted dataset silently
    changes what the number underneath the verdict means. Matched on the marker
    integrity.py stamps, not on the wording, so rewording a warning cannot
    quietly stop it being shown.
    """
    return error.startswith(INTEGRITY_MARKER)


def render_report(state: GraphState) -> str:
    """Build the full Markdown report from the finished state."""
    claims = state.get("claims", [])
    claims_by_id = {claim.id: claim for claim in claims}
    verdicts = state.get("verdicts", [])
    plans = state.get("plans", {})
    run_results = state.get("run_results", {})
    selected = set(state.get("selected_claim_ids", []))

    title = state.get("paper_title") or state.get("paper_id", "Unknown paper")
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    parts: list[str] = [
        f"# ClaimScope report: {title}",
        "",
        f"arXiv `{state.get('paper_id', '')}` · generated {generated}",
        "",
        "This report tests whether individual claims from the paper still hold **at reduced "
        "scale**. It is not a reproduction of the paper.",
        "",
    ]

    if repo := state.get("repo_url"):
        parts += [f"Official repository: {repo}", ""]

    # A verdict computed from data the plan did not name is not a weaker
    # result, it is a different question answered. That has to reach the reader
    # before the table, not from a list of problems underneath it.
    if integrity := [e for e in state.get("errors", []) if _is_integrity_warning(e)]:
        parts += [
            "> **Read the verdicts below with care.** The generated script did not run the "
            "experiment its plan describes:",
            ">",
        ]
        parts += [f"> - {warning}" for warning in integrity]
        parts += [
            ">",
            "> A verdict computed from data the plan does not name answers a different question "
            "from the one the claim asks. Read the script in the claim's workspace before "
            "relying on it.",
            "",
        ]

    # Summary table first: the reader wants the verdicts.
    parts += [
        "## Verdicts",
        "",
        "| Claim | Type | Verdict | Effect | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for verdict in verdicts:
        claim = claims_by_id.get(verdict.claim_id)
        if claim is not None:
            parts.append(_format_verdict_row(claim, verdict))
    parts.append("")

    budget_used = state.get("budget_minutes_used", 0.0)
    budget_total = state.get("budget_minutes_total", 0.0)
    if budget_total:
        parts += [f"Compute used: {budget_used:.1f} of {budget_total:.0f} minutes.", ""]

    # Then the detail, for claims that actually produced runs. A selected claim
    # that never ran belongs with the untested ones, not under "Tested claims".
    tested = [v for v in verdicts if v.claim_id in selected and run_results.get(v.claim_id)]
    if tested:
        parts += ["## Tested claims", ""]
        for verdict in tested:
            claim = claims_by_id.get(verdict.claim_id)
            if claim is None:
                continue
            parts += [
                f"### {claim.id}",
                "",
                f"> {claim.text}",
                "",
                f"Source: {claim.source_location} · Metric: {claim.metric} · "
                f"Expected: {claim.expected_direction}",
                "",
                f"**Verdict: {_VERDICT_LABELS.get(verdict.verdict, verdict.verdict)}**",
                "",
            ]
            if verdict.notes:
                parts += [verdict.notes, ""]
            if commit := state.get("repo_commits", {}).get(claim.id):
                # Without the commit the run is not reproducible.
                parts += [f"Ran against the official repository at commit `{commit}`.", ""]
            if plan := plans.get(claim.id):
                parts += [
                    "#### Reduction plan",
                    "",
                    "_This is what was **designed**. The experiment was then written by a model "
                    "from this description, so read the script in the workspace to see what "
                    "actually ran -- the two can differ, for instance where the sandbox has no "
                    "network and the data had to be generated instead of downloaded._",
                    "",
                    _plan_section(plan),
                    "",
                ]
            if results := run_results.get(claim.id):
                parts += ["#### Runs", "", _results_table(results), ""]

    # Claims that produced no runs, so the reader knows they were considered and
    # why nothing came of them. Selected-but-unrun is distinguished from
    # never-selected: one is a failure to execute, the other a triage decision.
    tested_ids = {v.claim_id for v in tested}
    untested = [v for v in verdicts if v.claim_id not in tested_ids]
    if untested:
        parts += ["## Claims not tested", ""]
        for verdict in untested:
            claim = claims_by_id.get(verdict.claim_id)
            if claim is None:
                continue
            marker = " (selected but not completed)" if claim.id in selected else ""
            parts += [
                f"- **`{claim.id}`**{marker} ({claim.claim_type}) — {claim.text} _{verdict.notes}_"
            ]
        parts.append("")

    if state.get("unconfined_execution"):
        parts += [
            "## How these experiments were run",
            "",
            "> **These results were produced without container isolation.** The experiments ran in "
            "a restricted subprocess, with network access blocked and resources capped, but not "
            "inside a sandbox that can contain what it runs. Treat the results as a demonstration "
            "of the pipeline rather than as a trustworthy measurement, and re-run under Docker "
            "before relying on them.",
            "",
        ]

    if errors := state.get("errors"):
        parts += ["## Problems encountered", ""]
        parts += [f"- {error}" for error in errors]
        parts.append("")

    parts.append(LIMITATIONS)
    return "\n".join(parts)


def report(state: GraphState, settings: Settings | None = None) -> GraphState:
    """Graph node: write the report to runs/<paper_id>/report.md."""
    settings = settings or get_settings()

    out_dir = Path(settings.runs_dir) / state["paper_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REPORT_FILENAME
    path.write_text(render_report(state), encoding="utf-8")

    logger.info("wrote report to %s", path)
    return {"report_path": str(path)}
