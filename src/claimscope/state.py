"""Graph state (PLAN.md section 5).

Phase 1 only populates the ingest and extraction fields; the rest are declared
now so later phases add nodes without reshaping the state.
"""

from __future__ import annotations

from typing import TypedDict

from claimscope.sandbox.runner import ExecutionFailure
from claimscope.schemas import Claim, ClaimVerdict, ReductionPlan, RunResult


class GraphState(TypedDict, total=False):
    paper_id: str
    paper_text: str
    paper_title: str
    pdf_path: str
    repo_url: str | None

    claims: list[Claim]
    selected_claim_ids: list[str]
    plans: dict[str, ReductionPlan]
    approved_plan_ids: list[str]
    plan_feedback: dict[str, str]
    """Reviewer feedback per claim, fed back into design_plan on a rejection."""

    workspace_dirs: dict[str, str]
    """Per-claim sandbox workspace; each is the only directory mounted for it."""

    run_results: dict[str, list[RunResult]]
    execution_failures: dict[str, ExecutionFailure]
    """Claims whose last run failed, awaiting the debug node."""

    debug_attempts: dict[str, int]
    abandoned_claim_ids: dict[str, str]
    """Claims given up on, mapped to why. Reported as inconclusive."""

    over_budget_claim_ids: list[str]
    """Claims whose measured cost exceeded the remaining budget."""

    budget_minutes_total: float
    budget_minutes_used: float

    verdicts: list[ClaimVerdict]
    report_path: str | None
    errors: list[str]
