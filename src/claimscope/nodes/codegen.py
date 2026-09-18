"""Generate the experiment script for each approved plan (PLAN.md section 6).

Phase 3 implements ``from_scratch`` only; ``official_repo`` arrives in phase 5.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from claimscope.config import Settings, get_settings
from claimscope.llm import ProviderStructuredLLM, StructuredLLM
from claimscope.prompts import load_prompt
from claimscope.schemas import Claim, GeneratedCode, ReductionPlan
from claimscope.state import GraphState

logger = logging.getLogger(__name__)

RUN_SCRIPT = "run.py"
RESULT_FILE = "result.json"

# Preinstalled in the sandbox image. Kept small: every package is install time
# on every build, and generated code should not reach for heavy dependencies.
SANDBOX_PACKAGES = ("numpy", "scikit-learn")

_FEEDBACK_TEMPLATE = """
## Your previous attempt failed

{feedback}

Fix the cause. Return the complete corrected script, not a diff.
"""


def workspace_for(settings: Settings, paper_id: str, claim_id: str) -> Path:
    """Where one claim's code and outputs live. This is the only mounted directory."""
    return Path(settings.runs_dir) / paper_id / "workspaces" / claim_id


DEFAULT_ARMS = ("treatment", "control")


def arms_for(claim: Claim) -> list[str]:
    """The arm names to run. A comparison needs two; fall back if the claim lacks them."""
    return list(claim.arms) if len(claim.arms) >= 2 else list(DEFAULT_ARMS)


def _per_run_seconds(arm_count: int, seeds: int, settings: Settings) -> float:
    """Rough time budget for a single (arm, seed) run."""
    return (settings.budget_minutes_total * 60) / max(1, arm_count * seeds)


def generate_code(
    claim: Claim,
    plan: ReductionPlan,
    settings: Settings,
    llm: StructuredLLM,
    feedback: str = "",
) -> GeneratedCode:
    """Ask the model for a run.py implementing this plan."""
    arms = arms_for(claim)
    total_runs = len(arms) * plan.seeds

    prompt = load_prompt(
        "codegen",
        claim=json.dumps(claim.model_dump(mode="json"), indent=2, ensure_ascii=False),
        plan=json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False),
        arms=", ".join(arms),
        packages=", ".join(SANDBOX_PACKAGES),
        total_runs=total_runs,
        budget_minutes=settings.budget_minutes_total,
        per_run_seconds=_per_run_seconds(len(arms), plan.seeds, settings),
        feedback_section=_FEEDBACK_TEMPLATE.format(feedback=feedback) if feedback else "",
    )
    return llm.invoke_structured(prompt, GeneratedCode)


def codegen(
    state: GraphState,
    settings: Settings | None = None,
    llm: StructuredLLM | None = None,
) -> GraphState:
    """Graph node: write run.py into each approved claim's workspace."""
    settings = settings or get_settings()
    llm = llm or ProviderStructuredLLM(settings)

    claims_by_id = {claim.id: claim for claim in state["claims"]}
    plans = state.get("plans", {})
    workspaces = dict(state.get("workspace_dirs", {}))

    for claim_id in state.get("approved_plan_ids", []):
        if claim_id in workspaces:
            continue  # already generated
        claim = claims_by_id.get(claim_id)
        plan = plans.get(claim_id)
        if claim is None or plan is None:
            logger.warning("cannot generate code for %s: missing claim or plan", claim_id)
            continue

        if plan.code_source == "official_repo":
            # Phase 5 territory. Fall back rather than fail the run.
            logger.info("official_repo not implemented yet; generating %s from scratch", claim_id)

        generated = generate_code(claim, plan, settings, llm)
        workspace = workspace_for(settings, state["paper_id"], claim_id)
        workspace.mkdir(parents=True, exist_ok=True)
        # Validation strips surrounding whitespace; restore the trailing newline
        # so the file is a well-formed text file.
        (workspace / RUN_SCRIPT).write_text(generated.code + "\n", encoding="utf-8")
        workspaces[claim_id] = str(workspace)
        logger.info("generated %s for %s: %s", RUN_SCRIPT, claim_id, generated.summary)

    return {"workspace_dirs": workspaces}
