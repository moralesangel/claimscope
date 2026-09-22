"""Design reduced-scale experiments for the selected claims (PLAN.md section 6)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from claimscope.config import Settings, get_settings
from claimscope.integrity import OFFLINE_DATASETS
from claimscope.llm import ProviderStructuredLLM, StructuredLLM
from claimscope.nodes.codegen import SANDBOX_PACKAGES
from claimscope.prompts import load_prompt
from claimscope.schemas import Claim, ReductionPlan
from claimscope.state import GraphState

logger = logging.getLogger(__name__)

_FEEDBACK_TEMPLATE = """
## Reviewer feedback on your previous plan

A human reviewer rejected your last plan for this claim with this feedback:

{feedback}

Address it directly in the new plan.
"""


def _offline_datasets() -> str:
    """The datasets the sandbox can load, as a bullet list for the prompt."""
    return "\n".join(f"- `{name}` -- {shape}" for name, shape in OFFLINE_DATASETS)


def _code_context(repo_url: str | None) -> str:
    if repo_url:
        return (
            f"The paper's official repository is available at {repo_url}. "
            "Prefer adapting it (code_source: official_repo) over reimplementing, "
            "provided its setup can realistically be shrunk to the budget."
        )
    return (
        "No official repository was found for this paper, so the experiment must be "
        "written from scratch (code_source: from_scratch). Keep it minimal."
    )


def _plan_one(
    claim: Claim,
    state: GraphState,
    settings: Settings,
    llm: StructuredLLM,
) -> ReductionPlan:
    """Design the plan for a single claim, folding in any review feedback."""
    feedback = state.get("plan_feedback", {}).get(claim.id, "")
    prompt = load_prompt(
        "design_plan",
        claim=json.dumps(claim.model_dump(mode="json"), indent=2, ensure_ascii=False),
        code_context=_code_context(state.get("repo_url")),
        budget_minutes=settings.budget_minutes_total,
        offline_datasets=_offline_datasets(),
        packages=", ".join(SANDBOX_PACKAGES),
        seeds=settings.seeds_per_arm,
        feedback_section=_FEEDBACK_TEMPLATE.format(feedback=feedback) if feedback else "",
    )

    plan = llm.invoke_structured(prompt, ReductionPlan)
    # The model sometimes echoes a different id; the claim we asked about wins.
    if plan.claim_id != claim.id:
        logger.warning("plan returned claim_id %r, expected %r", plan.claim_id, claim.id)
        plan = plan.model_copy(update={"claim_id": claim.id})
    return plan


def design_plan(
    state: GraphState,
    settings: Settings | None = None,
    llm: StructuredLLM | None = None,
) -> GraphState:
    """Graph node: produce a ReductionPlan for each selected claim.

    On a rejection this node runs again, so only claims still awaiting a plan
    are redesigned; already approved ones are left alone.
    """
    settings = settings or get_settings()
    llm = llm or ProviderStructuredLLM(settings)

    claims_by_id = {claim.id: claim for claim in state["claims"]}
    approved = set(state.get("approved_plan_ids", []))
    plans = dict(state.get("plans", {}))

    for claim_id in state.get("selected_claim_ids", []):
        if claim_id in approved:
            continue
        claim = claims_by_id.get(claim_id)
        if claim is None:
            logger.warning("selected claim %s is not in the claim list", claim_id)
            continue
        plans[claim_id] = _plan_one(claim, state, settings, llm)
        logger.info(
            "planned %s: %s, %d seeds, ~%.0f min",
            claim_id,
            plans[claim_id].code_source,
            plans[claim_id].seeds,
            plans[claim_id].estimated_minutes,
        )

    _write_plans(plans, Path(settings.runs_dir) / state["paper_id"])
    return {"plans": plans}


def _write_plans(plans: dict[str, ReductionPlan], out_dir: Path) -> None:
    """Persist the plans for inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {cid: plan.model_dump(mode="json") for cid, plan in plans.items()}
    (out_dir / "plans.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
