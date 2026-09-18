"""Human approval of reduction plans, via a LangGraph interrupt (PLAN.md section 6).

Resuming re-executes the node from the top, so everything before ``interrupt()``
runs twice. Keep that part free of side effects, and apply the decision only
after the value comes back.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.types import interrupt
from pydantic import BaseModel, Field

from claimscope.state import GraphState

logger = logging.getLogger(__name__)

ReviewAction = Literal["approve", "reject"]


class ReviewRequest(BaseModel):
    """What the CLI is asked to show the user while the graph is paused."""

    claim_id: str
    claim_text: str
    plan: dict[str, Any]


class ReviewResponse(BaseModel):
    """What the CLI sends back to resume the graph."""

    claim_id: str
    action: ReviewAction
    feedback: str = Field(default="", description="Required when rejecting.")


def _pending(state: GraphState) -> list[str]:
    """Selected claims that have a plan but no decision yet."""
    approved = set(state.get("approved_plan_ids", []))
    plans = state.get("plans", {})
    return [
        cid for cid in state.get("selected_claim_ids", []) if cid in plans and cid not in approved
    ]


def review(state: GraphState) -> GraphState:
    """Graph node: pause for human approval of each pending plan.

    Approving records the plan id; rejecting records the feedback and clears the
    plan so ``design_plan`` redesigns it.
    """
    pending = _pending(state)
    if not pending:
        return {}

    claims_by_id = {claim.id: claim for claim in state["claims"]}
    plans = state.get("plans", {})

    requests = [
        ReviewRequest(
            claim_id=cid,
            claim_text=claims_by_id[cid].text if cid in claims_by_id else "",
            plan=plans[cid].model_dump(mode="json"),
        ).model_dump(mode="json")
        for cid in pending
    ]

    # Execution stops here. On resume the node restarts and this returns the
    # value supplied via Command(resume=...).
    raw = interrupt({"type": "plan_review", "requests": requests})

    responses = _parse_responses(raw, pending)

    approved = list(state.get("approved_plan_ids", []))
    feedback = dict(state.get("plan_feedback", {}))
    remaining_plans = dict(plans)

    for response in responses:
        if response.action == "approve":
            if response.claim_id not in approved:
                approved.append(response.claim_id)
            feedback.pop(response.claim_id, None)
            logger.info("plan approved: %s", response.claim_id)
        else:
            feedback[response.claim_id] = response.feedback
            remaining_plans.pop(response.claim_id, None)
            logger.info("plan rejected: %s (%s)", response.claim_id, response.feedback[:80])

    return {
        "approved_plan_ids": approved,
        "plan_feedback": feedback,
        "plans": remaining_plans,
    }


def _parse_responses(raw: object, pending: list[str]) -> list[ReviewResponse]:
    """Normalise whatever the resume value was into ReviewResponse objects.

    Accepts a single response or a list, as dicts or models, so the CLI and
    tests can resume with the shape that reads most naturally.
    """
    items = raw if isinstance(raw, list) else [raw]
    responses: list[ReviewResponse] = []

    for item in items:
        if isinstance(item, ReviewResponse):
            responses.append(item)
        elif isinstance(item, dict):
            responses.append(ReviewResponse.model_validate(item))
        else:
            raise TypeError(f"cannot interpret review response of type {type(item).__name__}")

    unexpected = {r.claim_id for r in responses} - set(pending)
    if unexpected:
        raise ValueError(f"review response for claims that were not pending: {sorted(unexpected)}")
    return responses


def needs_redesign(state: GraphState) -> bool:
    """Whether any selected claim still lacks an approved plan.

    A rejection clears that claim's plan, so design_plan runs again for it.
    """
    approved = set(state.get("approved_plan_ids", []))
    return any(cid not in approved for cid in state.get("selected_claim_ids", []))
