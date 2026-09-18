"""Decide which claims are worth testing at reduced scale (PLAN.md section 6)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from claimscope.config import Settings, get_settings
from claimscope.llm import ProviderStructuredLLM, StructuredLLM
from claimscope.prompts import load_prompt
from claimscope.schemas import Claim, TriageDecision, TriageResult
from claimscope.state import GraphState

logger = logging.getLogger(__name__)

ABSOLUTE_REASON = (
    "Absolute claims report a specific headline number, which cannot be "
    "reproduced at reduced scale by construction."
)


def _format_claims(claims: list[Claim]) -> str:
    """Render claims for the prompt, omitting the fields triage is about to set."""
    payload = [
        claim.model_dump(exclude={"testable", "triage_reason"}, mode="json") for claim in claims
    ]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _apply_decisions(
    claims: list[Claim], decisions: list[TriageDecision]
) -> dict[str, TriageDecision]:
    """Write decisions onto the claims, enforcing the absolute-claim rule.

    The rule that ``absolute`` is never testable is structural, so it is applied
    here rather than trusted to the model.
    """
    by_id = {decision.claim_id: decision for decision in decisions}
    applied: dict[str, TriageDecision] = {}

    for claim in claims:
        decision = by_id.get(claim.id)
        if decision is None:
            logger.warning("triage returned no decision for %s; treating as untestable", claim.id)
            decision = TriageDecision(
                claim_id=claim.id,
                testable=False,
                reason="Triage returned no decision for this claim.",
            )

        if claim.claim_type == "absolute" and decision.testable:
            logger.info("overriding triage: %s is an absolute claim", claim.id)
            decision = decision.model_copy(
                update={"testable": False, "reason": ABSOLUTE_REASON, "priority": 0}
            )

        claim.testable = decision.testable
        claim.triage_reason = decision.reason
        applied[claim.id] = decision

    return applied


def _select(claims: list[Claim], decisions: dict[str, TriageDecision], limit: int) -> list[str]:
    """Pick at most ``limit`` testable claims, best priority first."""
    testable = [claim for claim in claims if claim.testable]
    testable.sort(key=lambda c: (decisions[c.id].priority or 999, c.id))
    return [claim.id for claim in testable[:limit]]


def triage(
    state: GraphState,
    settings: Settings | None = None,
    llm: StructuredLLM | None = None,
) -> GraphState:
    """Graph node: mark claims testable and select at most K of them."""
    settings = settings or get_settings()
    llm = llm or ProviderStructuredLLM(settings)

    claims = state["claims"]
    if not claims:
        logger.warning("triage received no claims")
        return {"claims": [], "selected_claim_ids": []}

    prompt = load_prompt(
        "triage",
        claims=_format_claims(claims),
        budget_minutes=settings.budget_minutes_total,
        seeds=settings.seeds_per_arm,
    )
    result = llm.invoke_structured(prompt, TriageResult)

    decisions = _apply_decisions(claims, result.decisions)
    selected = _select(claims, decisions, settings.max_claims)

    logger.info(
        "triage: %d/%d testable, selected %s",
        sum(1 for c in claims if c.testable),
        len(claims),
        selected,
    )
    _write_triage(claims, Path(settings.runs_dir) / state["paper_id"])
    return {"claims": claims, "selected_claim_ids": selected}


def _write_triage(claims: list[Claim], out_dir: Path) -> None:
    """Persist the triaged claims for inspection and evaluation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [claim.model_dump(mode="json") for claim in claims]
    (out_dir / "triage.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
