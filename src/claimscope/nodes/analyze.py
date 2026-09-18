"""Turn run results into verdicts (PLAN.md sections 6 and 8)."""

from __future__ import annotations

import logging
import re

from claimscope.config import Settings, get_settings
from claimscope.nodes.codegen import arms_for
from claimscope.schemas import Claim, ClaimVerdict, RunResult
from claimscope.state import GraphState
from claimscope.stats import Direction, InsufficientDataError, compare_arms

logger = logging.getLogger(__name__)

# Metrics where a lower number is better. Matched as whole words against the
# claim's metric text, so "test loss" and "error rate" are caught but
# "lossless" is not.
_LOWER_IS_BETTER_TERMS = (
    "loss",
    "error",
    "perplexity",
    "ppl",
    "rmse",
    "mse",
    "mae",
    "wer",
    "cer",
    "fid",
    "regret",
    "latency",
)


def metric_direction(claim: Claim) -> Direction:
    """Whether a higher or lower value of this claim's metric is better.

    The expected_direction text wins when it is explicit, because a claim can
    predict a *decrease* in a higher-is-better metric.
    """
    metric = claim.metric.lower()
    words = set(re.findall(r"[a-z]+", metric))
    if words & set(_LOWER_IS_BETTER_TERMS):
        return "lower_is_better"
    return "higher_is_better"


def _arm_values(results: list[RunResult], arm: str) -> list[float]:
    """Metric values for one arm, ordered by seed for reproducibility."""
    matching = [r for r in results if r.arm == arm]
    matching.sort(key=lambda r: r.seed)
    return [r.metric_value for r in matching]


def _untestable_verdict(claim: Claim) -> ClaimVerdict:
    return ClaimVerdict(
        claim_id=claim.id,
        verdict="not_testable",
        notes=claim.triage_reason or "Triage judged this claim not testable at reduced scale.",
    )


def _inconclusive(claim_id: str, reason: str) -> ClaimVerdict:
    return ClaimVerdict(claim_id=claim_id, verdict="inconclusive", notes=reason)


def analyze(state: GraphState, settings: Settings | None = None) -> GraphState:
    """Graph node: produce a verdict for every claim that was considered.

    Claims that were never testable, were abandoned, or ran out of budget all
    get a verdict too, with the reason recorded. Silence would read as success.
    """
    settings = settings or get_settings()

    claims = state.get("claims", [])
    run_results = state.get("run_results", {})
    abandoned = state.get("abandoned_claim_ids", {})
    over_budget = set(state.get("over_budget_claim_ids", []))
    selected = set(state.get("selected_claim_ids", []))

    verdicts: list[ClaimVerdict] = []

    for claim in claims:
        if claim.id not in selected:
            verdicts.append(_untestable_verdict(claim))
            continue

        if claim.id in abandoned:
            verdicts.append(_inconclusive(claim.id, abandoned[claim.id]))
            continue

        if claim.id in over_budget:
            verdicts.append(
                _inconclusive(
                    claim.id,
                    "The measured cost of this experiment exceeded the remaining compute budget.",
                )
            )
            continue

        results = run_results.get(claim.id)
        if not results:
            verdicts.append(_inconclusive(claim.id, "No runs completed for this claim."))
            continue

        verdicts.append(_verdict_for(claim, results))

    logger.info("produced %d verdicts", len(verdicts))
    return {"verdicts": verdicts}


def _verdict_for(claim: Claim, results: list[RunResult]) -> ClaimVerdict:
    """Compare the two arms of one claim and build its verdict."""
    arms = arms_for(claim)
    treatment_name, control_name = arms[0], arms[1]

    treatment = _arm_values(results, treatment_name)
    control = _arm_values(results, control_name)

    if not treatment or not control:
        return _inconclusive(
            claim.id,
            f"Missing results for one arm: {treatment_name} has {len(treatment)} runs, "
            f"{control_name} has {len(control)}.",
        )

    if len(treatment) != len(control):
        # Invariant 1: unmatched arms make the comparison meaningless.
        return _inconclusive(
            claim.id,
            f"Arms are not matched: {treatment_name} has {len(treatment)} runs but "
            f"{control_name} has {len(control)}. A comparison would not be valid.",
        )

    try:
        comparison = compare_arms(
            treatment,
            control,
            direction=metric_direction(claim),
            treatment_name=treatment_name,
            control_name=control_name,
        )
    except InsufficientDataError as exc:
        return _inconclusive(claim.id, str(exc))

    return ClaimVerdict(
        claim_id=claim.id,
        verdict=comparison.verdict,
        effect_estimate=comparison.effect,
        ci_low=comparison.ci_low,
        ci_high=comparison.ci_high,
        p_value=comparison.p_value,
        notes=comparison.notes(),
    )
