"""Score one run of the agent against a paper's annotation (PLAN.md section 9)."""

from __future__ import annotations

import logging

from claimscope.schemas import Claim, ClaimVerdict
from claimscope.state import GraphState
from eval.matching import match_claims
from eval.schemas import (
    ClassificationMetrics,
    CostMetrics,
    ExecutionMetrics,
    ExtractionMetrics,
    PaperAnnotation,
    PaperResult,
    VerdictMetrics,
)

logger = logging.getLogger(__name__)


def score_extraction(
    predicted: list[Claim], annotation: PaperAnnotation
) -> tuple[ExtractionMetrics, ClassificationMetrics]:
    """Precision and recall of extraction, plus accuracy on the matched claims.

    Classification is scored only over matches: penalising the type of a claim
    the agent never extracted would count the same miss twice.
    """
    result = match_claims(predicted, annotation.claims)

    extraction = ExtractionMetrics(
        matched=len(result.matches),
        predicted=len(predicted),
        annotated=len(annotation.claims),
    )

    classification = ClassificationMetrics(total=len(result.matches))
    for match in result.matches:
        if match.predicted.claim_type == match.annotated.claim_type:
            classification.type_correct += 1
        if match.predicted.testable is not None and (
            match.predicted.testable == match.annotated.testable
        ):
            classification.testable_correct += 1

    return extraction, classification


def score_verdicts(
    verdicts: list[ClaimVerdict],
    predicted: list[Claim],
    annotation: PaperAnnotation,
) -> VerdictMetrics:
    """Agreement with the expected verdict, over claims that have one.

    Only claims the annotator gave an expected verdict for are compared; an
    annotator who was unsure should not be counted as a disagreement.
    """
    result = match_claims(predicted, annotation.claims)
    by_claim_id = {verdict.claim_id: verdict for verdict in verdicts}

    metrics = VerdictMetrics()
    for match in result.matches:
        expected = match.annotated.expected_verdict
        verdict = by_claim_id.get(match.predicted.id)
        if expected is None or verdict is None:
            continue
        metrics.measured = True
        metrics.compared += 1
        if verdict.verdict == expected:
            metrics.agreed += 1

    return metrics


def score_execution(state: GraphState, sandbox_available: bool) -> ExecutionMetrics:
    """How much generated code ran, when a sandbox was there to run it.

    Without a sandbox this returns measured=False rather than zeros, so a report
    can distinguish "nothing ran" from "everything failed".
    """
    if not sandbox_available:
        return ExecutionMetrics(measured=False)

    approved = state.get("approved_plan_ids") or []
    run_results = state.get("run_results") or {}
    attempts = state.get("debug_attempts") or {}

    return ExecutionMetrics(
        measured=True,
        claims_attempted=len(approved),
        claims_completed=sum(1 for claim_id in approved if run_results.get(claim_id)),
        total_debug_attempts=sum(attempts.get(claim_id, 0) for claim_id in approved),
    )


def score_paper(
    annotation: PaperAnnotation,
    state: GraphState,
    cost: CostMetrics,
    sandbox_available: bool = False,
) -> PaperResult:
    """Score one paper end to end."""
    predicted = state.get("claims") or []
    verdicts = state.get("verdicts") or []

    extraction, classification = score_extraction(predicted, annotation)

    return PaperResult(
        paper_id=annotation.paper_id,
        extraction=extraction,
        classification=classification,
        execution=score_execution(state, sandbox_available),
        verdicts=score_verdicts(verdicts, predicted, annotation),
        cost=cost,
        errors=list(state.get("errors") or []),
    )
