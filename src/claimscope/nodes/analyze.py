"""Turn run results into verdicts (PLAN.md sections 6 and 8)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

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


_DIAGNOSTIC = re.compile(r"train_(?:error|loss)=([0-9.eE+-]+)\s+test_(?:error|loss)=([0-9.eE+-]+)")

_NEGLIGIBLE_GAP = 0.02
"""Below this train/test gap the baseline is not overfitting in any useful sense."""

_FAILED_TO_FIT = 0.5
"""Above this training error the baseline never learned its own training set.

A hand-written convnet came out at 80% training error on a 10-class task, which
is near chance. Comparing a regulariser against a model that did not train is
not a weak measurement, it is no measurement.
"""


def _control_diagnostics(control: list[RunResult]) -> list[tuple[float, float]]:
    """The (train, test) pairs the baseline arm's scripts printed.

    Scripts print a `diagnostic: train_error=... test_error=...` line. An absent
    diagnostic yields nothing, so the caller stays quiet rather than guessing.
    """
    pairs: list[tuple[float, float]] = []
    for run in control:
        try:
            text = Path(run.log_path).read_text(encoding="utf-8")
        except OSError:
            continue
        if match := _DIAGNOSTIC.search(text):
            try:
                pairs.append((float(match.group(1)), float(match.group(2))))
            except ValueError:
                continue
    return pairs


def _regime_note(control: list[RunResult]) -> str | None:
    """Whether the baseline arm was in a regime where the effect could appear.

    Two ways it is not, and they look identical in the verdict: the task is too
    easy, so there is no overfitting to reduce; or the model never fit its own
    training data, so the arms differ by noise between two broken runs.
    """
    pairs = _control_diagnostics(control)
    if not pairs:
        return None

    mean_train = sum(train for train, _ in pairs) / len(pairs)
    if mean_train >= _FAILED_TO_FIT:
        return (
            f"The baseline arm never fit its own training data (training error "
            f"{mean_train:.3f}), so this comparison is between two models that did not "
            "learn. The experiment needs redesigning, not more seeds."
        )

    mean_gap = sum(test - train for train, test in pairs) / len(pairs)
    if mean_gap < _NEGLIGIBLE_GAP:
        return (
            f"The baseline arm barely overfits (train/test gap {mean_gap:.3f}), so the reduced "
            "task may be too easy for this effect to appear at all. Treat this as a limit of the "
            "reduction rather than evidence about the claim."
        )

    return None


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


def _is_degenerate(values: list[float]) -> bool:
    """Whether an arm produced no usable signal.

    All-identical values across seeds mean the experiment did not measure
    anything: a task too easy to separate the arms, a metric that never moved,
    or a script that returned a constant. Reporting a zero effect would imply
    the arms were compared and found equal, which is not what happened.
    """
    return len(set(values)) == 1


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

    if _is_degenerate(treatment + control):
        constant = treatment[0]
        return _inconclusive(
            claim.id,
            f"Every run returned the same value ({constant:g}), so the experiment measured "
            "nothing. The reduced task is probably too easy or too hard to separate the "
            "arms, or the metric did not respond to the intervention. The experiment needs "
            "redesigning, not rerunning.",
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

    notes = comparison.notes()
    # An inconclusive result has two very different causes, and the reader
    # cannot tell them apart from the interval alone: the effect may be absent,
    # or the reduced task may never have been in a regime where it could appear.
    if comparison.verdict == "inconclusive":
        control_runs = [run for run in results if run.arm == control_name]
        if regime := _regime_note(control_runs):
            notes = f"{notes} {regime}"

    return ClaimVerdict(
        claim_id=claim.id,
        verdict=comparison.verdict,
        effect_estimate=comparison.effect,
        ci_low=comparison.ci_low,
        ci_high=comparison.ci_high,
        p_value=comparison.p_value,
        notes=notes,
    )
