"""Verdict assignment, including the paths where no experiment ran."""

from __future__ import annotations

import pytest

from claimscope.config import Settings
from claimscope.nodes.analyze import analyze, metric_direction
from claimscope.schemas import Claim, RunResult
from claimscope.state import GraphState


def _claim(
    claim_id: str = "c1",
    metric: str = "accuracy",
    claim_type: str = "comparative",
    arms: list[str] | None = None,
    **extra: object,
) -> Claim:
    return Claim(
        id=claim_id,
        text="Method A beats baseline B.",
        source_location="Table 1",
        claim_type=claim_type,  # type: ignore[arg-type]
        arms=arms if arms is not None else ["treatment", "control"],
        metric=metric,
        expected_direction="treatment > control",
        **extra,  # type: ignore[arg-type]
    )


def _runs(arm: str, values: list[float]) -> list[RunResult]:
    return [
        RunResult(arm=arm, seed=i, metric_value=v, runtime_s=1.0, log_path=f"{arm}_{i}.txt")
        for i, v in enumerate(values)
    ]


def _state(**overrides: object) -> GraphState:
    state: GraphState = {
        "paper_id": "2401.00001",
        "claims": [_claim()],
        "selected_claim_ids": ["c1"],
        "run_results": {
            "c1": _runs("treatment", [0.9, 0.91, 0.89]) + _runs("control", [0.7, 0.71, 0.69])
        },
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


class TestMetricDirection:
    @pytest.mark.parametrize(
        "metric",
        ["test loss", "error rate", "perplexity", "RMSE", "word error rate (WER)", "FID"],
    )
    def test_lower_is_better_metrics(self, metric: str) -> None:
        assert metric_direction(_claim(metric=metric)) == "lower_is_better"

    @pytest.mark.parametrize("metric", ["accuracy", "top-1 accuracy", "BLEU", "F1 score"])
    def test_higher_is_better_metrics(self, metric: str) -> None:
        assert metric_direction(_claim(metric=metric)) == "higher_is_better"

    def test_lossless_is_not_a_loss_metric(self) -> None:
        # Substring matching would get this wrong.
        assert metric_direction(_claim(metric="lossless compression ratio")) == "higher_is_better"


class TestVerdicts:
    def test_a_clear_win_is_consistent(self, settings: Settings) -> None:
        result = analyze(_state(), settings)

        verdict = result["verdicts"][0]
        assert verdict.verdict == "consistent_at_reduced_scale"
        assert verdict.effect_estimate is not None
        assert verdict.ci_low is not None

    def test_a_reversal_is_not_consistent(self, settings: Settings) -> None:
        state = _state(
            run_results={
                "c1": _runs("treatment", [0.7, 0.71, 0.69]) + _runs("control", [0.9, 0.91, 0.89])
            }
        )

        assert analyze(state, settings)["verdicts"][0].verdict == "not_consistent_at_reduced_scale"

    def test_a_loss_metric_is_interpreted_in_the_right_direction(self, settings: Settings) -> None:
        # Treatment has lower loss, which is the predicted improvement.
        state = _state(
            claims=[_claim(metric="test loss")],
            run_results={
                "c1": _runs("treatment", [0.1, 0.11, 0.09]) + _runs("control", [0.3, 0.31, 0.29])
            },
        )

        assert analyze(state, settings)["verdicts"][0].verdict == "consistent_at_reduced_scale"

    def test_notes_carry_the_power_warning(self, settings: Settings) -> None:
        assert "low power" in analyze(_state(), settings)["verdicts"][0].notes


class TestNonExecutedClaims:
    """Every claim gets a verdict; silence would read as success."""

    def test_unselected_claims_are_not_testable(self, settings: Settings) -> None:
        state = _state(
            claims=[
                _claim(),
                _claim("c2", claim_type="absolute", triage_reason="Absolute number."),
            ],
            selected_claim_ids=["c1"],
        )

        verdicts = {v.claim_id: v for v in analyze(state, settings)["verdicts"]}
        assert verdicts["c2"].verdict == "not_testable"
        assert verdicts["c2"].notes == "Absolute number."

    def test_abandoned_claims_are_inconclusive_with_the_reason(self, settings: Settings) -> None:
        state = _state(
            run_results={},
            abandoned_claim_ids={"c1": "Gave up after 3 debug attempts."},
        )

        verdict = analyze(state, settings)["verdicts"][0]
        assert verdict.verdict == "inconclusive"
        assert "3 debug attempts" in verdict.notes

    def test_over_budget_claims_are_inconclusive(self, settings: Settings) -> None:
        state = _state(run_results={}, over_budget_claim_ids=["c1"])

        verdict = analyze(state, settings)["verdicts"][0]
        assert verdict.verdict == "inconclusive"
        assert "budget" in verdict.notes

    def test_a_claim_with_no_runs_is_inconclusive(self, settings: Settings) -> None:
        verdict = analyze(_state(run_results={}), settings)["verdicts"][0]

        assert verdict.verdict == "inconclusive"
        assert "No runs completed" in verdict.notes


class TestArmIntegrity:
    def test_unmatched_arms_are_refused(self, settings: Settings) -> None:
        # Invariant 1: comparing 3 seeds against 2 is not a valid comparison.
        state = _state(
            run_results={
                "c1": _runs("treatment", [0.9, 0.91, 0.89]) + _runs("control", [0.7, 0.71])
            }
        )

        verdict = analyze(state, settings)["verdicts"][0]
        assert verdict.verdict == "inconclusive"
        assert "not matched" in verdict.notes

    def test_a_missing_arm_is_refused(self, settings: Settings) -> None:
        state = _state(run_results={"c1": _runs("treatment", [0.9, 0.91, 0.89])})

        verdict = analyze(state, settings)["verdicts"][0]
        assert verdict.verdict == "inconclusive"
        assert "Missing results" in verdict.notes


def test_every_claim_receives_a_verdict(settings: Settings) -> None:
    state = _state(
        claims=[_claim(), _claim("c2", claim_type="absolute"), _claim("c3")],
        selected_claim_ids=["c1"],
    )

    verdicts = analyze(state, settings)["verdicts"]

    assert {v.claim_id for v in verdicts} == {"c1", "c2", "c3"}
