"""Verdict assignment, including the paths where no experiment ran."""

from __future__ import annotations

from pathlib import Path

import pytest

from claimscope.config import Settings
from claimscope.nodes.analyze import _regime_note, analyze, metric_direction
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


class TestDegenerateResults:
    """An experiment that measured nothing must not read as "no difference".

    This came from a real run: the generated experiment used synthetic data so
    separable that both arms scored exactly zero errors, and the report showed a
    clean zero effect as though the arms had been compared and found equal.
    """

    def test_all_identical_values_are_called_out(self, settings: Settings) -> None:
        state = _state(
            run_results={
                "c1": _runs("treatment", [0.0, 0.0, 0.0]) + _runs("control", [0.0, 0.0, 0.0])
            }
        )

        verdict = analyze(state, settings)["verdicts"][0]

        assert verdict.verdict == "inconclusive"
        assert "measured nothing" in verdict.notes
        assert "redesigning" in verdict.notes

    def test_it_does_not_report_a_zero_effect(self, settings: Settings) -> None:
        state = _state(
            run_results={
                "c1": _runs("treatment", [1.0, 1.0, 1.0]) + _runs("control", [1.0, 1.0, 1.0])
            }
        )

        verdict = analyze(state, settings)["verdicts"][0]

        # A zero with a tight interval would look like a measured null result.
        assert verdict.effect_estimate is None
        assert verdict.ci_low is None

    def test_a_real_null_result_is_still_analysed(self, settings: Settings) -> None:
        """Arms that genuinely overlap are a measurement, not a degenerate run."""
        state = _state(
            run_results={
                "c1": _runs("treatment", [0.80, 0.82, 0.79]) + _runs("control", [0.81, 0.78, 0.83])
            }
        )

        verdict = analyze(state, settings)["verdicts"][0]

        assert verdict.verdict == "inconclusive"
        assert verdict.effect_estimate is not None
        assert "measured nothing" not in verdict.notes


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


class TestClaimShapesThisAnalysisCannotSettle:
    """Claim shapes the one-sided comparison would answer wrongly.

    Both came out of running real papers: Adam's `adam ~= sgd_nesterov >
    adagrad` was reported "not consistent" from a two-arm test that silently
    dropped adagrad and read the "~=" as ">".
    """

    def test_a_three_arm_claim_is_not_forced_into_a_two_arm_test(self, settings: Settings) -> None:
        claim = _claim(arms=["adam", "sgd_nesterov", "adagrad"])
        claim.expected_direction = "adam > sgd_nesterov > adagrad"
        state = _state(
            claims=[claim],
            run_results={
                "c1": _runs("adam", [0.10, 0.11, 0.09])
                + _runs("sgd_nesterov", [0.20, 0.21, 0.19])
                + _runs("adagrad", [0.30, 0.31, 0.29])
            },
        )

        verdict = analyze(state, settings)["verdicts"][0]
        assert verdict.verdict == "inconclusive"
        assert "3 arms" in verdict.notes
        # The point of refusing: no effect is reported for a comparison that
        # was never the one the claim asked for.
        assert verdict.effect_estimate is None

    def test_an_equivalence_claim_is_not_read_as_directional(self, settings: Settings) -> None:
        claim = _claim()
        claim.expected_direction = "treatment approximately equal to control"
        # Arms that separate cleanly: a directional test would call this
        # "consistent", which for a sameness claim is backwards.
        state = _state(
            claims=[claim],
            run_results={
                "c1": _runs("treatment", [0.90, 0.91, 0.89]) + _runs("control", [0.50, 0.51, 0.49])
            },
        )

        verdict = analyze(state, settings)["verdicts"][0]
        assert verdict.verdict == "inconclusive"
        assert "alike" in verdict.notes
        assert verdict.effect_estimate is None

    def test_a_no_improvement_claim_is_caught_too(self, settings: Settings) -> None:
        claim = _claim()
        claim.expected_direction = "A-LRN shows no improvement over A"

        verdict = analyze(_state(claims=[claim]), settings)["verdicts"][0]

        assert verdict.verdict == "inconclusive"
        assert "alike" in verdict.notes

    def test_an_ordinary_two_arm_claim_still_gets_a_verdict(self, settings: Settings) -> None:
        # The guard must not swallow the normal case.
        verdict = analyze(_state(), settings)["verdicts"][0]

        assert verdict.verdict == "consistent_at_reduced_scale"
        assert verdict.effect_estimate is not None


def test_every_claim_receives_a_verdict(settings: Settings) -> None:
    state = _state(
        claims=[_claim(), _claim("c2", claim_type="absolute"), _claim("c3")],
        selected_claim_ids=["c1"],
    )

    verdicts = analyze(state, settings)["verdicts"]

    assert {v.claim_id for v in verdicts} == {"c1", "c2", "c3"}


class TestRegimeNote:
    """An inconclusive verdict hides two very different failures.

    The effect may be absent, or the experiment may never have been in a regime
    where it could appear. The interval looks the same either way, so the
    scripts' train/test diagnostic is what separates them.
    """

    def _run(self, tmp_path: Path, arm: str, seed: int, train: float, test: float) -> RunResult:
        log = tmp_path / f"log_{arm}_{seed}.txt"
        log.write_text(f"diagnostic: train_error={train} test_error={test}\n", encoding="utf-8")
        return RunResult(arm=arm, seed=seed, metric_value=test, runtime_s=1.0, log_path=str(log))

    def test_a_baseline_that_never_learned_is_called_out(self, tmp_path: Path) -> None:
        control = [self._run(tmp_path, "control", i, 0.81, 0.85) for i in range(3)]

        note = _regime_note(control)

        assert note is not None
        assert "never fit its own training data" in note

    def test_a_task_too_easy_is_called_out(self, tmp_path: Path) -> None:
        control = [self._run(tmp_path, "control", i, 0.045, 0.048) for i in range(3)]

        note = _regime_note(control)

        assert note is not None
        assert "barely overfits" in note

    def test_a_healthy_overfitting_baseline_says_nothing(self, tmp_path: Path) -> None:
        """Train 0.2%, test 3.8%: exactly the regime a regulariser needs."""
        control = [self._run(tmp_path, "control", i, 0.002, 0.038) for i in range(3)]

        assert _regime_note(control) is None

    def test_no_diagnostic_says_nothing(self, tmp_path: Path) -> None:
        """An absent diagnostic is not evidence either way."""
        log = tmp_path / "log_plain.txt"
        log.write_text("nothing useful here\n", encoding="utf-8")
        control = [
            RunResult(arm="control", seed=0, metric_value=0.1, runtime_s=1.0, log_path=str(log))
        ]

        assert _regime_note(control) is None

    def test_a_missing_log_does_not_raise(self, tmp_path: Path) -> None:
        control = [
            RunResult(
                arm="control",
                seed=0,
                metric_value=0.1,
                runtime_s=1.0,
                log_path=str(tmp_path / "gone.txt"),
            )
        ]

        assert _regime_note(control) is None
