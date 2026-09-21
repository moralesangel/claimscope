"""Execution: dry-run calibration, the budget guard, and failure handling."""

from __future__ import annotations

import json
from pathlib import Path

from claimscope.config import Settings
from claimscope.nodes.codegen import RESULT_FILE
from claimscope.nodes.execute import (
    DRY_RUN_STEPS,
    dry_run,
    estimate_total_minutes,
    execute,
    run_study,
)
from claimscope.schemas import Claim, ReductionPlan
from claimscope.state import GraphState
from stubs import FakeRunner

PAPER_ID = "2401.00001"


def _claim(claim_id: str = "c1") -> Claim:
    return Claim(
        id=claim_id,
        text="Method A beats baseline B.",
        source_location="Table 1",
        claim_type="comparative",
        arms=["treatment", "control"],
        metric="accuracy",
        expected_direction="treatment > control",
    )


def _plan(claim_id: str = "c1", seeds: int = 3) -> ReductionPlan:
    return ReductionPlan(
        claim_id=claim_id,
        original_setup="Full.",
        reduced_setup="Small.",
        changes=["Less data -- safe."],
        preserved=["Architecture."],
        why_claim_should_transfer="Scale free.",
        seeds=seeds,
        estimated_minutes=5.0,
        code_source="from_scratch",
    )


def _workspace(settings: Settings, claim_id: str = "c1") -> Path:
    ws = settings.runs_dir / PAPER_ID / "workspaces" / claim_id
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "run.py").write_text("print('x')\n", encoding="utf-8")
    return ws


def _state(settings: Settings, **overrides: object) -> GraphState:
    state: GraphState = {
        "paper_id": PAPER_ID,
        "claims": [_claim()],
        "plans": {"c1": _plan()},
        "approved_plan_ids": ["c1"],
        "workspace_dirs": {"c1": str(_workspace(settings))},
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


class TestDryRun:
    def test_passes_a_step_limit(self, settings: Settings) -> None:
        runner = FakeRunner()

        dry_run(runner, _workspace(settings), "treatment")

        assert runner.calls[0] == [
            "--arm",
            "treatment",
            "--seed",
            "0",
            "--steps",
            str(DRY_RUN_STEPS),
        ]

    def test_measures_seconds_per_step(self, settings: Settings) -> None:
        runner = FakeRunner(seconds_per_call=10.0)

        _result, per_step = dry_run(runner, _workspace(settings), "treatment")

        assert per_step == 10.0 / DRY_RUN_STEPS


class TestEstimate:
    def test_extrapolates_the_whole_study(self) -> None:
        # 0.5s per step x 100 steps x 6 runs = 300s = 5 min.
        assert estimate_total_minutes(0.5, 100, 6) == 5.0


class TestBudgetGuard:
    def test_refuses_a_study_that_cannot_finish(self, settings: Settings) -> None:
        # Very slow: extrapolation will exceed a tiny budget.
        runner = FakeRunner(seconds_per_call=60.0)
        tight = settings.model_copy(update={"budget_minutes_total": 1.0})

        result = execute(_state(settings), tight, runner)

        assert result["over_budget_claim_ids"] == ["c1"]
        assert "c1" not in result["run_results"]
        assert any("exceeds" in e for e in result["errors"])

    def test_the_message_says_how_to_fix_it(self, settings: Settings) -> None:
        """Being told a run is too expensive is only useful with a next step."""
        runner = FakeRunner(seconds_per_call=60.0)
        tight = settings.model_copy(update={"budget_minutes_total": 1.0})

        result = execute(_state(settings), tight, runner)

        message = next(e for e in result["errors"] if "exceeds" in e)
        assert "CLAIMSCOPE_BUDGET_MINUTES_TOTAL" in message
        assert "reject the plan" in message

    def test_runs_a_study_that_fits(self, settings: Settings) -> None:
        runner = FakeRunner(seconds_per_call=0.01)

        result = execute(_state(settings), settings, runner)

        assert result["over_budget_claim_ids"] == []
        assert len(result["run_results"]["c1"]) == 6  # 2 arms x 3 seeds


class TestRunStudy:
    def test_runs_every_arm_and_seed(self, settings: Settings) -> None:
        runner = FakeRunner()

        results, failure = run_study(runner, _claim(), _plan(), _workspace(settings), 60)

        assert failure is None
        assert len(results) == 6
        assert {r.arm for r in results} == {"treatment", "control"}
        assert {r.seed for r in results} == {0, 1, 2}

    def test_both_arms_get_the_same_seeds(self, settings: Settings) -> None:
        # Invariant 1: arms must be matched, or the comparison is meaningless.
        runner = FakeRunner()

        results, _ = run_study(runner, _claim(), _plan(), _workspace(settings), 60)

        by_arm: dict[str, set[int]] = {}
        for r in results:
            by_arm.setdefault(r.arm, set()).add(r.seed)
        assert by_arm["treatment"] == by_arm["control"]

    def test_stops_at_the_first_failure(self, settings: Settings) -> None:
        # A broken script fails identically for every seed; running on wastes budget.
        runner = FakeRunner(fail_times=99)

        results, failure = run_study(runner, _claim(), _plan(), _workspace(settings), 60)

        assert results == []
        assert failure is not None
        assert len(runner.calls) == 1

    def test_records_the_metric_from_result_json(self, settings: Settings) -> None:
        runner = FakeRunner(metric_by_arm={"treatment": 0.9, "control": 0.4})

        results, _ = run_study(runner, _claim(), _plan(seeds=3), _workspace(settings), 60)

        treatment = [r.metric_value for r in results if r.arm == "treatment"]
        assert all(v >= 0.9 for v in treatment)

    def test_writes_a_log_per_run(self, settings: Settings) -> None:
        workspace = _workspace(settings)

        results, _ = run_study(FakeRunner(), _claim(), _plan(), workspace, 60)

        for result in results:
            assert Path(result.log_path).exists()


class TestDegenerateStudy:
    """An experiment where every run ties measured nothing, and should be fixed.

    From a real run: the generated task was so easy that both arms scored zero
    errors on every seed. Nothing crashed, so without this the study would be
    reported as a clean null result.
    """

    def test_identical_results_are_sent_to_debug(self, settings: Settings) -> None:
        class TiedRunner(FakeRunner):
            """Returns the same metric no matter the arm or seed."""

            def execute(self, request: object) -> object:  # type: ignore[override]
                result = super().execute(request)  # type: ignore[arg-type]
                (request.workspace / RESULT_FILE).write_text(  # type: ignore[attr-defined]
                    json.dumps({"arm": "x", "seed": 0, "metric": 0.0, "metric_name": "m"}),
                    encoding="utf-8",
                )
                return result

        results, failure = run_study(TiedRunner(), _claim(), _plan(), _workspace(settings), 60)

        assert len(results) == 6
        assert failure is not None
        assert "measured nothing" in failure.traceback_text()

    def test_the_message_tells_the_debugger_what_to_do(self, settings: Settings) -> None:
        class TiedRunner(FakeRunner):
            def execute(self, request: object) -> object:  # type: ignore[override]
                result = super().execute(request)  # type: ignore[arg-type]
                (request.workspace / RESULT_FILE).write_text(  # type: ignore[attr-defined]
                    json.dumps({"arm": "x", "seed": 0, "metric": 1.0, "metric_name": "m"}),
                    encoding="utf-8",
                )
                return result

        _results, failure = run_study(TiedRunner(), _claim(), _plan(), _workspace(settings), 60)

        assert failure is not None
        assert "too easy" in failure.traceback_text()
        assert "keeping the comparison fair" in failure.traceback_text()

    def test_varying_results_are_not_flagged(self, settings: Settings) -> None:
        # The normal case: FakeRunner varies the metric by arm and seed.
        results, failure = run_study(FakeRunner(), _claim(), _plan(), _workspace(settings), 60)

        assert failure is None
        assert len(results) == 6


class TestMissingMetric:
    def test_a_run_without_result_json_is_a_failure(self, settings: Settings) -> None:
        class SilentRunner(FakeRunner):
            def execute(self, request: object) -> object:  # type: ignore[override]
                from claimscope.sandbox.runner import ExecutionResult

                self.calls.append(list(request.args))  # type: ignore[attr-defined]
                return ExecutionResult(exit_code=0, stdout="done", stderr="", duration_s=0.1)

        results, failure = run_study(SilentRunner(), _claim(), _plan(), _workspace(settings), 60)

        assert results == []
        assert failure is not None
        assert "result.json" in failure.traceback_text()

    def test_malformed_result_json_is_a_failure(self, settings: Settings) -> None:
        workspace = _workspace(settings)

        class BadJsonRunner(FakeRunner):
            def execute(self, request: object) -> object:  # type: ignore[override]
                from claimscope.sandbox.runner import ExecutionResult

                (workspace / RESULT_FILE).write_text("{not json", encoding="utf-8")
                return ExecutionResult(exit_code=0, stdout="", stderr="", duration_s=0.1)

        results, failure = run_study(BadJsonRunner(), _claim(), _plan(), workspace, 60)

        assert results == []
        assert failure is not None


class TestExecuteNode:
    def test_records_a_failure_for_the_debug_node(self, settings: Settings) -> None:
        runner = FakeRunner(fail_times=99, fail_message="ValueError: bad shape")

        result = execute(_state(settings), settings, runner)

        failure = result["execution_failures"]["c1"]
        assert "bad shape" in failure.traceback_text()

    def test_skips_claims_that_already_ran(self, settings: Settings) -> None:
        runner = FakeRunner()
        state = _state(settings, run_results={"c1": []})

        execute(state, settings, runner)

        assert runner.calls == []

    def test_accumulates_budget_used(self, settings: Settings) -> None:
        runner = FakeRunner(seconds_per_call=6.0)

        result = execute(_state(settings), settings, runner)

        # 6 runs x 6s = 36s = 0.6 min (the dry run is not counted as study time).
        assert result["budget_minutes_used"] > 0

    def test_reports_a_missing_workspace(self, settings: Settings) -> None:
        result = execute(_state(settings, workspace_dirs={}), settings, FakeRunner())

        assert any("missing claim, plan or workspace" in e for e in result["errors"])

    def test_writes_result_json_the_node_can_read(self, settings: Settings) -> None:
        workspace = _workspace(settings)
        runner = FakeRunner(metric_by_arm={"treatment": 0.75, "control": 0.25})

        execute(_state(settings), settings, runner)

        payload = json.loads((workspace / RESULT_FILE).read_text(encoding="utf-8"))
        assert payload["metric_name"] == "accuracy"
