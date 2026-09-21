"""Run the generated experiments in the sandbox (PLAN.md section 6).

A short dry run measures real speed first. If extrapolating it exceeds the
remaining budget, the claim goes back to planning rather than burning the budget
on a study that cannot finish.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from claimscope.config import Settings, get_settings
from claimscope.nodes.codegen import REPO_DIR, RESULT_FILE, SANDBOX_PACKAGES, arms_for
from claimscope.sandbox.runner import (
    ExecutionFailure,
    ExecutionRequest,
    ExecutionResult,
    ImageSpec,
    Runner,
    SandboxError,
    is_contained,
)
from claimscope.schemas import Claim, ReductionPlan, RunResult
from claimscope.state import GraphState

logger = logging.getLogger(__name__)

DRY_RUN_STEPS = 20
"""Steps used to calibrate timing. Short enough to be cheap, long enough to measure."""

DRY_RUN_TIMEOUT_S = 300


def sandbox_packages(workspace_dirs: dict[str, str]) -> set[str]:
    """Packages the image needs: our defaults plus whatever the repos declare.

    Repository requirements are advisory. A paper's full dependency stack often
    cannot be installed on CPU in a small image, so anything that fails to
    resolve is dropped rather than failing the build; the codegen prompt tells
    the model to work with what is actually available.
    """
    packages = set(SANDBOX_PACKAGES)
    for workspace_dir in workspace_dirs.values():
        requirements = Path(workspace_dir) / REPO_DIR / "requirements.txt"
        if requirements.exists():
            packages |= _parse_requirements(requirements)
    return packages


def _parse_requirements(path: Path) -> set[str]:
    """Package names from a requirements file, ignoring options and comments."""
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        names.add(line)
    return names


def _read_metric(workspace: Path) -> float:
    """Read the metric the experiment wrote, failing loudly if it is malformed."""
    result_path = workspace / RESULT_FILE
    if not result_path.exists():
        raise ValueError(f"{RESULT_FILE} was not written")

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if "metric" not in payload:
        raise ValueError(f"{RESULT_FILE} has no 'metric' field")
    return float(payload["metric"])


def dry_run(
    runner: Runner, workspace: Path, arm: str, seed: int = 0
) -> tuple[ExecutionResult, float]:
    """Run a truncated experiment to measure how fast it really is.

    Returns the result and the measured seconds per step.
    """
    result = runner.execute(
        ExecutionRequest(
            workspace=workspace,
            args=["--arm", arm, "--seed", str(seed), "--steps", str(DRY_RUN_STEPS)],
            timeout_s=DRY_RUN_TIMEOUT_S,
        )
    )
    seconds_per_step = result.duration_s / DRY_RUN_STEPS if DRY_RUN_STEPS else 0.0
    return result, seconds_per_step


def estimate_total_minutes(seconds_per_step: float, steps: int, runs: int) -> float:
    """Extrapolate the full study from the dry run."""
    return (seconds_per_step * steps * runs) / 60.0


def _run_one(
    runner: Runner,
    workspace: Path,
    claim_id: str,
    arm: str,
    seed: int,
    timeout_s: int,
) -> tuple[RunResult | None, ExecutionFailure | None]:
    """Execute a single (arm, seed) and read its metric."""
    result = runner.execute(
        ExecutionRequest(
            workspace=workspace,
            args=["--arm", arm, "--seed", str(seed)],
            timeout_s=timeout_s,
        )
    )

    log_path = workspace / f"log_{arm}_{seed}.txt"
    log_path.write_text(
        f"$ run.py --arm {arm} --seed {seed}\n\n"
        f"[stdout]\n{result.stdout}\n"
        f"[stderr]\n{result.stderr}",
        encoding="utf-8",
    )

    if not result.ok:
        logger.warning(
            "run failed: %s %s seed=%d (%s)", claim_id, arm, seed, result.failure_summary()
        )
        return None, ExecutionFailure(claim_id, arm, seed, result)

    try:
        metric = _read_metric(workspace)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("run produced no usable metric: %s", exc)
        failed = ExecutionResult(
            exit_code=1,
            stdout=result.stdout,
            stderr=f"{result.stderr}\n\n{exc}",
            duration_s=result.duration_s,
        )
        return None, ExecutionFailure(claim_id, arm, seed, failed)

    return (
        RunResult(
            arm=arm,
            seed=seed,
            metric_value=metric,
            runtime_s=result.duration_s,
            log_path=str(log_path),
        ),
        None,
    )


def run_study(
    runner: Runner,
    claim: Claim,
    plan: ReductionPlan,
    workspace: Path,
    timeout_s: int,
) -> tuple[list[RunResult], ExecutionFailure | None]:
    """Run every arm and seed, stopping at the first failure.

    Stopping early matters: a bug in the script will fail identically for every
    seed, so continuing just wastes the budget before the debug node can fix it.
    """
    results: list[RunResult] = []

    for arm in arms_for(claim):
        for seed in range(plan.seeds):
            result, failure = _run_one(runner, workspace, claim.id, arm, seed, timeout_s)
            if failure is not None:
                return results, failure
            if result is not None:
                results.append(result)

    if degenerate := _degenerate_failure(claim, results):
        return results, degenerate

    return results, None


def _degenerate_failure(claim: Claim, results: list[RunResult]) -> ExecutionFailure | None:
    """Treat an experiment that measured nothing as a failure worth debugging.

    Every run returning the same value means the task could not separate the
    arms -- usually because it is too easy. The script did not crash, so nothing
    would otherwise notice, and the study would be reported as a clean null
    result. Sending it to the debug node gives it a chance to make the task
    harder instead.
    """
    values = [result.metric_value for result in results]
    if len(values) < 2 or len(set(values)) > 1:
        return None

    message = (
        f"Every run returned the same metric value ({values[0]:g}), so the experiment "
        "measured nothing: the task cannot separate the arms. It is most likely too "
        "easy, letting both arms score perfectly. Make the task harder while keeping "
        "the comparison fair."
    )
    logger.warning("%s: %s", claim.id, message)
    return ExecutionFailure(
        claim_id=claim.id,
        arm=results[0].arm,
        seed=results[0].seed,
        result=ExecutionResult(exit_code=1, stdout="", stderr=message, duration_s=0.0),
    )


def execute(
    state: GraphState,
    settings: Settings | None = None,
    runner: Runner | None = None,
) -> GraphState:
    """Graph node: calibrate, budget-check, then run each approved study."""
    settings = settings or get_settings()
    if runner is None:
        from claimscope.sandbox.runner import build_runner

        runner = build_runner(settings)

    claims_by_id = {claim.id: claim for claim in state["claims"]}
    plans = state.get("plans", {})
    workspaces = state.get("workspace_dirs", {})

    run_results = dict(state.get("run_results", {}))
    failures = dict(state.get("execution_failures", {}))
    over_budget = list(state.get("over_budget_claim_ids", []))
    errors = list(state.get("errors", []))
    used = state.get("budget_minutes_used", 0.0)

    # Dependencies are installed while building the image, which is the only
    # step allowed network access; the experiments themselves then run offline.
    try:
        packages = sandbox_packages(workspaces) | set(state.get("extra_packages", []))
        runner.prepare(ImageSpec(packages=sorted(packages)))
    except SandboxError as exc:
        message = f"could not prepare the sandbox: {exc}"
        logger.error(message)
        return {"errors": [*errors, message]}

    for claim_id in state.get("approved_plan_ids", []):
        if claim_id in run_results:
            continue  # already executed successfully
        claim, plan = claims_by_id.get(claim_id), plans.get(claim_id)
        workspace_dir = workspaces.get(claim_id)
        if claim is None or plan is None or workspace_dir is None:
            errors.append(f"cannot execute {claim_id}: missing claim, plan or workspace")
            continue

        workspace = Path(workspace_dir)
        arms = arms_for(claim)

        # 1. Calibrate on a short run before committing to the full study.
        probe, seconds_per_step = dry_run(runner, workspace, arms[0])
        if not probe.ok:
            failures[claim_id] = ExecutionFailure(claim_id, arms[0], 0, probe)
            logger.info("dry run failed for %s: %s", claim_id, probe.failure_summary())
            continue

        # 2. Refuse studies that cannot finish in the remaining budget.
        remaining = settings.budget_minutes_total - used
        projected = estimate_total_minutes(
            seconds_per_step, settings.full_run_steps, len(arms) * plan.seeds
        )
        if projected > remaining:
            logger.info(
                "%s needs ~%.1f min but only %.1f remain; sending back to planning",
                claim_id,
                projected,
                remaining,
            )
            over_budget.append(claim_id)
            errors.append(
                f"{claim_id}: estimated {projected:.1f} min exceeds the {remaining:.1f} min "
                f"left. Raise CLAIMSCOPE_BUDGET_MINUTES_TOTAL above {projected:.0f}, or reject "
                "the plan at review and ask for a smaller experiment."
            )
            continue

        # 3. Run the study.
        timeout_s = max(60, int(projected * 60 / max(1, len(arms) * plan.seeds)) * 3)
        results, failure = run_study(runner, claim, plan, workspace, timeout_s)
        used += sum(r.runtime_s for r in results) / 60.0

        if failure is not None:
            failures[claim_id] = failure
            continue

        run_results[claim_id] = results
        failures.pop(claim_id, None)
        logger.info("%s completed %d runs", claim_id, len(results))

    return {
        "run_results": run_results,
        "execution_failures": failures,
        "over_budget_claim_ids": over_budget,
        "budget_minutes_used": used,
        "errors": errors,
        # Recorded so the report can warn the reader, not just the operator.
        "unconfined_execution": not is_contained(settings),
    }
