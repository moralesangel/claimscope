"""Building the sandbox image before anything runs in it.

Dependencies are installed while the image is built, which is the only step with
network access; the experiments themselves then run offline.
"""

from __future__ import annotations

from pathlib import Path

from claimscope.config import Settings
from claimscope.nodes.codegen import REPO_DIR, SANDBOX_PACKAGES
from claimscope.nodes.execute import execute, sandbox_packages
from claimscope.sandbox.runner import ExecutionResult, ImageSpec, SandboxError
from claimscope.schemas import Claim, ReductionPlan
from claimscope.state import GraphState
from stubs import FakeRunner

PAPER_ID = "2401.00001"


def _claim() -> Claim:
    return Claim(
        id="c1",
        text="Method A beats baseline B.",
        source_location="Table 1",
        claim_type="comparative",
        arms=["treatment", "control"],
        metric="accuracy",
        expected_direction="treatment > control",
    )


def _plan() -> ReductionPlan:
    return ReductionPlan(
        claim_id="c1",
        original_setup="Full.",
        reduced_setup="Small.",
        changes=["Less data -- safe."],
        preserved=["Architecture."],
        why_claim_should_transfer="Scale free.",
        seeds=3,
        estimated_minutes=5.0,
        code_source="from_scratch",
    )


def _workspace(settings: Settings, requirements: str | None = None) -> Path:
    ws = settings.runs_dir / PAPER_ID / "workspaces" / "c1"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "run.py").write_text("print('x')\n", encoding="utf-8")
    if requirements is not None:
        repo = ws / REPO_DIR
        repo.mkdir(exist_ok=True)
        (repo / "requirements.txt").write_text(requirements, encoding="utf-8")
    return ws


def _state(workspace: Path) -> GraphState:
    return {
        "paper_id": PAPER_ID,
        "claims": [_claim()],
        "plans": {"c1": _plan()},
        "approved_plan_ids": ["c1"],
        "workspace_dirs": {"c1": str(workspace)},
    }


class TestSandboxPackages:
    def test_always_includes_the_defaults(self) -> None:
        assert set(SANDBOX_PACKAGES) <= sandbox_packages({})

    def test_adds_packages_a_repo_declares(self, settings: Settings) -> None:
        workspace = _workspace(settings, "torch==2.0\nmatplotlib\n")

        packages = sandbox_packages({"c1": str(workspace)})

        assert "torch==2.0" in packages
        assert "matplotlib" in packages

    def test_ignores_comments_and_pip_options(self, settings: Settings) -> None:
        workspace = _workspace(settings, "# a comment\n-r other.txt\n--index-url http://x\nnumpy\n")

        packages = sandbox_packages({"c1": str(workspace)})

        assert not any(p.startswith("-") or p.startswith("#") for p in packages)
        assert "numpy" in packages

    def test_a_workspace_without_a_repo_contributes_nothing(self, settings: Settings) -> None:
        workspace = _workspace(settings)

        assert sandbox_packages({"c1": str(workspace)}) == set(SANDBOX_PACKAGES)


class TestPrepareIsCalled:
    def test_the_image_is_built_before_running(self, settings: Settings) -> None:
        runner = FakeRunner(seconds_per_call=0.01)

        execute(_state(_workspace(settings)), settings, runner)

        assert len(runner.prepared) == 1
        assert isinstance(runner.prepared[0], ImageSpec)

    def test_repo_requirements_reach_the_image(self, settings: Settings) -> None:
        runner = FakeRunner(seconds_per_call=0.01)
        workspace = _workspace(settings, "scipy\n")

        execute(_state(workspace), settings, runner)

        spec = runner.prepared[0]
        assert isinstance(spec, ImageSpec)
        assert "scipy" in spec.packages

    def test_a_build_failure_stops_before_any_run(self, settings: Settings) -> None:
        class BrokenRunner(FakeRunner):
            def prepare(self, spec: object) -> None:
                raise SandboxError("docker daemon is not running")

        runner = BrokenRunner()

        result = execute(_state(_workspace(settings)), settings, runner)

        # Nothing was executed, and the reason is recorded rather than swallowed.
        assert runner.calls == []
        assert any("could not prepare the sandbox" in e for e in result["errors"])

    def test_a_build_failure_leaves_no_partial_results(self, settings: Settings) -> None:
        class BrokenRunner(FakeRunner):
            def prepare(self, spec: object) -> None:
                raise SandboxError("no daemon")

        result = execute(_state(_workspace(settings)), settings, BrokenRunner())

        assert "run_results" not in result or not result.get("run_results")


class TestOfflineExecution:
    def test_experiments_run_without_network(self, settings: Settings) -> None:
        """Only the image build may use the network."""
        seen: list[bool] = []

        class RecordingRunner(FakeRunner):
            def execute(self, request: object) -> ExecutionResult:  # type: ignore[override]
                seen.append(request.network)  # type: ignore[attr-defined]
                return super().execute(request)  # type: ignore[arg-type]

        execute(_state(_workspace(settings)), settings, RecordingRunner(seconds_per_call=0.01))

        assert seen and not any(seen)
