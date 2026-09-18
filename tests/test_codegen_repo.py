"""Code generation against the paper's own repository."""

from __future__ import annotations

import subprocess
from pathlib import Path

from claimscope.config import Settings
from claimscope.nodes.codegen import REPO_DIR, RUN_SCRIPT, codegen, workspace_for
from claimscope.schemas import Claim, GeneratedCode, ReductionPlan
from claimscope.state import GraphState
from stubs import StubLLM

PAPER_ID = "2401.00001"

TRAIN_PY = '''\
"""Train a model."""
import argparse
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--epochs", default=200, type=int)
args = parser.parse_args()

trainset = torchvision.datasets.CIFAR10(root="./data", download=True)
'''


def _git_repo(path: Path, files: dict[str, str]) -> str:
    path.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"],
    ):
        subprocess.run(command, cwd=path, check=True, capture_output=True)
    return str(path)


def _claim() -> Claim:
    return Claim(
        id="c1",
        text="Method A beats baseline B.",
        source_location="Table 1",
        claim_type="comparative",
        arms=["method_a", "baseline_b"],
        metric="accuracy",
        expected_direction="method_a > baseline_b",
    )


def _plan(code_source: str = "official_repo") -> ReductionPlan:
    return ReductionPlan(
        claim_id="c1",
        original_setup="Full scale.",
        reduced_setup="Small scale.",
        changes=["Fewer epochs -- the effect appears early."],
        preserved=["The architecture."],
        why_claim_should_transfer="Scale free.",
        seeds=3,
        estimated_minutes=5.0,
        code_source=code_source,  # type: ignore[arg-type]
    )


def _state(repo_url: str | None, code_source: str = "official_repo") -> GraphState:
    return {
        "paper_id": PAPER_ID,
        "claims": [_claim()],
        "plans": {"c1": _plan(code_source)},
        "approved_plan_ids": ["c1"],
        "repo_url": repo_url,
    }


class TestRepoMode:
    def test_clones_the_repository_into_the_workspace(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        url = _git_repo(tmp_path / "src", {"train.py": TRAIN_PY})
        llm = StubLLM([GeneratedCode(code="print('adapter')", summary="Drives train.py.")])

        codegen(_state(url), settings, llm)

        repo = workspace_for(settings, PAPER_ID, "c1") / REPO_DIR
        assert (repo / "train.py").exists()

    def test_prompt_describes_the_entrypoint_and_its_flags(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        url = _git_repo(tmp_path / "src", {"train.py": TRAIN_PY})
        llm = StubLLM([GeneratedCode(code="print('adapter')")])

        codegen(_state(url), settings, llm)

        prompt = llm.prompts[0]
        assert "train.py" in prompt
        assert "--lr" in prompt
        assert "--epochs" in prompt

    def test_prompt_warns_that_the_entrypoint_downloads_data(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """The sandbox has no network, so this warning has to reach the model."""
        url = _git_repo(tmp_path / "src", {"train.py": TRAIN_PY})
        llm = StubLLM([GeneratedCode(code="print('adapter')")])

        codegen(_state(url), settings, llm)

        assert "downloads data at runtime" in llm.prompts[0]

    def test_prompt_lists_config_files(self, settings: Settings, tmp_path: Path) -> None:
        url = _git_repo(tmp_path / "src", {"train.py": TRAIN_PY, "config/small.yaml": "epochs: 2"})
        llm = StubLLM([GeneratedCode(code="print('adapter')")])

        codegen(_state(url), settings, llm)

        assert "config/small.yaml" in llm.prompts[0]

    def test_prompt_includes_the_commit(self, settings: Settings, tmp_path: Path) -> None:
        # The report should be able to say exactly what was run.
        url = _git_repo(tmp_path / "src", {"train.py": TRAIN_PY})
        llm = StubLLM([GeneratedCode(code="print('adapter')")])

        codegen(_state(url), settings, llm)

        assert "at commit" in llm.prompts[0]

    def test_writes_the_adapter_beside_the_clone(self, settings: Settings, tmp_path: Path) -> None:
        url = _git_repo(tmp_path / "src", {"train.py": TRAIN_PY})
        llm = StubLLM([GeneratedCode(code="print('adapter')")])

        codegen(_state(url), settings, llm)

        workspace = workspace_for(settings, PAPER_ID, "c1")
        assert (workspace / RUN_SCRIPT).read_text(encoding="utf-8") == "print('adapter')\n"
        assert (workspace / REPO_DIR).is_dir()


class TestReproducibility:
    def test_records_the_commit_that_was_used(self, settings: Settings, tmp_path: Path) -> None:
        url = _git_repo(tmp_path / "src", {"train.py": TRAIN_PY})
        llm = StubLLM([GeneratedCode(code="print('adapter')")])

        result = codegen(_state(url), settings, llm)

        commit = result["repo_commits"]["c1"]
        assert len(commit) == 40  # a full git sha

    def test_the_report_names_the_commit(self) -> None:
        """A result that cannot be traced to a commit is not reproducible."""
        from claimscope.nodes.report import render_report
        from claimscope.schemas import ClaimVerdict, RunResult

        markdown = render_report(
            {
                "paper_id": PAPER_ID,
                "claims": [_claim()],
                "selected_claim_ids": ["c1"],
                "plans": {"c1": _plan()},
                "run_results": {
                    "c1": [
                        RunResult(
                            arm="method_a",
                            seed=0,
                            metric_value=0.9,
                            runtime_s=1.0,
                            log_path="l.txt",
                        )
                    ]
                },
                "verdicts": [ClaimVerdict(claim_id="c1", verdict="inconclusive")],
                "repo_commits": {"c1": "abc123def456"},
            }
        )

        assert "abc123def456" in markdown

    def test_no_commit_recorded_for_from_scratch(self, settings: Settings, tmp_path: Path) -> None:
        url = _git_repo(tmp_path / "src", {"train.py": TRAIN_PY})
        llm = StubLLM([GeneratedCode(code="print('x')")])

        result = codegen(_state(url, code_source="from_scratch"), settings, llm)

        assert result["repo_commits"] == {}


class TestDependencies:
    """A repo whose packages are missing fails on the first import of every run."""

    def test_imported_packages_are_collected_for_the_image(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        url = _git_repo(tmp_path / "src", {"train.py": TRAIN_PY})
        llm = StubLLM([GeneratedCode(code="print('adapter')")])

        result = codegen(_state(url), settings, llm)

        assert "torchvision" in result["extra_packages"]

    def test_they_reach_the_sandbox_image(self, settings: Settings, tmp_path: Path) -> None:
        from claimscope.nodes.execute import execute
        from stubs import FakeRunner

        url = _git_repo(tmp_path / "src", {"train.py": TRAIN_PY})
        llm = StubLLM([GeneratedCode(code="print('adapter')")])
        state = _state(url)
        generated = codegen(state, settings, llm)

        runner = FakeRunner(seconds_per_call=0.01)
        execute({**state, **generated}, settings, runner)  # type: ignore[arg-type]

        assert "torchvision" in runner.prepared[0].packages  # type: ignore[attr-defined]

    def test_the_prompt_names_the_available_packages(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        url = _git_repo(tmp_path / "src", {"train.py": TRAIN_PY})
        llm = StubLLM([GeneratedCode(code="print('adapter')")])

        codegen(_state(url), settings, llm)

        # The model must not be told torch is unavailable when it will be there.
        assert "torchvision" in llm.prompts[0]


class TestFallback:
    """A repository problem must not cost us the claim."""

    def test_an_unclonable_repo_falls_back_to_from_scratch(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        llm = StubLLM([GeneratedCode(code="print('from scratch')")])

        result = codegen(_state(str(tmp_path / "missing")), settings, llm)

        assert result["workspace_dirs"]["c1"]
        # The from-scratch prompt, not the repo one.
        assert "self-contained Python script" in llm.prompts[0]

    def test_the_fallback_is_recorded_as_an_error(self, settings: Settings, tmp_path: Path) -> None:
        llm = StubLLM([GeneratedCode(code="print('from scratch')")])

        result = codegen(_state(str(tmp_path / "missing")), settings, llm)

        assert any("could not use the official repo" in e for e in result["errors"])

    def test_a_from_scratch_plan_ignores_the_repo(self, settings: Settings, tmp_path: Path) -> None:
        url = _git_repo(tmp_path / "src", {"train.py": TRAIN_PY})
        llm = StubLLM([GeneratedCode(code="print('from scratch')")])

        codegen(_state(url, code_source="from_scratch"), settings, llm)

        assert "self-contained Python script" in llm.prompts[0]
        assert not (workspace_for(settings, PAPER_ID, "c1") / REPO_DIR).exists()

    def test_no_repo_url_uses_from_scratch(self, settings: Settings) -> None:
        llm = StubLLM([GeneratedCode(code="print('from scratch')")])

        codegen(_state(None), settings, llm)

        assert "self-contained Python script" in llm.prompts[0]
