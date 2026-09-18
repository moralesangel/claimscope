"""The execute/debug loop driven through the real graph."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from claimscope.config import Settings
from claimscope.graph import build_graph
from claimscope.schemas import (
    ClaimList,
    CodePatch,
    GeneratedCode,
    PaperSource,
    ReductionPlan,
    TriageResult,
)
from claimscope.session import thread_config
from stubs import FakeRunner, StubLLM

PAPER_ID = "2401.00001"
APPROVE = [{"claim_id": "residual_beats_plain", "action": "approve"}]


@pytest.fixture
def fake_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    source = PaperSource(
        paper_id=PAPER_ID,
        title="A Test Paper",
        text="Body text.",
        pdf_path="paper.pdf",
        repo_url=None,
    )
    monkeypatch.setattr("claimscope.nodes.ingest.fetch_paper", lambda paper_id, runs_dir: source)


def _llm(
    claims: list,
    triage: TriageResult,
    plan: ReductionPlan,
    *extra: object,
) -> StubLLM:
    return StubLLM([ClaimList(claims=claims), triage, plan, *extra])  # type: ignore[list-item]


def test_full_pipeline_reaches_results(
    settings: Settings,
    fake_paper: None,
    sample_claims: list,
    sample_triage: TriageResult,
    sample_plan: ReductionPlan,
) -> None:
    llm = _llm(sample_claims, sample_triage, sample_plan, GeneratedCode(code="print(1)"))
    runner = FakeRunner(seconds_per_call=0.01)
    graph = build_graph(settings, llm, InMemorySaver(), runner)
    config = thread_config("t-1")

    graph.invoke({"paper_id": PAPER_ID}, config)
    final = graph.invoke(Command(resume=APPROVE), config)

    results = final["run_results"]["residual_beats_plain"]
    assert len(results) == 6  # 2 arms x 3 seeds
    assert final["execution_failures"] == {}


def test_a_failing_script_is_patched_and_rerun(
    settings: Settings,
    fake_paper: None,
    sample_claims: list,
    sample_triage: TriageResult,
    sample_plan: ReductionPlan,
) -> None:
    llm = _llm(
        sample_claims,
        sample_triage,
        sample_plan,
        GeneratedCode(code="raise ValueError('boom')"),
        CodePatch(diagnosis="Fixed the shape.", code="print('fixed')"),
    )
    # Fail the dry run once, then succeed.
    runner = FakeRunner(seconds_per_call=0.01, fail_times=1)
    graph = build_graph(settings, llm, InMemorySaver(), runner)
    config = thread_config("t-1")

    graph.invoke({"paper_id": PAPER_ID}, config)
    final = graph.invoke(Command(resume=APPROVE), config)

    assert final["debug_attempts"]["residual_beats_plain"] == 1
    assert len(final["run_results"]["residual_beats_plain"]) == 6
    assert final["abandoned_claim_ids"] == {}


def test_a_persistently_broken_script_is_abandoned(
    settings: Settings,
    fake_paper: None,
    sample_claims: list,
    sample_triage: TriageResult,
    sample_plan: ReductionPlan,
) -> None:
    capped = settings.model_copy(update={"max_debug_attempts": 2})
    llm = _llm(
        sample_claims,
        sample_triage,
        sample_plan,
        GeneratedCode(code="raise ValueError('boom')"),
        CodePatch(diagnosis="try 1", code="raise ValueError('still')"),
        CodePatch(diagnosis="try 2", code="raise ValueError('again')"),
    )
    runner = FakeRunner(seconds_per_call=0.01, fail_times=999)
    graph = build_graph(capped, llm, InMemorySaver(), runner)
    config = thread_config("t-1")

    graph.invoke({"paper_id": PAPER_ID}, config)
    final = graph.invoke(Command(resume=APPROVE), config)

    # It gives up rather than looping forever.
    assert "residual_beats_plain" in final["abandoned_claim_ids"]
    assert final["debug_attempts"]["residual_beats_plain"] == 2
    assert "residual_beats_plain" not in final["run_results"]


def test_generated_code_lands_in_the_workspace(
    settings: Settings,
    fake_paper: None,
    sample_claims: list,
    sample_triage: TriageResult,
    sample_plan: ReductionPlan,
) -> None:
    from pathlib import Path

    llm = _llm(sample_claims, sample_triage, sample_plan, GeneratedCode(code="print(1)"))
    graph = build_graph(settings, llm, InMemorySaver(), FakeRunner(seconds_per_call=0.01))
    config = thread_config("t-1")

    graph.invoke({"paper_id": PAPER_ID}, config)
    final = graph.invoke(Command(resume=APPROVE), config)

    workspace = Path(final["workspace_dirs"]["residual_beats_plain"])
    assert (workspace / "run.py").read_text(encoding="utf-8") == "print(1)\n"


def test_the_sandbox_only_ever_sees_the_workspace(
    settings: Settings,
    fake_paper: None,
    sample_claims: list,
    sample_triage: TriageResult,
    sample_plan: ReductionPlan,
) -> None:
    """Generated code must never run outside its own workspace."""
    llm = _llm(sample_claims, sample_triage, sample_plan, GeneratedCode(code="print(1)"))

    class RecordingRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__(seconds_per_call=0.01)
            self.workspaces: list[str] = []

        def execute(self, request: object) -> object:  # type: ignore[override]
            self.workspaces.append(str(request.workspace))  # type: ignore[attr-defined]
            return super().execute(request)  # type: ignore[arg-type]

    runner = RecordingRunner()
    graph = build_graph(settings, llm, InMemorySaver(), runner)
    config = thread_config("t-1")

    graph.invoke({"paper_id": PAPER_ID}, config)
    final = graph.invoke(Command(resume=APPROVE), config)

    expected = final["workspace_dirs"]["residual_beats_plain"]
    assert set(runner.workspaces) == {expected}
