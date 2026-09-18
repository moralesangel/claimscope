"""End-to-end tests of the graph, with arXiv and the LLM stubbed out."""

from __future__ import annotations

import json

import pymupdf
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from claimscope.config import Settings
from claimscope.graph import build_graph
from claimscope.schemas import GeneratedCode, PaperSource, ReductionPlan
from claimscope.session import thread_config
from stubs import FakeRunner, StubLLM

PAPER_ID = "2401.00001"


@pytest.fixture
def fake_paper(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> PaperSource:
    """Replace the network fetch with a locally generated PDF."""
    paper_dir = settings.runs_dir / PAPER_ID
    paper_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = paper_dir / "paper.pdf"

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Our code is available at https://github.com/lab/repo")
    doc.save(pdf_path)
    doc.close()

    source = PaperSource(
        paper_id=PAPER_ID,
        title="A Test Paper",
        text="Our code is available at https://github.com/lab/repo",
        pdf_path=str(pdf_path),
        repo_url="https://github.com/lab/repo",
    )
    monkeypatch.setattr("claimscope.nodes.ingest.fetch_paper", lambda paper_id, runs_dir: source)
    return source


def test_graph_runs_through_triage_and_planning(
    settings: Settings, planning_llm: StubLLM, fake_paper: PaperSource
) -> None:
    graph = build_graph(settings, planning_llm, checkpointer=InMemorySaver())

    result = graph.invoke({"paper_id": PAPER_ID}, thread_config("t-1"))

    assert result["paper_title"] == "A Test Paper"
    assert result["repo_url"] == "https://github.com/lab/repo"
    assert [c.id for c in result["claims"]] == ["residual_beats_plain", "imagenet_top1"]
    # The absolute claim is rejected, so only the comparative one is selected.
    assert result["selected_claim_ids"] == ["residual_beats_plain"]
    assert "residual_beats_plain" in result["plans"]


def test_graph_pauses_for_review(
    settings: Settings, planning_llm: StubLLM, fake_paper: PaperSource
) -> None:
    graph = build_graph(settings, planning_llm, checkpointer=InMemorySaver())

    result = graph.invoke({"paper_id": PAPER_ID}, thread_config("t-1"))

    payload = result["__interrupt__"][0].value
    assert payload["type"] == "plan_review"
    assert [r["claim_id"] for r in payload["requests"]] == ["residual_beats_plain"]


def test_approving_moves_on_to_execution(
    settings: Settings, planning_llm: StubLLM, fake_paper: PaperSource
) -> None:
    # Approval now leads into codegen and execute, so both need stand-ins.
    planning_llm.responses.append(GeneratedCode(code="print(1)"))
    graph = build_graph(settings, planning_llm, InMemorySaver(), FakeRunner(seconds_per_call=0.01))
    config = thread_config("t-1")
    graph.invoke({"paper_id": PAPER_ID}, config)

    final = graph.invoke(
        Command(resume=[{"claim_id": "residual_beats_plain", "action": "approve"}]),
        config,
    )

    assert final["approved_plan_ids"] == ["residual_beats_plain"]
    assert "__interrupt__" not in final
    assert final["run_results"]["residual_beats_plain"]


def test_rejecting_sends_the_plan_back_for_redesign(
    settings: Settings,
    planning_llm: StubLLM,
    sample_plan: ReductionPlan,
    fake_paper: PaperSource,
) -> None:
    # One more plan response, for the redesign that follows the rejection.
    planning_llm.responses.append(sample_plan)
    graph = build_graph(settings, planning_llm, checkpointer=InMemorySaver())
    config = thread_config("t-1")
    graph.invoke({"paper_id": PAPER_ID}, config)

    after_rejection = graph.invoke(
        Command(
            resume=[
                {
                    "claim_id": "residual_beats_plain",
                    "action": "reject",
                    "feedback": "Use more seeds.",
                }
            ]
        ),
        config,
    )

    # It pauses again on a freshly designed plan, and the feedback reached the planner.
    assert after_rejection["__interrupt__"]
    assert "Use more seeds." in planning_llm.prompts[-1]
    assert after_rejection["approved_plan_ids"] == []


def test_graph_writes_artifacts(
    settings: Settings, planning_llm: StubLLM, fake_paper: PaperSource
) -> None:
    graph = build_graph(settings, planning_llm, checkpointer=InMemorySaver())
    graph.invoke({"paper_id": PAPER_ID}, thread_config("t-1"))

    out_dir = settings.runs_dir / PAPER_ID
    assert len(json.loads((out_dir / "claims.json").read_text(encoding="utf-8"))) == 2
    assert len(json.loads((out_dir / "triage.json").read_text(encoding="utf-8"))) == 2
    assert "residual_beats_plain" in json.loads(
        (out_dir / "plans.json").read_text(encoding="utf-8")
    )


def test_extraction_sees_the_ingested_text(
    settings: Settings, planning_llm: StubLLM, fake_paper: PaperSource
) -> None:
    graph = build_graph(settings, planning_llm, checkpointer=InMemorySaver())
    graph.invoke({"paper_id": PAPER_ID}, thread_config("t-1"))

    assert "https://github.com/lab/repo" in planning_llm.prompts[0]
