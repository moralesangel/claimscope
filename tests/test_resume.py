"""Phase 2 acceptance: a run survives the process that started it.

The graph, the checkpointer connection and the LLM stub are all discarded
between the two halves, so the second half recovers purely from the SQLite file.
"""

from __future__ import annotations

import pytest
from langgraph.types import Command

from claimscope.config import Settings
from claimscope.graph import build_graph
from claimscope.schemas import ClaimList, PaperSource, ReductionPlan, TriageResult
from claimscope.session import checkpointer, list_threads, new_thread_id, thread_config
from stubs import StubLLM

PAPER_ID = "2401.00001"


@pytest.fixture
def fake_paper(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    source = PaperSource(
        paper_id=PAPER_ID,
        title="A Test Paper",
        text="Body text.",
        pdf_path=str(settings.runs_dir / PAPER_ID / "paper.pdf"),
        repo_url=None,
    )
    monkeypatch.setattr("claimscope.nodes.ingest.fetch_paper", lambda paper_id, runs_dir: source)


def _fresh_llm(claims: ClaimList, triage: TriageResult, plan: ReductionPlan) -> StubLLM:
    return StubLLM([claims, triage, plan])


def test_interrupted_run_resumes_from_disk(
    settings: Settings,
    fake_paper: None,
    sample_claims: list,
    sample_triage: TriageResult,
    sample_plan: ReductionPlan,
) -> None:
    thread_id = new_thread_id(PAPER_ID)
    config = thread_config(thread_id)

    # --- first process: run until the review interrupt ---
    with checkpointer(settings) as saver:
        graph = build_graph(
            settings,
            _fresh_llm(ClaimList(claims=sample_claims), sample_triage, sample_plan),
            checkpointer=saver,
        )
        first = graph.invoke({"paper_id": PAPER_ID}, config)
        assert first["__interrupt__"], "expected the run to pause for review"

    # Everything from the first half is now out of scope, as if the CLI exited.

    # --- second process: reopen the database and finish the run ---
    with checkpointer(settings) as saver:
        graph = build_graph(settings, StubLLM([]), checkpointer=saver)

        snapshot = graph.get_state(config)
        assert snapshot.values["paper_id"] == PAPER_ID
        assert snapshot.values["selected_claim_ids"] == ["residual_beats_plain"]

        pending = snapshot.tasks[0].interrupts
        assert pending[0].value["requests"][0]["claim_id"] == "residual_beats_plain"

        final = graph.invoke(
            Command(resume=[{"claim_id": "residual_beats_plain", "action": "approve"}]),
            config,
        )

    assert final["approved_plan_ids"] == ["residual_beats_plain"]
    assert "__interrupt__" not in final


def test_the_checkpoint_file_is_created_under_runs(
    settings: Settings,
    fake_paper: None,
    sample_claims: list,
    sample_triage: TriageResult,
    sample_plan: ReductionPlan,
) -> None:
    with checkpointer(settings) as saver:
        graph = build_graph(
            settings,
            _fresh_llm(ClaimList(claims=sample_claims), sample_triage, sample_plan),
            checkpointer=saver,
        )
        graph.invoke({"paper_id": PAPER_ID}, thread_config(new_thread_id(PAPER_ID)))

    assert (settings.runs_dir / "checkpoints.sqlite").exists()


def test_threads_are_listed_for_resume(
    settings: Settings,
    fake_paper: None,
    sample_claims: list,
    sample_triage: TriageResult,
    sample_plan: ReductionPlan,
) -> None:
    thread_id = new_thread_id(PAPER_ID)

    with checkpointer(settings) as saver:
        graph = build_graph(
            settings,
            _fresh_llm(ClaimList(claims=sample_claims), sample_triage, sample_plan),
            checkpointer=saver,
        )
        graph.invoke({"paper_id": PAPER_ID}, thread_config(thread_id))

    with checkpointer(settings) as saver:
        assert thread_id in list_threads(saver)


def test_thread_ids_are_prefixed_with_the_paper() -> None:
    assert new_thread_id("1706.03762").startswith("1706.03762-")
    assert new_thread_id("x") != new_thread_id("x")
