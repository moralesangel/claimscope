"""The review node's decision handling, exercised through a real interrupt."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from claimscope.nodes.review import ReviewResponse, needs_redesign, review
from claimscope.schemas import Claim, ReductionPlan
from claimscope.session import thread_config
from claimscope.state import GraphState


def _claim(claim_id: str) -> Claim:
    return Claim(
        id=claim_id,
        text=f"Claim {claim_id}.",
        source_location="Table 1",
        claim_type="comparative",
        arms=["a", "b"],
        metric="accuracy",
        expected_direction="a > b",
    )


def _plan(claim_id: str) -> ReductionPlan:
    return ReductionPlan(
        claim_id=claim_id,
        original_setup="Full scale.",
        reduced_setup="Small scale.",
        changes=["Smaller data -- safe."],
        preserved=["Architecture."],
        why_claim_should_transfer="Scale free mechanism.",
        seeds=3,
        estimated_minutes=5.0,
        code_source="from_scratch",
    )


def _run(state: GraphState, resume: object) -> dict[str, object]:
    """Drive the review node through a real interrupt and resume."""
    builder: StateGraph[GraphState] = StateGraph(GraphState)
    builder.add_node("review", review)
    builder.add_edge(START, "review")
    graph = builder.compile(checkpointer=InMemorySaver())

    config = thread_config("t-1")
    graph.invoke(state, config)
    return dict(graph.invoke(Command(resume=resume), config))


def _state(claim_ids: list[str], **overrides: object) -> GraphState:
    state: GraphState = {
        "paper_id": "p",
        "claims": [_claim(c) for c in claim_ids],
        "selected_claim_ids": claim_ids,
        "plans": {c: _plan(c) for c in claim_ids},
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_approval_records_the_plan() -> None:
    result = _run(_state(["c1"]), [{"claim_id": "c1", "action": "approve"}])

    assert result["approved_plan_ids"] == ["c1"]
    assert "c1" in result["plans"]


def test_rejection_stores_feedback_and_drops_the_plan() -> None:
    result = _run(
        _state(["c1"]),
        [{"claim_id": "c1", "action": "reject", "feedback": "Too few seeds."}],
    )

    assert result["approved_plan_ids"] == []
    assert result["plan_feedback"] == {"c1": "Too few seeds."}
    # The plan is cleared so design_plan produces a new one.
    assert "c1" not in result["plans"]


def test_handles_a_mix_of_decisions() -> None:
    result = _run(
        _state(["c1", "c2"]),
        [
            {"claim_id": "c1", "action": "approve"},
            {"claim_id": "c2", "action": "reject", "feedback": "Wrong dataset."},
        ],
    )

    assert result["approved_plan_ids"] == ["c1"]
    assert result["plan_feedback"] == {"c2": "Wrong dataset."}


def test_approving_clears_earlier_feedback() -> None:
    state = _state(["c1"], plan_feedback={"c1": "Old complaint."})

    result = _run(state, [{"claim_id": "c1", "action": "approve"}])

    assert result["plan_feedback"] == {}


def test_accepts_a_single_response_rather_than_a_list() -> None:
    result = _run(_state(["c1"]), {"claim_id": "c1", "action": "approve"})

    assert result["approved_plan_ids"] == ["c1"]


def test_rejects_a_response_for_an_unrelated_claim() -> None:
    with pytest.raises(ValueError, match="not pending"):
        _run(_state(["c1"]), [{"claim_id": "other", "action": "approve"}])


def test_interrupt_payload_describes_each_plan() -> None:
    builder: StateGraph[GraphState] = StateGraph(GraphState)
    builder.add_node("review", review)
    builder.add_edge(START, "review")
    graph = builder.compile(checkpointer=InMemorySaver())

    result = graph.invoke(_state(["c1"]), thread_config("t-1"))

    payload = result["__interrupt__"][0].value
    request = payload["requests"][0]
    assert request["claim_id"] == "c1"
    assert request["claim_text"] == "Claim c1."
    assert request["plan"]["seeds"] == 3


def test_no_pending_plans_is_a_no_op() -> None:
    # Already approved, so the node should not interrupt at all.
    state = _state(["c1"], approved_plan_ids=["c1"])

    assert review(state) == {}


class TestNeedsRedesign:
    def test_true_while_a_selected_claim_is_unapproved(self) -> None:
        assert needs_redesign(_state(["c1"]))

    def test_false_once_everything_is_approved(self) -> None:
        assert not needs_redesign(_state(["c1"], approved_plan_ids=["c1"]))

    def test_false_when_nothing_was_selected(self) -> None:
        assert not needs_redesign({"selected_claim_ids": []})


class TestReviewResponse:
    def test_defaults_feedback_to_empty(self) -> None:
        assert ReviewResponse(claim_id="c1", action="approve").feedback == ""

    def test_rejects_an_unknown_action(self) -> None:
        with pytest.raises(ValueError):
            ReviewResponse(claim_id="c1", action="maybe")  # type: ignore[arg-type]
