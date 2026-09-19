"""Plan review in the terminal, including when there is nobody to ask."""

from __future__ import annotations

from typing import Any

import pytest
from rich.console import Console

from claimscope.review_ui import NoReviewerError, collect_decisions, render_plan

console = Console(width=100)

PLAN: dict[str, Any] = {
    "claim_id": "c1",
    "claim_text": "Dropout reduces the generalization gap.",
    "plan": {
        "original_setup": "MLP on full MNIST.",
        "reduced_setup": "Small MLP on a 10k subset.",
        "changes": ["Fewer epochs -- the effect appears early."],
        "preserved": ["The dropout layer itself."],
        "why_claim_should_transfer": "Regularisation is scale free.",
        "seeds": 3,
        "estimated_minutes": 5.0,
        "code_source": "from_scratch",
    },
}

PAYLOAD: dict[str, Any] = {"type": "plan_review", "requests": [PLAN]}


class TestAssumeYes:
    def test_approves_without_asking(self) -> None:
        decisions = collect_decisions(console, PAYLOAD, assume_yes=True)

        assert decisions == [{"claim_id": "c1", "action": "approve", "feedback": ""}]

    def test_still_shows_the_plan(self) -> None:
        """Approving unattended is not a reason to hide what was approved."""
        recorder = Console(width=100, record=True)

        collect_decisions(recorder, PAYLOAD, assume_yes=True)

        output = recorder.export_text()
        assert "Small MLP on a 10k subset." in output
        assert "Approving automatically" in output

    def test_approves_every_pending_plan(self) -> None:
        payload = {"requests": [PLAN, {**PLAN, "claim_id": "c2"}]}

        decisions = collect_decisions(console, payload, assume_yes=True)

        assert [d["claim_id"] for d in decisions] == ["c1", "c2"]


class TestNoReviewer:
    """A redirected stdin used to loop forever on an exhausted stream."""

    def test_exhausted_input_raises_rather_than_looping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_input(*_args: object, **_kwargs: object) -> bool:
            raise EOFError

        monkeypatch.setattr("claimscope.review_ui.typer.confirm", no_input)

        with pytest.raises(NoReviewerError, match="--yes"):
            collect_decisions(console, PAYLOAD)

    def test_an_aborted_prompt_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import typer

        def aborted(*_args: object, **_kwargs: object) -> bool:
            raise typer.Abort

        monkeypatch.setattr("claimscope.review_ui.typer.confirm", aborted)

        with pytest.raises(NoReviewerError):
            collect_decisions(console, PAYLOAD)

    def test_exhausted_input_while_giving_feedback_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("claimscope.review_ui.typer.confirm", lambda *a, **k: False)

        def no_input(*_args: object, **_kwargs: object) -> str:
            raise EOFError

        monkeypatch.setattr("claimscope.review_ui.typer.prompt", no_input)

        with pytest.raises(NoReviewerError, match="feedback"):
            collect_decisions(console, PAYLOAD)


class TestInteractive:
    def test_approving_records_the_decision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("claimscope.review_ui.typer.confirm", lambda *a, **k: True)

        decisions = collect_decisions(console, PAYLOAD)

        assert decisions[0]["action"] == "approve"

    def test_rejecting_collects_feedback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("claimscope.review_ui.typer.confirm", lambda *a, **k: False)
        monkeypatch.setattr("claimscope.review_ui.typer.prompt", lambda *a, **k: "Use more seeds.")

        decisions = collect_decisions(console, PAYLOAD)

        assert decisions[0] == {
            "claim_id": "c1",
            "action": "reject",
            "feedback": "Use more seeds.",
        }

    def test_empty_feedback_is_asked_for_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A rejection with no reason gives design_plan nothing to work with.
        answers = iter(["", "   ", "Actually explain the problem."])
        monkeypatch.setattr("claimscope.review_ui.typer.confirm", lambda *a, **k: False)
        monkeypatch.setattr("claimscope.review_ui.typer.prompt", lambda *a, **k: next(answers))

        decisions = collect_decisions(console, PAYLOAD)

        assert decisions[0]["feedback"] == "Actually explain the problem."


class TestRendering:
    def test_shows_what_the_reviewer_needs_to_judge(self) -> None:
        recorder = Console(width=100, record=True)

        render_plan(recorder, PLAN)

        output = recorder.export_text()
        assert "MLP on full MNIST." in output
        assert "Fewer epochs" in output
        assert "The dropout layer itself." in output
        assert "Regularisation is scale free." in output
        assert "3" in output  # seeds

    def test_handles_a_plan_with_no_changes_listed(self) -> None:
        recorder = Console(width=100, record=True)
        sparse = {**PLAN, "plan": {**PLAN["plan"], "changes": [], "preserved": []}}

        render_plan(recorder, sparse)

        assert "Small MLP on a 10k subset." in recorder.export_text()
