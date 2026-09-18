"""Plan design, including how reviewer feedback re-enters the prompt."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from claimscope.config import Settings
from claimscope.nodes.design_plan import design_plan
from claimscope.schemas import Claim, ReductionPlan
from claimscope.state import GraphState
from stubs import StubLLM


def _claim(claim_id: str = "c1") -> Claim:
    return Claim(
        id=claim_id,
        text="Method A beats baseline B.",
        source_location="Table 1",
        claim_type="comparative",
        arms=["a", "b"],
        metric="accuracy",
        expected_direction="a > b",
    )


def _plan(claim_id: str = "c1") -> ReductionPlan:
    return ReductionPlan(
        claim_id=claim_id,
        original_setup="Full scale.",
        reduced_setup="Small scale.",
        changes=["Smaller dataset -- the effect is not dataset specific."],
        preserved=["The architecture."],
        why_claim_should_transfer="The mechanism is scale free.",
        seeds=3,
        estimated_minutes=5.0,
        code_source="from_scratch",
    )


def _state(**overrides: object) -> GraphState:
    state: GraphState = {
        "paper_id": "2401.00001",
        "claims": [_claim()],
        "selected_claim_ids": ["c1"],
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_produces_a_plan_per_selected_claim(settings: Settings) -> None:
    llm = StubLLM([_plan()])

    result = design_plan(_state(), settings, llm)

    assert "c1" in result["plans"]
    assert result["plans"]["c1"].seeds == 3


def test_skips_claims_that_are_already_approved(settings: Settings) -> None:
    # No canned response: planning an approved claim again would raise.
    llm = StubLLM([])

    result = design_plan(_state(approved_plan_ids=["c1"]), settings, llm)

    assert result["plans"] == {}
    assert llm.prompts == []


def test_feedback_is_included_in_the_prompt(settings: Settings) -> None:
    llm = StubLLM([_plan()])

    design_plan(_state(plan_feedback={"c1": "Use five seeds instead."}), settings, llm)

    assert "Use five seeds instead." in llm.prompts[0]
    assert "Reviewer feedback" in llm.prompts[0]


def test_no_feedback_section_on_a_first_attempt(settings: Settings) -> None:
    llm = StubLLM([_plan()])

    design_plan(_state(), settings, llm)

    assert "Reviewer feedback" not in llm.prompts[0]


def test_prefers_the_official_repo_when_there_is_one(settings: Settings) -> None:
    llm = StubLLM([_plan()])

    design_plan(_state(repo_url="https://github.com/lab/repo"), settings, llm)

    assert "https://github.com/lab/repo" in llm.prompts[0]
    assert "official_repo" in llm.prompts[0]


def test_says_from_scratch_without_a_repo(settings: Settings) -> None:
    llm = StubLLM([_plan()])

    design_plan(_state(), settings, llm)

    assert "from_scratch" in llm.prompts[0]


def test_corrects_a_mismatched_claim_id(settings: Settings) -> None:
    llm = StubLLM([_plan(claim_id="something_else")])

    result = design_plan(_state(), settings, llm)

    assert result["plans"]["c1"].claim_id == "c1"


def test_writes_plans_json(settings: Settings) -> None:
    design_plan(_state(), settings, StubLLM([_plan()]))

    written = json.loads(
        (settings.runs_dir / "2401.00001" / "plans.json").read_text(encoding="utf-8")
    )
    assert written["c1"]["seeds"] == 3


class TestInvariants:
    def test_rejects_fewer_than_three_seeds(self) -> None:
        # Invariant 2 from PLAN.md section 2.
        with pytest.raises(ValidationError):
            ReductionPlan.model_validate({**_plan().model_dump(), "seeds": 2})

    def test_rejects_a_non_positive_runtime_estimate(self) -> None:
        with pytest.raises(ValidationError):
            ReductionPlan.model_validate({**_plan().model_dump(), "estimated_minutes": 0})

    def test_rejects_an_unknown_code_source(self) -> None:
        with pytest.raises(ValidationError):
            ReductionPlan.model_validate({**_plan().model_dump(), "code_source": "borrowed"})
