"""Code generation: the run.py contract and what the prompt tells the model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from claimscope.config import Settings
from claimscope.nodes.codegen import (
    DEFAULT_ARMS,
    RUN_SCRIPT,
    arms_for,
    codegen,
    workspace_for,
)
from claimscope.schemas import Claim, GeneratedCode, ReductionPlan
from claimscope.state import GraphState
from stubs import StubLLM

SCRIPT = "import json\nprint('ok')\n"


def _claim(claim_id: str = "c1", arms: list[str] | None = None) -> Claim:
    return Claim(
        id=claim_id,
        text="Method A beats baseline B.",
        source_location="Table 1",
        claim_type="comparative",
        arms=arms if arms is not None else ["method_a", "baseline_b"],
        metric="accuracy",
        expected_direction="method_a > baseline_b",
    )


def _plan(claim_id: str = "c1", seeds: int = 3) -> ReductionPlan:
    return ReductionPlan(
        claim_id=claim_id,
        original_setup="Full scale.",
        reduced_setup="Small scale.",
        changes=["Smaller data -- safe."],
        preserved=["Architecture."],
        why_claim_should_transfer="Scale free.",
        seeds=seeds,
        estimated_minutes=5.0,
        code_source="from_scratch",
    )


def _state(**overrides: object) -> GraphState:
    state: GraphState = {
        "paper_id": "2401.00001",
        "claims": [_claim()],
        "plans": {"c1": _plan()},
        "approved_plan_ids": ["c1"],
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_writes_run_py_into_the_workspace(settings: Settings) -> None:
    llm = StubLLM([GeneratedCode(code=SCRIPT, summary="trains a tiny model")])

    result = codegen(_state(), settings, llm)

    workspace = workspace_for(settings, "2401.00001", "c1")
    assert result["workspace_dirs"]["c1"] == str(workspace)
    assert (workspace / RUN_SCRIPT).read_text(encoding="utf-8") == SCRIPT


def test_skips_claims_that_already_have_a_workspace(settings: Settings) -> None:
    llm = StubLLM([])  # calling the model would raise

    result = codegen(_state(workspace_dirs={"c1": "/somewhere"}), settings, llm)

    assert result["workspace_dirs"] == {"c1": "/somewhere"}
    assert llm.prompts == []


def test_prompt_states_the_arm_names(settings: Settings) -> None:
    llm = StubLLM([GeneratedCode(code=SCRIPT)])

    codegen(_state(), settings, llm)

    assert "method_a, baseline_b" in llm.prompts[0]


def test_prompt_states_the_required_interface(settings: Settings) -> None:
    llm = StubLLM([GeneratedCode(code=SCRIPT)])

    codegen(_state(), settings, llm)

    prompt = llm.prompts[0]
    assert "--arm" in prompt
    assert "--seed" in prompt
    assert "result.json" in prompt


def test_prompt_forbids_network_access(settings: Settings) -> None:
    llm = StubLLM([GeneratedCode(code=SCRIPT)])

    codegen(_state(), settings, llm)

    assert "No network access" in llm.prompts[0]


def test_prompt_derives_a_per_run_time_budget(settings: Settings) -> None:
    llm = StubLLM([GeneratedCode(code=SCRIPT)])

    # 2 arms x 3 seeds = 6 runs inside the total budget.
    codegen(_state(), settings.model_copy(update={"budget_minutes_total": 60.0}), llm)

    # 3600s / 6 runs = 600s per run.
    assert "600 seconds" in llm.prompts[0]


def test_skips_a_claim_without_a_plan(settings: Settings) -> None:
    llm = StubLLM([])

    result = codegen(_state(plans={}), settings, llm)

    assert result["workspace_dirs"] == {}


class TestArmsFor:
    def test_uses_the_claims_arms(self) -> None:
        assert arms_for(_claim(arms=["a", "b"])) == ["a", "b"]

    def test_falls_back_when_the_claim_has_too_few(self) -> None:
        assert arms_for(_claim(arms=["only_one"])) == list(DEFAULT_ARMS)

    def test_falls_back_when_the_claim_has_none(self) -> None:
        assert arms_for(_claim(arms=[])) == list(DEFAULT_ARMS)


class TestGeneratedCodeValidation:
    def test_strips_a_fenced_block(self) -> None:
        fence = "`" * 3
        code = GeneratedCode(code=f"{fence}python\nprint(1)\n{fence}").code

        assert code == "print(1)"

    def test_keeps_backticks_inside_the_code(self) -> None:
        fence = "`" * 3
        source = f'doc = "{fence}"\nprint(1)'

        assert GeneratedCode(code=source).code == source

    def test_rejects_empty_code(self) -> None:
        with pytest.raises(ValidationError):
            GeneratedCode(code="   ")


class TestIntegrityWarnings:
    """A substituted dataset must reach the reader, not just the log.

    The report names the plan's dataset. When the script used another one and
    nothing said so, the report described an experiment that never ran.
    """

    def _mnist_plan(self) -> ReductionPlan:
        return ReductionPlan(
            claim_id="c1",
            original_setup="Train on full MNIST for 200 epochs.",
            reduced_setup="Subsample MNIST to 5000 images, 5 epochs.",
            changes=["Fewer images -- fits the CPU budget."],
            preserved=["The architecture comparison."],
            why_claim_should_transfer="Dropout is scale free.",
            seeds=3,
            estimated_minutes=5.0,
            code_source="from_scratch",
        )

    def test_a_substituted_dataset_is_recorded_as_an_error(self, settings: Settings) -> None:
        substituted = "from sklearn.datasets import load_digits\nX, y = load_digits()\n"
        llm = StubLLM([GeneratedCode(code=substituted, summary="trains on digits")])

        result = codegen(_state(plans={"c1": self._mnist_plan()}), settings, llm)

        errors = result.get("errors", [])
        assert any("load_digits" in e and "mnist" in e for e in errors), errors

    def test_the_script_is_still_written(self, settings: Settings) -> None:
        """The check is advisory: a flagged claim still runs and reports."""
        substituted = "from sklearn.datasets import load_digits\nX, y = load_digits()\n"
        llm = StubLLM([GeneratedCode(code=substituted, summary="trains on digits")])

        result = codegen(_state(plans={"c1": self._mnist_plan()}), settings, llm)

        assert "c1" in result.get("workspace_dirs", {})

    def test_a_faithful_script_records_nothing(self, settings: Settings) -> None:
        faithful = "X, y = load_mnist_from_disk()\nX_train = X[:5000]\n"
        llm = StubLLM([GeneratedCode(code=faithful, summary="trains on mnist")])

        result = codegen(_state(plans={"c1": self._mnist_plan()}), settings, llm)

        assert result.get("errors", []) == []
