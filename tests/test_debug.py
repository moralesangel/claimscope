"""The debug loop: patching, the attempt cap, and giving up cleanly."""

from __future__ import annotations

from pathlib import Path

from claimscope.config import Settings
from claimscope.nodes.codegen import RUN_SCRIPT
from claimscope.nodes.debug import debug, needs_debugging
from claimscope.sandbox.runner import ExecutionFailure, ExecutionResult
from claimscope.schemas import Claim, CodePatch
from claimscope.state import GraphState
from stubs import StubLLM

PAPER_ID = "2401.00001"
BROKEN = "raise ValueError('boom')\n"
FIXED = "print('fixed')"


def _claim(claim_id: str = "c1") -> Claim:
    return Claim(
        id=claim_id,
        text="Method A beats baseline B.",
        source_location="Table 1",
        claim_type="comparative",
        arms=["treatment", "control"],
        metric="accuracy",
        expected_direction="treatment > control",
    )


def _failure(claim_id: str = "c1", stderr: str = "ValueError: boom") -> ExecutionFailure:
    return ExecutionFailure(
        claim_id=claim_id,
        arm="treatment",
        seed=0,
        result=ExecutionResult(exit_code=1, stdout="", stderr=stderr, duration_s=1.0),
    )


def _workspace(settings: Settings, claim_id: str = "c1") -> Path:
    ws = settings.runs_dir / PAPER_ID / "workspaces" / claim_id
    ws.mkdir(parents=True, exist_ok=True)
    (ws / RUN_SCRIPT).write_text(BROKEN, encoding="utf-8")
    return ws


def _state(settings: Settings, **overrides: object) -> GraphState:
    state: GraphState = {
        "paper_id": PAPER_ID,
        "claims": [_claim()],
        "workspace_dirs": {"c1": str(_workspace(settings))},
        "execution_failures": {"c1": _failure()},
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_overwrites_the_script_with_the_patch(settings: Settings) -> None:
    llm = StubLLM([CodePatch(diagnosis="Wrong shape.", code=FIXED)])
    workspace = _workspace(settings)

    debug(_state(settings), settings, llm)

    assert (workspace / RUN_SCRIPT).read_text(encoding="utf-8") == FIXED + "\n"


def test_counts_the_attempt(settings: Settings) -> None:
    llm = StubLLM([CodePatch(diagnosis="d", code=FIXED)])

    result = debug(_state(settings), settings, llm)

    assert result["debug_attempts"] == {"c1": 1}


def test_clears_the_failure_so_execute_retries(settings: Settings) -> None:
    llm = StubLLM([CodePatch(diagnosis="d", code=FIXED)])

    result = debug(_state(settings), settings, llm)

    assert result["execution_failures"] == {}


def test_prompt_includes_the_traceback_and_code(settings: Settings) -> None:
    llm = StubLLM([CodePatch(diagnosis="d", code=FIXED)])

    state = _state(settings, execution_failures={"c1": _failure(stderr="KeyError: arm")})
    debug(state, settings, llm)

    prompt = llm.prompts[0]
    assert "KeyError: arm" in prompt
    assert BROKEN.strip() in prompt


def test_prompt_forbids_network_access(settings: Settings) -> None:
    llm = StubLLM([CodePatch(diagnosis="d", code=FIXED)])

    debug(_state(settings), settings, llm)

    assert "No network access" in llm.prompts[0]


class TestAttemptCap:
    def test_abandons_the_claim_at_the_cap(self, settings: Settings) -> None:
        capped = settings.model_copy(update={"max_debug_attempts": 2})
        llm = StubLLM([])  # must not be called
        state = _state(settings, debug_attempts={"c1": 2})

        result = debug(state, capped, llm)

        assert "c1" in result["abandoned_claim_ids"]
        assert "Gave up after 2 debug attempts" in result["abandoned_claim_ids"]["c1"]
        assert llm.prompts == []

    def test_abandoning_clears_the_failure(self, settings: Settings) -> None:
        capped = settings.model_copy(update={"max_debug_attempts": 1})
        state = _state(settings, debug_attempts={"c1": 1})

        result = debug(state, capped, StubLLM([]))

        assert result["execution_failures"] == {}

    def test_records_why_it_gave_up(self, settings: Settings) -> None:
        capped = settings.model_copy(update={"max_debug_attempts": 1})
        state = _state(settings, debug_attempts={"c1": 1})

        result = debug(state, capped, StubLLM([]))

        assert any("Gave up" in e for e in result["errors"])

    def test_patches_while_under_the_cap(self, settings: Settings) -> None:
        capped = settings.model_copy(update={"max_debug_attempts": 3})
        llm = StubLLM([CodePatch(diagnosis="d", code=FIXED)])
        state = _state(settings, debug_attempts={"c1": 2})

        result = debug(state, capped, llm)

        assert result["debug_attempts"]["c1"] == 3
        assert result["abandoned_claim_ids"] == {}


class TestNeedsDebugging:
    def test_true_with_a_failure(self, settings: Settings) -> None:
        assert needs_debugging({"execution_failures": {"c1": _failure()}})

    def test_false_without_failures(self) -> None:
        assert not needs_debugging({"execution_failures": {}})

    def test_false_when_absent(self) -> None:
        assert not needs_debugging({})


class TestCodePatchValidation:
    def test_strips_fences(self) -> None:
        fence = "`" * 3
        patch = CodePatch(diagnosis="d", code=f"{fence}python\nprint(1)\n{fence}")

        assert patch.code == "print(1)"
