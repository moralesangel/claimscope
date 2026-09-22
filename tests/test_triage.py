"""Triage marks claims testable and selects at most K of them."""

from __future__ import annotations

import json

import pytest

from claimscope.config import Settings
from claimscope.nodes.triage import triage
from claimscope.schemas import Claim, TriageDecision, TriageResult
from claimscope.state import GraphState
from stubs import StubLLM


def _claim(claim_id: str, claim_type: str = "comparative") -> Claim:
    return Claim(
        id=claim_id,
        text=f"Claim {claim_id}.",
        source_location="Table 1",
        claim_type=claim_type,  # type: ignore[arg-type]
        arms=["a", "b"],
        metric="accuracy",
        expected_direction="a > b",
    )


def _state(claims: list[Claim]) -> GraphState:
    return {"paper_id": "2401.00001", "claims": claims}


def _decision(claim_id: str, testable: bool = True, priority: int = 1) -> TriageDecision:
    return TriageDecision(
        claim_id=claim_id,
        testable=testable,
        reason="Because of reasons.",
        priority=priority if testable else 0,
    )


def test_marks_claims_with_the_decision(settings: Settings) -> None:
    claims = [_claim("a"), _claim("b")]
    llm = StubLLM([TriageResult(decisions=[_decision("a"), _decision("b", testable=False)])])

    result = triage(_state(claims), settings, llm)

    assert result["claims"][0].testable is True
    assert result["claims"][1].testable is False
    assert result["claims"][1].triage_reason == "Because of reasons."


def test_absolute_claims_are_never_testable(settings: Settings) -> None:
    # Even if the model says otherwise: the rule is structural.
    claims = [_claim("headline", claim_type="absolute")]
    llm = StubLLM([TriageResult(decisions=[_decision("headline", testable=True)])])

    result = triage(_state(claims), settings, llm)

    assert result["claims"][0].testable is False
    assert "cannot be reproduced at reduced scale" in (result["claims"][0].triage_reason or "")
    assert result["selected_claim_ids"] == []


def test_selects_at_most_max_claims(settings: Settings) -> None:
    claims = [_claim(c) for c in "abcd"]
    llm = StubLLM(
        [
            TriageResult(
                decisions=[
                    _decision("a", priority=3),
                    _decision("b", priority=1),
                    _decision("c", priority=2),
                    _decision("d", priority=4),
                ]
            )
        ]
    )

    result = triage(_state(claims), settings.model_copy(update={"max_claims": 2}), llm)

    # Best priority first, capped at K.
    assert result["selected_claim_ids"] == ["b", "c"]


def test_untestable_claims_are_never_selected(settings: Settings) -> None:
    claims = [_claim("a"), _claim("b")]
    llm = StubLLM(
        [TriageResult(decisions=[_decision("a", testable=False), _decision("b", priority=1)])]
    )

    result = triage(_state(claims), settings, llm)

    assert result["selected_claim_ids"] == ["b"]


def test_a_missing_decision_defaults_to_untestable(settings: Settings) -> None:
    claims = [_claim("a"), _claim("forgotten")]
    llm = StubLLM([TriageResult(decisions=[_decision("a")])])

    result = triage(_state(claims), settings, llm)

    assert result["claims"][1].testable is False


def test_a_missing_decision_is_not_reported_as_a_judgement(settings: Settings) -> None:
    """It sits in the report beside real reasons like "ImageNet is too expensive".

    Worded as a verdict, a reader takes it for one. The claim was not assessed,
    which is a gap in the triage output and should read as one.
    """
    claims = [_claim("a"), _claim("forgotten")]
    llm = StubLLM([TriageResult(decisions=[_decision("a")])])

    result = triage(_state(claims), settings, llm)

    reason = result["claims"][1].triage_reason or ""
    assert "not assessed" in reason.lower()
    assert "not a judgement" in reason.lower()


def test_writes_triage_json(settings: Settings) -> None:
    claims = [_claim("a")]
    llm = StubLLM([TriageResult(decisions=[_decision("a")])])

    triage(_state(claims), settings, llm)

    written = json.loads(
        (settings.runs_dir / "2401.00001" / "triage.json").read_text(encoding="utf-8")
    )
    assert written[0]["testable"] is True


def test_handles_no_claims(settings: Settings) -> None:
    result = triage(_state([]), settings, StubLLM([]))

    assert result["selected_claim_ids"] == []


def test_prompt_states_the_budget(settings: Settings) -> None:
    llm = StubLLM([TriageResult(decisions=[_decision("a")])])

    triage(_state([_claim("a")]), settings, llm)

    prompt = llm.prompts[0]
    assert str(settings.budget_minutes_total) in prompt
    assert str(settings.seeds_per_arm) in prompt


@pytest.mark.parametrize("blank", ["", "   "])
def test_reason_must_not_be_blank(blank: str) -> None:
    with pytest.raises(ValueError, match="blank"):
        TriageDecision(claim_id="a", testable=True, reason=blank)
