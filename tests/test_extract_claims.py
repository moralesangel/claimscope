"""Tests for the claim extraction node, with the LLM stubbed out."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from claimscope.config import Settings
from claimscope.nodes.extract_claims import MAX_PAPER_CHARS, extract_claims
from claimscope.schemas import Claim, ClaimList
from claimscope.state import GraphState
from stubs import StubLLM


def _state(text: str = "Paper body.", paper_id: str = "1512.03385") -> GraphState:
    return {"paper_id": paper_id, "paper_text": text, "paper_title": "Deep Residual Learning"}


def test_returns_validated_claims(settings: Settings, stub_llm: StubLLM) -> None:
    result = extract_claims(_state(), settings, stub_llm)

    claims = result["claims"]
    assert len(claims) == 2
    assert claims[0].id == "residual_beats_plain"
    assert claims[0].claim_type == "comparative"


def test_writes_claims_json(settings: Settings, stub_llm: StubLLM) -> None:
    extract_claims(_state(), settings, stub_llm)

    written = settings.runs_dir / "1512.03385" / "claims.json"
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert [c["id"] for c in payload] == ["residual_beats_plain", "imagenet_top1"]
    assert payload[0]["source_location"] == "Fig. 4"


def test_prompt_includes_title_and_text(settings: Settings, stub_llm: StubLLM) -> None:
    extract_claims(_state(text="Distinctive body text."), settings, stub_llm)

    prompt = stub_llm.prompts[0]
    assert "Deep Residual Learning" in prompt
    assert "Distinctive body text." in prompt
    assert str(settings.max_claims_extracted) in prompt


def test_truncates_long_papers(settings: Settings, stub_llm: StubLLM) -> None:
    marker = "UNIQUEMARKER"
    long_text = marker * ((MAX_PAPER_CHARS + 5_000) // len(marker))
    extract_claims(_state(text=long_text), settings, stub_llm)

    # The prompt template itself contains prose, so count only the body marker.
    prompt = stub_llm.prompts[0]
    kept = prompt.count(marker) * len(marker)
    assert kept <= MAX_PAPER_CHARS
    assert kept > MAX_PAPER_CHARS - len(marker)


def test_deduplicates_repeated_ids(settings: Settings) -> None:
    duplicate = Claim(
        id="same_id",
        text="A claim.",
        source_location="Sec. 1",
        claim_type="comparative",
        arms=["a", "b"],
        metric="accuracy",
        expected_direction="a > b",
    )
    other = duplicate.model_copy(update={"text": "Another claim."})
    llm = StubLLM([ClaimList(claims=[duplicate, other])])

    result = extract_claims(_state(), settings, llm)

    assert [c.id for c in result["claims"]] == ["same_id", "same_id_2"]


class TestClaimValidation:
    def test_rejects_blank_source_location(self) -> None:
        with pytest.raises(ValidationError):
            Claim(
                id="x",
                text="A claim.",
                source_location="   ",
                claim_type="comparative",
                arms=["a", "b"],
                metric="accuracy",
                expected_direction="a > b",
            )

    def test_rejects_unknown_claim_type(self) -> None:
        with pytest.raises(ValidationError):
            Claim(
                id="x",
                text="A claim.",
                source_location="Sec. 1",
                claim_type="speculative",  # type: ignore[arg-type]
                arms=["a"],
                metric="accuracy",
                expected_direction="up",
            )

    def test_strips_surrounding_whitespace(self) -> None:
        claim = Claim(
            id="x",
            text="  A claim.  ",
            source_location=" Table 2 ",
            claim_type="ablation",
            arms=["a", "b"],
            metric=" F1 ",
            expected_direction=" a > b ",
        )
        assert claim.text == "A claim."
        assert claim.source_location == "Table 2"
