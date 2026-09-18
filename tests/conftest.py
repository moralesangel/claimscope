"""Shared fixtures. Tests never hit the network or the real LLM by default."""

from __future__ import annotations

from pathlib import Path

import pytest

from claimscope.config import Settings
from claimscope.schemas import Claim, ClaimList
from stubs import StubLLM


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a temporary runs directory."""
    return Settings(runs_dir=tmp_path / "runs", anthropic_api_key="test-key")


@pytest.fixture
def sample_claims() -> list[Claim]:
    return [
        Claim(
            id="residual_beats_plain",
            text="Residual networks reach lower training error than plain networks of equal depth.",
            source_location="Fig. 4",
            claim_type="comparative",
            arms=["resnet34", "plain34"],
            metric="top-1 error",
            expected_direction="resnet34 < plain34",
        ),
        Claim(
            id="imagenet_top1",
            text="The 152-layer model reaches 78.57% top-1 accuracy on ImageNet.",
            source_location="Table 4",
            claim_type="absolute",
            arms=["resnet152"],
            metric="top-1 accuracy",
            expected_direction="equals 78.57",
        ),
    ]


@pytest.fixture
def stub_llm(sample_claims: list[Claim]) -> StubLLM:
    return StubLLM([ClaimList(claims=sample_claims)])
