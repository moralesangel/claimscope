"""Shared fixtures. Tests never hit the network or the real LLM by default."""

from __future__ import annotations

from pathlib import Path

import pytest

from claimscope.config import Settings
from claimscope.schemas import (
    Claim,
    ClaimList,
    ReductionPlan,
    TriageDecision,
    TriageResult,
)
from stubs import StubLLM

# Every environment variable Settings reads. Cleared for every test so results
# do not depend on the developer's .env or shell.
_SETTINGS_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "CLAIMSCOPE_PROVIDER",
    "CLAIMSCOPE_MODEL_NAME",
    "CLAIMSCOPE_RUNS_DIR",
    "CLAIMSCOPE_MAX_CLAIMS",
    "CLAIMSCOPE_MAX_CLAIMS_EXTRACTED",
    "CLAIMSCOPE_BUDGET_MINUTES_TOTAL",
    "CLAIMSCOPE_SEEDS_PER_ARM",
    "CLAIMSCOPE_MAX_DEBUG_ATTEMPTS",
    "CLAIMSCOPE_DOCKER_IMAGE",
    "CLAIMSCOPE_SANDBOX_CPUS",
    "CLAIMSCOPE_SANDBOX_MEMORY_MB",
    "CLAIMSCOPE_SANDBOX_TIMEOUT_S",
    "CLAIMSCOPE_TRACING_ENABLED",
)


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the real .env and shell environment out of every test.

    Settings loads .env by default, so without this a developer's local keys and
    model choice leak into assertions and the suite passes only on their machine.
    """
    for name in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Point the .env lookup at an empty directory.
    monkeypatch.chdir(tmp_path)


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


@pytest.fixture
def sample_triage() -> TriageResult:
    """Triage that accepts the comparative claim and rejects the absolute one."""
    return TriageResult(
        decisions=[
            TriageDecision(
                claim_id="residual_beats_plain",
                testable=True,
                reason="A depth-matched comparison on a small image dataset shrinks cleanly.",
                priority=1,
            ),
            TriageDecision(
                claim_id="imagenet_top1",
                testable=False,
                reason="Absolute ImageNet numbers cannot be reproduced at reduced scale.",
                priority=0,
            ),
        ]
    )


@pytest.fixture
def sample_plan() -> ReductionPlan:
    return ReductionPlan(
        claim_id="residual_beats_plain",
        original_setup="ResNet-34 vs plain-34 on ImageNet.",
        reduced_setup="ResNet-8 vs plain-8 on a 5k-image CIFAR-10 subset, 2 epochs.",
        changes=["ImageNet -> CIFAR-10 subset -- the degradation effect is not dataset specific."],
        preserved=["The residual connections themselves, which the claim is about."],
        why_claim_should_transfer="Plain nets degrade with depth even at small scale.",
        seeds=3,
        estimated_minutes=8.0,
        code_source="from_scratch",
    )


@pytest.fixture
def planning_llm(
    sample_claims: list[Claim], sample_triage: TriageResult, sample_plan: ReductionPlan
) -> StubLLM:
    """Canned responses for a full extract -> triage -> plan run."""
    return StubLLM([ClaimList(claims=sample_claims), sample_triage, sample_plan])
