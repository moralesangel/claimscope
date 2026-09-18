"""Pydantic schemas shared across the graph (PLAN.md section 5).

Every LLM output is validated against one of these models. Keep them strict:
a malformed claim that slips through here becomes a wasted experiment later.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ClaimType = Literal["absolute", "comparative", "ablation", "scaling_trend"]

Verdict = Literal[
    "consistent_at_reduced_scale",
    "not_consistent_at_reduced_scale",
    "inconclusive",
    "not_testable",
]

CodeSource = Literal["official_repo", "from_scratch"]


class Claim(BaseModel):
    """A single falsifiable statement extracted from a paper."""

    id: str
    text: str = Field(description="Paraphrased claim, never a long verbatim quote.")
    source_location: str = Field(description='Where it came from, e.g. "Table 2", "Sec. 4.1".')
    claim_type: ClaimType
    arms: list[str] = Field(
        default_factory=list,
        description='What is being compared, e.g. ["method_A", "baseline_B"].',
    )
    metric: str
    expected_direction: str = Field(description='e.g. "A > B", "decreases without X".')
    testable: bool | None = None
    triage_reason: str | None = None

    @field_validator("text", "source_location", "metric", "expected_direction")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class ClaimList(BaseModel):
    """Wrapper so the LLM returns an object rather than a bare array."""

    claims: list[Claim]


class ReductionPlan(BaseModel):
    """How a claim will be tested at reduced scale (PLAN.md section 2 invariants)."""

    claim_id: str
    original_setup: str
    reduced_setup: str
    changes: list[str] = Field(description="Each change to the paper's setup, with justification.")
    preserved: list[str] = Field(description="What is deliberately kept intact.")
    why_claim_should_transfer: str
    seeds: int = Field(default=3, ge=3, description="At least 3 seeds per arm (invariant 2).")
    estimated_minutes: float = Field(gt=0)
    code_source: CodeSource


class RunResult(BaseModel):
    """One (arm, seed) execution."""

    arm: str
    seed: int
    metric_value: float
    runtime_s: float
    log_path: str


class ClaimVerdict(BaseModel):
    """The statistical conclusion for one claim (PLAN.md section 8)."""

    claim_id: str
    verdict: Verdict
    effect_estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    notes: str = ""


class PaperSource(BaseModel):
    """Result of the ingest node."""

    paper_id: str
    title: str
    text: str
    pdf_path: str
    repo_url: str | None = None
