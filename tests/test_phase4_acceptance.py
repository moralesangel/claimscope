"""Phase 4 acceptance: a complete report for the toy claim.

The toy claim from PLAN.md section 10: "dropout reduces the gap between train
and test loss in a small MLP". Everything but the sandbox and the model is real
-- the statistics, the verdict rule and the report are the production code.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from claimscope.config import Settings
from claimscope.graph import build_graph
from claimscope.nodes.report import REPORT_FILENAME
from claimscope.schemas import (
    Claim,
    ClaimList,
    GeneratedCode,
    PaperSource,
    ReductionPlan,
    TriageDecision,
    TriageResult,
)
from claimscope.session import thread_config
from stubs import FakeRunner, StubLLM

PAPER_ID = "1207.0580"
CLAIM_ID = "dropout_reduces_generalization_gap"


@pytest.fixture
def toy_claim() -> Claim:
    return Claim(
        id=CLAIM_ID,
        text="Dropout reduces the gap between training and test loss in a small MLP.",
        source_location="Fig. 2",
        claim_type="ablation",
        arms=["with_dropout", "without_dropout"],
        metric="train/test loss gap",
        expected_direction="with_dropout < without_dropout",
    )


@pytest.fixture
def toy_plan() -> ReductionPlan:
    return ReductionPlan(
        claim_id=CLAIM_ID,
        original_setup="MLP on MNIST with and without dropout, trained to convergence.",
        reduced_setup="2-layer MLP, 5k MNIST images, 3 epochs; arms differ only in dropout.",
        changes=[
            "Subsample MNIST to 5k images -- overfitting is easier to observe, not harder.",
            "3 epochs instead of training to convergence -- the gap appears early.",
        ],
        preserved=["The dropout layer itself, which is what the claim is about."],
        why_claim_should_transfer="Regularisation reduces overfitting at any scale.",
        seeds=3,
        estimated_minutes=6.0,
        code_source="from_scratch",
    )


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch, toy_claim: Claim, toy_plan: ReductionPlan) -> StubLLM:
    monkeypatch.setattr(
        "claimscope.nodes.ingest.fetch_paper",
        lambda paper_id, runs_dir: PaperSource(
            paper_id=PAPER_ID,
            title="Improving neural networks by preventing co-adaptation",
            text="Paper body.",
            pdf_path="paper.pdf",
            repo_url=None,
        ),
    )
    return StubLLM(
        [
            ClaimList(claims=[toy_claim]),
            TriageResult(
                decisions=[
                    TriageDecision(
                        claim_id=CLAIM_ID,
                        testable=True,
                        reason="A small MLP on MNIST trains in minutes on CPU.",
                        priority=1,
                    )
                ]
            ),
            toy_plan,
            GeneratedCode(code="print('experiment')", summary="Trains a small MLP."),
        ]
    )


def _run_pipeline(settings: Settings, llm: StubLLM, runner: FakeRunner) -> dict:
    graph = build_graph(settings, llm, InMemorySaver(), runner)
    config = thread_config("acceptance")
    graph.invoke({"paper_id": PAPER_ID}, config)
    return dict(
        graph.invoke(
            Command(resume=[{"claim_id": CLAIM_ID, "action": "approve"}]),
            config,
        )
    )


def test_produces_a_complete_report(settings: Settings, pipeline: StubLLM) -> None:
    # Dropout narrows the gap, so the claim should hold.
    runner = FakeRunner(
        seconds_per_call=0.01,
        metric_by_arm={"with_dropout": 0.05, "without_dropout": 0.30},
    )

    final = _run_pipeline(settings, pipeline, runner)

    report_path = settings.runs_dir / PAPER_ID / REPORT_FILENAME
    assert final["report_path"] == str(report_path)
    markdown = report_path.read_text(encoding="utf-8")

    # It reaches a verdict, in the right direction.
    assert final["verdicts"][0].verdict == "consistent_at_reduced_scale"
    assert "Consistent at reduced scale" in markdown

    # And the report carries everything the plan asks for.
    assert "Improving neural networks" in markdown  # the paper
    assert CLAIM_ID in markdown  # the claim
    assert "2-layer MLP, 5k MNIST images" in markdown  # the reduction plan
    assert "overfitting is easier to observe" in markdown  # the justification
    assert "| with_dropout | 0 |" in markdown  # the runs
    assert "Welch p=" in markdown  # the statistics
    assert "does not refute the paper" in markdown  # the limitation


def test_a_reversed_result_is_reported_without_overclaiming(
    settings: Settings, pipeline: StubLLM
) -> None:
    # Dropout makes the gap worse: the claim does not hold at this scale.
    runner = FakeRunner(
        seconds_per_call=0.01,
        metric_by_arm={"with_dropout": 0.30, "without_dropout": 0.05},
    )

    final = _run_pipeline(settings, pipeline, runner)

    assert final["verdicts"][0].verdict == "not_consistent_at_reduced_scale"
    markdown = (settings.runs_dir / PAPER_ID / REPORT_FILENAME).read_text(encoding="utf-8")
    assert "Not consistent at reduced scale" in markdown
    # The crucial line: a negative result is not presented as a refutation.
    assert "does not refute the paper" in markdown


def test_the_gap_metric_is_read_as_lower_is_better(
    settings: Settings, pipeline: StubLLM, toy_claim: Claim
) -> None:
    """A smaller loss gap is an improvement; reading it the other way inverts the verdict."""
    from claimscope.nodes.analyze import metric_direction

    assert metric_direction(toy_claim) == "lower_is_better"


def test_every_seed_is_run_for_both_arms(settings: Settings, pipeline: StubLLM) -> None:
    runner = FakeRunner(seconds_per_call=0.01)

    final = _run_pipeline(settings, pipeline, runner)

    results = final["run_results"][CLAIM_ID]
    assert len(results) == 6
    by_arm: dict[str, set[int]] = {}
    for result in results:
        by_arm.setdefault(result.arm, set()).add(result.seed)
    # Invariant 1: matched arms.
    assert by_arm["with_dropout"] == by_arm["without_dropout"] == {0, 1, 2}
