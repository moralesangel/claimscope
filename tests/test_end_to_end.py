"""The whole pipeline, with a real experiment really executing.

Everything except the model is real: the graph, the sandbox, the statistics and
the report. The experiment below is the shape codegen is asked to produce, so
this is the closest thing to a full run that works without Docker or an API key.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from claimscope.config import Settings
from claimscope.graph import build_graph
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
from stubs import StubLLM

PAPER_ID = "1207.0580"
CLAIM_ID = "dropout_reduces_generalization_gap"

# A real, self-contained experiment: a one-hidden-layer network on synthetic
# data, overfitting a small sample, with and without dropout. Deliberately tiny
# so the test stays fast, and offline so it runs in the sandbox.
EXPERIMENT = """
import argparse
import json

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=300)
args = p.parse_args()

rng = np.random.default_rng(args.seed)

n_train, n_test, dim, hidden = 60, 400, 40, 64
w_true = rng.normal(size=dim)


def make(n):
    x = rng.normal(size=(n, dim))
    y = (x @ w_true + rng.normal(scale=0.5, size=n) > 0).astype(float)
    return x, y


x_train, y_train = make(n_train)
x_test, y_test = make(n_test)

w1 = rng.normal(scale=0.3, size=(dim, hidden))
b1 = np.zeros(hidden)
w2 = rng.normal(scale=0.3, size=hidden)
b2 = 0.0

use_dropout = args.arm == "with_dropout"
keep = 0.5


def forward(x, training):
    h = np.maximum(0.0, x @ w1 + b1)
    if training and use_dropout:
        h = h * ((rng.random(h.shape) < keep) / keep)
    return h, 1.0 / (1.0 + np.exp(-(h @ w2 + b2)))


def loss_of(x, y):
    _, pred = forward(x, training=False)
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred)))


lr = 0.05
for _ in range(args.steps):
    h, pred = forward(x_train, training=True)
    error = (pred - y_train) / len(y_train)
    grad_h = np.outer(error, w2) * (h > 0)
    w2 -= lr * (h.T @ error)
    b2 -= lr * error.sum()
    w1 -= lr * (x_train.T @ grad_h)
    b1 -= lr * grad_h.sum(axis=0)

gap = loss_of(x_test, y_test) - loss_of(x_train, y_train)

json.dump(
    {"arm": args.arm, "seed": args.seed, "metric": gap, "metric_name": "train/test loss gap"},
    open("result.json", "w"),
)
print(f"{args.arm} seed={args.seed} gap={gap:.4f}")
"""


@pytest.fixture
def subprocess_settings(tmp_path: Path) -> Settings:
    return Settings(
        runs_dir=tmp_path / "runs",
        gemini_api_key="stub",
        provider="google",
        sandbox_backend="subprocess",
        budget_minutes_total=15.0,
        full_run_steps=300,
    )


@pytest.fixture
def stubbed_model(monkeypatch: pytest.MonkeyPatch) -> StubLLM:
    claim = Claim(
        id=CLAIM_ID,
        text="Dropout reduces the gap between training and test loss in a small network.",
        source_location="Fig. 1",
        claim_type="ablation",
        arms=["with_dropout", "without_dropout"],
        metric="train/test loss gap",
        expected_direction="with_dropout < without_dropout",
    )
    plan = ReductionPlan(
        claim_id=CLAIM_ID,
        original_setup="MLP on MNIST with and without 50% dropout.",
        reduced_setup="One hidden layer on 60 synthetic points; arms differ only in dropout.",
        changes=["MNIST -> synthetic data: the sandbox has no network."],
        preserved=["The dropout layer, and identical data and seeds across arms."],
        why_claim_should_transfer="Regularisation reduces overfitting at any scale.",
        seeds=3,
        estimated_minutes=1.0,
        code_source="from_scratch",
    )
    monkeypatch.setattr(
        "claimscope.nodes.ingest.fetch_paper",
        lambda paper_id, runs_dir: PaperSource(
            paper_id=PAPER_ID,
            title="Improving neural networks by preventing co-adaptation",
            text="Body.",
            pdf_path="paper.pdf",
            repo_url=None,
        ),
    )
    return StubLLM(
        [
            ClaimList(claims=[claim]),
            TriageResult(
                decisions=[
                    TriageDecision(
                        claim_id=CLAIM_ID,
                        testable=True,
                        reason="A tiny network trains in seconds on CPU.",
                        priority=1,
                    )
                ]
            ),
            plan,
            GeneratedCode(code=EXPERIMENT, summary="Small MLP with and without dropout."),
        ]
    )


def _run(settings: Settings, llm: StubLLM) -> dict:
    graph = build_graph(settings, llm, InMemorySaver())
    config = thread_config("e2e")
    graph.invoke({"paper_id": PAPER_ID}, config)
    return dict(graph.invoke(Command(resume=[{"claim_id": CLAIM_ID, "action": "approve"}]), config))


def test_a_real_experiment_runs_and_reaches_a_verdict(
    subprocess_settings: Settings, stubbed_model: StubLLM
) -> None:
    """The end-to-end property: generated code executes and produces a verdict."""
    state = _run(subprocess_settings, stubbed_model)

    results = state["run_results"][CLAIM_ID]
    assert len(results) == 6  # 2 arms x 3 seeds
    assert all(result.metric_value != 0 for result in results)

    verdict = state["verdicts"][0]
    assert verdict.verdict in {
        "consistent_at_reduced_scale",
        "not_consistent_at_reduced_scale",
        "inconclusive",
    }
    assert verdict.effect_estimate is not None


def test_dropout_narrows_the_gap_as_the_paper_says(
    subprocess_settings: Settings, stubbed_model: StubLLM
) -> None:
    """A sanity check on the whole chain: the known answer comes out."""
    state = _run(subprocess_settings, stubbed_model)

    results = state["run_results"][CLAIM_ID]
    with_dropout = [r.metric_value for r in results if r.arm == "with_dropout"]
    without = [r.metric_value for r in results if r.arm == "without_dropout"]

    assert sum(with_dropout) / len(with_dropout) < sum(without) / len(without)


def test_both_arms_run_the_same_seeds(
    subprocess_settings: Settings, stubbed_model: StubLLM
) -> None:
    state = _run(subprocess_settings, stubbed_model)

    by_arm: dict[str, set[int]] = {}
    for result in state["run_results"][CLAIM_ID]:
        by_arm.setdefault(result.arm, set()).add(result.seed)

    assert by_arm["with_dropout"] == by_arm["without_dropout"] == {0, 1, 2}


def test_unconfined_execution_is_recorded(
    subprocess_settings: Settings, stubbed_model: StubLLM
) -> None:
    state = _run(subprocess_settings, stubbed_model)

    assert state["unconfined_execution"] is True


def test_the_report_discloses_the_weaker_sandbox(
    subprocess_settings: Settings, stubbed_model: StubLLM
) -> None:
    """A reader must not mistake this for a contained run."""
    state = _run(subprocess_settings, stubbed_model)

    report = Path(state["report_path"]).read_text(encoding="utf-8")
    assert "without container isolation" in report
    assert "re-run under Docker" in report


def test_the_report_contains_the_statistics(
    subprocess_settings: Settings, stubbed_model: StubLLM
) -> None:
    state = _run(subprocess_settings, stubbed_model)

    report = Path(state["report_path"]).read_text(encoding="utf-8")
    assert "Welch p=" in report
    assert "| with_dropout | 0 |" in report
    assert "does not refute the paper" in report
