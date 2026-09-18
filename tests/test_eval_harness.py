"""The harness end to end, against a stubbed agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.run_eval import _report_json, evaluate_paper, load_annotations, render_report
from eval.schemas import (
    AnnotatedClaim,
    EvalReport,
    ExtractionMetrics,
    PaperAnnotation,
    PaperResult,
)

from claimscope.config import Settings
from claimscope.schemas import ClaimList, PaperSource, ReductionPlan, TriageResult
from stubs import StubLLM

PAPER_ID = "2401.00001"


@pytest.fixture
def annotation() -> PaperAnnotation:
    return PaperAnnotation(
        paper_id=PAPER_ID,
        title="A Test Paper",
        claims=[
            AnnotatedClaim(
                text="Residual networks reach lower training error than plain networks",
                claim_type="comparative",
                testable=True,
                expected_verdict="consistent_at_reduced_scale",
            ),
            AnnotatedClaim(
                text="The model reaches 78.57 percent top-1 accuracy on ImageNet",
                claim_type="absolute",
                testable=False,
            ),
        ],
    )


@pytest.fixture
def stubbed_agent(
    monkeypatch: pytest.MonkeyPatch,
    sample_claims: list,
    sample_triage: TriageResult,
    sample_plan: ReductionPlan,
) -> None:
    """Replace ingest and the LLM so no network or API key is needed.

    Three calls: extraction, triage and planning. The run then pauses at the
    review interrupt, which is where a sandbox-less evaluation stops.
    """
    monkeypatch.setattr(
        "claimscope.nodes.ingest.fetch_paper",
        lambda paper_id, runs_dir: PaperSource(
            paper_id=PAPER_ID,
            title="A Test Paper",
            text="Body.",
            pdf_path="paper.pdf",
            repo_url=None,
        ),
    )
    monkeypatch.setattr(
        "claimscope.llm.ProviderStructuredLLM",
        lambda settings: StubLLM([ClaimList(claims=sample_claims), sample_triage, sample_plan]),
    )


def test_evaluates_a_paper_without_a_sandbox(
    settings: Settings, annotation: PaperAnnotation, stubbed_agent: None
) -> None:
    result = evaluate_paper(annotation, settings, sandbox_available=False)

    assert result.paper_id == PAPER_ID
    # The sample claims line up with the annotation, so extraction scores well.
    assert result.extraction.matched == 2
    assert result.extraction.f1 == 1.0


def test_execution_is_not_measured_without_a_sandbox(
    settings: Settings, annotation: PaperAnnotation, stubbed_agent: None
) -> None:
    """The point of the phase 6 caveat: unmeasured is not the same as zero."""
    result = evaluate_paper(annotation, settings, sandbox_available=False)

    assert not result.execution.measured
    assert not result.verdicts.measured


def test_it_stops_before_running_unreviewed_code(
    settings: Settings, annotation: PaperAnnotation, stubbed_agent: None
) -> None:
    result = evaluate_paper(annotation, settings, sandbox_available=False)

    assert any("no sandbox" in error for error in result.errors)


def test_counts_llm_calls(
    settings: Settings, annotation: PaperAnnotation, stubbed_agent: None
) -> None:
    result = evaluate_paper(annotation, settings, sandbox_available=False)

    # extract_claims, triage and design_plan.
    assert result.cost.llm_calls == 3


def test_a_failing_paper_does_not_end_the_evaluation(
    settings: Settings, annotation: PaperAnnotation, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(paper_id: str, runs_dir: Path) -> None:
        raise RuntimeError("arXiv is down")

    monkeypatch.setattr("claimscope.nodes.ingest.fetch_paper", explode)
    monkeypatch.setattr("claimscope.llm.ProviderStructuredLLM", lambda settings: StubLLM([]))

    result = evaluate_paper(annotation, settings, sandbox_available=False)

    assert any("arXiv is down" in error for error in result.errors)
    assert result.extraction.recall == 0.0


class TestReporting:
    def test_renders_a_table(self) -> None:
        report = EvalReport(results=[PaperResult(paper_id="a")])

        table = render_report(report)

        assert table.row_count >= 1

    def test_unmeasured_metrics_render_as_a_dash(self) -> None:
        """A zero would claim the agent failed; a dash says we did not look."""
        from rich.console import Console

        report = EvalReport(results=[PaperResult(paper_id="a")])
        console = Console(width=200, record=True)
        console.print(render_report(report))

        assert "—" in console.export_text()

    def test_json_includes_derived_rates(self) -> None:
        result = PaperResult(paper_id="a")
        result.extraction = ExtractionMetrics(matched=1, predicted=2, annotated=2)

        payload = json.loads(_report_json(EvalReport(results=[result])))

        assert payload["papers"][0]["derived"]["precision"] == 0.5
        assert payload["overall"]["recall"] == 0.5
        assert payload["overall"]["execution_measured"] is False


class TestLoading:
    def test_skips_a_malformed_annotation(self, tmp_path: Path) -> None:
        (tmp_path / "good.json").write_text(
            json.dumps({"paper_id": "p", "claims": []}), encoding="utf-8"
        )
        (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")

        annotations = load_annotations(tmp_path)

        assert [a.paper_id for a in annotations] == ["p"]

    def test_an_empty_directory_yields_nothing(self, tmp_path: Path) -> None:
        assert load_annotations(tmp_path) == []
