"""The Markdown report, especially the claims it must never overstate."""

from __future__ import annotations

from claimscope.config import Settings
from claimscope.nodes.report import REPORT_FILENAME, render_report, report
from claimscope.schemas import Claim, ClaimVerdict, ReductionPlan, RunResult
from claimscope.state import GraphState


def _claim(claim_id: str = "c1", claim_type: str = "comparative") -> Claim:
    return Claim(
        id=claim_id,
        text="Residual networks beat plain networks of equal depth.",
        source_location="Fig. 4",
        claim_type=claim_type,  # type: ignore[arg-type]
        arms=["resnet", "plain"],
        metric="top-1 error",
        expected_direction="resnet < plain",
    )


def _plan(claim_id: str = "c1") -> ReductionPlan:
    return ReductionPlan(
        claim_id=claim_id,
        original_setup="ResNet-34 vs plain-34 on ImageNet.",
        reduced_setup="8-layer variants on a CIFAR-10 subset.",
        changes=["ImageNet -> CIFAR-10 subset -- degradation is not dataset specific."],
        preserved=["The residual connections, which the claim is about."],
        why_claim_should_transfer="Plain nets degrade with depth at small scale too.",
        seeds=3,
        estimated_minutes=8.0,
        code_source="from_scratch",
    )


def _verdict(claim_id: str = "c1", verdict: str = "consistent_at_reduced_scale") -> ClaimVerdict:
    return ClaimVerdict(
        claim_id=claim_id,
        verdict=verdict,  # type: ignore[arg-type]
        effect_estimate=0.2,
        ci_low=0.15,
        ci_high=0.25,
        p_value=0.004,
        notes="resnet: 0.9 (sd 0.01, n=3); plain: 0.7 (sd 0.01, n=3).",
    )


def _runs() -> list[RunResult]:
    return [
        RunResult(arm=arm, seed=s, metric_value=v, runtime_s=12.5, log_path=f"{arm}_{s}.txt")
        for arm, values in (("resnet", [0.9, 0.91, 0.89]), ("plain", [0.7, 0.71, 0.69]))
        for s, v in enumerate(values)
    ]


def _state(**overrides: object) -> GraphState:
    state: GraphState = {
        "paper_id": "1512.03385",
        "paper_title": "Deep Residual Learning",
        "claims": [_claim()],
        "selected_claim_ids": ["c1"],
        "plans": {"c1": _plan()},
        "run_results": {"c1": _runs()},
        "verdicts": [_verdict()],
        "budget_minutes_total": 60.0,
        "budget_minutes_used": 4.2,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


class TestHonesty:
    """PLAN.md section 2: the report must never let a result be over-read."""

    def test_says_a_negative_result_does_not_refute_the_paper(self) -> None:
        markdown = render_report(
            _state(verdicts=[_verdict(verdict="not_consistent_at_reduced_scale")])
        )

        assert "does not refute the paper" in markdown

    def test_the_disclaimer_is_present_even_on_a_positive_result(self) -> None:
        # It is unconditional, not something shown only on bad news.
        markdown = render_report(_state())

        assert "does not refute the paper" in markdown
        assert "not a reproduction" in markdown

    def test_warns_about_low_seed_counts(self) -> None:
        assert "statistical power is low" in render_report(_state())

    def test_states_it_is_not_a_reproduction_up_front(self) -> None:
        markdown = render_report(_state())
        intro = markdown.split("## Verdicts")[0]

        assert "not a reproduction of the paper" in intro


class TestContent:
    def test_includes_the_paper_title(self) -> None:
        assert "Deep Residual Learning" in render_report(_state())

    def test_lists_the_verdict_in_the_summary_table(self) -> None:
        markdown = render_report(_state())

        assert "Consistent at reduced scale" in markdown
        assert "`c1`" in markdown

    def test_shows_the_effect_and_interval(self) -> None:
        markdown = render_report(_state())

        assert "+0.2" in markdown
        assert "+0.15" in markdown and "+0.25" in markdown

    def test_documents_the_reduction_plan(self) -> None:
        markdown = render_report(_state())

        assert "ResNet-34 vs plain-34 on ImageNet." in markdown
        assert "degradation is not dataset specific" in markdown
        assert "The residual connections" in markdown

    def test_includes_a_row_per_run(self) -> None:
        markdown = render_report(_state())

        for arm in ("resnet", "plain"):
            for seed in range(3):
                assert f"| {arm} | {seed} |" in markdown

    def test_reports_budget_usage(self) -> None:
        assert "4.2 of 60 minutes" in render_report(_state())

    def test_includes_the_repository_when_known(self) -> None:
        markdown = render_report(_state(repo_url="https://github.com/lab/repo"))

        assert "https://github.com/lab/repo" in markdown

    def test_lists_untested_claims_with_their_reason(self) -> None:
        state = _state(
            claims=[_claim(), _claim("c2", claim_type="absolute")],
            verdicts=[
                _verdict(),
                ClaimVerdict(
                    claim_id="c2",
                    verdict="not_testable",
                    notes="Absolute numbers cannot be reproduced at reduced scale.",
                ),
            ],
        )

        markdown = render_report(state)

        assert "## Claims not tested" in markdown
        assert "Absolute numbers cannot be reproduced" in markdown

    def test_surfaces_errors_encountered(self) -> None:
        markdown = render_report(_state(errors=["c9: gave up after 3 debug attempts"]))

        assert "## Problems encountered" in markdown
        assert "gave up after 3 debug attempts" in markdown

    def test_handles_a_paper_where_nothing_was_testable(self) -> None:
        state = _state(
            selected_claim_ids=[],
            plans={},
            run_results={},
            verdicts=[
                ClaimVerdict(claim_id="c1", verdict="not_testable", notes="Scale dependent.")
            ],
        )

        markdown = render_report(state)

        assert "Not testable at reduced scale" in markdown
        assert "does not refute the paper" in markdown


class TestWriting:
    def test_writes_the_file_and_returns_its_path(self, settings: Settings) -> None:
        result = report(_state(), settings)

        path = settings.runs_dir / "1512.03385" / REPORT_FILENAME
        assert result["report_path"] == str(path)
        assert path.exists()

    def test_the_written_file_is_the_rendered_report(self, settings: Settings) -> None:
        state = _state()

        report(state, settings)

        written = (settings.runs_dir / "1512.03385" / REPORT_FILENAME).read_text(encoding="utf-8")
        assert written.startswith("# ClaimScope report: Deep Residual Learning")
