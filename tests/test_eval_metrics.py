"""Evaluation metrics: matching, scoring, and what happens with no sandbox."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.matching import match_claims, similarity
from eval.metrics import score_execution, score_extraction, score_paper, score_verdicts
from eval.run_eval import load_annotations
from eval.schemas import AnnotatedClaim, CostMetrics, EvalReport, PaperAnnotation, PaperResult

from claimscope.schemas import Claim, ClaimVerdict

ANNOTATIONS = Path(__file__).parents[1] / "eval" / "annotations"


def _claim(
    claim_id: str,
    text: str,
    metric: str = "accuracy",
    arms: list[str] | None = None,
    claim_type: str = "comparative",
    testable: bool | None = None,
) -> Claim:
    return Claim(
        id=claim_id,
        text=text,
        source_location="Table 1",
        claim_type=claim_type,  # type: ignore[arg-type]
        arms=arms or ["a", "b"],
        metric=metric,
        expected_direction="a > b",
        testable=testable,
    )


def _annotated(
    text: str,
    claim_type: str = "comparative",
    testable: bool = True,
    expected_verdict: str | None = None,
) -> AnnotatedClaim:
    return AnnotatedClaim(
        text=text,
        claim_type=claim_type,  # type: ignore[arg-type]
        testable=testable,
        expected_verdict=expected_verdict,  # type: ignore[arg-type]
    )


class TestSimilarity:
    def test_the_same_claim_scores_high(self) -> None:
        predicted = _claim("c1", "Residual networks reach lower training error than plain networks")
        annotated = _annotated("Plain networks show higher training error than residual networks")

        assert similarity(predicted, annotated) > 0.35

    def test_unrelated_claims_score_low(self) -> None:
        predicted = _claim("c1", "Dropout reduces overfitting in small MLPs")
        annotated = _annotated("The model reaches 92 BLEU on English to French translation")

        assert similarity(predicted, annotated) < 0.2

    def test_sharing_the_metric_helps(self) -> None:
        with_metric = _claim("c1", "Method A wins", metric="top-1 error")
        without = _claim("c2", "Method A wins", metric="unrelated")
        annotated = _annotated("Method A achieves lower top-1 error")

        assert similarity(with_metric, annotated) > similarity(without, annotated)

    def test_empty_text_scores_zero(self) -> None:
        assert similarity(_claim("c1", "the and of"), _annotated("a to in")) == 0.0


class TestMatching:
    def test_matches_one_to_one(self) -> None:
        predicted = [
            _claim("c1", "Residual networks beat plain networks on training error"),
            _claim("c2", "Dropout reduces the generalization gap"),
        ]
        annotated = [
            _annotated("Dropout reduces overfitting and the generalization gap"),
            _annotated("Plain networks have higher training error than residual networks"),
        ]

        result = match_claims(predicted, annotated)

        assert len(result.matches) == 2
        pairs = {(m.predicted.id, m.annotated.text[:7]) for m in result.matches}
        assert ("c1", "Plain n") in pairs
        assert ("c2", "Dropout") in pairs

    def test_a_claim_matches_at_most_one_annotation(self) -> None:
        predicted = [_claim("c1", "Dropout reduces the generalization gap")]
        annotated = [
            _annotated("Dropout reduces the generalization gap"),
            _annotated("Dropout reduces the generalization gap on MNIST"),
        ]

        result = match_claims(predicted, annotated)

        assert len(result.matches) == 1
        assert len(result.unmatched_annotated) == 1

    def test_reports_what_was_missed(self) -> None:
        predicted = [_claim("c1", "Dropout reduces overfitting")]
        annotated = [_annotated("A completely unrelated statement about parsing scores")]

        result = match_claims(predicted, annotated)

        assert result.matches == []
        assert len(result.unmatched_predicted) == 1
        assert len(result.unmatched_annotated) == 1


class TestExtractionScoring:
    def test_a_perfect_run_scores_one(self) -> None:
        predicted = [_claim("c1", "Dropout reduces the generalization gap")]
        annotation = PaperAnnotation(
            paper_id="p", claims=[_annotated("Dropout reduces the generalization gap")]
        )

        extraction, _ = score_extraction(predicted, annotation)

        assert extraction.precision == 1.0
        assert extraction.recall == 1.0
        assert extraction.f1 == 1.0

    def test_a_spurious_claim_lowers_precision(self) -> None:
        predicted = [
            _claim("c1", "Dropout reduces the generalization gap"),
            _claim("c2", "Something entirely invented about parsing"),
        ]
        annotation = PaperAnnotation(
            paper_id="p", claims=[_annotated("Dropout reduces the generalization gap")]
        )

        extraction, _ = score_extraction(predicted, annotation)

        assert extraction.precision == 0.5
        assert extraction.recall == 1.0

    def test_a_missed_claim_lowers_recall(self) -> None:
        predicted = [_claim("c1", "Dropout reduces the generalization gap")]
        annotation = PaperAnnotation(
            paper_id="p",
            claims=[
                _annotated("Dropout reduces the generalization gap"),
                _annotated("Batch norm speeds up convergence considerably"),
            ],
        )

        extraction, _ = score_extraction(predicted, annotation)

        assert extraction.precision == 1.0
        assert extraction.recall == 0.5

    def test_classification_is_scored_only_on_matches(self) -> None:
        """Penalising the type of a claim never found would count one miss twice."""
        predicted = [_claim("c1", "Dropout reduces the generalization gap", claim_type="ablation")]
        annotation = PaperAnnotation(
            paper_id="p",
            claims=[
                _annotated("Dropout reduces the generalization gap", claim_type="ablation"),
                _annotated("An unrelated claim about parsing", claim_type="absolute"),
            ],
        )

        _extraction, classification = score_extraction(predicted, annotation)

        assert classification.total == 1
        assert classification.type_accuracy == 1.0

    def test_a_wrong_type_is_counted(self) -> None:
        predicted = [_claim("c1", "Dropout reduces the generalization gap", claim_type="absolute")]
        annotation = PaperAnnotation(
            paper_id="p",
            claims=[_annotated("Dropout reduces the generalization gap", claim_type="ablation")],
        )

        _extraction, classification = score_extraction(predicted, annotation)

        assert classification.type_accuracy == 0.0

    def test_testable_accuracy_is_counted(self) -> None:
        predicted = [
            _claim("c1", "Dropout reduces the generalization gap", testable=True),
        ]
        annotation = PaperAnnotation(
            paper_id="p",
            claims=[_annotated("Dropout reduces the generalization gap", testable=True)],
        )

        _extraction, classification = score_extraction(predicted, annotation)

        assert classification.testable_accuracy == 1.0


class TestVerdictScoring:
    def test_agreement_is_counted(self) -> None:
        predicted = [_claim("c1", "Dropout reduces the generalization gap")]
        annotation = PaperAnnotation(
            paper_id="p",
            claims=[
                _annotated(
                    "Dropout reduces the generalization gap",
                    expected_verdict="consistent_at_reduced_scale",
                )
            ],
        )
        verdicts = [ClaimVerdict(claim_id="c1", verdict="consistent_at_reduced_scale")]

        metrics = score_verdicts(verdicts, predicted, annotation)

        assert metrics.measured
        assert metrics.agreement == 1.0

    def test_disagreement_is_counted(self) -> None:
        predicted = [_claim("c1", "Dropout reduces the generalization gap")]
        annotation = PaperAnnotation(
            paper_id="p",
            claims=[
                _annotated(
                    "Dropout reduces the generalization gap",
                    expected_verdict="consistent_at_reduced_scale",
                )
            ],
        )
        verdicts = [ClaimVerdict(claim_id="c1", verdict="inconclusive")]

        assert score_verdicts(verdicts, predicted, annotation).agreement == 0.0

    def test_claims_without_an_expected_verdict_are_skipped(self) -> None:
        """An unsure annotator must not be scored as a disagreement."""
        predicted = [_claim("c1", "Dropout reduces the generalization gap")]
        annotation = PaperAnnotation(
            paper_id="p", claims=[_annotated("Dropout reduces the generalization gap")]
        )
        verdicts = [ClaimVerdict(claim_id="c1", verdict="inconclusive")]

        metrics = score_verdicts(verdicts, predicted, annotation)

        assert not metrics.measured
        assert metrics.compared == 0


class TestExecutionScoring:
    def test_not_measured_without_a_sandbox(self) -> None:
        """Zero would read as "everything failed"; this says "nothing was run"."""
        metrics = score_execution({"approved_plan_ids": ["c1"]}, sandbox_available=False)

        assert not metrics.measured
        assert metrics.claims_attempted == 0

    def test_counts_completed_claims(self) -> None:
        state = {
            "approved_plan_ids": ["c1", "c2"],
            "run_results": {"c1": [object()]},
            "debug_attempts": {"c1": 1, "c2": 3},
        }

        metrics = score_execution(state, sandbox_available=True)

        assert metrics.measured
        assert metrics.claims_attempted == 2
        assert metrics.claims_completed == 1
        assert metrics.completion_rate == 0.5
        assert metrics.mean_debug_attempts == 2.0


class TestPaperScoring:
    def test_scores_a_whole_paper(self) -> None:
        annotation = PaperAnnotation(
            paper_id="p", claims=[_annotated("Dropout reduces the generalization gap")]
        )
        state = {
            "claims": [_claim("c1", "Dropout reduces the generalization gap")],
            "verdicts": [],
            "errors": ["something went wrong"],
        }

        result = score_paper(annotation, state, CostMetrics(llm_calls=4))

        assert result.paper_id == "p"
        assert result.extraction.f1 == 1.0
        assert result.cost.llm_calls == 4
        assert result.errors == ["something went wrong"]

    def test_an_empty_run_scores_zero_without_crashing(self) -> None:
        annotation = PaperAnnotation(paper_id="p", claims=[_annotated("A claim")])

        result = score_paper(annotation, {}, CostMetrics())

        assert result.extraction.recall == 0.0
        assert result.extraction.f1 == 0.0


class TestAggregation:
    def test_micro_averages_across_papers(self) -> None:
        report = EvalReport(
            results=[
                PaperResult(paper_id="a"),
                PaperResult(paper_id="b"),
            ]
        )
        report.results[0].extraction.matched = 1
        report.results[0].extraction.predicted = 2
        report.results[0].extraction.annotated = 1
        report.results[1].extraction.matched = 3
        report.results[1].extraction.predicted = 3
        report.results[1].extraction.annotated = 6

        overall = report.extraction

        assert overall.matched == 4
        assert overall.precision == pytest.approx(4 / 5)
        assert overall.recall == pytest.approx(4 / 7)

    def test_execution_is_unmeasured_when_no_paper_ran(self) -> None:
        report = EvalReport(results=[PaperResult(paper_id="a")])

        assert not report.execution.measured


class TestRealAnnotations:
    """The checked-in annotations must stay loadable and internally consistent."""

    def test_annotations_load(self) -> None:
        annotations = load_annotations(ANNOTATIONS)

        assert len(annotations) >= 2
        assert {a.paper_id for a in annotations} >= {"1512.03385", "1706.03762"}

    def test_every_annotation_has_claims(self) -> None:
        for annotation in load_annotations(ANNOTATIONS):
            assert annotation.claims, f"{annotation.paper_id} has no claims"

    def test_absolute_claims_are_never_marked_testable(self) -> None:
        # The same structural rule triage enforces.
        for annotation in load_annotations(ANNOTATIONS):
            for claim in annotation.claims:
                if claim.claim_type == "absolute":
                    assert not claim.testable, f"{annotation.paper_id}: {claim.text[:40]}"

    def test_untestable_claims_have_no_expected_verdict(self) -> None:
        for annotation in load_annotations(ANNOTATIONS):
            for claim in annotation.claims:
                if not claim.testable:
                    assert claim.expected_verdict is None

    def test_files_are_valid_json(self) -> None:
        for path in ANNOTATIONS.glob("*.json"):
            json.loads(path.read_text(encoding="utf-8"))

    def test_the_filename_matches_the_paper_id(self) -> None:
        for path in ANNOTATIONS.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["paper_id"] == path.stem

    def test_every_testable_claim_has_an_expected_verdict(self) -> None:
        # A testable claim with no expectation contributes nothing to the score.
        for annotation in load_annotations(ANNOTATIONS):
            for claim in annotation.testable_claims:
                assert claim.expected_verdict is not None, (
                    f"{annotation.paper_id}: {claim.text[:50]}"
                )


class TestCorpusBalance:
    """A lopsided corpus produces flattering metrics.

    These are deliberately loose: they catch a corpus drifting into one shape,
    not small fluctuations as papers are added.
    """

    def test_there_are_enough_papers(self) -> None:
        # PLAN.md section 9 asks for 10-15.
        assert len(load_annotations(ANNOTATIONS)) >= 10

    def test_all_four_claim_types_appear(self) -> None:
        types = {
            claim.claim_type
            for annotation in load_annotations(ANNOTATIONS)
            for claim in annotation.claims
        }

        assert types == {"absolute", "comparative", "ablation", "scaling_trend"}

    def test_testable_and_untestable_are_both_well_represented(self) -> None:
        claims = [c for a in load_annotations(ANNOTATIONS) for c in a.claims]
        testable = sum(1 for claim in claims if claim.testable)

        assert 0.3 < testable / len(claims) < 0.85

    def test_some_claims_are_expected_to_be_contradicted(self) -> None:
        """Without these, the corpus cannot tell whether the agent ever says no.

        A harness that only ever rewards agreement would score a sycophantic
        agent perfectly, which is the opposite of what this tool is for.
        """
        verdicts = [
            claim.expected_verdict
            for annotation in load_annotations(ANNOTATIONS)
            for claim in annotation.claims
            if claim.expected_verdict
        ]

        assert "not_consistent_at_reduced_scale" in verdicts

    def test_all_three_testable_verdicts_are_represented(self) -> None:
        verdicts = {
            claim.expected_verdict
            for annotation in load_annotations(ANNOTATIONS)
            for claim in annotation.claims
            if claim.expected_verdict
        }

        assert verdicts == {
            "consistent_at_reduced_scale",
            "not_consistent_at_reduced_scale",
            "inconclusive",
        }

    def test_at_least_one_paper_is_almost_entirely_untestable(self) -> None:
        # Triage must be able to refuse a whole paper, not just individual claims.
        annotations = load_annotations(ANNOTATIONS)

        assert any(len(a.testable_claims) == 0 for a in annotations)
