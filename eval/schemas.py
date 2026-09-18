"""Annotation and metric schemas for evaluating the agent (PLAN.md section 9)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from claimscope.schemas import ClaimType, Verdict


class AnnotatedClaim(BaseModel):
    """One claim as a human annotator recorded it."""

    text: str
    claim_type: ClaimType
    testable: bool
    expected_verdict: Verdict | None = None
    """What a correct run should conclude. Absent when the annotator is unsure."""

    source_location: str = ""
    notes: str = ""


class PaperAnnotation(BaseModel):
    """The gold standard for one paper: ``eval/annotations/<paper_id>.json``."""

    paper_id: str
    title: str = ""
    claims: list[AnnotatedClaim] = Field(default_factory=list)
    notes: str = ""

    @property
    def testable_claims(self) -> list[AnnotatedClaim]:
        return [claim for claim in self.claims if claim.testable]


class ExtractionMetrics(BaseModel):
    """How well the agent found the claims a human found."""

    matched: int = 0
    predicted: int = 0
    annotated: int = 0

    @property
    def precision(self) -> float:
        """Of the claims the agent extracted, how many were real."""
        return self.matched / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        """Of the claims a human found, how many the agent also found."""
        return self.matched / self.annotated if self.annotated else 0.0

    @property
    def f1(self) -> float:
        if not (self.precision + self.recall):
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


class ClassificationMetrics(BaseModel):
    """Accuracy of claim_type and testable, over matched claims only.

    Scoring a claim the agent never found would conflate two different failures.
    """

    type_correct: int = 0
    testable_correct: int = 0
    total: int = 0

    @property
    def type_accuracy(self) -> float:
        return self.type_correct / self.total if self.total else 0.0

    @property
    def testable_accuracy(self) -> float:
        return self.testable_correct / self.total if self.total else 0.0


class ExecutionMetrics(BaseModel):
    """How often generated code actually ran.

    Requires a sandbox. Without one these stay at zero and ``measured`` is
    False, so a report can say "not measured" instead of implying a zero score.
    """

    measured: bool = False
    claims_attempted: int = 0
    claims_completed: int = 0
    total_debug_attempts: int = 0

    @property
    def completion_rate(self) -> float:
        return self.claims_completed / self.claims_attempted if self.claims_attempted else 0.0

    @property
    def mean_debug_attempts(self) -> float:
        return self.total_debug_attempts / self.claims_attempted if self.claims_attempted else 0.0


class VerdictMetrics(BaseModel):
    """Agreement with the verdict the annotator expected."""

    measured: bool = False
    agreed: int = 0
    compared: int = 0

    @property
    def agreement(self) -> float:
        return self.agreed / self.compared if self.compared else 0.0


class CostMetrics(BaseModel):
    """What one paper cost to process."""

    wall_clock_s: float = 0.0
    compute_minutes: float = 0.0
    llm_calls: int = 0


class PaperResult(BaseModel):
    """Everything measured for one paper."""

    paper_id: str
    extraction: ExtractionMetrics = Field(default_factory=ExtractionMetrics)
    classification: ClassificationMetrics = Field(default_factory=ClassificationMetrics)
    execution: ExecutionMetrics = Field(default_factory=ExecutionMetrics)
    verdicts: VerdictMetrics = Field(default_factory=VerdictMetrics)
    cost: CostMetrics = Field(default_factory=CostMetrics)
    errors: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    """The aggregate across every annotated paper."""

    results: list[PaperResult] = Field(default_factory=list)

    @property
    def extraction(self) -> ExtractionMetrics:
        """Micro-averaged: every claim counts the same, regardless of its paper."""
        total = ExtractionMetrics()
        for result in self.results:
            total.matched += result.extraction.matched
            total.predicted += result.extraction.predicted
            total.annotated += result.extraction.annotated
        return total

    @property
    def classification(self) -> ClassificationMetrics:
        total = ClassificationMetrics()
        for result in self.results:
            total.type_correct += result.classification.type_correct
            total.testable_correct += result.classification.testable_correct
            total.total += result.classification.total
        return total

    @property
    def execution(self) -> ExecutionMetrics:
        total = ExecutionMetrics()
        for result in self.results:
            if not result.execution.measured:
                continue
            total.measured = True
            total.claims_attempted += result.execution.claims_attempted
            total.claims_completed += result.execution.claims_completed
            total.total_debug_attempts += result.execution.total_debug_attempts
        return total

    @property
    def verdicts(self) -> VerdictMetrics:
        total = VerdictMetrics()
        for result in self.results:
            if not result.verdicts.measured:
                continue
            total.measured = True
            total.agreed += result.verdicts.agreed
            total.compared += result.verdicts.compared
        return total
