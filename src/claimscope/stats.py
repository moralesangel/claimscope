"""Statistics and the verdict rule (PLAN.md section 8).

The effect is a difference of means across seeds, signed so that positive always
means "in the direction the paper predicts". A bootstrap confidence interval
decides the verdict; Welch's t-test is reported alongside as support, never as
the deciding test.

With three seeds power is very low. Callers must surface that: an
``inconclusive`` verdict here usually means "not enough evidence", not "no
effect".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats as scipy_stats

Verdict = Literal[
    "consistent_at_reduced_scale",
    "not_consistent_at_reduced_scale",
    "inconclusive",
]

Direction = Literal["higher_is_better", "lower_is_better"]

DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95

# Below this many seeds per arm a bootstrap over seeds is not meaningful.
MIN_SEEDS_FOR_INFERENCE = 3


class InsufficientDataError(ValueError):
    """Not enough runs to say anything, so no verdict is invented."""


@dataclass(frozen=True)
class ArmSummary:
    """Per-arm descriptive statistics."""

    name: str
    n: int
    mean: float
    std: float

    @classmethod
    def from_values(cls, name: str, values: list[float]) -> ArmSummary:
        array = np.asarray(values, dtype=float)
        return cls(
            name=name,
            n=int(array.size),
            mean=float(array.mean()),
            # Sample standard deviation: these seeds are a sample of possible runs.
            std=float(array.std(ddof=1)) if array.size > 1 else 0.0,
        )


@dataclass(frozen=True)
class ComparisonResult:
    """The full statistical picture for one claim."""

    treatment: ArmSummary
    control: ArmSummary
    effect: float
    """Difference of means, signed so positive means "as the paper predicts"."""

    ci_low: float
    ci_high: float
    confidence: float
    p_value: float
    verdict: Verdict
    bootstrap_samples: int
    underpowered: bool
    """True when the seed count is too low for the interval to mean much."""

    def notes(self) -> str:
        """A short, honest summary of what this result does and does not show."""
        parts = [
            f"{self.treatment.name}: {self.treatment.mean:.4g} "
            f"(sd {self.treatment.std:.3g}, n={self.treatment.n}); "
            f"{self.control.name}: {self.control.mean:.4g} "
            f"(sd {self.control.std:.3g}, n={self.control.n}).",
            f"Effect in the predicted direction: {self.effect:+.4g} "
            f"[{int(self.confidence * 100)}% CI {self.ci_low:+.4g}, {self.ci_high:+.4g}], "
            f"Welch p={self.p_value:.3g}.",
        ]
        if self.underpowered:
            parts.append(
                f"With {self.treatment.n} seeds per arm this test has low power: "
                "an inconclusive result means insufficient evidence, not absence of an effect."
            )
        return " ".join(parts)


def signed_effect(treatment_mean: float, control_mean: float, direction: Direction) -> float:
    """Difference of means, signed so positive always supports the claim.

    For a metric where lower is better (loss, error rate), the paper predicting
    "treatment beats control" means treatment should be *lower*, so the
    difference is negated.
    """
    raw = treatment_mean - control_mean
    return raw if direction == "higher_is_better" else -raw


def bootstrap_ci(
    treatment: list[float],
    control: list[float],
    direction: Direction,
    confidence: float = DEFAULT_CONFIDENCE,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the signed effect, resampling seeds.

    Seeds are the unit of resampling: the question is how much the effect would
    move if the experiment were repeated with different random seeds.
    """
    rng = np.random.default_rng(seed)
    treatment_array = np.asarray(treatment, dtype=float)
    control_array = np.asarray(control, dtype=float)

    treatment_draws = rng.choice(treatment_array, size=(samples, treatment_array.size))
    control_draws = rng.choice(control_array, size=(samples, control_array.size))

    raw = treatment_draws.mean(axis=1) - control_draws.mean(axis=1)
    effects = raw if direction == "higher_is_better" else -raw

    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(effects, [tail, 1.0 - tail])
    return float(low), float(high)


def welch_p_value(treatment: list[float], control: list[float]) -> float:
    """Two-sided Welch's t-test, reported as supporting evidence only."""
    if len({*treatment, *control}) == 1:
        # Both arms identical and constant: t-test is undefined, effect is zero.
        return 1.0
    result = scipy_stats.ttest_ind(treatment, control, equal_var=False)
    p_value = float(result.pvalue)
    return 1.0 if np.isnan(p_value) else p_value


def verdict_from_interval(ci_low: float, ci_high: float) -> Verdict:
    """The rule from PLAN.md section 8, applied to the signed interval.

    The interval is already signed so that positive means "as predicted", so:
    entirely above zero supports the claim, entirely below contradicts it, and
    straddling zero is inconclusive.
    """
    if ci_low > 0:
        return "consistent_at_reduced_scale"
    if ci_high < 0:
        return "not_consistent_at_reduced_scale"
    return "inconclusive"


def compare_arms(
    treatment_values: list[float],
    control_values: list[float],
    direction: Direction = "higher_is_better",
    treatment_name: str = "treatment",
    control_name: str = "control",
    confidence: float = DEFAULT_CONFIDENCE,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> ComparisonResult:
    """Compare two arms and return the verdict with its supporting statistics."""
    if not treatment_values or not control_values:
        raise InsufficientDataError("both arms need at least one run")

    treatment = ArmSummary.from_values(treatment_name, treatment_values)
    control = ArmSummary.from_values(control_name, control_values)

    effect = signed_effect(treatment.mean, control.mean, direction)
    ci_low, ci_high = bootstrap_ci(
        treatment_values,
        control_values,
        direction,
        confidence=confidence,
        samples=bootstrap_samples,
        seed=seed,
    )

    return ComparisonResult(
        treatment=treatment,
        control=control,
        effect=effect,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        p_value=welch_p_value(treatment_values, control_values),
        verdict=verdict_from_interval(ci_low, ci_high),
        bootstrap_samples=bootstrap_samples,
        underpowered=min(treatment.n, control.n) <= MIN_SEEDS_FOR_INFERENCE,
    )
