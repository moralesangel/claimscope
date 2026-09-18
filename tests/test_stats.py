"""The verdict rule, on synthetic data whose right answer is known in advance."""

from __future__ import annotations

import numpy as np
import pytest

from claimscope.stats import (
    ArmSummary,
    InsufficientDataError,
    bootstrap_ci,
    compare_arms,
    signed_effect,
    verdict_from_interval,
    welch_p_value,
)


class TestVerdictFromInterval:
    """PLAN.md section 8, applied to an already-signed interval."""

    def test_interval_entirely_positive_supports_the_claim(self) -> None:
        assert verdict_from_interval(0.1, 0.3) == "consistent_at_reduced_scale"

    def test_interval_entirely_negative_contradicts_the_claim(self) -> None:
        assert verdict_from_interval(-0.3, -0.1) == "not_consistent_at_reduced_scale"

    def test_interval_crossing_zero_is_inconclusive(self) -> None:
        assert verdict_from_interval(-0.1, 0.2) == "inconclusive"

    @pytest.mark.parametrize(("low", "high"), [(0.0, 0.3), (-0.3, 0.0), (0.0, 0.0)])
    def test_touching_zero_is_inconclusive(self, low: float, high: float) -> None:
        # A bound exactly at zero is not evidence of a direction.
        assert verdict_from_interval(low, high) == "inconclusive"


class TestSignedEffect:
    def test_higher_is_better_keeps_the_sign(self) -> None:
        assert signed_effect(0.9, 0.7, "higher_is_better") == pytest.approx(0.2)

    def test_lower_is_better_flips_the_sign(self) -> None:
        # Treatment has lower loss, which is what the paper predicted.
        assert signed_effect(0.1, 0.3, "lower_is_better") == pytest.approx(0.2)

    def test_lower_is_better_penalises_a_higher_value(self) -> None:
        assert signed_effect(0.3, 0.1, "lower_is_better") == pytest.approx(-0.2)


class TestKnownAnswers:
    """Each case has an unambiguous correct verdict."""

    def test_a_large_separation_is_consistent(self) -> None:
        result = compare_arms([0.90, 0.91, 0.89, 0.92], [0.70, 0.71, 0.69, 0.70])

        assert result.verdict == "consistent_at_reduced_scale"
        assert result.effect > 0
        assert result.ci_low > 0

    def test_a_reversed_separation_is_not_consistent(self) -> None:
        result = compare_arms([0.70, 0.71, 0.69], [0.90, 0.91, 0.89])

        assert result.verdict == "not_consistent_at_reduced_scale"
        assert result.effect < 0
        assert result.ci_high < 0

    def test_overlapping_arms_are_inconclusive(self) -> None:
        result = compare_arms([0.80, 0.85, 0.75], [0.79, 0.84, 0.76])

        assert result.verdict == "inconclusive"
        assert result.ci_low < 0 < result.ci_high

    def test_identical_arms_are_inconclusive(self) -> None:
        result = compare_arms([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        assert result.verdict == "inconclusive"
        assert result.effect == pytest.approx(0.0)
        assert result.p_value == 1.0

    def test_a_loss_metric_improving_is_consistent(self) -> None:
        # The claim is that treatment lowers loss, and it does.
        result = compare_arms([0.10, 0.11, 0.09], [0.30, 0.31, 0.29], direction="lower_is_better")

        assert result.verdict == "consistent_at_reduced_scale"
        assert result.effect > 0

    def test_a_loss_metric_worsening_is_not_consistent(self) -> None:
        result = compare_arms([0.30, 0.31, 0.29], [0.10, 0.11, 0.09], direction="lower_is_better")

        assert result.verdict == "not_consistent_at_reduced_scale"


class TestBootstrap:
    def test_is_deterministic_for_a_given_seed(self) -> None:
        args = ([0.9, 0.8, 0.85], [0.7, 0.6, 0.65], "higher_is_better")

        assert bootstrap_ci(*args, seed=7) == bootstrap_ci(*args, seed=7)

    def test_interval_brackets_the_point_estimate(self) -> None:
        treatment, control = [0.9, 0.85, 0.88], [0.7, 0.72, 0.68]
        low, high = bootstrap_ci(treatment, control, "higher_is_better")
        effect = signed_effect(
            float(np.mean(treatment)), float(np.mean(control)), "higher_is_better"
        )

        assert low <= effect <= high

    def test_a_wider_confidence_level_gives_a_wider_interval(self) -> None:
        treatment, control = [0.9, 0.8, 0.85, 0.87], [0.7, 0.6, 0.65, 0.68]

        narrow = bootstrap_ci(treatment, control, "higher_is_better", confidence=0.80)
        wide = bootstrap_ci(treatment, control, "higher_is_better", confidence=0.99)

        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_more_seeds_narrow_the_interval(self) -> None:
        """The power point the report has to make."""
        rng = np.random.default_rng(0)
        widths = []
        for n in (3, 30):
            treatment = list(rng.normal(0.62, 0.05, n))
            control = list(rng.normal(0.60, 0.05, n))
            low, high = bootstrap_ci(treatment, control, "higher_is_better")
            widths.append(high - low)

        assert widths[1] < widths[0]


class TestWelch:
    def test_separated_arms_give_a_small_p_value(self) -> None:
        assert welch_p_value([0.9, 0.91, 0.89], [0.5, 0.51, 0.49]) < 0.01

    def test_overlapping_arms_give_a_large_p_value(self) -> None:
        assert welch_p_value([0.5, 0.6, 0.4], [0.52, 0.58, 0.42]) > 0.5

    def test_constant_identical_arms_do_not_produce_nan(self) -> None:
        assert welch_p_value([0.5, 0.5], [0.5, 0.5]) == 1.0


class TestArmSummary:
    def test_computes_mean_and_sample_std(self) -> None:
        summary = ArmSummary.from_values("treatment", [1.0, 2.0, 3.0])

        assert summary.n == 3
        assert summary.mean == pytest.approx(2.0)
        assert summary.std == pytest.approx(1.0)  # ddof=1

    def test_a_single_run_has_zero_std(self) -> None:
        assert ArmSummary.from_values("a", [1.0]).std == 0.0


class TestPowerWarning:
    def test_three_seeds_are_flagged_as_underpowered(self) -> None:
        result = compare_arms([0.9, 0.8, 0.85], [0.7, 0.6, 0.65])

        assert result.underpowered
        assert "low power" in result.notes()

    def test_many_seeds_are_not_flagged(self) -> None:
        values = [0.9, 0.88, 0.91, 0.89, 0.92, 0.9]
        result = compare_arms(values, [v - 0.2 for v in values])

        assert not result.underpowered
        assert "low power" not in result.notes()


class TestGuards:
    def test_an_empty_arm_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            compare_arms([], [0.5, 0.6])

    def test_both_arms_empty_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            compare_arms([], [])


class TestNotes:
    def test_mentions_both_arms_by_name(self) -> None:
        result = compare_arms(
            [0.9, 0.8, 0.85],
            [0.7, 0.6, 0.65],
            treatment_name="with_dropout",
            control_name="without_dropout",
        )

        notes = result.notes()
        assert "with_dropout" in notes
        assert "without_dropout" in notes
        assert "Welch p=" in notes
