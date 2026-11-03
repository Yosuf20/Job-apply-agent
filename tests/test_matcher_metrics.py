"""Tests for matcher evaluation metrics."""

from jobagent.evaluation.matcher_metrics import (
    band_accuracy,
    band_for_score,
    mean_absolute_error,
    spearman_rank_correlation,
)


def test_bands_and_midpoint_mae() -> None:
    assert band_for_score(80) == "high"
    assert band_for_score(79) == "medium"
    assert band_for_score(44) == "low"
    assert mean_absolute_error([90, 62, 22], ["high", "medium", "low"]) == 0.0


def test_band_accuracy_and_spearman() -> None:
    assert band_accuracy([90, 62, 10], ["high", "medium", "low"]) == 1.0
    assert spearman_rank_correlation([10, 20, 30], ["low", "medium", "high"]) == 1.0
    assert spearman_rank_correlation([1, 1, 2], [1, 2, 3]) is not None


def test_spearman_handles_small_and_constant_inputs() -> None:
    assert spearman_rank_correlation([1, 2], [1, 2]) is None
    assert spearman_rank_correlation([1, 1, 1], [1, 2, 3]) is None
