"""Pure evaluation metrics for matcher scores."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean

BAND_MIDPOINTS = {"high": 90.0, "medium": 62.0, "low": 22.0}


def band_for_score(score: float) -> str:
    """Classify a score using the documented high/medium/low ranges."""
    if score >= 80:
        return "high"
    if score >= 45:
        return "medium"
    return "low"


def mean_absolute_error(
    predicted: Sequence[float], expected: Sequence[float | str]
) -> float:
    """Return MAE, converting expected bands to their midpoint."""
    if len(predicted) != len(expected):
        raise ValueError("predicted and expected must have the same length")
    if not predicted:
        return 0.0
    targets = [_expected_value(value) for value in expected]
    return mean(abs(actual - target) for actual, target in zip(predicted, targets))


def band_accuracy(predicted: Sequence[float], expected: Sequence[str]) -> float:
    """Return the share of predicted scores in the expected band."""
    if len(predicted) != len(expected):
        raise ValueError("predicted and expected must have the same length")
    if not predicted:
        return 0.0
    return mean(
        band_for_score(score) == band for score, band in zip(predicted, expected)
    )


def spearman_rank_correlation(
    predicted: Sequence[float], expected: Sequence[float | str]
) -> float | None:
    """Calculate Spearman correlation with average ranks for ties."""
    if len(predicted) != len(expected) or len(predicted) < 3:
        return None
    expected_values = [_expected_value(value) for value in expected]
    if len(set(predicted)) < 2 or len(set(expected_values)) < 2:
        return None
    predicted_ranks = _average_ranks(predicted)
    expected_ranks = _average_ranks(expected_values)
    predicted_mean = mean(predicted_ranks)
    expected_mean = mean(expected_ranks)
    numerator = sum(
        (left - predicted_mean) * (right - expected_mean)
        for left, right in zip(predicted_ranks, expected_ranks)
    )
    left_sum = sum((rank - predicted_mean) ** 2 for rank in predicted_ranks)
    right_sum = sum((rank - expected_mean) ** 2 for rank in expected_ranks)
    return numerator / (left_sum * right_sum) ** 0.5


def _expected_value(value: float | str) -> float:
    if isinstance(value, str):
        try:
            return BAND_MIDPOINTS[value]
        except KeyError as error:
            raise ValueError(f"Unknown score band: {value}") from error
    return float(value)


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2
        for position in range(index, end):
            ranks[ordered[position][0]] = rank
        index = end
    return ranks
