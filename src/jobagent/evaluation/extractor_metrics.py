"""Pure metrics for comparing extractor output with hand-written labels."""

from __future__ import annotations

import re
import string
from statistics import mean
from typing import NamedTuple

from pydantic import BaseModel

from jobagent.models import JobRequirements

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


class MatchLists(NamedTuple):
    """One-to-one skill matching results."""

    matched: list[tuple[str, str]]
    unmatched_predicted: list[str]
    unmatched_gold: list[str]


class ExtractionScore(BaseModel):
    """Metrics and qualitative errors for one extraction."""

    must_have_precision: float
    must_have_recall: float
    must_have_f1: float
    nice_to_have_precision: float
    nice_to_have_recall: float
    nice_to_have_f1: float
    hallucinated_skills: list[str]
    misplaced_skills: list[str]
    seniority_correct: bool
    role_type_correct: bool


def normalize(text: str) -> str:
    """Lowercase text, remove punctuation, and collapse whitespace."""
    without_punctuation = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(without_punctuation.lower().split())


def skills_match(a: str, b: str) -> bool:
    """Return whether two skill labels are equivalent under relaxed matching."""
    left = normalize(a)
    right = normalize(b)
    if not left or not right:
        return left == right
    if left == right:
        return True
    if left.replace(" ", "") == right.replace(" ", ""):
        return True
    if f" {left} " in f" {right} " or f" {right} " in f" {left} ":
        return True

    left_words = set(left.split())
    right_words = set(right.split())
    return bool(left_words and right_words) and (
        len(left_words & right_words) / len(left_words | right_words) >= 0.5
    )


def match_lists(predicted: list[str], gold: list[str]) -> MatchLists:
    """Greedily match each predicted skill to at most one gold skill."""
    matched: list[tuple[str, str]] = []
    unmatched_predicted: list[str] = []
    remaining_gold = list(gold)

    for prediction in predicted:
        match_index = next(
            (index for index, expected in enumerate(remaining_gold) if skills_match(prediction, expected)),
            None,
        )
        if match_index is None:
            unmatched_predicted.append(prediction)
        else:
            matched.append((prediction, remaining_gold.pop(match_index)))

    return MatchLists(matched, unmatched_predicted, remaining_gold)


def precision_recall_f1(
    predicted: list[str],
    gold: list[str],
) -> tuple[float, float, float]:
    """Calculate precision, recall, and F1 from one-to-one skill matches."""
    matched_count = len(match_lists(predicted, gold).matched)
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 1.0, 0.0, 0.0
    precision = matched_count / len(predicted)
    recall = matched_count / len(gold) if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def is_hallucinated(skill: str, posting_text: str) -> bool:
    """Return whether fewer than half of the meaningful skill words appear."""
    skill_words = {word.casefold() for word in _WORD_RE.findall(normalize(skill))}
    skill_words = {word for word in skill_words if len(word) > 2}
    posting_words = {
        word.casefold() for word in _WORD_RE.findall(normalize(posting_text))
    }
    return bool(skill_words) and (
        len(skill_words & posting_words) / len(skill_words) < 0.5
    )


def misplaced(predicted_must: list[str], gold_nice: list[str]) -> list[str]:
    """Return predicted must-have skills that belong in the gold nice-to-have list."""
    return [
        skill
        for skill in predicted_must
        if any(skills_match(skill, expected) for expected in gold_nice)
    ]


def score_extraction(
    result: JobRequirements,
    expected: dict[str, object],
    posting_text: str,
) -> ExtractionScore:
    """Score one extractor result against a gold entry."""
    gold_must = _skill_list(expected, "must_have_skills")
    gold_nice = _skill_list(expected, "nice_to_have_skills")
    must_metrics = precision_recall_f1(result.must_have_skills, gold_must)
    nice_metrics = precision_recall_f1(result.nice_to_have_skills, gold_nice)
    predicted_skills = [*result.must_have_skills, *result.nice_to_have_skills]
    hallucinated = [
        skill for skill in predicted_skills if is_hallucinated(skill, posting_text)
    ]
    return ExtractionScore(
        must_have_precision=must_metrics[0],
        must_have_recall=must_metrics[1],
        must_have_f1=must_metrics[2],
        nice_to_have_precision=nice_metrics[0],
        nice_to_have_recall=nice_metrics[1],
        nice_to_have_f1=nice_metrics[2],
        hallucinated_skills=hallucinated,
        misplaced_skills=misplaced(result.must_have_skills, gold_nice),
        seniority_correct=normalize(result.seniority)
        == normalize(_string_value(expected, "seniority")),
        role_type_correct=normalize(result.role_type)
        == normalize(_string_value(expected, "role_type")),
    )


def aggregate(scores: list[ExtractionScore]) -> dict[str, float | int]:
    """Aggregate score metrics and total hallucination/misplacement counts."""
    if not scores:
        return {
            "must_have_precision": 0.0,
            "must_have_recall": 0.0,
            "must_have_f1": 0.0,
            "nice_to_have_precision": 0.0,
            "nice_to_have_recall": 0.0,
            "nice_to_have_f1": 0.0,
            "seniority_accuracy": 0.0,
            "role_type_accuracy": 0.0,
            "hallucinated_count": 0,
            "misplaced_count": 0,
        }
    return {
        "must_have_precision": mean(score.must_have_precision for score in scores),
        "must_have_recall": mean(score.must_have_recall for score in scores),
        "must_have_f1": mean(score.must_have_f1 for score in scores),
        "nice_to_have_precision": mean(score.nice_to_have_precision for score in scores),
        "nice_to_have_recall": mean(score.nice_to_have_recall for score in scores),
        "nice_to_have_f1": mean(score.nice_to_have_f1 for score in scores),
        "seniority_accuracy": mean(score.seniority_correct for score in scores),
        "role_type_accuracy": mean(score.role_type_correct for score in scores),
        "hallucinated_count": sum(len(score.hallucinated_skills) for score in scores),
        "misplaced_count": sum(len(score.misplaced_skills) for score in scores),
    }


def _skill_list(expected: dict[str, object], key: str) -> list[str]:
    value = expected.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"Expected {key} to be a list of strings")
    return value


def _string_value(expected: dict[str, object], key: str) -> str:
    value = expected.get(key)
    if not isinstance(value, str):
        raise TypeError(f"Expected {key} to be a string")
    return value
