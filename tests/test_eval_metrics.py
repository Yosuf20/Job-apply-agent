"""Tests for pure extractor evaluation metrics."""

from jobagent.evaluation.extractor_metrics import (
    ExtractionScore,
    aggregate,
    is_hallucinated,
    match_lists,
    misplaced,
    normalize,
    precision_recall_f1,
    score_extraction,
    skills_match,
)
from jobagent.models import JobRequirements


def test_normalize_and_skill_matching() -> None:
    assert normalize(" PCI-DSS ") == "pcidss"
    assert skills_match("PCI-DSS", "pci dss")
    assert not skills_match("Python", "Jython")
    # The shared word is 1/5 of the union, so this pair is below the 0.5 threshold.
    assert not skills_match(
        "Government relations", "Government relationship management"
    )


def test_match_lists_is_one_to_one() -> None:
    result = match_lists(["Python", "Python 3"], ["Python"])
    assert result.matched == [("Python", "Python")]
    assert result.unmatched_predicted == ["Python 3"]
    assert result.unmatched_gold == []


def test_precision_recall_f1_handles_simple_and_empty_cases() -> None:
    assert precision_recall_f1(["Python"], ["Python"]) == (1.0, 1.0, 1.0)
    assert precision_recall_f1([], []) == (1.0, 1.0, 1.0)
    assert precision_recall_f1([], ["Python"]) == (1.0, 0.0, 0.0)
    assert precision_recall_f1(["Python"], []) == (0.0, 0.0, 0.0)


def test_hallucination_and_misplacement() -> None:
    assert is_hallucinated("Kubernetes", "Build services with Python.")
    assert not is_hallucinated("Python", "Build services with Python.")
    assert misplaced(["GST", "Python"], ["GST"])


def test_score_extraction_and_aggregate() -> None:
    result = JobRequirements(
        must_have_skills=["Python", "GST"],
        nice_to_have_skills=["AWS"],
        seniority="mid",
        role_type="backend",
        keywords=[],
    )
    expected = {
        "must_have_skills": ["Python"],
        "nice_to_have_skills": ["GST"],
        "seniority": "mid",
        "role_type": "backend",
    }

    score = score_extraction(result, expected, "Python backend role.")

    assert isinstance(score, ExtractionScore)
    assert score.must_have_precision == 0.5
    assert score.must_have_recall == 1.0
    assert score.misplaced_skills == ["GST"]
    assert score.hallucinated_skills == ["GST", "AWS"]
    summary = aggregate([score])
    assert summary["must_have_f1"] == 2 / 3
    assert summary["hallucinated_count"] == 2
    assert summary["misplaced_count"] == 1
