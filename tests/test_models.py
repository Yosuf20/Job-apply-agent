"""Tests for data models and configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from jobagent.config import get_settings
from jobagent.models import (
    Bullet,
    Education,
    Experience,
    MatchResult,
    Profile,
    Project,
    Skill,
    load_profile,
)


def test_example_profile_loads_and_validates() -> None:
    """The example profile YAML loads cleanly into a validated Profile instance."""
    profile_path = Path("data/profile.example.yaml")
    profile = load_profile(profile_path)

    assert profile.name == "Alex Morgan"
    assert profile.email == "alex.morgan.fake@example.com"
    assert len(profile.experience) >= 2
    assert len(profile.projects) >= 2

    # Check total bullet count >= 8
    exp_bullets = [b for exp in profile.experience for b in exp.bullets]
    proj_bullets = [b for proj in profile.projects for b in proj.bullets]
    total_bullets = exp_bullets + proj_bullets
    assert len(total_bullets) >= 8

    # All bullets should have tags and some should have metrics
    assert all(len(b.tags) > 0 for b in total_bullets)
    assert any(b.metrics is not None for b in total_bullets)


def test_match_result_score_upper_bound() -> None:
    """A MatchResult with score > 100 raises a ValidationError."""
    with pytest.raises(ValidationError):
        MatchResult(
            job_id="job-1",
            score=150,
            reasoning="Fit is unexpectedly high",
            matched_skills=["Python"],
            missing_skills=[],
        )


def test_match_result_score_lower_bound() -> None:
    """A MatchResult with score < 0 raises a ValidationError."""
    with pytest.raises(ValidationError):
        MatchResult(
            job_id="job-2",
            score=-1,
            reasoning="Fit is below minimum bound",
            matched_skills=[],
            missing_skills=["Python"],
        )


def test_match_result_valid_boundary_scores() -> None:
    """MatchResult accepts boundary values 0 and 100."""
    result_min = MatchResult(
        job_id="job-min",
        score=0,
        reasoning="No match",
    )
    result_max = MatchResult(
        job_id="job-max",
        score=100,
        reasoning="Perfect match",
    )
    assert result_min.score == 0
    assert result_max.score == 100


def test_profile_duplicate_bullet_ids_raises_validation_error() -> None:
    """Duplicate bullet IDs across experience and projects raise a ValidationError."""
    shared_id = "bullet-duplicate-1"

    with pytest.raises(ValidationError, match="Duplicate bullet ids found"):
        Profile(
            name="Test User",
            email="test@example.com",
            phone="123-456-7890",
            location="Remote",
            links=[],
            summary="Test candidate summary",
            skills=[Skill(name="Python", tags=["backend"])],
            experience=[
                Experience(
                    company="Company A",
                    title="Engineer",
                    start="2022-01",
                    bullets=[Bullet(id=shared_id, text="Did work", tags=["tag1"])],
                )
            ],
            projects=[
                Project(
                    name="Project B",
                    description="A side project",
                    tech=["Python"],
                    bullets=[Bullet(id=shared_id, text="Another claim", tags=["tag2"])],
                )
            ],
            education=[
                Education(
                    institution="Test University",
                    degree="B.S. CS",
                    start="2018-09",
                    end="2022-06",
                )
            ],
        )


def test_load_profile_missing_file_raises_error() -> None:
    """Attempting to load a nonexistent profile file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_profile("data/nonexistent_profile.yaml")


def test_get_settings_defaults_when_no_env_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """get_settings returns sensible defaults when no environment variables or .env file are set."""
    env_vars_to_clear = [
        "LLM_API_KEY",
        "LLM_MODEL_SMALL",
        "LLM_MODEL_STRONG",
        "PROFILE_PATH",
        "OUTPUT_DIR",
        "DATABASE_PATH",
        "SOURCES_PATH",
    ]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)

    nonexistent_env_file = tmp_path / "nonexistent.env"
    settings = get_settings(env_file=nonexistent_env_file)

    assert settings.llm_api_key == ""
    assert settings.model_small == ""
    assert settings.model_strong == ""
    assert settings.profile_path == Path("data/profile.yaml")
    assert settings.output_dir == Path("data/output")
    assert settings.database_path == Path("data/tracker.db")
    assert settings.sources_path == Path("data/sources.yaml")
