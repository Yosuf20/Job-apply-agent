"""Tests for pure skill matching."""

from jobagent.matching.skills import (
    compare_skills,
    coverage,
    normalize,
    profile_skill_index,
    skill_in_profile,
)
from jobagent.models import (
    Bullet,
    Experience,
    JobRequirements,
    Profile,
    Project,
    Skill,
)


def _profile() -> Profile:
    return Profile(
        name="Alex Morgan",
        email="alex@example.com",
        phone="555",
        location="Secret Location",
        summary="",
        skills=[Skill(name="Python", tags=["language"]), Skill(name="JavaScript")],
        experience=[
            Experience(
                company="Fake",
                title="Engineer",
                start="2020",
                bullets=[
                    Bullet(
                        id="b1", text="Built CI/CD pipelines with C++.", tags=["devops"]
                    )
                ],
            )
        ],
        projects=[
            Project(
                name="Project",
                description="",
                tech=["PostgreSQL"],
                bullets=[Bullet(id="b2", text="Used React for dashboards.")],
            )
        ],
    )


def test_normalize_preserves_special_skill_tokens() -> None:
    assert normalize("C++") == "cplusplus"
    assert normalize("C#") == "csharp"
    assert normalize("CI/CD") == "cicd"


def test_whole_word_matching_avoids_java_and_sql_false_positives() -> None:
    index = {"javascript", "nosql"}
    assert not skill_in_profile("Java", index)
    assert not skill_in_profile("SQL", index)


def test_matches_skill_name_tag_project_tech_and_bullet_text() -> None:
    profile = _profile()
    index = profile_skill_index(profile)
    assert skill_in_profile("Python", index)
    assert skill_in_profile("DevOps", index)
    assert skill_in_profile("PostgreSQL", index)
    assert skill_in_profile("React", index)
    assert skill_in_profile("C++", index)
    assert skill_in_profile("CI/CD", index)


def test_compare_preserves_spelling_order_and_removes_duplicates() -> None:
    matched, missing = compare_skills(
        [" python ", "Python", "PostgreSQL", "Missing"], _profile()
    )
    assert matched == [" python ", "PostgreSQL"]
    assert missing == ["Missing"]


def test_coverage_with_no_must_haves_is_one() -> None:
    requirements = JobRequirements(
        seniority="unspecified", role_type="other", must_have_skills=[]
    )
    assert coverage(requirements, _profile()) == 1.0
