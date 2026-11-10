import pytest
from jobagent.models import Bullet, Experience, JobRequirements, Profile
from jobagent.tailoring.selector import SelectionLimits, score_bullet, select_content


# ---------- helpers ----------

def bullet(bid: str, text: str, tags: list[str] | None = None) -> Bullet:
    return Bullet(id=bid, text=text, tags=tags or [], metrics=None)


def job(company: str, bullets: list[Bullet]) -> Experience:
    return Experience(
        company=company, title="Engineer", start="2020-01", end=None, bullets=bullets
    )


def make_profile(experiences: list[Experience]) -> Profile:
    return Profile(
        name="Test Person",
        email="test@example.com",
        phone="000",
        location="Nowhere",
        links=[],
        summary="",
        skills=[],
        experience=experiences,
        projects=[],
        education=[],
    )


def make_req() -> JobRequirements:
    # "alpha", "beta", "gamma" are must-haves, so a bullet's score grows with
    # how many of the three words it contains.
    return JobRequirements(
        must_have_skills=["alpha", "beta", "gamma"],
        nice_to_have_skills=[],
        seniority="mid",
        role_type="backend",
        keywords=[],
    )


def limits(per_exp: int = 4, total: int = 14) -> SelectionLimits:
    return SelectionLimits(
        max_bullets_per_experience=per_exp,
        max_bullets_per_project=2,
        max_projects=3,
        max_total_bullets=total,
    )


def ids_by_job(selection) -> list[list[str]]:
    return [[b.source_id for b in e.bullets] for e in selection.selected_experience]


def all_ids(selection) -> list[str]:
    return [i for job_ids in ids_by_job(selection) for i in job_ids]


# ---------- cycle 1: guaranteed bullet ----------

def test_every_job_keeps_one_bullet_even_when_all_score_zero():
    profile = make_profile([
        job("A", [bullet("a1", "nothing relevant"), bullet("a2", "still nothing")]),
        job("B", [bullet("b1", "unrelated work")]),
    ])
    result = select_content(profile, make_req(), limits())
    assert [len(ids) for ids in ids_by_job(result)] == [1, 1]


def test_all_zero_job_keeps_exactly_one_and_the_earlier_one_wins_the_tie():
    profile = make_profile([
        job("A", [bullet("a1", "nothing"), bullet("a2", "nothing"), bullet("a3", "nothing")]),
    ])
    result = select_content(profile, make_req(), limits())
    assert ids_by_job(result) == [["a1"]]


def test_job_keeps_its_highest_scoring_bullet():
    profile = make_profile([
        job("A", [bullet("a1", "alpha"), bullet("a2", "alpha beta gamma"), bullet("a3", "alpha beta")]),
    ])
    result = select_content(profile, make_req(), limits(per_exp=1))
    assert ids_by_job(result) == [["a2"]]


def test_job_with_a_single_bullet_works():
    profile = make_profile([job("A", [bullet("a1", "alpha")])])
    result = select_content(profile, make_req(), limits())
    assert ids_by_job(result) == [["a1"]]


# ---------- cycle 2: pool and per-job limit ----------

def test_per_job_limit_is_respected_and_best_bullets_win():
    profile = make_profile([
        job("A", [
            bullet("a1", "alpha"),
            bullet("a2", "alpha beta gamma"),
            bullet("a3", "alpha beta"),
            bullet("a4", "nothing"),
        ]),
    ])
    result = select_content(profile, make_req(), limits(per_exp=2))
    assert sorted(ids_by_job(result)[0]) == ["a2", "a3"]


def test_zero_score_bullets_are_not_added_from_the_pool():
    profile = make_profile([
        job("A", [bullet("a1", "alpha"), bullet("a2", "nothing relevant")]),
    ])
    result = select_content(profile, make_req(), limits(per_exp=4))
    assert ids_by_job(result) == [["a1"]]


def test_no_bullet_is_selected_twice():
    profile = make_profile([
        job("A", [bullet("a1", "alpha beta"), bullet("a2", "alpha")]),
        job("B", [bullet("b1", "alpha gamma"), bullet("b2", "beta")]),
    ])
    result = select_content(profile, make_req(), limits())
    ids = all_ids(result)
    assert len(ids) == len(set(ids))


# ---------- cycle 3: total limit ----------

def test_total_limit_counts_guaranteed_bullets_and_best_pool_bullet_gets_the_slot():
    profile = make_profile([
        job("A", [bullet("a1", "alpha beta gamma"), bullet("a2", "alpha beta"), bullet("a3", "alpha")]),
        job("B", [bullet("b1", "alpha beta"), bullet("b2", "alpha")]),
        job("C", [bullet("c1", "alpha")]),
    ])
    # Guaranteed: a1, b1, c1 (3 of 4). One slot left: a2 (two skills) beats a3 and b2.
    result = select_content(profile, make_req(), limits(per_exp=3, total=4))
    assert sorted(all_ids(result)) == ["a1", "a2", "b1", "c1"]


def test_higher_score_wins_the_last_slot_across_jobs():
    profile = make_profile([
        job("A", [bullet("a1", "alpha beta gamma"), bullet("a2", "alpha")]),
        job("B", [bullet("b1", "alpha beta"), bullet("b2", "alpha beta")]),
    ])
    result = select_content(profile, make_req(), limits(per_exp=4, total=3))
    assert sorted(all_ids(result)) == ["a1", "b1", "b2"]


def test_total_never_exceeds_the_limit():
    profile = make_profile([
        job("A", [bullet(f"a{i}", "alpha beta") for i in range(5)]),
        job("B", [bullet(f"b{i}", "alpha beta") for i in range(5)]),
    ])
    result = select_content(profile, make_req(), limits(per_exp=5, total=6))
    assert len(all_ids(result)) <= 6


# ---------- cycle 4: original order ----------

def test_best_bullet_that_is_last_in_the_profile_stays_last():
    profile = make_profile([
        job("A", [bullet("a1", "alpha"), bullet("a2", "nothing"), bullet("a3", "alpha beta gamma")]),
    ])
    result = select_content(profile, make_req(), limits(per_exp=2))
    assert ids_by_job(result) == [["a1", "a3"]]


def test_equal_scores_keep_profile_order():
    profile = make_profile([
        job("A", [bullet("a1", "alpha"), bullet("a2", "alpha"), bullet("a3", "alpha")]),
    ])
    result = select_content(profile, make_req(), limits(per_exp=2))
    assert ids_by_job(result) == [["a1", "a2"]]


def test_job_order_is_unchanged():
    profile = make_profile([
        job("First", [bullet("f1", "nothing")]),
        job("Second", [bullet("s1", "alpha beta gamma")]),
        job("Third", [bullet("t1", "alpha")]),
    ])
    result = select_content(profile, make_req(), limits())
    assert [e.company for e in result.selected_experience] == ["First", "Second", "Third"]


# ---------- data carried through ----------

def test_selected_bullet_keeps_source_id_original_text_and_score():
    original = bullet("a1", "alpha beta")
    profile = make_profile([job("A", [original])])
    req = make_req()
    result = select_content(profile, req, limits())
    chosen = result.selected_experience[0].bullets[0]
    assert chosen.source_id == "a1"
    assert chosen.text == "alpha beta"
    assert chosen.score == score_bullet(original, req)


def test_company_title_and_dates_are_copied():
    profile = make_profile([job("Acme", [bullet("a1", "alpha")])])
    result = select_content(profile, make_req(), limits())
    entry = result.selected_experience[0]
    assert (entry.company, entry.title, entry.start, entry.end) == ("Acme", "Engineer", "2020-01", None)


# ---------- safety ----------

def test_input_profile_is_not_modified():
    profile = make_profile([
        job("A", [bullet("a1", "alpha"), bullet("a2", "alpha beta gamma"), bullet("a3", "nothing")]),
    ])
    before = profile.model_dump()
    select_content(profile, make_req(), limits(per_exp=1))
    assert profile.model_dump() == before


def test_same_input_gives_same_output():
    profile = make_profile([
        job("A", [bullet("a1", "alpha"), bullet("a2", "alpha beta")]),
        job("B", [bullet("b1", "beta")]),
    ])
    req = make_req()
    first = select_content(profile, req, limits())
    second = select_content(profile, req, limits())
    assert first.model_dump() == second.model_dump()