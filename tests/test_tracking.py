from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from jobagent.models import Job, JobRequirements, MatchResult
from jobagent.tracking import TrackerRepository, init_db
from jobagent.tracking import db as db_module
from jobagent.tracking.hashing import file_sha256, text_sha256
from jobagent.tracking.tables import (
    ApplicationStatus,
    ApplicationTable,
    JobRequirementTable,
    JobTable,
    MatchResultTable,
)


@pytest.fixture
def session_factory() -> sessionmaker[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    yield factory
    engine.dispose()


@pytest.fixture
def repository(session_factory: sessionmaker[Session]) -> TrackerRepository:
    return TrackerRepository(session_factory)


def make_job(job_id: str = "job-1", title: str = "Backend Engineer") -> Job:
    return Job(
        id=job_id,
        company="Example Co",
        title=title,
        location="Remote",
        url=f"https://example.test/{job_id}",
        description="Python and SQL experience required.",
        source="greenhouse",
        posted_at="2026-11-04T00:00:00Z",
    )


def make_requirements() -> JobRequirements:
    return JobRequirements(
        must_have_skills=["Python"],
        nice_to_have_skills=["SQL"],
        seniority="mid",
        role_type="full-time",
        keywords=["backend"],
    )


def make_match(job_id: str = "job-1", score: int = 80) -> MatchResult:
    return MatchResult(
        job_id=job_id,
        score=score,
        reasoning=f"score {score}",
        matched_skills=["Python"],
        missing_skills=["Kubernetes"],
    )


def test_init_db_is_idempotent_and_creates_all_tables() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    init_db(engine)
    assert set(inspect(engine).get_table_names()) == {
        "jobs",
        "job_requirements",
        "match_results",
        "applications",
    }
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
    engine.dispose()


def test_create_engine_makes_database_parent(tmp_path) -> None:
    settings = db_module.Settings(database_path=tmp_path / "nested" / "tracker.db")
    engine = db_module.create_engine(settings)
    assert (tmp_path / "nested").is_dir()
    engine.dispose()


def test_upsert_job_is_idempotent_and_updates_posting(
    repository: TrackerRepository, session_factory: sessionmaker[Session]
) -> None:
    job = make_job()
    assert repository.upsert_job(job) is True
    with session_factory() as session:
        first = session.get(JobTable, job.id)
        assert first is not None
        first_seen = first.first_seen_at
        last_seen = first.last_seen_at
        assert first_seen.tzinfo == UTC
        assert last_seen.tzinfo == UTC

    updated = job.model_copy(update={"title": "Senior Backend Engineer"})
    assert repository.upsert_job(updated) is False
    with session_factory() as session:
        row = session.get(JobTable, job.id)
        application = session.get(ApplicationTable, job.id)
        assert row is not None
        assert row.first_seen_at == first_seen
        assert row.last_seen_at > last_seen
        assert row.title == "Senior Backend Engineer"
        assert application is not None
        assert application.status == ApplicationStatus.DISCOVERED


def test_upsert_jobs_counts_new_and_existing(repository: TrackerRepository) -> None:
    assert repository.upsert_jobs([make_job("a"), make_job("b")]) == (2, 0)
    assert repository.upsert_jobs([make_job("b"), make_job("c")]) == (1, 1)


def test_requirements_and_matches_round_trip_json(
    repository: TrackerRepository,
) -> None:
    repository.upsert_job(make_job())
    requirements = make_requirements()
    repository.save_requirements("job-1", requirements, "prompt", "model")
    repository.save_match(make_match(), "profile-hash", "prompt", "model")
    assert repository.get_requirements("job-1") == requirements
    assert repository.latest_match("job-1") == make_match()


def test_match_history_uses_id_not_timestamp(
    repository: TrackerRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository.upsert_job(make_job())
    fixed = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("jobagent.tracking.repository._utc_now", lambda: fixed)
    repository.save_match(make_match(score=10), "profile", "prompt", "model")
    repository.save_match(make_match(score=90), "profile", "prompt", "model")
    assert repository.latest_match("job-1") == make_match(score=90)
    assert len(repository.list_jobs()) == 1
    assert repository.list_jobs()[0].latest_score == 90


def test_status_transitions_and_unknown_job(repository: TrackerRepository) -> None:
    repository.upsert_job(make_job())
    repository.set_status("job-1", "shortlisted")
    repository.set_status("job-1", "rejected")
    repository.set_status("job-1", "shortlisted")
    repository.set_status("job-1", "skipped")
    with pytest.raises(ValueError, match="Illegal status move"):
        repository.set_status("job-1", "tailored")
    repository.set_status("job-1", "discovered")
    repository.set_status("job-1", "tailored")
    with pytest.raises(ValueError, match="Unknown job"):
        repository.set_status("missing", "discovered")
    with pytest.raises(ValueError, match="Unknown application status"):
        repository.set_status("job-1", "not-a-status")


def test_status_blocks_backward_moves_and_allows_terminal_moves(
    repository: TrackerRepository,
) -> None:
    repository.upsert_job(make_job())
    for status in ("shortlisted", "tailored", "approved"):
        repository.set_status("job-1", status)
    with pytest.raises(ValueError, match="Illegal status move"):
        repository.set_status("job-1", "shortlisted")
    repository.set_status("job-1", "skipped")
    with pytest.raises(ValueError, match="Illegal status move"):
        repository.set_status("job-1", "tailored")


def test_list_jobs_filters_and_sorts(repository: TrackerRepository) -> None:
    repository.upsert_jobs([make_job("low"), make_job("high"), make_job("none")])
    repository.save_match(make_match("low", 60), "p1", "p", "m")
    repository.save_match(make_match("high", 90), "p1", "p", "m")
    repository.set_status("high", "shortlisted")
    summaries = repository.list_jobs()
    assert [summary.job_id for summary in summaries] == ["high", "low", "none"]
    assert [summary.job_id for summary in repository.list_jobs(min_score=70)] == [
        "high"
    ]
    assert [
        summary.job_id for summary in repository.list_jobs(status="shortlisted")
    ] == ["high"]


def test_jobs_needing_extraction_and_match(repository: TrackerRepository) -> None:
    repository.upsert_jobs([make_job("a"), make_job("b")])
    repository.save_requirements("a", make_requirements(), "p", "m")
    assert [job.id for job in repository.jobs_needing_extraction()] == ["b"]
    assert [job.id for job in repository.jobs_needing_match("profile")] == ["a"]
    repository.save_match(make_match("a"), "profile", "p", "m")
    assert repository.jobs_needing_match("profile") == []
    assert [job.id for job in repository.jobs_needing_match("other")] == ["a"]


def test_delete_job_removes_all_dependent_rows(
    repository: TrackerRepository, session_factory: sessionmaker[Session]
) -> None:
    repository.upsert_job(make_job())
    repository.save_requirements("job-1", make_requirements(), "p", "m")
    repository.save_match(make_match(), "profile", "p", "m")
    repository.delete_job("job-1")
    with session_factory() as session:
        assert session.execute(select(JobTable)).all() == []
        assert session.execute(select(JobRequirementTable)).all() == []
        assert session.execute(select(MatchResultTable)).all() == []
        assert session.execute(select(ApplicationTable)).all() == []


def test_database_does_not_store_profile_personal_data(
    repository: TrackerRepository, session_factory: sessionmaker[Session], tmp_path
) -> None:
    profile_file = tmp_path / "profile.yaml"
    distinctive = (
        "Distinctive Fake Name",
        "fake-email-privacy@example.test",
        "+15550199",
    )
    profile_file.write_text(
        f"name: {distinctive[0]}\nemail: {distinctive[1]}\nphone: '{distinctive[2]}'\n",
        encoding="utf-8",
    )
    repository.upsert_job(make_job())
    repository.save_match(make_match(), file_sha256(profile_file), "p", "m")

    with session_factory() as session:
        for table in (
            JobTable,
            JobRequirementTable,
            MatchResultTable,
            ApplicationTable,
        ):
            for row in session.execute(select(table)).scalars():
                serialized = json.dumps(
                    {
                        column.name: getattr(row, column.name)
                        for column in table.__table__.columns
                    },
                    default=str,
                )
                assert all(value not in serialized for value in distinctive)


def test_hash_helpers(tmp_path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("hello", encoding="utf-8")
    assert file_sha256(path) == text_sha256("hello")
