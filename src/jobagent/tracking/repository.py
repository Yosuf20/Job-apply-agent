"""Repository operations for the tracker database."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import delete, desc, func, select
from sqlalchemy.orm import Session, sessionmaker

from jobagent.models import Job, JobRequirements, JobSummary, MatchResult
from jobagent.tracking.tables import (
    ApplicationStatus,
    ApplicationTable,
    JobRequirementTable,
    JobTable,
    MatchResultTable,
)

SessionFactory = sessionmaker[Session]
_FORWARD_STATUSES: tuple[ApplicationStatus, ...] = (
    ApplicationStatus.DISCOVERED,
    ApplicationStatus.SHORTLISTED,
    ApplicationStatus.TAILORED,
    ApplicationStatus.APPROVED,
    ApplicationStatus.APPLIED,
    ApplicationStatus.INTERVIEW,
    ApplicationStatus.OFFER,
)
_REOPENABLE_STATUSES = frozenset(
    {ApplicationStatus.DISCOVERED, ApplicationStatus.SHORTLISTED}
)
_TERMINAL_STATUSES = frozenset({ApplicationStatus.REJECTED, ApplicationStatus.SKIPPED})


def _utc_now() -> datetime:
    return datetime.now(UTC)


class TrackerRepository:
    """Typed persistence operations for jobs and their pipeline outputs."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self.session_factory = session_factory

    def upsert_job(self, job: Job) -> bool:
        """Insert a job or update its latest posting fields."""
        with self.session_factory() as session, session.begin():
            existing = session.get(JobTable, job.id)
            now = _utc_now()
            if existing is None:
                session.add(
                    JobTable(
                        id=job.id,
                        company=job.company,
                        title=job.title,
                        location=job.location,
                        url=job.url,
                        description=job.description,
                        source=job.source,
                        posted_at=job.posted_at,
                        first_seen_at=now,
                        last_seen_at=now,
                    )
                )
                session.flush()
                session.add(
                    ApplicationTable(
                        job_id=job.id,
                        status=ApplicationStatus.DISCOVERED,
                        updated_at=now,
                    )
                )
                return True

            existing.company = job.company
            existing.title = job.title
            existing.location = job.location
            existing.url = job.url
            existing.description = job.description
            existing.source = job.source
            existing.posted_at = job.posted_at
            existing.last_seen_at = now
            return False

    def upsert_jobs(self, jobs: Sequence[Job]) -> tuple[int, int]:
        """Upsert jobs and return counts for new and existing rows."""
        new_count = sum(self.upsert_job(job) for job in jobs)
        return new_count, len(jobs) - new_count

    def save_requirements(
        self,
        job_id: str,
        requirements: JobRequirements,
        prompt_hash: str,
        model_name: str,
    ) -> None:
        """Save or replace the latest extracted requirements for a job."""
        with self.session_factory() as session, session.begin():
            self._require_job(session, job_id)
            row = session.get(JobRequirementTable, job_id)
            values = {
                "data": requirements.model_dump(mode="json"),
                "prompt_hash": prompt_hash,
                "model_name": model_name,
                "created_at": _utc_now(),
            }
            if row is None:
                session.add(JobRequirementTable(job_id=job_id, **values))
            else:
                for key, value in values.items():
                    setattr(row, key, value)

    def get_requirements(self, job_id: str) -> JobRequirements | None:
        """Return the latest requirements for a job, if extracted."""
        with self.session_factory() as session:
            row = session.get(JobRequirementTable, job_id)
            return JobRequirements.model_validate(row.data) if row else None

    def save_match(
        self,
        result: MatchResult,
        profile_hash: str,
        prompt_hash: str,
        model_name: str,
    ) -> None:
        """Append a match result, retaining all prior results."""
        with self.session_factory() as session, session.begin():
            self._require_job(session, result.job_id)
            session.add(
                MatchResultTable(
                    job_id=result.job_id,
                    profile_hash=profile_hash,
                    score=result.score,
                    reasoning=result.reasoning,
                    matched_skills=result.matched_skills,
                    missing_skills=result.missing_skills,
                    prompt_hash=prompt_hash,
                    model_name=model_name,
                    created_at=_utc_now(),
                )
            )

    def latest_match(self, job_id: str) -> MatchResult | None:
        """Return the newest match by database insertion ID."""
        with self.session_factory() as session:
            row = session.scalars(
                select(MatchResultTable)
                .where(MatchResultTable.job_id == job_id)
                .order_by(desc(MatchResultTable.id))
                .limit(1)
            ).first()
            return self._to_match(row) if row else None

    def set_status(
        self,
        job_id: str,
        status: ApplicationStatus | str,
        notes: str | None = None,
    ) -> None:
        """Change application status if the transition is allowed."""
        target = self._parse_status(status)
        with self.session_factory() as session, session.begin():
            self._require_job(session, job_id)
            application = session.get(ApplicationTable, job_id)
            if application is None:
                raise ValueError(f"No application row exists for job '{job_id}'.")
            current = self._parse_status(application.status)
            if not self._status_allowed(current, target):
                raise ValueError(
                    f"Illegal status move for job '{job_id}': "
                    f"{current.value} -> {target.value}."
                )
            application.status = target
            if notes is not None:
                application.notes = notes
            application.updated_at = _utc_now()

    def list_jobs(
        self,
        status: ApplicationStatus | str | None = None,
        min_score: int | None = None,
        limit: int = 50,
    ) -> list[JobSummary]:
        """List jobs with optional status and latest-score filters."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        parsed_status = self._parse_status(status) if status is not None else None
        latest_id = (
            select(
                MatchResultTable.job_id,
                func.max(MatchResultTable.id).label("latest_id"),
            )
            .group_by(MatchResultTable.job_id)
            .subquery()
        )
        with self.session_factory() as session:
            query = (
                select(
                    JobTable,
                    ApplicationTable.status,
                    MatchResultTable.score,
                )
                .join(ApplicationTable, ApplicationTable.job_id == JobTable.id)
                .outerjoin(latest_id, latest_id.c.job_id == JobTable.id)
                .outerjoin(
                    MatchResultTable,
                    MatchResultTable.id == latest_id.c.latest_id,
                )
                .order_by(
                    desc(MatchResultTable.score).nullslast(),
                    desc(JobTable.last_seen_at),
                )
                .limit(limit)
            )
            if parsed_status is not None:
                query = query.where(ApplicationTable.status == parsed_status)
            if min_score is not None:
                query = query.where(MatchResultTable.score >= min_score)

            rows = session.execute(query).all()
            return [
                JobSummary(
                    job_id=job.id,
                    company=job.company,
                    title=job.title,
                    status=self._parse_status(app_status).value,
                    latest_score=score,
                )
                for job, app_status, score in rows
            ]

    def jobs_needing_extraction(self) -> list[Job]:
        """Return jobs without stored requirements."""
        with self.session_factory() as session:
            rows = session.scalars(
                select(JobTable)
                .outerjoin(
                    JobRequirementTable,
                    JobRequirementTable.job_id == JobTable.id,
                )
                .where(JobRequirementTable.job_id.is_(None))
                .order_by(JobTable.id)
            ).all()
            return [self._to_job(row) for row in rows]

    def jobs_needing_match(self, profile_hash: str) -> list[Job]:
        """Return jobs without a match for the requested profile version."""
        matching_jobs = (
            select(MatchResultTable.job_id)
            .where(MatchResultTable.profile_hash == profile_hash)
            .distinct()
        )
        with self.session_factory() as session:
            rows = session.scalars(
                select(JobTable)
                .join(JobRequirementTable, JobRequirementTable.job_id == JobTable.id)
                .where(JobTable.id.not_in(matching_jobs))
                .order_by(JobTable.id)
            ).all()
            return [self._to_job(row) for row in rows]

    def delete_job(self, job_id: str) -> None:
        """Delete a job and all explicitly related tracker rows."""
        with self.session_factory() as session, session.begin():
            self._require_job(session, job_id)
            session.execute(
                delete(JobRequirementTable).where(JobRequirementTable.job_id == job_id)
            )
            session.execute(
                delete(MatchResultTable).where(MatchResultTable.job_id == job_id)
            )
            session.execute(
                delete(ApplicationTable).where(ApplicationTable.job_id == job_id)
            )
            session.execute(delete(JobTable).where(JobTable.id == job_id))

    @staticmethod
    def _require_job(session: Session, job_id: str) -> JobTable:
        job = session.get(JobTable, job_id)
        if job is None:
            raise ValueError(f"Unknown job '{job_id}'.")
        return job

    @staticmethod
    def _parse_status(status: ApplicationStatus | str) -> ApplicationStatus:
        try:
            return (
                status
                if isinstance(status, ApplicationStatus)
                else ApplicationStatus(status)
            )
        except ValueError as error:
            raise ValueError(f"Unknown application status: {status!r}.") from error

    @staticmethod
    def _status_allowed(current: ApplicationStatus, target: ApplicationStatus) -> bool:
        if current == target:
            return True
        if target in _TERMINAL_STATUSES:
            return True
        if current in _TERMINAL_STATUSES:
            return target in _REOPENABLE_STATUSES
        try:
            return _FORWARD_STATUSES.index(target) > _FORWARD_STATUSES.index(current)
        except ValueError:
            return False

    @staticmethod
    def _to_job(row: JobTable) -> Job:
        return Job(
            id=row.id,
            company=row.company,
            title=row.title,
            location=row.location,
            url=row.url,
            description=row.description,
            source=row.source,
            posted_at=row.posted_at,
        )

    @staticmethod
    def _to_match(row: MatchResultTable) -> MatchResult:
        return MatchResult(
            job_id=row.job_id,
            score=row.score,
            reasoning=row.reasoning,
            matched_skills=list(row.matched_skills),
            missing_skills=list(row.missing_skills),
        )
