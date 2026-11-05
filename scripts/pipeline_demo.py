"""Manually run discovery, extraction, matching, and tracker persistence."""

from __future__ import annotations

import warnings
from pathlib import Path

from jobagent.agents.extractor import ExtractorAgent
from jobagent.agents.matcher import MatcherAgent
from jobagent.config import get_settings
from jobagent.discovery.service import discover, load_sources
from jobagent.llm.client import LLMError, get_client
from jobagent.models import MatchInput, load_profile
from jobagent.tracking.db import create_engine, create_session_factory, init_db
from jobagent.tracking.hashing import file_sha256, text_sha256
from jobagent.tracking.repository import TrackerRepository


def main() -> None:
    settings = get_settings()
    engine = create_engine(settings)
    init_db(engine)
    repository = TrackerRepository(create_session_factory(engine))

    discovery = discover(load_sources(settings.sources_path))
    repository.upsert_jobs(discovery.jobs)
    if discovery.errors:
        for error in discovery.errors:
            print(f"Discovery warning: {error}")

    client = get_client(settings)
    extractor = ExtractorAgent(client)
    extraction_jobs = repository.jobs_needing_extraction()[:3]
    for job in extraction_jobs:
        try:
            requirements = extractor.run(job)
            repository.save_requirements(
                job.id,
                requirements,
                text_sha256(extractor.build_prompt(job)),
                settings.model_small,
            )
        except (LLMError, ValueError) as error:
            print(f"Extraction failed for {job.id}: {error}")

    profile_path = Path(settings.profile_path)
    if not profile_path.is_file():
        profile_path = Path("data/profile.example.yaml")
        warnings.warn(
            f"Profile not found; using example profile at {profile_path}.",
            stacklevel=1,
        )
    profile = load_profile(profile_path)
    profile_hash = file_sha256(profile_path)
    matcher = MatcherAgent(client)
    for job in repository.jobs_needing_match(profile_hash)[:3]:
        try:
            requirements = repository.get_requirements(job.id)
            if requirements is None:
                print(f"Match skipped for {job.id}: requirements are missing.")
                continue
            match_input = MatchInput(
                profile=profile,
                requirements=requirements,
                job_id=job.id,
                job_title=job.title,
            )
            result = matcher.run(match_input)
            repository.save_match(
                result,
                profile_hash,
                text_sha256(matcher.build_prompt(match_input)),
                settings.model_small,
            )
        except (LLMError, ValueError) as error:
            print(f"Match failed for {job.id}: {error}")

    for summary in repository.list_jobs():
        print(
            f"{summary.job_id}\t{summary.company}\t{summary.title}\t"
            f"{summary.status}\t{summary.latest_score}"
        )


if __name__ == "__main__":
    main()
