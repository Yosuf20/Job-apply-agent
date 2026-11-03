"""Manually run discovery, extraction, and matching for three jobs."""

from __future__ import annotations

from jobagent.agents.extractor import ExtractorAgent
from jobagent.agents.matcher import MatcherAgent
from jobagent.config import get_settings
from jobagent.discovery.service import discover, load_sources
from jobagent.llm.client import LLMTransientError, StructuredOutputError, get_client
from jobagent.models import MatchInput, load_profile


def main() -> None:
    settings = get_settings()
    profile = load_profile(settings.profile_path)
    result = discover(load_sources(settings.sources_path))
    client = get_client(settings)
    extractor = ExtractorAgent(client)
    matcher = MatcherAgent(client)

    for job in result.jobs[:3]:
        try:
            requirements = extractor.run(job)
            match = matcher.run(
                MatchInput(
                    profile=profile,
                    requirements=requirements,
                    job_id=job.id,
                    job_title=job.title,
                )
            )
        except (LLMTransientError, StructuredOutputError) as error:
            print(f"{job.title}: FAILED - {error}")
            continue
        print(f"{job.title}: {match.score}")
        print(f"  matched: {match.matched_skills}")
        print(f"  missing: {match.missing_skills}")
        print(f"  reasoning: {match.reasoning}")


if __name__ == "__main__":
    main()
