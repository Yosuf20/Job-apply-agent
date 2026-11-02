"""Manually extract requirements from the first three configured jobs."""

from __future__ import annotations

from jobagent.agents.extractor import ExtractorAgent
from jobagent.config import get_settings
from jobagent.discovery.service import discover, load_sources
from jobagent.llm.client import get_client


def main() -> None:
    settings = get_settings()
    sources = load_sources(settings.sources_path)
    result = discover(sources)
    extractor = ExtractorAgent(get_client(settings))

    for job in result.jobs[:3]:
        print(f"{job.title} - {job.company}")
        print(extractor.run(job).model_dump_json(indent=2))
        print()


if __name__ == "__main__":
    main()
