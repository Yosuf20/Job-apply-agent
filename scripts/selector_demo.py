from jobagent.tailoring.selector import select_content, score_bullet
from jobagent.config import get_settings
from jobagent.discovery.service import discover, load_sources
from jobagent.llm.client import LLMTransientError, StructuredOutputError, get_client
from jobagent.models import MatchInput, load_profile, SelectionLimits
from jobagent.agents.extractor import ExtractorAgent
from jobagent.agents.matcher import MatcherAgent

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
            selected_content = select_content(profile, requirements, SelectionLimits(
                max_bullets_per_experience = 4,
                min_bullets_per_experience = 2,
                min_bullets_per_project = 1,
                max_bullets_per_project = 4,
                max_projects = 3,
                max_total_bullets = 10
            ))
        except (LLMTransientError, StructuredOutputError) as error:
                    print(f"{job.title}: FAILED - {error}")
                    continue
        print(f"{job.title}: {match.score}")
        print(f"  matched: {match.matched_skills}")
        print(f"  missing: {match.missing_skills}")
        print(f"  reasoning: {match.reasoning}")
        print(f" Selected Project and Experience: {selected_content.bullet_to_list()}")


if __name__ == "__main__":
    main()
            

