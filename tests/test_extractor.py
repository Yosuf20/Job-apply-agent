"""Tests for the extractor agent."""

from jobagent.agents.extractor import ExtractorAgent
from jobagent.llm.client import LLMClient
from jobagent.models import Job, JobRequirements
from tests.conftest import FakeProvider


def _job(
    description: str = "Build backend services with Python, SQL, and AWS.",
) -> Job:
    return Job(
        id="greenhouse:board1:1",
        company="Acme",
        title="Senior Backend Engineer",
        location="Remote",
        url="https://example.com/job",
        description=description,
        source="greenhouse",
    )


def _agent(responses: list[str]) -> tuple[ExtractorAgent, FakeProvider]:
    provider = FakeProvider(responses=responses)
    client = LLMClient(provider=provider, model_small="test-small")
    return ExtractorAgent(client), provider


def test_valid_json_returns_job_requirements() -> None:
    agent, _ = _agent(
        [
            (
                '{"must_have_skills":["Python"],"nice_to_have_skills":["AWS"],'
                '"seniority":"senior","role_type":"backend","keywords":["APIs"]}'
            )
        ]
    )

    result = agent.run(_job())

    assert isinstance(result, JobRequirements)
    assert result.must_have_skills == ["Python"]
    assert result.nice_to_have_skills == ["AWS"]
    assert result.seniority == "senior"
    assert result.role_type == "backend"
    assert result.keywords == ["APIs"]


def test_json_code_fences_are_supported() -> None:
    agent, _ = _agent(
        [
            (
                '```json\n{"must_have_skills":[],"nice_to_have_skills":[],"seniority":'
                '"intern","role_type":"other","keywords":[]}\n```'
            )
        ]
    )

    assert agent.run(_job()).seniority == "intern"


def test_duplicates_cross_list_overlaps_and_keyword_skill_duplicates_are_cleaned() -> None:
    agent, _ = _agent(
        [
            (
                '{"must_have_skills":[" Python ","python","SQL"],'
                '"nice_to_have_skills":["sql"," AWS ","AWS",""],"seniority":"mid",'
                '"role_type":" backend ","keywords":[" APIs ","apis","python",""]}'
            )
        ]
    )

    result = agent.run(_job())

    assert result.must_have_skills == ["Python", "SQL"]
    assert result.nice_to_have_skills == ["AWS"]
    assert result.role_type == "backend"
    assert result.keywords == ["APIs"]


def test_ungrounded_skills_are_dropped_and_logged(caplog) -> None:
    agent, _ = _agent(
        [
            (
                '{"must_have_skills":["Python","Kubernetes"],'
                '"nice_to_have_skills":["Machine Learning"],"seniority":"mid",'
                '"role_type":"backend","keywords":[]}'
            )
        ]
    )

    with caplog.at_level("INFO", logger="jobagent.agents.extractor"):
        result = agent.run(
            _job(
                "Build backend services with Python, machine vision, and "
                "learning systems."
            )
        )

    assert result.must_have_skills == ["Python"]
    assert result.nice_to_have_skills == ["Machine Learning"]
    assert "Kubernetes" in caplog.text


def test_invalid_seniority_becomes_unspecified() -> None:
    agent, _ = _agent(
        [
            (
                '{"must_have_skills":[],"nice_to_have_skills":[],"seniority":"principal",'
                '"role_type":"other","keywords":[]}'
            )
        ]
    )

    assert agent.run(_job()).seniority == "unspecified"


def test_long_description_is_truncated() -> None:
    agent, provider = _agent(
        [
            (
                '{"must_have_skills":[],"nice_to_have_skills":[],"seniority":"unspecified",'
                '"role_type":"other","keywords":[]}'
            )
        ]
    )

    agent = ExtractorAgent(agent.client, description_limit=20)
    agent.run(_job("A" * 100))

    prompt = provider.calls[0]["prompt"]
    assert "A" * 20 in prompt
    assert "A" * 21 not in prompt
    assert "truncated to 20 characters" in prompt


def test_invalid_json_triggers_one_repair_retry() -> None:
    agent, provider = _agent(
        [
            "not json",
            (
                '{"must_have_skills":["Python"],"nice_to_have_skills":[],"seniority":'
                '"junior","role_type":"backend","keywords":[]}'
            ),
        ]
    )

    result = agent.run(_job())

    assert result.must_have_skills == ["Python"]
    assert len(provider.calls) == 2
