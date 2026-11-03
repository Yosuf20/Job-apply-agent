"""Tests for MatcherAgent."""

from jobagent.agents.matcher import MatcherAgent
from jobagent.llm.client import LLMClient
from jobagent.models import JobRequirements, MatchInput, Profile, Skill
from tests.conftest import FakeProvider


def _profile() -> Profile:
    return Profile(
        name="Alex Morgan",
        email="alex@example.com",
        phone="555-1234",
        location="Private City",
        links=["https://private.example"],
        summary="",
        skills=[Skill(name="Python")],
    )


def _input() -> MatchInput:
    return MatchInput(
        profile=_profile(),
        requirements=JobRequirements(
            must_have_skills=["Python", "Kubernetes"],
            nice_to_have_skills=["AWS"],
            seniority="senior",
            role_type="backend",
        ),
        job_id="job-1",
        job_title="Senior Backend Engineer",
    )


def _agent(responses: list[str]) -> tuple[MatcherAgent, FakeProvider]:
    provider = FakeProvider(responses=responses)
    return MatcherAgent(
        LLMClient(provider=provider, model_small="test-small")
    ), provider


def test_prompt_excludes_personal_data_and_separates_missing_lists() -> None:
    agent, provider = _agent(['{"score": 72, "reasoning": "Good fit."}'])
    agent.run(_input())
    prompt = provider.calls[0]["prompt"]
    for private_value in [
        "Alex Morgan",
        "alex@example.com",
        "555-1234",
        "Private City",
        "private.example",
    ]:
        assert private_value not in prompt
    assert "Missing must-have skills: ['Kubernetes']" in prompt
    assert "Missing nice-to-have skills: ['AWS']" in prompt


def test_code_owns_matched_and_missing_lists() -> None:
    agent, _ = _agent(
        ['{"score": 72, "reasoning": "Some fit.", "matched_skills": ["AWS"]}']
    )
    result = agent.run(_input())
    assert result.matched_skills == ["Python"]
    assert result.missing_skills == ["Kubernetes", "AWS"]


def test_score_is_clamped_and_float_is_accepted() -> None:
    high_agent, _ = _agent(['{"score": 120, "reasoning": "Excellent."}'])
    low_agent, _ = _agent(['{"score": -5, "reasoning": "Weak."}'])
    assert high_agent.run(_input()).score == 100
    assert low_agent.run(_input()).score == 0


def test_invalid_json_uses_one_repair_retry() -> None:
    agent, provider = _agent(["not json", '{"score": 72.0, "reasoning": "Good fit."}'])
    assert agent.run(_input()).score == 72
    assert len(provider.calls) == 2
