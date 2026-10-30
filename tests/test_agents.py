"""Tests for base Agent class."""

from pydantic import BaseModel, Field

from jobagent.agents.base import Agent
from jobagent.llm.client import LLMClient
from tests.conftest import FakeProvider


class DummyInput(BaseModel):
    """Input model for dummy agent."""

    job_title: str
    company: str


class DummyOutput(BaseModel):
    """Output model for dummy agent."""

    normalized_title: str
    is_technical: bool
    summary: str = Field(description="One-sentence summary")


class DummyRoleAgent(Agent[DummyInput, DummyOutput]):
    """Sample concrete agent subclass for testing."""

    name = "DummyRoleAgent"
    system_prompt = "You normalize job titles."
    tier = "small"
    input_type = DummyInput
    output_type = DummyOutput

    def build_prompt(self, input_data: DummyInput) -> str:
        return f"Company: {input_data.company}\nTitle: {input_data.job_title}"


def test_agent_runs_end_to_end_without_langgraph() -> None:
    """Agent runs end-to-end with FakeProvider and returns a validated output model."""
    canned_json = '{"normalized_title": "Senior Software Engineer", "is_technical": true, "summary": "Core engineering role"}'
    provider = FakeProvider(responses=[canned_json])
    client = LLMClient(provider=provider, model_small="test-haiku")

    agent = DummyRoleAgent(client=client)
    input_data = DummyInput(job_title="Sr Dev", company="Acme Corp")
    output = agent.run(input_data)

    assert isinstance(output, DummyOutput)
    assert output.normalized_title == "Senior Software Engineer"
    assert output.is_technical is True
    assert len(provider.calls) == 1
    assert "Company: Acme Corp" in provider.calls[0]["prompt"]
    assert provider.calls[0]["system"] == "You normalize job titles."
