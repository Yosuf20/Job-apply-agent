"""Match extracted job requirements against a candidate profile."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from jobagent.agents.base import Agent
from jobagent.matching.skills import compare_skills
from jobagent.models import MatchInput, MatchResult, Profile

_PROMPT_PATH = Path(__file__).parent / "prompts" / "matcher.txt"


class _MatcherOutput(BaseModel):
    """The only fields the language model is allowed to decide."""

    score: float = Field(ge=-1000, le=1000)
    reasoning: str


class MatcherAgent(Agent[MatchInput, MatchResult]):
    """Use an LLM only for fit judgment and reasoning."""

    name = "MatcherAgent"
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    tier = "small"
    input_type = MatchInput
    output_type = _MatcherOutput

    def __init__(self, client: Any, *, summary_limit: int = 6000) -> None:
        if summary_limit < 1:
            raise ValueError("summary_limit must be at least 1")
        super().__init__(client)
        self.summary_limit = summary_limit

    def build_prompt(self, input_data: MatchInput) -> str:
        """Build a compact prompt without personal profile information."""
        must_matched, must_missing = compare_skills(
            input_data.requirements.must_have_skills, input_data.profile
        )
        nice_matched, nice_missing = compare_skills(
            input_data.requirements.nice_to_have_skills, input_data.profile
        )
        return (
            f"Job title: {input_data.job_title}\n"
            f"Seniority: {input_data.requirements.seniority}\n"
            f"Role type: {input_data.requirements.role_type}\n"
            f"Must-have skills: {input_data.requirements.must_have_skills}\n"
            f"Nice-to-have skills: {input_data.requirements.nice_to_have_skills}\n"
            f"Matched must-have skills: {must_matched}\n"
            f"Missing must-have skills: {must_missing}\n"
            f"Matched nice-to-have skills: {nice_matched}\n"
            f"Missing nice-to-have skills: {nice_missing}\n"
            f"Profile summary:\n{build_profile_summary(input_data.profile, self.summary_limit)}"
        )

    def run(self, input_data: MatchInput) -> MatchResult:
        """Compute skill lists in code and merge them with model judgment."""
        matched_must, missing_must = compare_skills(
            input_data.requirements.must_have_skills, input_data.profile
        )
        matched_nice, missing_nice = compare_skills(
            input_data.requirements.nice_to_have_skills, input_data.profile
        )
        model_output = super().run(input_data)
        score = max(0, min(100, round(model_output.score)))
        return MatchResult(
            job_id=input_data.job_id,
            score=score,
            reasoning=model_output.reasoning,
            matched_skills=[*matched_must, *matched_nice],
            missing_skills=[*missing_must, *missing_nice],
        )


def build_profile_summary(profile: Profile, limit: int = 6000) -> str:
    """Build a privacy-safe summary from skills, titles, dates, and bullets."""
    sections = ["Skills: " + ", ".join(skill.name for skill in profile.skills)]
    for experience in profile.experience:
        sections.append(
            f"Role: {experience.title} ({experience.start}-{experience.end or 'present'})"
        )
        sections.extend(f"Bullet: {bullet.text}" for bullet in experience.bullets)
    for project in profile.projects:
        sections.append(f"Project: {project.name}")
        sections.extend(f"Bullet: {bullet.text}" for bullet in project.bullets)
    summary = "\n".join(sections)
    return summary[:limit]
