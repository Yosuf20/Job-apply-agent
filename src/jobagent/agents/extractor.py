"""Extract structured requirements from a job posting."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from jobagent.agents.base import Agent
from jobagent.models import Job, JobRequirements

_PROMPT_PATH = Path(__file__).parent / "prompts" / "extractor.txt"
_ALLOWED_SENIORITY = frozenset(
    {"intern", "junior", "mid", "senior", "staff", "lead", "unspecified"}
)
_WORD_RE = re.compile(r"\b\w+\b", re.IGNORECASE)
logger = logging.getLogger(__name__)


class ExtractorAgent(Agent[Job, JobRequirements]):
    """Extract hiring requirements using the configured small language model."""

    name = "ExtractorAgent"
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    tier = "small"
    input_type = Job
    output_type = JobRequirements

    def __init__(
        self,
        client: Any,
        *,
        description_limit: int = 12000,
    ) -> None:
        if description_limit < 1:
            raise ValueError("description_limit must be at least 1")
        super().__init__(client)
        self.description_limit = description_limit

    def build_prompt(self, input_data: Job) -> str:
        """Build a prompt containing the job fields and bounded description."""
        description = input_data.description
        truncated = len(description) > self.description_limit
        if truncated:
            description = description[: self.description_limit]
            truncation_note = (
                f"\n[Description truncated to {self.description_limit} characters.]"
            )
        else:
            truncation_note = ""

        return (
            f"Job title: {input_data.title}\n"
            f"Company: {input_data.company}\n"
            f"Location: {input_data.location}\n"
            f"Description:\n{description}{truncation_note}"
        )

    def run(self, input_data: Job) -> JobRequirements:
        """Extract and normalize the model output."""
        job_text = (
            f"{input_data.title} {input_data.company} {input_data.location} "
            f"{input_data.description}"
        )
        return _normalize_requirements(super().run(input_data), job_text)


def _normalize_requirements(
    requirements: JobRequirements,
    job_text: str,
) -> JobRequirements:
    text_words = {word.casefold() for word in _WORD_RE.findall(job_text)}
    must_have = _ground_skills(
        _unique_trimmed(requirements.must_have_skills), text_words
    )
    nice_to_have = _ground_skills(
        _unique_trimmed(requirements.nice_to_have_skills), text_words
    )
    must_have_keys = {skill.casefold() for skill in must_have}
    nice_to_have = [
        skill for skill in nice_to_have if skill.casefold() not in must_have_keys
    ]
    skill_keys = {skill.casefold() for skill in (*must_have, *nice_to_have)}
    keywords = [
        keyword
        for keyword in _unique_trimmed(requirements.keywords)
        if keyword.casefold() not in skill_keys
    ]

    seniority = requirements.seniority.strip().casefold()
    if seniority not in _ALLOWED_SENIORITY:
        seniority = "unspecified"

    return requirements.model_copy(
        update={
            "must_have_skills": must_have,
            "nice_to_have_skills": nice_to_have,
            "seniority": seniority,
            "role_type": requirements.role_type.strip(),
            "keywords": keywords,
        }
    )


def _unique_trimmed(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        trimmed = value.strip()
        key = trimmed.casefold()
        if trimmed and key not in seen:
            result.append(trimmed)
            seen.add(key)
    return result


def _ground_skills(skills: list[str], text_words: set[str]) -> list[str]:
    grounded: list[str] = []
    for skill in skills:
        skill_words = {word.casefold() for word in _WORD_RE.findall(skill)}
        if skill_words and skill_words <= text_words:
            grounded.append(skill)
        else:
            logger.info("Dropped ungrounded extracted skill: %s", skill)
    return grounded
