"""Pydantic data models for job application assistant."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class Bullet(BaseModel):
    """A single resume bullet point with a traceable ID."""

    id: str
    text: str
    tags: list[str] = Field(default_factory=list)
    metrics: str | None = None


class Skill(BaseModel):
    """A skill categorized with tags."""

    name: str
    tags: list[str] = Field(default_factory=list)


class Experience(BaseModel):
    """Work experience entry."""

    company: str
    title: str
    start: str
    end: str | None = None
    bullets: list[Bullet] = Field(default_factory=list)


class Project(BaseModel):
    """Project entry."""

    name: str
    description: str
    tech: list[str] = Field(default_factory=list)
    bullets: list[Bullet] = Field(default_factory=list)


class Education(BaseModel):
    """Education credential entry."""

    institution: str
    degree: str
    start: str
    end: str | None = None


class Profile(BaseModel):
    """Master candidate profile containing all experience, projects, and skills."""

    name: str
    email: str
    phone: str
    location: str
    links: list[str] = Field(default_factory=list)
    summary: str
    skills: list[Skill] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)

    @model_validator(mode="after")
    def verify_unique_bullet_ids(self) -> "Profile":
        """Ensure all bullet IDs across experience and projects are unique."""
        seen_ids: set[str] = set()
        duplicates: list[str] = []

        for exp in self.experience:
            for bullet in exp.bullets:
                if bullet.id in seen_ids:
                    duplicates.append(bullet.id)
                seen_ids.add(bullet.id)

        for proj in self.projects:
            for bullet in proj.bullets:
                if bullet.id in seen_ids:
                    duplicates.append(bullet.id)
                seen_ids.add(bullet.id)

        if duplicates:
            raise ValueError(f"Duplicate bullet ids found: {sorted(set(duplicates))}")
        return self


class Job(BaseModel):
    """Discovered job posting."""

    id: str
    company: str
    title: str
    location: str
    url: str
    description: str
    source: str
    posted_at: str | None = None


class JobRequirements(BaseModel):
    """Extracted requirements from a job posting."""

    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    seniority: str
    role_type: str
    keywords: list[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    """Evaluation of candidate fit against job requirements."""

    job_id: str
    score: int = Field(ge=0, le=100)
    reasoning: str
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)


class JobSummary(BaseModel):
    """Compact tracker view of a discovered job."""

    job_id: str
    company: str
    title: str
    status: str
    latest_score: int | None = None


def load_profile(path: str | Path) -> Profile:
    """Load and validate a Profile from a YAML file.

    Args:
        path: Path to the YAML profile file.

    Returns:
        Validated Profile instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If file content is not a YAML mapping.
        ValidationError: If content does not match Profile schema.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Profile file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data: Any = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a YAML mapping in {file_path}, got {type(data).__name__}"
        )

    return Profile.model_validate(data)


class MatchInput(BaseModel):
    profile: Profile
    requirements: JobRequirements
    job_id: str
    job_title: str

class TailoredBullet(BaseModel):
    """Custom Bullets points of any experince or project"""
    source_id : str
    text : str

class TailoredExperience(BaseModel):
    """Tailored experience for specific job """

    company : str
    start : str
    end : str | None = None
    title : str
    bullets : list[TailoredBullet]

class TailoredProject(BaseModel):
    """One project, with only the bullets chosen for a specific target job."""
    name : str
    tech : list[str]
    bullets : list[TailoredBullet]

class TailoredResume(BaseModel):
    """Complete tailored resume content for one job. Contains no contact details."""

    job_id : str
    role_label : str
    summary : str
    skills : list[str]
    experience : list[TailoredExperience]
    projects : list[TailoredProject]

class TailorInput(BaseModel):
    profile: Profile
    requirements: JobRequirements
    job_id: str
    job_description: str
    feedback: list[str] = []
 
    
