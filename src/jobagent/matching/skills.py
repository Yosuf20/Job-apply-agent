"""Pure profile skill matching functions."""

from __future__ import annotations

import re
import string

from jobagent.models import JobRequirements, Profile

_SPECIAL_TOKENS = (
    (re.compile(r"(?<!\w)c\+\+(?!\w)", re.IGNORECASE), "cplusplus"),
    (re.compile(r"(?<!\w)c#(?!\w)", re.IGNORECASE), "csharp"),
    (re.compile(r"(?<!\w)ci[-/]cd(?!\w)", re.IGNORECASE), "cicd"),
)


def normalize(text: str) -> str:
    """Normalize text while preserving common punctuation-based skill names."""
    normalized = text.casefold()
    for pattern, replacement in _SPECIAL_TOKENS:
        normalized = pattern.sub(replacement, normalized)
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    return " ".join(normalized.split())


def profile_skill_index(profile: Profile) -> set[str]:
    """Build normalized skill and bullet-text entries from a profile."""
    entries: set[str] = set()
    for skill in profile.skills:
        entries.add(normalize(skill.name))
        entries.update(normalize(tag) for tag in skill.tags)
    for experience in profile.experience:
        for bullet in experience.bullets:
            entries.add(normalize(bullet.text))
            entries.update(normalize(tag) for tag in bullet.tags)
    for project in profile.projects:
        entries.update(normalize(tech) for tech in project.tech)
        for bullet in project.bullets:
            entries.add(normalize(bullet.text))
            entries.update(normalize(tag) for tag in bullet.tags)
    return {entry for entry in entries if entry}


def skill_in_profile(skill: str, index: set[str] | Profile) -> bool:
    """Return whether a normalized skill occurs as a whole-word phrase."""
    normalized_skill = normalize(skill)
    if not normalized_skill:
        return False
    entries = profile_skill_index(index) if isinstance(index, Profile) else index
    pattern = re.compile(rf"(?<!\w){re.escape(normalized_skill)}(?!\w)")
    return any(pattern.search(entry) is not None for entry in entries)


def compare_skills(
    required: list[str], profile: Profile
) -> tuple[list[str], list[str]]:
    """Compare required skills, preserving first spelling and input order."""
    index = profile_skill_index(profile)
    matched: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for skill in required:
        key = normalize(skill)
        if not key or key in seen:
            continue
        seen.add(key)
        (matched if skill_in_profile(skill, index) else missing).append(skill)
    return matched, missing


def coverage(requirements: JobRequirements, profile: Profile) -> float:
    """Return the share of unique must-have skills found in the profile."""
    matched, missing = compare_skills(requirements.must_have_skills, profile)
    total = len(matched) + len(missing)
    return len(matched) / total if total else 1.0
