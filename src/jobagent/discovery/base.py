"""Shared discovery types: source protocol, filters, and board errors."""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel, Field

from jobagent.models import Job

_BOARD_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def is_valid_board_token(token: str) -> bool:
    """Return True if token uses only letters, digits, hyphens, and underscores."""
    return bool(token) and _BOARD_TOKEN_RE.fullmatch(token) is not None


def whole_word_match(term: str, text: str) -> bool:
    """Case-insensitive whole-word match using \\b and re.escape."""
    if not term:
        return False
    pattern = rf"\b{re.escape(term)}\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


class BoardNotFoundError(Exception):
    """Raised when a Greenhouse board token does not exist (HTTP 404)."""

    def __init__(self, board: str) -> None:
        self.board = board
        super().__init__(f"Greenhouse board not found: {board}")


class BoardFetchError(Exception):
    """Raised when a board request fails or the response cannot be used."""

    def __init__(self, board: str, message: str) -> None:
        self.board = board
        super().__init__(f"{board}: {message}")


class JobSource(Protocol):
    """A job board source that can list postings for one board token."""

    def fetch_jobs(self, board: str) -> list[Job]:
        """Return jobs from a single board."""
        ...


class JobFilter(BaseModel):
    """Optional title and location restrictions. Empty lists mean no restriction."""

    title_include: list[str] = Field(default_factory=list)
    title_exclude: list[str] = Field(default_factory=list)
    location_include: list[str] = Field(default_factory=list)

    def matches(self, job: Job) -> bool:
        """Return True if the job passes include/exclude whole-word filters."""
        include_terms = _nonempty(self.title_include)
        if include_terms and not any(
            whole_word_match(term, job.title) for term in include_terms
        ):
            return False

        exclude_terms = _nonempty(self.title_exclude)
        if any(whole_word_match(term, job.title) for term in exclude_terms):
            return False

        location_terms = _nonempty(self.location_include)
        if location_terms:
            location = job.location.strip() if job.location else ""
            # Unknown / whitespace-only locations stay in the result.
            if location and not any(
                whole_word_match(term, location) for term in location_terms
            ):
                return False

        return True


def _nonempty(terms: list[str]) -> list[str]:
    return [term for term in terms if term.strip()]
