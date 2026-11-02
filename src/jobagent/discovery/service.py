"""Load source config and run discovery across boards with error isolation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from jobagent.discovery.base import (
    BoardFetchError,
    BoardNotFoundError,
    JobFilter,
    JobSource,
    is_valid_board_token,
)
from jobagent.discovery.greenhouse import GreenhouseSource
from jobagent.models import Job


class GreenhouseBoards(BaseModel):
    """Greenhouse boards to query."""

    boards: list[str] = Field(default_factory=list)

    @field_validator("boards")
    @classmethod
    def validate_board_tokens(cls, boards: list[str]) -> list[str]:
        invalid = [token for token in boards if not is_valid_board_token(token)]
        if invalid:
            raise ValueError(
                "Invalid Greenhouse board token(s) "
                f"{invalid!r}; use only letters, digits, '-' and '_'"
            )
        return boards


class SourcesConfig(BaseModel):
    """Discovery sources and shared filters loaded from YAML."""

    greenhouse: GreenhouseBoards = Field(default_factory=GreenhouseBoards)
    filters: JobFilter = Field(default_factory=JobFilter)


class DiscoveryResult(BaseModel):
    """Jobs that passed filters, plus per-board errors that did not stop the run."""

    jobs: list[Job] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def load_sources(path: str | Path) -> SourcesConfig:
    """Load and validate a SourcesConfig from a YAML file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Sources file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        data: Any = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a YAML mapping in {file_path}, got {type(data).__name__}"
        )

    return SourcesConfig.model_validate(data)


def discover(
    config: SourcesConfig,
    source: JobSource | None = None,
) -> DiscoveryResult:
    """Fetch, filter, and de-duplicate jobs. Failures on one board do not stop others."""
    if source is not None:
        return _discover_with(config, source)

    with GreenhouseSource() as greenhouse:
        return _discover_with(config, greenhouse)


def _discover_with(config: SourcesConfig, source: JobSource) -> DiscoveryResult:
    jobs: list[Job] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    for board in config.greenhouse.boards:
        try:
            fetched = source.fetch_jobs(board)
        except (BoardNotFoundError, BoardFetchError) as exc:
            errors.append(str(exc))
            continue

        for job in fetched:
            if job.id in seen_ids:
                continue
            if not config.filters.matches(job):
                continue
            seen_ids.add(job.id)
            jobs.append(job)

    return DiscoveryResult(jobs=jobs, errors=errors)
