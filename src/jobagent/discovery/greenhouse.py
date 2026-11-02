"""Greenhouse Job Board API client (public JSON endpoint only)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Self

import httpx

from jobagent.discovery.base import BoardFetchError, BoardNotFoundError
from jobagent.discovery.text import clean_html
from jobagent.models import Job

logger = logging.getLogger(__name__)

_JOBS_URL = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs"
_DEFAULT_USER_AGENT = (
    "jobagent/0.1 (Greenhouse Job Board API; personal job-search assistant)"
)
_MAX_ATTEMPTS = 3
_RETRY_AFTER_CAP_SECONDS = 30.0
_TRANSIENT_STATUS = frozenset({429}) | frozenset(range(500, 600))


class GreenhouseSource:
    """Fetch jobs from one Greenhouse board via the public Job Board API."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        timeout_seconds: float = 30.0,
        user_agent: str = _DEFAULT_USER_AGENT,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._owns_client = client is None
        self._timeout = timeout_seconds
        self._user_agent = user_agent
        self._sleep_fn = sleep_fn
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"User-Agent": user_agent},
        )

    def close(self) -> None:
        """Close the HTTP client when this instance created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def fetch_jobs(self, board: str) -> list[Job]:
        """GET /v1/boards/{board}/jobs?content=true and map the payload to Job models."""
        url = _JOBS_URL.format(board=board)
        last_error: Exception | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._client.get(
                    url,
                    params={"content": "true"},
                    headers={"User-Agent": self._user_agent},
                    timeout=self._timeout,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt < _MAX_ATTEMPTS:
                    delay = _backoff_seconds(attempt)
                    logger.warning(
                        "Greenhouse request for %s timed out (attempt %d/%d); retrying in %.1fs",
                        board,
                        attempt,
                        _MAX_ATTEMPTS,
                        delay,
                    )
                    self._sleep_fn(delay)
                    continue
                raise BoardFetchError(
                    board, f"request failed after {_MAX_ATTEMPTS} attempts: {exc}"
                ) from exc

            status = response.status_code
            if status == 404:
                raise BoardNotFoundError(board)
            if status in _TRANSIENT_STATUS:
                last_error = BoardFetchError(board, f"HTTP {status}")
                if attempt < _MAX_ATTEMPTS:
                    delay = _retry_delay(attempt, response)
                    logger.warning(
                        "Greenhouse %s for %s (attempt %d/%d); retrying in %.1fs",
                        status,
                        board,
                        attempt,
                        _MAX_ATTEMPTS,
                        delay,
                    )
                    self._sleep_fn(delay)
                    continue
                raise BoardFetchError(
                    board, f"HTTP {status} after {_MAX_ATTEMPTS} attempts"
                )
            if 400 <= status < 500:
                raise BoardFetchError(board, f"HTTP {status}")

            return _parse_jobs_payload(board, response)

        raise BoardFetchError(
            board, f"request failed after {_MAX_ATTEMPTS} attempts: {last_error}"
        )


def _parse_jobs_payload(board: str, response: httpx.Response) -> list[Job]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise BoardFetchError(board, "response body was not valid JSON") from exc

    if not isinstance(payload, dict) or "jobs" not in payload:
        raise BoardFetchError(board, "response JSON missing 'jobs' key")

    raw_jobs = payload["jobs"]
    if not isinstance(raw_jobs, list):
        raise BoardFetchError(board, "'jobs' must be a list")

    jobs: list[Job] = []
    for item in raw_jobs:
        if not isinstance(item, dict):
            continue
        mapped = _map_job(board, item)
        if mapped is not None:
            jobs.append(mapped)
    return jobs


def _map_job(board: str, item: dict[str, Any]) -> Job | None:
    raw_id = item.get("id")
    if raw_id is None:
        return None

    title = item.get("title")
    if not isinstance(title, str) or not title.strip():
        return None

    company = item.get("company_name")
    if not isinstance(company, str) or not company.strip():
        company = board

    location_name = ""
    location = item.get("location")
    if isinstance(location, dict):
        name = location.get("name")
        if isinstance(name, str):
            location_name = name

    url = item.get("absolute_url")
    if not isinstance(url, str):
        url = ""

    content = item.get("content")
    description = clean_html(content) if isinstance(content, str) else ""

    updated_at = item.get("updated_at")
    posted_at = updated_at if isinstance(updated_at, str) else None

    return Job(
        id=f"greenhouse:{board}:{raw_id}",
        company=company,
        title=title,
        location=location_name,
        url=url,
        description=description,
        source="greenhouse",
        # Greenhouse list endpoint exposes updated_at (last edit), not original posting date.
        posted_at=posted_at,
    )


def _retry_delay(attempt: int, response: httpx.Response) -> float:
    if response.status_code == 429:
        retry_after = _parse_retry_after_seconds(response.headers.get("Retry-After"))
        if retry_after is not None:
            return retry_after
    return _backoff_seconds(attempt)


def _parse_retry_after_seconds(header: str | None) -> float | None:
    """Parse Retry-After as integer seconds. Ignore HTTP-dates and other values."""
    if header is None:
        return None
    try:
        seconds = int(header.strip())
    except ValueError:
        return None
    if seconds < 0:
        return None
    return min(float(seconds), _RETRY_AFTER_CAP_SECONDS)


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff after a failed attempt (1-indexed): 1s, 2s, 4s."""
    return float(2 ** (attempt - 1))
