"""Tests for Greenhouse job discovery (no real network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from jobagent.discovery.base import BoardFetchError, BoardNotFoundError, JobFilter
from jobagent.discovery.greenhouse import GreenhouseSource
from jobagent.discovery.service import SourcesConfig, discover, load_sources
from jobagent.discovery.text import clean_html
from jobagent.models import Job

FIXTURES = Path(__file__).parent / "fixtures"


def _payload() -> dict[str, Any]:
    return json.loads((FIXTURES / "greenhouse_jobs.json").read_text(encoding="utf-8"))


def _job(
    *,
    job_id: str = "greenhouse:board1:1",
    title: str = "Software Engineer",
    location: str = "Remote - US",
) -> Job:
    return Job(
        id=job_id,
        company="Acme",
        title=title,
        location=location,
        url="https://example.com/job",
        description="Build things.",
        source="greenhouse",
        posted_at="2016-01-14T10:55:28-05:00",
    )


def _source(
    handler: Callable[[httpx.Request], httpx.Response],
    sleep_fn: Callable[[float], None] | None = None,
) -> GreenhouseSource:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GreenhouseSource(client=client, sleep_fn=sleep_fn or (lambda _: None))


def _board_from(request: httpx.Request) -> str:
    return request.url.path.rstrip("/").split("/")[-2]


class FakeSource:
    def __init__(self, by_board: dict[str, list[Job] | BaseException]) -> None:
        self.by_board = by_board
        self.calls: list[str] = []

    def fetch_jobs(self, board: str) -> list[Job]:
        self.calls.append(board)
        result = self.by_board[board]
        if isinstance(result, BaseException):
            raise result
        return result


def test_clean_html_skips_script_style_and_formats_blocks() -> None:
    raw = """
    <html><body>
      <script>alert("x")</script>
      <style>.x{color:red}</style>
      <noscript>enable js</noscript>
      <p>Hello   world</p>
      <p>Second&nbsp;paragraph</p>
      <ul><li>First</li><li>Second</li></ul>
      Line<br>break
    </body></html>
    """
    text = clean_html(raw)
    assert "alert" not in text
    assert "color:red" not in text
    assert "enable js" not in text
    assert "Hello world" in text
    assert "Second paragraph" in text
    assert "- First" in text
    assert "- Second" in text
    lines = text.splitlines()
    assert any(line == "Line" for line in lines) or "Line" in text
    assert "break" in text


def test_clean_html_unescapes_greenhouse_entities() -> None:
    text = clean_html("Intro. &lt;p&gt;Build things.&lt;/p&gt;")
    assert "<p>" not in text
    assert "Intro." in text
    assert "Build things." in text


def test_title_exclude_intern_is_whole_word() -> None:
    filt = JobFilter(title_exclude=["intern"])
    assert not filt.matches(_job(title="Software Engineering Intern"))
    assert filt.matches(_job(title="Internal Tools Engineer"))
    assert filt.matches(_job(title="International Payments Engineer"))


def test_title_include_is_whole_word() -> None:
    filt = JobFilter(title_include=["intern"])
    assert filt.matches(_job(title="Software Engineering Intern"))
    assert not filt.matches(_job(title="Internal Tools Engineer"))


def test_empty_filters_keep_every_job() -> None:
    filt = JobFilter()
    assert filt.matches(_job(title="Anything", location=""))


def test_location_india_vs_indiana() -> None:
    filt = JobFilter(location_include=["india"])
    assert filt.matches(_job(location="Bengaluru, India"))
    assert not filt.matches(_job(location="Indiana, USA"))


def test_location_remote_is_whole_word() -> None:
    filt = JobFilter(location_include=["remote"])
    assert filt.matches(_job(location="Remote - US"))
    assert filt.matches(_job(location="US Remote"))
    assert not filt.matches(_job(location="Remotely operated warehouse, NYC"))


def test_whitespace_only_location_kept_when_location_include_set() -> None:
    filt = JobFilter(location_include=["india"])
    assert filt.matches(_job(location=""))
    assert filt.matches(_job(location="   "))


def test_map_greenhouse_job_fields() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_payload())

    source = _source(handler)
    jobs = source.fetch_jobs("board1")

    intern = next(job for job in jobs if "Intern" in job.title)
    tools = next(job for job in jobs if "Internal Tools" in job.title)

    assert intern.id == "greenhouse:board1:127817"
    assert intern.company == "board1"
    assert intern.source == "greenhouse"
    assert intern.posted_at == "2016-01-14T10:55:28-05:00"
    assert intern.location == "NYC"
    assert "Build things." in intern.description
    assert "<p>" not in intern.description

    assert tools.company == "Acme Corp"
    assert tools.id == "greenhouse:board1:127818"

    assert captured
    assert captured[0].url.params.get("content") == "true"
    assert "jobagent" in captured[0].headers["user-agent"].lower()


def test_404_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="missing")

    source = _source(handler)
    with pytest.raises(BoardNotFoundError, match="missing-board"):
        source.fetch_jobs("missing-board")
    assert calls["n"] == 1


def test_429_retries_using_integer_retry_after() -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, text="slow")
        return httpx.Response(200, json=_payload())

    source = _source(handler, sleep_fn=sleeps.append)
    jobs = source.fetch_jobs("board1")

    assert calls["n"] == 2
    assert sleeps == [7.0]
    assert len(jobs) == 2


def test_invalid_retry_after_falls_back_to_exponential_backoff() -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"},
                text="slow",
            )
        return httpx.Response(200, json=_payload())

    source = _source(handler, sleep_fn=sleeps.append)
    source.fetch_jobs("board1")

    assert calls["n"] == 2
    assert sleeps == [1.0]


def test_non_json_200_raises_board_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    source = _source(handler)
    with pytest.raises(BoardFetchError, match="not valid JSON"):
        source.fetch_jobs("board1")


def test_json_without_jobs_key_raises_board_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"meta": {"total": 0}})

    source = _source(handler)
    with pytest.raises(BoardFetchError, match="missing 'jobs'"):
        source.fetch_jobs("board1")


def test_404_isolated_while_other_boards_succeed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        board = _board_from(request)
        if board == "missing-board":
            return httpx.Response(404, text="missing")
        return httpx.Response(200, json=_payload())

    result = discover(
        SourcesConfig.model_validate(
            {"greenhouse": {"boards": ["board1", "missing-board"]}}
        ),
        source=_source(handler),
    )

    assert len(result.jobs) == 2
    assert any("missing-board" in error for error in result.errors)
    assert all(job.source == "greenhouse" for job in result.jobs)


def test_non_json_body_is_per_board_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        board = _board_from(request)
        if board == "broken":
            return httpx.Response(200, text="nope")
        return httpx.Response(200, json=_payload())

    result = discover(
        SourcesConfig.model_validate({"greenhouse": {"boards": ["broken", "board1"]}}),
        source=_source(handler),
    )

    assert len(result.jobs) == 2
    assert any("broken" in error and "JSON" in error for error in result.errors)


def test_json_without_jobs_is_per_board_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        board = _board_from(request)
        if board == "empty-shape":
            return httpx.Response(200, json={"hello": True})
        return httpx.Response(200, json=_payload())

    result = discover(
        SourcesConfig.model_validate(
            {"greenhouse": {"boards": ["empty-shape", "board1"]}}
        ),
        source=_source(handler),
    )

    assert len(result.jobs) == 2
    assert any("empty-shape" in error for error in result.errors)


def test_dedup_by_job_id() -> None:
    shared = _job(job_id="greenhouse:board1:99")
    result = discover(
        SourcesConfig.model_validate({"greenhouse": {"boards": ["board1", "board2"]}}),
        source=FakeSource({"board1": [shared], "board2": [shared]}),
    )
    assert len(result.jobs) == 1
    assert result.jobs[0].id == "greenhouse:board1:99"


def test_bad_board_token_rejected() -> None:
    with pytest.raises(ValidationError, match="Invalid Greenhouse board token"):
        SourcesConfig.model_validate({"greenhouse": {"boards": ["bad board"]}})


def test_example_yaml_validates() -> None:
    config = load_sources(Path("data/sources.example.yaml"))
    assert config.greenhouse.boards == ["board1", "board2"]
    assert config.filters.title_include == []
    assert config.filters.title_exclude == []
    assert config.filters.location_include == []
