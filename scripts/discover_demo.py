"""Print a table of jobs from configured Greenhouse boards.

Manual-only helper. Do not run from tests or CI.
"""

from __future__ import annotations

from jobagent.config import get_settings
from jobagent.discovery.service import discover, load_sources


def _cell(value: str, width: int) -> str:
    if len(value) <= width:
        return value.ljust(width)
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def main() -> None:
    settings = get_settings()
    config = load_sources(settings.sources_path)
    result = discover(config)

    headers = ("Title", "Company", "Location", "ID")
    rows: list[tuple[str, str, str, str]] = [headers]
    for job in result.jobs:
        location = job.location.strip() or "(unknown)"
        rows.append((job.title, job.company, location, job.id))

    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    widths = [min(width, 40) for width in widths]

    print("  ".join(_cell(headers[i], widths[i]) for i in range(4)))
    print("  ".join("-" * widths[i] for i in range(4)))
    for row in rows[1:]:
        print("  ".join(_cell(row[i], widths[i]) for i in range(4)))

    print()
    print(f"{len(result.jobs)} job(s)")
    if result.errors:
        print("Board errors:")
        for error in result.errors:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
