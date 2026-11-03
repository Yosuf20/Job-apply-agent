"""Run the matcher against labeled jobs with the real model."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jobagent.agents.matcher import MatcherAgent
from jobagent.config import get_settings
from jobagent.evaluation.matcher_metrics import (
    BAND_MIDPOINTS,
    band_accuracy,
    band_for_score,
    mean_absolute_error,
    spearman_rank_correlation,
)
from jobagent.llm.client import LLMTransientError, StructuredOutputError, get_client
from jobagent.matching.skills import coverage
from jobagent.models import JobRequirements, MatchInput, load_profile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    labels_path = _choose_labels(args.labels)
    profile_path = _choose_profile(args.profile, settings.profile_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    profile = load_profile(profile_path)
    client = get_client(settings)
    matcher = MatcherAgent(client)
    model_scores: list[float] = []
    baseline_scores: list[float] = []
    expected_bands: list[str] = []
    results: list[dict[str, Any]] = []

    for index, label in enumerate(labels):
        requirements = JobRequirements.model_validate(label["requirements"])
        match_input = MatchInput(
            profile=profile,
            requirements=requirements,
            job_id=label["job_id"],
            job_title=label["job_title"],
        )
        baseline = round(coverage(requirements, profile) * 100)
        expected_band = label["expected_band"]
        try:
            result = matcher.run(match_input)
        except (LLMTransientError, StructuredOutputError) as error:
            print(f"{label['job_id']}: FAILED - {error}")
            results.append(
                {"job_id": label["job_id"], "status": "failed", "error": str(error)}
            )
        else:
            model_scores.append(result.score)
            baseline_scores.append(baseline)
            expected_bands.append(expected_band)
            results.append(
                {
                    "job_id": label["job_id"],
                    "status": "ok",
                    "expected_band": expected_band,
                    "model_score": result.model_dump(),
                    "baseline_score": baseline,
                }
            )
            print(
                f"{label['job_id']}: expected={expected_band} "
                f"model={result.score} ({band_for_score(result.score)}) "
                f"baseline={baseline}"
            )
            print(f"  matched: {result.matched_skills}")
            print(f"  missing: {result.missing_skills}")
            print(f"  reasoning: {result.reasoning}")
        if index < len(labels) - 1:
            time.sleep(args.delay)

    summary = _summary(model_scores, baseline_scores, expected_bands)
    _print_summary(summary)
    output_path = _resolve(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "provider": settings.llm_provider,
                "models": {
                    "small": settings.model_small,
                    "strong": settings.model_strong,
                },
                "matcher_prompt_sha256": _prompt_hash(),
                "results": results,
                "aggregate": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="evals/matcher_labeled.json")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument(
        "--out",
        default=f"evals/results/matcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    return parser.parse_args()


def _choose_labels(path: str) -> Path:
    candidate = _resolve(path)
    if candidate.exists():
        return candidate
    fallback = _resolve("evals/matcher_labeled.example.json")
    print(f"Warning: {candidate} not found; using {fallback}")
    return fallback


def _choose_profile(path: str | None, configured: Path) -> Path:
    candidate = _resolve(path) if path else configured
    if candidate.exists():
        return candidate
    fallback = _resolve("data/profile.example.yaml")
    print(f"Warning: {candidate} not found; using {fallback}")
    return fallback


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _prompt_hash() -> str:
    return hashlib.sha256(
        (ROOT / "src/jobagent/agents/prompts/matcher.txt").read_bytes()
    ).hexdigest()


def _summary(
    model_scores: list[float],
    baseline_scores: list[float],
    expected_bands: list[str],
) -> dict[str, dict[str, float | None]]:
    expected_values = [BAND_MIDPOINTS[band] for band in expected_bands]
    return {
        "model": _metrics(model_scores, expected_bands, expected_values),
        "baseline": _metrics(baseline_scores, expected_bands, expected_values),
    }


def _metrics(
    scores: list[float],
    expected_bands: list[str],
    expected_values: list[float],
) -> dict[str, float | None]:
    return {
        "mae": mean_absolute_error(scores, expected_values),
        "band_accuracy": band_accuracy(scores, expected_bands),
        "spearman": spearman_rank_correlation(scores, expected_values),
    }


def _print_summary(summary: dict[str, dict[str, float | None]]) -> None:
    print("\nAggregate")
    print(f"{'Metric':<18} {'Model':>12} {'Coverage baseline':>20}")
    print("-" * 54)
    for metric in ("mae", "band_accuracy", "spearman"):
        print(
            f"{metric:<18} {summary['model'][metric]!s:>12} "
            f"{summary['baseline'][metric]!s:>20}"
        )


if __name__ == "__main__":
    main()
