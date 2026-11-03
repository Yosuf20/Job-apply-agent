"""Run the extractor against hand-written gold labels using the real model."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jobagent.agents.extractor import ExtractorAgent
from jobagent.config import get_settings
from jobagent.evaluation.extractor_metrics import (
    aggregate,
    score_extraction,
)
from jobagent.llm.client import LLMTransientError, StructuredOutputError, get_client
from jobagent.models import Job

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "src" / "jobagent" / "agents" / "prompts" / "extractor.txt"


def main() -> None:
    args = _parse_args()
    gold_path = _resolve_path(args.gold)
    gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
    selected_ids = _selected_ids(gold_data, args.ids)
    settings = get_settings()
    client = get_client(settings)
    extractor = ExtractorAgent(client)
    scores = []
    results: list[dict[str, Any]] = []

    for index, posting_id in enumerate(selected_ids):
        entry = gold_data[posting_id]
        posting_path = _resolve_path(entry["posting_file"])
        posting_text = posting_path.read_text(encoding="utf-8")
        job = Job(
            id=f"evaluation:{posting_id}",
            title=entry["title"],
            company=entry["company"],
            location=entry["location"],
            url="",
            description=posting_text,
            source="evaluation",
        )
        try:
            output = extractor.run(job)
            score = score_extraction(output, entry["expected"], posting_text)
            scores.append(score)
            results.append(
                {
                    "id": posting_id,
                    "status": "ok",
                    "output": output.model_dump(),
                    "score": score.model_dump(),
                }
            )
            _print_result(posting_id, output, score, entry["expected"])
        except (LLMTransientError, StructuredOutputError) as error:
            print(f"{posting_id}: FAILED - {error}")
            results.append({"id": posting_id, "status": "failed", "error": str(error)})
        if index < len(selected_ids) - 1:
            time.sleep(args.delay)

    timestamp = datetime.now(UTC).isoformat()
    output_path = _resolve_path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": timestamp,
        "provider": settings.llm_provider,
        "models": {
            "small": settings.model_small,
            "strong": settings.model_strong,
        },
        "extractor_prompt_sha256": _prompt_hash(),
        "results": results,
        "aggregate": aggregate(scores),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\nAggregate:")
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"\nSaved results to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="evals/extractor_gold.json")
    parser.add_argument("--ids", help="Comma-separated posting IDs to evaluate")
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument(
        "--out",
        default=f"evals/results/extractor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    return parser.parse_args()


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _selected_ids(gold_data: dict[str, Any], ids: str | None) -> list[str]:
    if not ids:
        return list(gold_data)
    selected = [
        posting_id.strip() for posting_id in ids.split(",") if posting_id.strip()
    ]
    missing = [posting_id for posting_id in selected if posting_id not in gold_data]
    if missing:
        raise ValueError(f"Unknown posting IDs: {', '.join(missing)}")
    return selected


def _prompt_hash() -> str:
    return hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()


def _print_result(
    posting_id: str,
    output: Any,
    score: Any,
    expected: dict[str, Any],
) -> None:
    must_missing = score_missing(output.must_have_skills, expected["must_have_skills"])
    nice_missing = score_missing(
        output.nice_to_have_skills, expected["nice_to_have_skills"]
    )
    extra_must = score_extra(output.must_have_skills, expected["must_have_skills"])
    extra_nice = score_extra(
        output.nice_to_have_skills, expected["nice_to_have_skills"]
    )
    print(
        f"{posting_id}: must F1={score.must_have_f1:.2f}, "
        f"nice F1={score.nice_to_have_f1:.2f}, "
        f"seniority={'yes' if score.seniority_correct else 'no'}, "
        f"role={'yes' if score.role_type_correct else 'no'}"
    )
    print(f"  hallucinated: {score.hallucinated_skills or '-'}")
    print(f"  misplaced: {score.misplaced_skills or '-'}")
    print(f"  missing must: {must_missing or '-'}; extra must: {extra_must or '-'}")
    print(f"  missing nice: {nice_missing or '-'}; extra nice: {extra_nice or '-'}")


def score_missing(predicted: list[str], gold: list[str]) -> list[str]:
    from jobagent.evaluation.extractor_metrics import match_lists

    return [skill for skill in match_lists(predicted, gold).unmatched_gold]


def score_extra(predicted: list[str], gold: list[str]) -> list[str]:
    from jobagent.evaluation.extractor_metrics import match_lists

    return [skill for skill in match_lists(predicted, gold).unmatched_predicted]


if __name__ == "__main__":
    main()
