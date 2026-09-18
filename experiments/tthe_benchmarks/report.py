"""Aggregation and paired comparison helpers for TTHE benchmark runs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .spec import Domain


@dataclass(frozen=True)
class Prediction:
    task_id: str
    correct: bool
    domain: str


def _extract_predictions(result: Mapping[str, Any], domain: Domain) -> list[Prediction]:
    per_problem = result.get("per_problem")
    if not isinstance(per_problem, list):
        return []
    key_by_domain = {
        Domain.BIRD: "db_id",
        Domain.LIVECODEBENCH: "qid",
        Domain.SWE: "instance_id",
        Domain.DS1000: "pid",
    }
    key = key_by_domain[domain]
    predictions: list[Prediction] = []
    for item in per_problem:
        if not isinstance(item, Mapping):
            continue
        task_id = item.get(key)
        if task_id is None:
            continue
        predictions.append(Prediction(
            task_id=str(task_id), correct=bool(item.get("correct")), domain=domain.value,
        ))
    return predictions


def load_result(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_result(result: Mapping[str, Any], domain: Domain) -> dict[str, Any]:
    correct = int(result.get("tt_correct", 0))
    total = int(result.get("tt_total", 0))
    batches = result.get("batches", [])
    return {
        "domain": domain.value,
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else 0.0,
        "batches": len(batches) if isinstance(batches, list) else None,
        "final_harness": result.get("final_harness"),
    }


def paired_predictions(
    none_predictions: Iterable[Prediction],
    mmh_predictions: Iterable[Prediction],
) -> dict[str, Any]:
    none_map = {item.task_id: item for item in none_predictions}
    mmh_map = {item.task_id: item for item in mmh_predictions}
    shared = sorted(set(none_map) & set(mmh_map))
    both_correct = sum(1 for task in shared if none_map[task].correct and mmh_map[task].correct)
    none_only = sum(1 for task in shared if none_map[task].correct and not mmh_map[task].correct)
    mmh_only = sum(1 for task in shared if not none_map[task].correct and mmh_map[task].correct)
    both_wrong = sum(1 for task in shared if not none_map[task].correct and not mmh_map[task].correct)
    return {
        "paired_tasks": len(shared),
        "both_correct": both_correct,
        "none_only": none_only,
        "mmh_only": mmh_only,
        "both_wrong": both_wrong,
        "delta_mmh_minus_none": mmh_only - none_only,
        "mcnemar_exact_two_sided": mcnemar_exact_two_sided(none_only, mmh_only),
    }


def mcnemar_exact_two_sided(none_only: int, mmh_only: int) -> float:
    """Exact two-sided McNemar p-value without SciPy."""
    discordant = int(none_only) + int(mmh_only)
    if discordant == 0:
        return 1.0
    # Under H0 each discordant pair is correct with p=0.5.
    tail = sum(math.comb(discordant, k) for k in range(0, min(none_only, mmh_only) + 1))
    p = min(1.0, 2.0 * tail / (2 ** discordant))
    return p


def compare_result_files(
    *,
    domain: Domain | str,
    none_result_path: str | Path,
    mmh_result_path: str | Path,
) -> dict[str, Any]:
    domain = Domain.coerce(domain)
    none_result = load_result(none_result_path)
    mmh_result = load_result(mmh_result_path)
    comparison: dict[str, Any] = {
        "domain": domain.value,
        "none": summarize_result(none_result, domain),
        "mmh": summarize_result(mmh_result, domain),
    }
    none_predictions = _extract_predictions(none_result, domain)
    mmh_predictions = _extract_predictions(mmh_result, domain)
    if none_predictions and mmh_predictions:
        comparison["paired"] = paired_predictions(none_predictions, mmh_predictions)
    else:
        comparison["paired"] = {
            "available": False,
            "reason": "one or both result files lack per_problem predictions",
        }
    return comparison


def aggregate_domain_comparisons(comparisons: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    comparisons = list(comparisons)
    total_none_correct = sum(int(item["none"]["correct"]) for item in comparisons)
    total_none_total = sum(int(item["none"]["total"]) for item in comparisons)
    total_mmh_correct = sum(int(item["mmh"]["correct"]) for item in comparisons)
    total_mmh_total = sum(int(item["mmh"]["total"]) for item in comparisons)
    return {
        "domains": len(comparisons),
        "none_correct": total_none_correct,
        "none_total": total_none_total,
        "none_accuracy": total_none_correct / total_none_total if total_none_total else 0.0,
        "mmh_correct": total_mmh_correct,
        "mmh_total": total_mmh_total,
        "mmh_accuracy": total_mmh_correct / total_mmh_total if total_mmh_total else 0.0,
        "delta_accuracy": (
            (total_mmh_correct / total_mmh_total) - (total_none_correct / total_none_total)
            if total_none_total and total_mmh_total else 0.0
        ),
        "per_domain": list(comparisons),
    }
