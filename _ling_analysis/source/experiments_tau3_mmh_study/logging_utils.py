"""Structured logging for experiment runs.

Mirrors the lab's observability contract so an offline sweep and a real tau3 run produce
the same artifacts and can be analysed by the same code:

    <run_dir>/
      manifest.json     frozen comparison basis (config, schedule, code revision, seeds)
      command.txt       exact invocation
      schedule.json     the round plan and its viability verdict
      rounds.json       per-round metrics
      lifetimes.json    per-guideline ledger
      patches.json      every staged/resolved patch with the subset that resolved it
      traits.csv        per-rule trait vector at staging (volatile-vs-stable analysis)
      exit.json         exit code and finish time

Written incrementally (after every round) so an interrupted run still leaves analysable
evidence -- the archived tau3 run lost its partial state because the cache only flushed
at the end.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def code_revision(root: Path | None = None) -> dict[str, str]:
    """Best-effort code revision, so a run is attributable after the fact."""
    root = root or Path(__file__).resolve().parents[2]
    info = {"revision": "unknown", "dirty": "unknown"}
    try:
        info["revision"] = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        info["dirty"] = str(len(status.splitlines()))
    except Exception:  # noqa: BLE001 - revision is metadata, never fatal
        pass
    return info


class RunLogger:
    """Append-only structured logger for one experiment run."""

    def __init__(self, run_dir: str | Path, *, run_name: str) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.started = time.time()
        self._rounds: list[dict[str, Any]] = []
        self._traits: list[dict[str, Any]] = []
        self._exit: dict[str, Any] | None = None

    # ------------------------------------------------------------- manifest
    def write_manifest(self, **fields: Any) -> None:
        payload = {
            "run_name": self.run_name,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **code_revision(),
            **fields,
        }
        self._write("manifest.json", payload)

    def write_command(self, argv: Sequence[str]) -> None:
        (self.dir / "command.txt").write_text(" ".join(argv) + "\n", encoding="utf-8")

    def write_schedule(self, plans: Sequence[Any], viability: Mapping[str, Any]) -> None:
        rows = []
        for plan in plans:
            rows.append({
                "round_id": plan.round_id,
                "regime": plan.regime.name,
                "domain": plan.regime.domain,
                "regime_round_index": plan.regime_round_index,
                "validation_label": plan.validation_label,
                "validation_task_keys": list(plan.validation_task_keys),
                "collects_failures": plan.collects_failures,
            })
        self._write("schedule.json", {"rounds": rows, "viability": dict(viability)})

    # --------------------------------------------------------------- rounds
    def log_round(self, record: Any, *, traits: Iterable[Mapping[str, Any]] = ()) -> None:
        self._rounds.append(record.as_dict() if hasattr(record, "as_dict") else dict(record))
        self._write("rounds.json", self._rounds)
        new_traits = [dict(t) for t in traits]
        if new_traits:
            self._traits.extend(new_traits)
            self._write_csv("traits.csv", self._traits)

    def log_lifetimes(self, lifetimes: Iterable[Any]) -> None:
        rows = []
        for life in lifetimes:
            row = asdict(life) if hasattr(life, "__dataclass_fields__") else dict(life)
            if hasattr(life, "hit_rate"):
                row["hit_rate"] = life.hit_rate
            if hasattr(life, "lifespan_rounds"):
                row["lifespan_rounds"] = life.lifespan_rounds
            rows.append(row)
        self._write("lifetimes.json", rows)

    def log_patches(self, patches: Sequence[Mapping[str, Any]]) -> None:
        self._write("patches.json", [dict(p) for p in patches])

    def log_promotion_report(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._write("promotion_report.json", [dict(r) for r in rows])

    # ----------------------------------------------------------------- exit
    def finish(self, exit_code: int, **extra: Any) -> None:
        self._exit = {
            "exit_code": int(exit_code),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_time_s": round(time.time() - self.started, 3),
            **extra,
        }
        self._write("exit.json", self._exit)

    # --------------------------------------------------------------- helpers
    def _write(self, name: str, payload: Any) -> None:
        (self.dir / name).write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

    def _write_csv(self, name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        keys = sorted({k for row in rows for k in row})
        lines = [",".join(keys)]
        for row in rows:
            cells = []
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (list, tuple, set)):
                    value = "|".join(str(v) for v in value)
                text = str(value).replace('"', "'")
                cells.append(f'"{text}"' if "," in text else text)
            lines.append(",".join(cells))
        (self.dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    @property
    def rounds(self) -> list[dict[str, Any]]:
        return list(self._rounds)
