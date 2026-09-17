"""tau3 binding: scores from the real benchmark become MMH evidence.

This is the piece that turns the harness from a synthetic simulator into a real
experiment.  It does three things:

  1. **Renders the current guidelines** into the file the harness reads, so the agent's
     behaviour is driven by memory rather than by a hard-coded prompt.
  2. **Runs one round** via the benchmark's own eval entry point, parsing each task's
     reward into ``TaskObservation``.
  3. **Classifies failures honestly**: a task that crashed or timed out becomes an
     ``error`` observation, NOT a reward of 0. The archived tau3 run folded 12 terminal
     ReadTimeouts into its score, which inflated variance and made a serving fault look
     like a capability gap.

Nothing here decides *whether* a task passed -- that is the benchmark's grader.  This
module only transports its verdict.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .guidelines import TaskObservation

# Error markers the benchmark/runtime may emit.  A task whose trace contains one of these
# did not fail the task -- it failed to run, and must not be scored.
_INFRA_MARKERS = (
    "ReadTimeout",
    "ConnectError",
    "APIConnectionError",
    "InternalServerError",
    "TimeoutError",
    "SandboxError",
    "OOM",
    "CUDA out of memory",
)


@dataclass(frozen=True)
class TaskRun:
    """One task's raw outcome as reported by the benchmark."""

    task_key: str
    reward: float
    error: str | None
    latency_s: float | None = None
    turns: int | None = None
    cost_usd: float | None = None


def classify_error(record: Mapping[str, Any]) -> str | None:
    """Return an infrastructure error string, or None if the task genuinely ran.

    A task counts as errored when the runtime reports an error field, the trace shows a
    transport/timeout marker, OR the record shows the task never actually executed.

    That last case was learned the hard way: when the user simulator's model alias was
    missing, every task was recorded as ``reward: 0.0`` with ``num_turns: null``,
    ``wall_time_s: 0.0`` and no trace files at all. Those are *unrun* tasks, and scoring
    them as failures feeds the memory fabricated negative evidence -- the benchmarking
    error masquerading as a model failure that this module exists to prevent.

    Deliberately conservative in one direction only: it never reclassifies a genuine
    failure as an error, because hiding real model failures is the worse mistake.
    """
    explicit = record.get("error") or record.get("exception")
    if explicit:
        return str(explicit)
    blob = " ".join(
        str(record.get(key, "")) for key in ("stderr", "traceback", "status", "reason")
    )
    for marker in _INFRA_MARKERS:
        if marker in blob:
            return marker
    if _never_ran(record):
        return "task never executed (no turns, no elapsed time, no trial artifacts)"
    return None


def _never_ran(record: Mapping[str, Any]) -> bool:
    """True when a record shows a task that produced no execution at all.

    Requires positive evidence of non-execution rather than merely missing fields, so a
    benchmark that legitimately omits metadata is not misread as an infrastructure fault.
    """
    if record.get("passed") is True or (record.get("reward") or 0) > 0:
        return False
    signals = 0
    if record.get("num_turns") is None:
        signals += 1
    elapsed = record.get("wall_time_s", record.get("duration_ms"))
    if elapsed is None or float(elapsed or 0) == 0.0:
        signals += 1
    trial = record.get("trial_dir")
    if trial is not None and str(trial).strip() == "":
        signals += 1
    # All three absent/zero: the task was listed but never ran.
    return signals >= 3


def parse_scores(scores: Mapping[str, Any], per_task: Iterable[Mapping[str, Any]] = ()) -> list[TaskObservation]:
    """Convert a benchmark result into observations, keeping errors separate.

    ``per_task`` entries are expected to carry at least ``task``/``task_name`` and either
    ``reward`` or ``passed``.  A task present in the scores but absent from per-task detail
    is not invented -- it is skipped, because fabricating a reward would be worse than a
    smaller sample.
    """
    round_id = int(scores.get("round_id", 0))
    observations: list[TaskObservation] = []
    for record in per_task:
        key = str(record.get("task") or record.get("task_name") or record.get("short_name") or "")
        if not key:
            continue
        error = classify_error(record)
        if "reward" in record and record["reward"] is not None:
            reward = float(record["reward"])
        elif "passed" in record:
            reward = 1.0 if record["passed"] else 0.0
        else:
            continue
        observations.append(
            TaskObservation(
                task_key=key,
                round_id=round_id,
                reward=reward,
                error=error,
                latency_s=_as_float(record.get("wall_time_s") or record.get("duration_s")),
                turns=_as_int(record.get("num_turns")),
                cost_tokens=None,
            )
        )
    return observations


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------- rendering


def render_guideline_block(rules: Sequence[Any]) -> str:
    """Render active guidelines into the harness's prompt block.

    Kept byte-stable for a given rule set so a diff of two rounds shows exactly what the
    memory changed, which is the audit trail for a harness-edit experiment.
    """
    from .tau3_harness.harness import BASELINE_GUIDELINES
    if not rules:
        return BASELINE_GUIDELINES
    return BASELINE_GUIDELINES + "\nLearned operating rules (apply when the condition matches):\n" + "\n".join(
        f"- When {rule.phi}: {rule.psi}" for rule in rules
    ) + "\n"


def write_guidelines(rules: Sequence[Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_guideline_block(rules), encoding="utf-8")
    return target


# ------------------------------------------------------------ eval invocation


@dataclass
class EvalInvocation:
    """Parameters for one round's benchmark evaluation."""

    repo: Path
    benchmark: str
    config_dir: Path
    name: str
    tasks: Sequence[str]
    model: str
    concurrency: int = 4
    python: str = sys.executable
    timeout_s: int = 7200
    extra_args: Sequence[str] = ()

    def command(self) -> list[str]:
        cmd = [
            self.python, "-m", "meta_agent", "eval",
            "--benchmark", self.benchmark,
            "--config", str(self.config_dir),
            "--name", self.name,
            "--model", self.model,
            "--concurrency", str(self.concurrency),
            "--tasks", ",".join(self.tasks),
            "--keep-failed",
        ]
        cmd.extend(self.extra_args)
        return cmd


def run_eval(invocation: EvalInvocation, *, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Execute one evaluation and return its parsed scores.

    Raises ``RuntimeError`` on a non-zero exit rather than returning empty scores: an eval
    that crashed is an infrastructure fault and must not be silently read as "all tasks
    failed", which would feed the gate false failure evidence.
    """
    cmd = invocation.command()
    full_env = {**os.environ, **(env or {})}
    completed = subprocess.run(
        cmd, cwd=str(invocation.repo), env=full_env,
        capture_output=True, text=True, timeout=invocation.timeout_s,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "eval failed (rc={}): {}\n--- stderr tail ---\n{}".format(
                completed.returncode, shlex.join(cmd), completed.stderr[-4000:]
            )
        )
    scores_path = _locate_scores(invocation)
    if scores_path is None:
        raise RuntimeError(
            f"eval succeeded but no scores.json was found for {invocation.name}; "
            f"stdout tail: {completed.stdout[-2000:]}"
        )
    return json.loads(scores_path.read_text(encoding="utf-8"))


def _locate_scores(invocation: EvalInvocation) -> Path | None:
    """Find the scores file the eval wrote, without guessing its exact schema."""
    root = invocation.repo / "experience"
    if not root.exists():
        return None
    matches = sorted(root.rglob(f"{invocation.name}/scores.json"))
    if matches:
        return matches[-1]
    return None


def load_per_task(scores_path: Path) -> list[dict[str, Any]]:
    """Per-task records adjacent to a scores.json, when the adapter emitted them."""
    for candidate in (scores_path.parent / "per_task", scores_path.parent / "tasks"):
        if candidate.is_dir():
            out = []
            for path in sorted(candidate.glob("*.json")):
                try:
                    out.append(json.loads(path.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    continue
            if out:
                return out
    return []
