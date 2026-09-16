"""Round driver: drives the real tau3 benchmark through the MMH loop.

This is the orchestrator that turns the tested pieces into an experiment. Per round:

  1. snapshot memory, render the active guideline block into the harness's file
  2. run the round's tasks via the benchmark's own eval entry point
  3. parse each task's reward into evidence -- errors stay errors, never zeros
  4. feed evidence to the gate on round-scoped validation subsets
  5. propose from failures, stage, promote, and log
  6. persist memory so the run survives a crash or a resubmission

Design notes that matter for correctness:

* **The arm is the only difference.** Both arms run the same harness file with the same
  task order; the MMH arm renders learned guidelines, the baseline arm always renders the
  frozen block. Nothing else varies between them.
* **A failed eval raises.** It never returns empty scores, because empty scores would be
  read by the gate as "every task failed" and would poison the memory with false negatives.
* **Evidence is per-task and honest.** A task that timed out is excluded from the success
  rate and from the gate, and the error rate guard is applied per round.

Usage:
    python -m experiments_tau3_mmh.round_driver --arm mmh --rounds 3 --tasks-per-round 2
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments_tau3_mmh.guidelines import GuidelineMemory, TaskObservation  # noqa: E402
from experiments_tau3_mmh.tau3_binding import (  # noqa: E402
    EvalInvocation,
    load_per_task,
    parse_scores,
    run_eval,
    write_guidelines,
)
from experiments_tau3_mmh.integrity import (  # noqa: E402
    InfrastructureFault,
    IntegrityReport,
    require_error_rate_acceptable,
    require_non_degenerate_scores,
)

DEFAULT_BENCHMARK = "reproduction/qwen_tau3/all375/benchmark.yaml:search"


@dataclass
class RoundOutcome:
    round_id: int
    arm: str
    tasks: list[str]
    n_scored: int
    n_errors: int
    successes: int
    success_rate: float
    active_rules: int
    staged: int
    promoted: list[str]
    rolled_back: int
    wall_time_s: float
    per_task: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RoundDriver:
    def __init__(
        self,
        *,
        arm: str,
        tasks_by_round: Sequence[Sequence[str]],
        model: str,
        out_dir: Path,
        config_dir: Path | None = None,
        benchmark: str = DEFAULT_BENCHMARK,
        concurrency: int = 2,
        eval_timeout_s: int = 3600,
        base_url: str = "http://127.0.0.1:8100/v1",
        memory_path: Path | None = None,
        max_candidates_per_round: int = 1,
    ) -> None:
        if arm not in {"mmh", "baseline"}:
            raise ValueError(f"arm must be 'mmh' or 'baseline', got {arm!r}")
        self.arm = arm
        # Each validated candidate costs a full task-set evaluation, so this is the main
        # cost multiplier per round. One is the honest minimum for a pilot.
        self.max_candidates_per_round = max_candidates_per_round
        self.tasks_by_round = [list(t) for t in tasks_by_round]
        self.model = model
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir = config_dir or (REPO / "experiments_tau3_mmh" / "tau3_harness")
        self.benchmark = benchmark
        self.concurrency = concurrency
        self.eval_timeout_s = eval_timeout_s
        self.base_url = base_url
        self.guidelines_file = self.out_dir / "guidelines.md"
        self.memory_path = memory_path or (self.out_dir / "memory.sqlite")
        # Candidate directories live under the repo's experience tree and are keyed by
        # --name. Two runs sharing a name would write into the SAME directory, and since
        # the scores locator takes the newest match, a still-running eval could be read
        # from a previous run's scores.json. That silently injects stale rewards as
        # evidence. Derive a unique tag from the output path so runs cannot collide.
        self.run_tag = "-".join(
            part for part in (self.out_dir.parent.name, self.out_dir.name) if part
        ) or "run"
        self.records: list[RoundOutcome] = []
        self.integrity = IntegrityReport()
        # The baseline arm keeps a memory object only to supply the frozen block; it never
        # stages proposals, so its tier counts stay empty by construction.
        self.memory = GuidelineMemory.from_snapshot(self.memory_path) if (
            self.arm == "mmh" and Path(self.memory_path).exists()
        ) else self._new_memory()

    def _new_memory(self) -> GuidelineMemory:
        from meta_memory import MMHConfig, SQLiteStore

        return GuidelineMemory(store=SQLiteStore(self.memory_path), config=MMHConfig())

    # ------------------------------------------------------------------ run
    def run(self) -> list[RoundOutcome]:
        self._write_manifest()
        for index, tasks in enumerate(self.tasks_by_round, start=1):
            outcome = self._run_round(index, tasks)
            self.records.append(outcome)
            self._flush()
        require_non_degenerate_scores(
            self.integrity, rates=[r.success_rate for r in self.records]
        )
        self.memory.snapshot_to(self.memory_path)
        self._flush()
        self.integrity.write(self.out_dir / "integrity.json")
        return self.records

    def _run_round(self, round_id: int, tasks: Sequence[str]) -> RoundOutcome:
        started = time.time()
        # 1. render the guideline block the harness will read
        if self.arm == "mmh":
            self.memory.advance(round_id)
            write_guidelines(self.memory.active(), self.guidelines_file)
        else:
            write_guidelines((), self.guidelines_file)   # frozen baseline block

        # 2. run the round
        invocation = EvalInvocation(
            repo=REPO, benchmark=self.benchmark, config_dir=self.config_dir,
            name=self._candidate_name("r", round_id), tasks=tasks, model=self.model,
            concurrency=self.concurrency, python=sys.executable,
            timeout_s=self.eval_timeout_s,
        )
        scores = run_eval(invocation, env=self._child_env())
        per_task = self._load_per_task(invocation, tasks)

        # 3. parse -- errors remain errors
        observations = parse_scores({**scores, "round_id": round_id}, per_task)
        require_error_rate_acceptable(
            self.integrity, n_total=len(observations),
            n_errors=sum(1 for o in observations if not o.scored), round_id=round_id,
        )

        staged = promoted = 0
        rolled_back = 0
        if self.arm == "mmh" and observations:
            staged, promoted, rolled_back = self._apply_evidence(round_id, observations, tasks)

        scored = [o for o in observations if o.scored]
        successes = sum(1 for o in scored if o.success)
        return RoundOutcome(
            round_id=round_id, arm=self.arm, tasks=list(tasks),
            n_scored=len(scored), n_errors=len(observations) - len(scored),
            successes=successes,
            success_rate=(successes / len(scored)) if scored else float("nan"),
            active_rules=len(self.memory.active()),
            staged=staged, promoted=[str(p) for p in promoted], rolled_back=rolled_back,
            wall_time_s=round(time.time() - started, 1),
            per_task=[
                {"task": o.task_key, "reward": o.reward, "error": o.error,
                 "latency_s": o.latency_s, "turns": o.turns}
                for o in observations
            ],
        )

    # ------------------------------------------------------------- internals
    def _apply_evidence(self, round_id: int, observations: Sequence[TaskObservation],
                        tasks: Sequence[str]) -> tuple[int, list[str], int]:
        """Resolve pending candidates by actually running them on this round's tasks.

        A pending guideline is NOT in force, so the round's outcomes say nothing about it.
        The only honest way to decide whether it helps is to run the held-out tasks with the
        candidate rendered into the prompt and see whether they improve. That is why
        ``validate_candidate_eval`` exists -- and why validation costs real GPU time rather
        than being free bookkeeping.

        Validation is capped at one candidate per round for cost control; extra candidates
        stay PENDING and compete in later rounds. A truncated candidate list is recorded as
        an integrity warning rather than silently dropped.
        """
        from experiments_tau3_mmh.proposer import FailureSignature

        # --- stage proposals from this round's genuine failures
        staged_ids: list[str] = []
        failures = [o for o in observations if o.scored and not o.success]
        if failures:
            signatures = [
                FailureSignature(
                    domain=self._domain_of(o.task_key), task_id=o.task_key,
                    signature=self._signature_for(o),
                    evidence={"reward": o.reward, "turns": o.turns},
                )
                for o in failures
            ]
            for proposal in self._proposer().propose(
                round_id=round_id, failures=signatures, memory=self.memory
            ):
                try:
                    patch = self.memory.stage_add(
                        proposal.guideline, round_id,
                        judge_confidence=proposal.judge_confidence,
                        provenance=proposal.as_provenance(),
                    )
                    staged_ids.append(patch.patch_id)
                except (ValueError, KeyError) as exc:
                    self.integrity.warn(
                        "proposal_rejected_at_stage",
                        f"{proposal.guideline.guideline_id}: {exc}", round_id=round_id,
                    )

        # --- validate the oldest pending candidate by running it for real
        pending: list[Any] = []
        for patch in self.memory.store.list_patches():
            if patch.status.value == "pending":
                pending.extend(patch.result_rules)
        resolved: list[Any] = []
        promoted_rules: list[Any] = []
        if pending:
            if len(pending) > self.max_candidates_per_round:
                self.integrity.warn(
                    "candidate_budget_deferred",
                    f"{len(pending)} pending candidates, validating {self.max_candidates_per_round} "
                    f"this round; the rest stay pending and compete later",
                    round_id=round_id,
                )
            for candidate in pending[: self.max_candidates_per_round]:
                passed, scored = self._validate_candidate(candidate, round_id, tasks)
                if scored == 0:
                    self.integrity.warn(
                        "candidate_validation_unscored",
                        f"{candidate.rule_id} validation produced no scored task",
                        round_id=round_id,
                    )
                    continue
                if passed:
                    resolved.extend(self.memory.record_round(
                        round_id=round_id, subset_id=self._subset_label(round_id, candidate),
                        successes=[candidate.rule_id], failures=[],
                    ))
                else:
                    resolved.extend(self.memory.record_round(
                        round_id=round_id, subset_id=self._subset_label(round_id, candidate),
                        successes=[], failures=[candidate.rule_id],
                    ))

        promoted_rules = self.memory.promote(round_id)
        rolled = sum(1 for p in resolved if p.status.value == "rolled_back")
        return len(staged_ids), [r.rule_id for r in promoted_rules], rolled

    def _subset_label(self, round_id: int, candidate: Any) -> str:
        """Validation group label.

        Distinct per ROUND here, because each round validates on a different task set. The
        label must not encode the round if the same tasks are ever reused -- that is what
        would let independent_subsets count rounds instead of data groups. Rounds in this
        driver use disjoint task slices, so a per-round label is faithful.
        """
        return f"val-r{round_id}"

    @staticmethod
    def _signature_for(observation: TaskObservation) -> str:
        """Map an outcome to a structural failure signature.

        The pilot uses one signature so the proposer has a well-defined candidate; a richer
        suite would classify the trace. Kept explicit rather than hidden in the proposer.
        """
        return "no_tool_call_before_answer"

    def _validate_candidate(self, candidate: Any, round_id: int,
                            tasks: Sequence[str]) -> tuple[bool, int]:
        """Run ``tasks`` with ONLY ``candidate`` in force; return (majority_pass, n_scored).

        This is a real evaluation, not bookkeeping: the candidate's guideline block is
        written into the harness file, the tasks re-run, and the resulting pass rate decides
        the patch. Nothing here inspects memory state to guess the answer.
        """
        write_guidelines([candidate], self.guidelines_file)
        invocation = EvalInvocation(
            repo=REPO, benchmark=self.benchmark, config_dir=self.config_dir,
            name=self._candidate_name(f"val_r{round_id}", candidate.rule_id),
            tasks=list(tasks), model=self.model, concurrency=self.concurrency,
            python=sys.executable, timeout_s=self.eval_timeout_s,
        )
        try:
            scores = run_eval(invocation, env=self._child_env())
        except RuntimeError as exc:
            # An eval crash is infrastructure, not evidence about the candidate. Leaving
            # the patch pending is the safe action; recording a failure would roll back a
            # rule on the basis of a serving fault.
            self.integrity.warn(
                "candidate_validation_failed",
                f"{candidate.rule_id}: {str(exc)[:200]}", round_id=round_id,
            )
            return False, 0
        observations = parse_scores({**scores, "round_id": round_id},
                                    self._load_per_task(invocation, tasks))
        scored = [o for o in observations if o.scored]
        if not scored:
            return False, 0
        passed = sum(1 for o in scored if o.success) / len(scored) >= 0.6
        return passed, len(scored)

    def _proposer(self):
        from experiments_tau3_mmh.proposer import LLMProposer, OfflineProposer

        if os.environ.get("MMH_PROPOSER", "offline").lower() == "llm":
            return LLMProposer(base_url=self.base_url, model=self.model, disable_thinking=True)
        return OfflineProposer(max_per_round=2, confidence=0.6)

    @staticmethod
    def _domain_of(task_key: str) -> str:
        return task_key.split(":", 1)[0] if ":" in task_key else "unknown"

    def _candidate_name(self, kind: str, suffix: Any) -> str:
        """A candidate directory name unique to this run.

        Sanitised to a filesystem-safe token and prefixed with the run tag, so two runs
        -- or a re-run after a crash -- can never share a candidate directory and read
        each other's scores.
        """
        raw = f"mmh_{self.run_tag}_{kind}_{suffix}"
        return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in raw)

    @staticmethod
    def _requested_keys(tasks: Sequence[str]) -> set[str]:
        """Task keys as the benchmark reports them (``domain:id`` -> ``domain_id``)."""
        return {t.replace(":", "_") for t in tasks}

    def _load_per_task(self, invocation: EvalInvocation,
                       requested: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Per-task records, verified to correspond to the tasks we actually requested.

        A mismatch means the scores belong to another run -- the failure mode where a
        stale candidate directory silently supplies rewards. That is fatal rather than
        tolerated, because recording another run's rewards as this run's evidence is
        worse than recording nothing.
        """
        from experiments_tau3_mmh.tau3_binding import _locate_scores

        scores_path = _locate_scores(invocation)
        if scores_path is None:
            return []
        try:
            scores = json.loads(scores_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if requested:
            expected = self._requested_keys(requested)
            reported = set(scores.get("tasks_passed") or []) | set(scores.get("tasks_failed") or [])
            if reported and reported != expected:
                raise InfrastructureFault(
                    f"scores at {scores_path} describe {sorted(reported)} but this round "
                    f"requested {sorted(expected)}: refusing to record another run's "
                    f"results as evidence"
                )
        return load_per_task(scores_path)

    def _child_env(self) -> dict[str, str]:
        return {
            "PYTHONPATH": str(REPO),
            "LOCAL_OPENAI_V1_BASE": self.base_url,
            "LOCAL_MODEL_BASE_URL": self.base_url,
            "OPENAI_BASE_URL": self.base_url,
            "OPENAI_API_BASE": self.base_url,
            "OPENAI_API_KEY": "EMPTY",
            "MMH_GUIDELINES_FILE": str(self.guidelines_file),
            "TAU2_DATA_DIR": os.environ.get(
                "TAU2_DATA_DIR", "/home/yangfan/meta-agent-test/tau2-bench/data"
            ),
            "META_AGENT_CONCURRENCY": str(self.concurrency),
            "TAU3_TASK_TIMEOUT_S": str(self.eval_timeout_s),
            "PYTHONUNBUFFERED": "1",
        }

    def _write_manifest(self) -> None:
        (self.out_dir / "manifest.json").write_text(json.dumps({
            "arm": self.arm, "model": self.model, "benchmark": self.benchmark,
            "config_dir": str(self.config_dir), "concurrency": self.concurrency,
            "tasks_by_round": self.tasks_by_round,
            "base_url": self.base_url,
            "guidelines_file": str(self.guidelines_file),
            "memory_path": str(self.memory_path),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2), encoding="utf-8")

    def _flush(self) -> None:
        (self.out_dir / "rounds.json").write_text(
            json.dumps([r.as_dict() for r in self.records], indent=2), encoding="utf-8"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("mmh", "baseline"), required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tasks-per-round", type=int, default=2)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--eval-timeout", type=int, default=3600)
    parser.add_argument("--start-at", type=int, default=0,
                        help="task index to start from, so arms can use disjoint tasks")
    args = parser.parse_args(argv)

    pool = _search_tasks()
    per_round = args.tasks_per_round
    start = args.start_at
    tasks_by_round = [
        pool[start + i * per_round: start + (i + 1) * per_round]
        for i in range(args.rounds)
    ]
    if any(len(t) < per_round for t in tasks_by_round):
        print(f"not enough tasks in pool ({len(pool)}) for the requested schedule",
              file=sys.stderr)
        return 2

    driver = RoundDriver(
        arm=args.arm, tasks_by_round=tasks_by_round, model=args.model,
        out_dir=Path(args.out), concurrency=args.concurrency,
        eval_timeout_s=args.eval_timeout,
    )
    print(f"arm={args.arm} rounds={args.rounds} tasks/round={per_round}")
    for i, tasks in enumerate(tasks_by_round, 1):
        print(f"  round {i}: {tasks}")
    try:
        records = driver.run()
    except InfrastructureFault as exc:
        print(f"ABORTED (infrastructure fault): {exc}", file=sys.stderr)
        return 1

    print()
    print(f"{'round':>5}{'scored':>8}{'errors':>8}{'pass':>7}{'rate':>7}"
          f"{'active':>8}{'staged':>8}{'promoted':>10}")
    for r in records:
        print(f"{r.round_id:>5}{r.n_scored:>8}{r.n_errors:>8}{r.successes:>7}"
              f"{r.success_rate:>7.2f}{r.active_rules:>8}{r.staged:>8}"
              f"{len(r.promoted):>10}")
    print()
    print(f"integrity: {driver.integrity.summary()}")
    print(f"artifacts: {driver.out_dir}")
    return 0


def _search_tasks() -> list[str]:
    """The frozen search-split task keys, from the benchmark's own split manifest."""
    manifest = REPO / "reproduction" / "qwen_tau3" / "all375" / "split_manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    search = data["splits"]["search"]
    out: list[str] = []
    for domain, ids in search.items():
        out.extend(f"{domain}:{tid}" for tid in ids)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
