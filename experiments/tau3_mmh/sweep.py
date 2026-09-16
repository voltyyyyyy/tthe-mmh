"""Seeded multi-run sweep with the full logging structure.

Two purposes:

  1. Validate the whole analysis path at volume, producing the same artifact layout a real
     tau3 run produces.
  2. Measure what the *design* can detect: does the schedule see a planted distribution
     shift, and does the tiered arm behave differently from a flat arm?

The world is synthetic and its ground truth is planted, so nothing here is evidence about
real agent behaviour.  It bounds the design's sensitivity.

Mechanism being exercised: each regime has a failure cause.  Proposals address either the
real cause (valid) or a plausible-but-wrong one (spurious).  Only valid proposals pass
validation, so the promotion gate has something to filter -- without that, the tiered and
flat arms are identical by construction and the experiment measures nothing.

    python -m experiments.tau3_mmh.sweep --seeds 0 1 2 3 --out DIR
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meta_memory import MMHConfig, SQLiteStore  # noqa: E402

from experiments.tau3_mmh import (  # noqa: E402
    ExperimentRunner,
    FailureSignature,
    Guideline,
    GuidelineMemory,
    OfflineProposer,
    Proposal,
    RegimeSpec,
    Schedule,
    TaskObservation,
    assess_schedule,
    build_subsets,
    detect_change_points,
)
from experiments.tau3_mmh.logging_utils import RunLogger  # noqa: E402
from experiments.tau3_mmh.traits import (  # noqa: E402
    auc,
    cliffs_delta,
    summarise_traits,
)

# Regimes must exceed promotion_age or nothing can promote (see README).
REGIMES = [
    RegimeSpec(name="A1-airline", domain="airline", rounds=5),
    RegimeSpec(name="A2-retail", domain="retail", rounds=5),
    RegimeSpec(name="B1-telecom", domain="telecom", rounds=5),
    RegimeSpec(name="B2-banking", domain="banking_knowledge", rounds=5),
]

# Failure cause each regime is built around.
REGIME_CAUSE = {
    "A1-airline": "no_tool_call_before_answer",
    "A2-retail": "unconfirmed_mutation",
    "B1-telecom": "missing_escalation",
    "B2-banking": "partial_refund_math",
}

# Causes that sound plausible but do not repair anything in this world. Proposals for
# these must fail validation, which is what gives the gate a job.
SPURIOUS_CAUSES = ["repeated_tool_call", "identity_not_verified", "stale_cache_reuse"]

# How many guidelines the agent's prompt can carry. The cap is what makes a spurious
# rule harmful rather than merely useless: it displaces a useful one.
ACTIVE_CAP = 4


class SeededWorld:
    """Task outcomes with noise, a planted shift, and valid-vs-spurious proposals."""

    def __init__(self, *, seed: int, shift_round: int, repair_prob: float = 0.85,
                 stale_prob: float = 0.20, tasks_per_round: int = 12,
                 val_tasks: int = 6) -> None:
        self.rng = random.Random(seed)
        self.shift_round = shift_round
        self.repair_prob = repair_prob
        self.stale_prob = stale_prob
        self.tasks_per_round = tasks_per_round
        self.val_tasks = val_tasks
        self.phase = "pre"

    def _valid_ids(self, active_rules) -> set[str]:
        """Which *valid* causes are currently covered by an in-force guideline."""
        covered = set()
        for cause in REGIME_CAUSE.values():
            if any(cause in r.rule_id for r in active_rules):
                covered.add(cause)
        return covered

    def _rate(self, plan, active_rules) -> float:
        cause = REGIME_CAUSE[plan.regime.name]
        covered = cause in self._valid_ids(active_rules)
        if plan.round_id < self.shift_round:
            return self.repair_prob if covered else 0.15
        return self.stale_prob if covered else 0.05

    def _observe(self, plan, active_rules, n):
        rate = self._rate(plan, active_rules)
        if rate < 0.4:
            self.phase = "failing"
        return [
            TaskObservation(
                task_key=f"{plan.regime.domain}:r{plan.round_id}:{i}",
                round_id=plan.round_id,
                reward=1.0 if self.rng.random() < rate else 0.0,
                latency_s=self.rng.uniform(60.0, 200.0),
                cost_tokens=self.rng.randint(600, 1400),
                turns=self.rng.randint(4, 9),
            )
            for i in range(n)
        ]

    def evaluate(self, plan, active_rules):
        # The agent sees the top-N highest-confidence guidelines, simulating a prompt cap.
        ranked = sorted(active_rules, key=lambda r: (-r.confidence, r.rule_id))[:ACTIVE_CAP]
        if ranked is not active_rules:
            plan = replace(plan)
        return self._observe(plan, ranked, self.tasks_per_round)

    def validate(self, plan, active_rules):
        ranked = sorted(active_rules, key=lambda r: (-r.confidence, r.rule_id))[:ACTIVE_CAP]
        return self._observe(plan, ranked, self.val_tasks)

    class _Source:
        def failures_for(self, plan, observations):
            if any(o.scored and not o.success for o in observations):
                return [FailureSignature(domain=plan.regime.domain,
                                         task_id=f"t{plan.round_id}",
                                         signature=REGIME_CAUSE[plan.regime.name],
                                         evidence={"round": plan.round_id})]
            return []

    @property
    def trace_source(self):
        return SeededWorld._Source()


class MixedProposer:
    """Proposes the observed cause plus a plausible-but-wrong one.

    Mirrors a real proposer: it sees failures, offers a hypothesis, and is sometimes
    wrong.  The spurious half is what validation must reject.
    """

    def __init__(self, *, seed: int) -> None:
        self.rng = random.Random(seed + 999)

    def propose(self, *, round_id: int, failures, memory):
        existing = {r.rule_id for r in memory.store.list_rules()}
        out = []
        for failure in failures:
            if failure.signature not in REGIME_CAUSE.values():
                continue
            real_id = f"g-{round_id}-{failure.signature}"
            if real_id not in existing:
                out.append(self._make(real_id, failure.signature, confidence=0.9, valid=True))
            spurious = self.rng.choice(SPURIOUS_CAUSES)
            spurious_id = f"g-{round_id}-{spurious}"
            if spurious_id not in existing:
                out.append(self._make(spurious_id, spurious, confidence=0.55, valid=False))
        return out

    @staticmethod
    def _make(guideline_id: str, cause: str, *, confidence: float, valid: bool) -> Proposal:
        offline = OfflineProposer()
        phi, psi = OfflineProposer._signature_text(cause)
        return Proposal(
            guideline=Guideline(
                guideline_id=guideline_id, phi=phi, psi=psi, omega="pending",
                initial_confidence=confidence,
                provenance={"cause": cause, "planted_valid": valid},
            ),
            hypothesis=f"failures match cause '{cause}'",
            expected_benefit="recurrence of this cause should stop failing"
            if valid else "suspected but unverified cause",
            risks="may be a spurious correlate of the observed failures"
            if valid else "planted spurious candidate; should fail validation",
            source_tasks=(cause,),
            judge_confidence=confidence,
        )


def _config(arm: str, base: MMHConfig) -> MMHConfig:
    if arm == "tiered":
        return replace(base)
    if arm == "flat":
        # No gate at all: everything promotes on sight. Isolates the gate from merely
        # having a guideline list, which is the comparison that matters.
        return replace(base, promotion_age=0, promotion_subsets=0, promotion_confidence=0.0)
    raise ValueError(f"unknown arm: {arm}")


def run_one(*, arm: str, seed: int, base: MMHConfig, out_dir: Path,
            shift_round: int, regimes=REGIMES) -> dict:
    run_name = f"{arm}-s{seed}"
    logger = RunLogger(out_dir / run_name, run_name=run_name)
    config = _config(arm, base)
    schedule = Schedule(regimes=regimes,
                        subsets=build_subsets([f"v{i}" for i in range(10)], count=5))
    verdict = assess_schedule(regimes, config)
    logger.write_manifest(
        arm=arm, seed=seed, shift_round=shift_round, synthetic=True,
        mmh_config={k: getattr(config, k) for k in (
            "promotion_age", "promotion_subsets", "promotion_confidence",
            "influence_threshold", "stable_failure_rounds")},
        regimes=[r.as_dict() for r in regimes], viability=verdict,
    )
    logger.write_command(sys.argv)
    plans = schedule.build()
    logger.write_schedule(plans, verdict)

    memory = GuidelineMemory(store=SQLiteStore(":memory:"), config=config)
    world = SeededWorld(seed=seed, shift_round=shift_round)
    runner = ExperimentRunner(
        memory=memory, proposer=MixedProposer(seed=seed), schedule=schedule,
        evaluate=world.evaluate, validate=world.validate,
        trace_source=world.trace_source, output_dir=out_dir / run_name,
    )

    logged_traits: list[dict] = []
    trait_cursor = 0
    for plan in plans:
        record = runner._run_round(plan)
        runner.records.append(record)
        new_rows = [t.as_dict() for t in runner.staged_traits[trait_cursor:]]
        trait_cursor = len(runner.staged_traits)
        logged_traits.extend(new_rows)
        logger.log_round(record, traits=new_rows)

    # Label is attached after the run; the trait VALUES stay frozen at staging so the
    # predictor is not contaminated by the outcome. planted_valid is read from the
    # proposal provenance via the stored patch, not inferred.
    planted: dict[str, int] = {}
    for patch in memory.store.list_patches():
        for rule in patch.result_rules:
            planted[rule.rule_id] = int(bool(
                (rule.provenance or {}).get("planted_valid")
                or (patch.provenance or {}).get("planted_valid")
            ))
    promoted_ids = {r.rule_id for r in memory.stable()}
    rolled_back_ids = {
        rule.rule_id for patch in memory.store.list_patches()
        if patch.status.value == "rolled_back" for rule in patch.result_rules
    }
    for row in logged_traits:
        row["promoted"] = int(row["rule_id"] in promoted_ids)
        row["rolled_back"] = int(row["rule_id"] in rolled_back_ids)
        row["is_stable_now"] = "stable" if row["rule_id"] in promoted_ids else "volatile"
        row["planted_valid"] = planted.get(row["rule_id"], -1)
    logger._write_csv("traits.csv", logged_traits)
    logger.log_lifetimes(runner.lifetimes.values())
    logger.log_patches(runner.resolved_patches)
    logger.log_promotion_report(memory.promotion_report())

    rates = [r.success_rate for r in runner.records]
    points = detect_change_points(rates)
    # The runner was driven round-by-round for per-round logging, so the end-of-run
    # integrity verdict must be requested explicitly. Without this the run has NO
    # integrity report and would be reported as healthy by default -- fail closed instead.
    runner.finish()
    runner.require_finished()
    integrity = runner.integrity.as_dict()
    summary = {
        "arm": arm, "seed": seed, "run_dir": str(out_dir / run_name),
        "analysable": runner.integrity.analysable,
        "integrity_clean": runner.integrity.clean,
        "integrity_summary": runner.integrity.summary(),
        "integrity": integrity,
        "rounds": len(runner.records),
        "staged": sum(r.staged for r in runner.records),
        "promotions": sum(len(r.promoted) for r in runner.records),
        "rollbacks": sum(1 for p in runner.resolved_patches if p["status"] == "rolled_back"),
        "rate_pre_shift": statistics.fmean(rates[: shift_round - 1]) if rates else float("nan"),
        "rate_post_shift": statistics.fmean(rates[shift_round - 1:]) if rates else float("nan"),
        "final_stable": len(memory.stable()),
        "final_volatile": len(memory.volatile()),
        "detected_change_points": points,
        "detected_shift": any(abs(p - shift_round) <= 2 for p in points),
        "n_staged_traits": len(logged_traits),
        "total_cost_tokens": sum(r.total_cost_tokens for r in runner.records),
    }
    logger.finish(0, **summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--arms", nargs="+", default=["tiered", "flat"])
    parser.add_argument("--shift-round", type=int, default=11)
    parser.add_argument("--out", default="experiments/tau3_mmh/_runs/sweep")
    args = parser.parse_args(argv)

    base = MMHConfig(promotion_age=3, promotion_subsets=2, promotion_confidence=0.6,
                     influence_threshold=0.0, context_match_confidence=0.0)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    verdict = assess_schedule(REGIMES, base)
    print(f"schedule: {verdict['total_rounds']} rounds, viable={verdict['viable']}")
    if not verdict["viable"]:
        print("REFUSING: " + "; ".join(verdict["problems"]), file=sys.stderr)
        return 2

    summaries = []
    for arm in args.arms:
        for seed in args.seeds:
            summary = run_one(arm=arm, seed=seed, base=base, out_dir=out_dir,
                              shift_round=args.shift_round)
            summaries.append(summary)
            print(f"  {arm:7} seed={seed:<3} staged={summary['staged']:<3} "
                  f"prom={summary['promotions']:<3} rb={summary['rollbacks']:<3} "
                  f"pre={summary['rate_pre_shift']:.2f} post={summary['rate_post_shift']:.2f} "
                  f"shift={summary['detected_shift']}")

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))

    # Refuse to aggregate compromised runs into a headline number. A fatally broken run
    # aborts inside the runner; a run with defects still completes but its result cannot
    # be interpreted as evidence about the mechanism, so it is excluded and reported.
    # Fail closed: a missing integrity report means the verdict was never computed, which
    # must NOT be read as "no problems found". Only an explicit analysable=true counts.
    compromised = [s for s in summaries if s.get("analysable") is not True]
    if compromised:
        print()
        print(f"WARNING: {len(compromised)}/{len(summaries)} runs are not analysable "
              f"(integrity defects). Excluded from aggregate:")
        for s in compromised:
            print(f"  {s['arm']}-s{s['seed']}: {s.get('integrity_summary', 'unspecified')}")

    analysable = [s for s in summaries if s.get("analysable") is True]
    if not analysable:
        print("REFUSING to report aggregates: no analysable runs", file=sys.stderr)
        return 3

    print()
    print(f"{'arm':8}{'n':>3}{'staged':>8}{'promoted':>10}{'rollback':>10}"
          f"{'pre':>7}{'post':>7}{'stable':>8}{'shift':>8}")
    for arm in args.arms:
        rows = [s for s in analysable if s["arm"] == arm]
        if not rows:
            continue
        print(f"{arm:8}{len(rows):>3}"
              f"{statistics.fmean([r['staged'] for r in rows]):>8.1f}"
              f"{statistics.fmean([r['promotions'] for r in rows]):>10.2f}"
              f"{statistics.fmean([r['rollbacks'] for r in rows]):>10.2f}"
              f"{statistics.fmean([r['rate_pre_shift'] for r in rows]):>7.3f}"
              f"{statistics.fmean([r['rate_post_shift'] for r in rows]):>7.3f}"
              f"{statistics.fmean([r['final_stable'] for r in rows]):>8.2f}"
              f"{sum(1 for r in rows if r['detected_shift'])}/{len(rows):>5}")

    traits = _load_traits(out_dir)
    if traits:
        print()
        print(f"trait analysis (frozen at staging): {len(traits)} staged observations")
        # Two labels. Promotion is the gate's own decision; planted_valid is ground truth
        # about whether the candidate could ever work. They answer different questions, so
        # both are reported rather than conflated.
        for label_name in ("promoted", "planted_valid"):
            usable = [t for t in traits if t.get(label_name) not in ("", None, "-1")]
            if not usable:
                continue
            labels = [int(float(t[label_name])) for t in usable]
            if len(set(labels)) < 2:
                continue
            print(f"  label={label_name}  n={len(usable)}  positive={statistics.fmean(labels):.2f}")
            print(f"  {'trait':<12}{'AUC':>8}{'cliffs_delta':>14}")
            for name in ("novelty", "overlap", "uniqueness", "sparsity", "surprise", "lifespan"):
                try:
                    values = [float(t[name]) for t in usable]
                except (KeyError, ValueError):
                    continue
                pos = [v for v, y in zip(values, labels) if y]
                neg = [v for v, y in zip(values, labels) if not y]
                print(f"  {name:<12}{auc(values, labels):>8.3f}"
                      f"{cliffs_delta(pos, neg):>14.3f}")
            print()
    return 0


def _load_traits(out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for csv_path in sorted(out_dir.glob("*/traits.csv")):
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) < 2:
            continue
        header = lines[0].split(",")
        for line in lines[1:]:
            cells = [c.strip('"') for c in line.split(",")]
            rows.append(dict(zip(header, cells)))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
