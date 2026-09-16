"""End-to-end demo of the tau3 x MMH loop on a synthetic non-stationary stream.

Runs without a model, network, or tau3 checkout.  Its purpose is to exercise the whole
pipeline -- regime shifts, proposal, staging, round-scoped validation, promotion,
rollback -- and emit the same artifacts a real run produces, so the analysis path can be
validated before GPU hours are spent.

The stream is built to contain the phenomenon under study: guidelines that help in one
regime and are *invalidated* by the next.  That is what a real drift experiment is looking
for, and it is what the MMH paper's stationary benchmarks never exercise.

    python -m experiments.tau3_mmh.demo [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meta_memory import MMHConfig, SQLiteStore  # noqa: E402

from experiments.tau3_mmh import (  # noqa: E402
    ExperimentRunner,
    FailureSignature,
    GuidelineMemory,
    OfflineProposer,
    RegimeSpec,
    Schedule,
    TaskObservation,
    ValidationSubset,
    assess_schedule,
    build_subsets,
    detect_change_points,
)

# A guideline that repairs symptom "no_tool_call" is proposed in the first regime and
# works; the world then changes so that it no longer does.  Regimes are long enough to
# promote (>= 4 rounds, since elapsed_age lags successful_lifespan by one).
REGIMES = [
    RegimeSpec(name="A1-airline", domain="airline", rounds=5),
    RegimeSpec(name="A2-retail", domain="retail", rounds=5),
    RegimeSpec(name="B1-telecom", domain="telecom", rounds=4),
    RegimeSpec(name="B2-banking", domain="banking_knowledge", rounds=4),
]


class DriftingWorld:
    """Failures are repairable in early regimes and unrepairable after the shift."""

    def __init__(self, *, shift_at_round: int) -> None:
        self.shift_at_round = shift_at_round
        self.history: list[dict[str, object]] = []

    @staticmethod
    def _helped(rules) -> bool:
        # The learned guideline addresses the pre-shift symptom only.
        return any("no_tool_call" in r.rule_id for r in rules)

    def evaluate(self, plan, active_rules):
        self.history.append({"round": plan.round_id, "phase": "eval",
                             "n_rules": len(active_rules)})
        pre_shift = plan.round_id < self.shift_at_round
        ok = self._helped(active_rules) if pre_shift else False
        return [
            TaskObservation(task_key=f"{plan.regime.domain}:s{plan.round_id}:{i}",
                            round_id=plan.round_id, reward=1.0 if ok else 0.0,
                            latency_s=120.0, cost_tokens=900)
            for i in range(4)
        ]

    def validate(self, plan, active_rules):
        self.history.append({"round": plan.round_id, "phase": "val",
                             "n_rules": len(active_rules)})
        pre_shift = plan.round_id < self.shift_at_round
        ok = self._helped(active_rules) if pre_shift else False
        return [
            TaskObservation(task_key=f"{plan.validation_label}:{i}",
                            round_id=plan.round_id, reward=1.0 if ok else 0.0)
            for i in range(2)
        ]

    class _Source:
        def failures_for(self, plan, observations):
            if any(o.scored and not o.success for o in observations):
                return [FailureSignature(domain=plan.regime.domain,
                                         task_id=f"t{plan.round_id}",
                                         signature="no_tool_call_before_answer",
                                         evidence={"round": plan.round_id})]
            return []

    @property
    def trace_source(self):
        return DriftingWorld._Source()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="experiments/tau3_mmh/_demo_run")
    parser.add_argument("--shift-at-round", type=int, default=7,
                        help="round at which the distribution changes")
    args = parser.parse_args(argv)

    config = MMHConfig(
        promotion_age=3,
        promotion_subsets=2,
        promotion_confidence=0.6,
        influence_threshold=0.0,
        context_match_confidence=0.0,
    )
    verdict = assess_schedule(REGIMES, config)
    print(f"schedule viable: {verdict['viable']}  rounds={verdict['total_rounds']}")
    for entry in verdict["regimes"]:
        print(f"  {entry['regime']:14} rounds={entry['rounds']} "
              f"promote={entry['can_promote']} retire={entry['can_retire_stable']}")
    if not verdict["viable"]:
        print("REFUSING: schedule cannot exercise the mechanism", file=sys.stderr)
        return 2

    validation_keys = [f"v{i}" for i in range(10)]
    schedule = Schedule(regimes=REGIMES, subsets=build_subsets(validation_keys, count=5))
    print(f"validation subsets: {schedule.subset_coverage()}")

    memory = GuidelineMemory(store=SQLiteStore(":memory:"), config=config)
    world = DriftingWorld(shift_at_round=args.shift_at_round)
    runner = ExperimentRunner(
        memory=memory,
        proposer=OfflineProposer(max_per_round=2, confidence=0.9),
        schedule=schedule,
        evaluate=world.evaluate,
        validate=world.validate,
        trace_source=world.trace_source,
        output_dir=args.out,
    )
    records = runner.run()

    print()
    print("round regime          rate  promoted  tier(s/v)")
    for r in records:
        print(f"  {r.round_id:>3} {r.regime:14} {r.success_rate:5.2f}  "
              f"{','.join(r.promoted) or '-':<10} "
              f"{r.tier_counts['stable']}/{r.tier_counts['volatile']}")

    rates = [r.success_rate for r in records]
    points = detect_change_points(rates)
    print()
    print(f"detected change points (round index): {points}")
    print(f"planted shift at round {args.shift_at_round}")

    rolled_back = [p for p in runner.resolved_patches if p["status"] == "rolled_back"]
    print(f"rollbacks: {len(rolled_back)}")
    print(f"stable tier at end: {[r.rule_id for r in memory.stable()]}")
    print(f"artifacts written to: {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
