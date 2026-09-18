"""Runner for the separate LiveCodeBench online-MMH experiment.

The default command is credential-free and synthetic.  It exercises the same
typed lifecycle, budget ledger, retrieval, and persistence that an LCB runner
would use; it does not launch a paid benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, build_budget_ledger, build_memory, load_experiment_config
from .fixture import run_offline_demo
from .types import Arm, ProposalCard, stable_hash


def _flat_card() -> ProposalCard:
    return ProposalCard(
        candidate="flat_candidate",
        parent="bare",
        parent_sha256=stable_hash("bare"),
        candidate_sha256=stable_hash("flat_candidate"),
        branch_id=0,
        generation_round="b0r0",
        peer_candidates=(),
        role="conservative repair",
        behavior_change={"change": "flat chronological intervention summary"},
        rationale="flat control fixture",
        expected_effect="reuse public trace lessons without lifecycle",
        origin_task_ids=("origin-flat",),
        trace_refs=("trace:flat",),
        new_hypothesis="flat mode keeps summaries only",
    )


def run_offline(config: ExperimentConfig) -> dict[str, Any]:
    config.validate()
    if config.arm is Arm.NONE:
        return {
            "arm": "none",
            "memory": None,
            "total_budget": config.total_budget,
            "note": "ordinary TTHE; no memory or validation budget required",
        }
    if config.arm is Arm.FLAT:
        flat = build_memory(config)
        assert flat is not None
        card = _flat_card()
        flat.append_card_summary(card)
        flat.record_outcome(card.candidate, "inconclusive", {"n_pass": 0, "n_total": 1})
        package = flat.retrieve_package("public intervention summary")
        return {
            "arm": "flat",
            "memory": flat.summary(),
            "package": package.to_dict(),
            "total_budget": config.total_budget,
            "memory_fraction": config.memory_budget_fraction,
            "note": "flat mode does not consume validation budget for lifecycle transitions",
        }
    # MMH arm: run the deterministic synthetic stream in the configured run dir.
    result = run_offline_demo(Path(config.run_dir))
    ledger = build_budget_ledger(config)
    if ledger is not None:
        result["budget_report"] = ledger.report()
        ledger.close()
    result["arm"] = "mmh"
    result["config"] = config.to_dict()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Separate LiveCodeBench online-MMH experiment (offline by default)")
    parser.add_argument("--arm", choices=[arm.value for arm in Arm], default="mmh")
    parser.add_argument("--config", default=None, help="JSON experiment config")
    parser.add_argument("--run-name", default="lcb_online_mmh")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--stream-seed", type=int, default=0)
    parser.add_argument("--total-budget", type=float, default=None,
                        help="finite total inference budget denominator")
    parser.add_argument("--memory-budget-fraction", type=float, default=0.10)
    parser.add_argument("--retrieval-limit", type=int, default=5)
    parser.add_argument("--output", default=None, help="report JSON path")
    args = parser.parse_args(argv)

    config = load_experiment_config(
        args.config,
        arm=args.arm,
        run_name=args.run_name,
        run_dir=args.run_dir or f"runs/{args.run_name}_{args.arm}",
        stream_seed=args.stream_seed,
        total_budget=args.total_budget,
        memory_budget_fraction=args.memory_budget_fraction,
        retrieval_limit=args.retrieval_limit,
    )
    report = run_offline(config)
    output = Path(args.output) if args.output else Path(config.run_dir) / "offline_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
