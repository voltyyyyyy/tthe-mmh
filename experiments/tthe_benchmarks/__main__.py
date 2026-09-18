"""CLI for TTHE benchmark data preparation and run planning."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .commands import (
    UnsupportedArm,
    mixed_study_plan,
    single_study_commands,
    write_plan,
)
from .prepare import prepare_all
from .spec import Arm, Domain, arms_for_study, validate_slices
from .streams import mixed_domain_stream, single_domain_stream, summarize_stream


def _cmd_validate(args: argparse.Namespace) -> int:
    print(json.dumps(validate_slices(), indent=2, sort_keys=True))
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    report = prepare_all(args.data_root, bird_root=args.bird_root, version=args.version)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _cmd_plan_single(args: argparse.Namespace) -> int:
    stream = single_domain_stream(args.domain, batch_size=args.batch_size, seed=args.seed)
    payload = {
        "study": "single",
        "stream": summarize_stream(stream),
        "stream_detail": stream.to_dict(),
        "arms": [arm.value for arm in arms_for_study()],
    }
    if args.output:
        write_plan(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_plan_mixed(args: argparse.Namespace) -> int:
    payload = mixed_study_plan(
        batch_size=args.batch_size, repeats=args.repeats, seed=args.seed, run_name=args.run_name,
    )
    if args.output:
        write_plan(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_commands_single(args: argparse.Namespace) -> int:
    try:
        commands = single_study_commands(
            args.domain,
            run_name=args.run_name,
            batch_size=args.batch_size,
            group=args.group,
            max_rounds=args.max_rounds,
            model=args.model,
            total_budget=args.total_budget,
            memory_budget_fraction=args.memory_budget_fraction,
            initial_harness=args.initial_harness,
            fresh=not args.no_fresh,
        )
    except (UnsupportedArm, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2
    payload = [command.to_dict() for command in commands]
    if args.output:
        write_plan(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    for command in commands:
        print(command.shell())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TTHE benchmark suite utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-slices", help="validate the in-repo hard slices")
    validate.set_defaults(func=_cmd_validate)

    prepare = sub.add_parser("prepare", help="pull/verify external benchmark datasets")
    prepare.add_argument("--data-root", default="benchmark_data")
    prepare.add_argument("--bird-root", default=None)
    prepare.add_argument("--version", default="test6")
    prepare.set_defaults(func=_cmd_prepare)

    plan_single = sub.add_parser("plan-single", help="write a single-domain stream plan")
    plan_single.add_argument("--domain", choices=[domain.value for domain in Domain], required=True)
    plan_single.add_argument("--batch-size", type=int, default=5)
    plan_single.add_argument("--seed", type=int, default=0)
    plan_single.add_argument("--output", default=None)
    plan_single.set_defaults(func=_cmd_plan_single)

    plan_mixed = sub.add_parser("plan-mixed", help="write the mixed-domain stream plan")
    plan_mixed.add_argument("--batch-size", type=int, default=5)
    plan_mixed.add_argument("--repeats", type=int, default=3)
    plan_mixed.add_argument("--seed", type=int, default=0)
    plan_mixed.add_argument("--run-name", default="mixed_all")
    plan_mixed.add_argument("--output", default=None)
    plan_mixed.set_defaults(func=_cmd_plan_mixed)

    commands_single = sub.add_parser("commands-single", help="print the none/MMH commands for one domain")
    commands_single.add_argument("--domain", choices=[domain.value for domain in Domain], required=True)
    commands_single.add_argument("--run-name", default="hard")
    commands_single.add_argument("--batch-size", type=int, default=5)
    commands_single.add_argument("--group", type=int, default=2)
    commands_single.add_argument("--max-rounds", type=int, default=3)
    commands_single.add_argument("--model", default="deepseek-v4-flash")
    commands_single.add_argument("--total-budget", type=float, default=None)
    commands_single.add_argument("--memory-budget-fraction", type=float, default=0.10)
    commands_single.add_argument("--initial-harness", default="bare")
    commands_single.add_argument("--no-fresh", action="store_true")
    commands_single.add_argument("--output", default=None)
    commands_single.set_defaults(func=_cmd_commands_single)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
