"""Live-runner entry point for the separate online-MMH experiment.

It reuses the existing LiveCodeBench optimizer loop unmodified and swaps its
memory adapter through a legacy-compatible shim.  The default ``none`` path is
untouched; ``flat`` and ``mmh`` require a finite total budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ExperimentConfig, load_experiment_config
from .legacy_shim import LegacyMemoryShim, set_active_config
from .types import Arm


def _build_extra_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm", choices=[arm.value for arm in Arm], default="mmh",
                        help="none (ordinary TTHE), flat (chronological advice), or mmh")
    parser.add_argument("--total-budget", type=float, default=None,
                        help="finite inference budget denominator shared by all arms")
    parser.add_argument("--memory-budget-fraction", type=float, default=0.10,
                        help="fraction of total budget available to memory validation")
    parser.add_argument("--retrieval-limit", type=int, default=5,
                        help="maximum positive/tentative lessons returned to the proposer")
    parser.add_argument("--online-policy", default=None,
                        help="JSON object or path to a JSON online policy config")
    parser.add_argument("--experiment-config", default=None,
                        help="optional JSON experiment config with arm/budget/policy")
    return parser


def _load_policy(value: str | None) -> dict:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return json.loads(value)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(flag in argv for flag in ("--help", "-h")):
        print("online-MMH options: --arm {none,flat,mmh} --total-budget N "
              "--memory-budget-fraction F --retrieval-limit N --online-policy JSON_OR_PATH "
              "--experiment-config JSON_OR_PATH")
        from livecodebench import lcb_optimize
        sys.argv = [sys.argv[0], "--help"]
        return int(lcb_optimize.main() or 0)
    extra, remaining = _build_extra_parser().parse_known_args(argv)
    config = load_experiment_config(
        extra.experiment_config,
        arm=extra.arm,
        total_budget=extra.total_budget,
        memory_budget_fraction=extra.memory_budget_fraction,
        retrieval_limit=extra.retrieval_limit,
        online_policy=_load_policy(extra.online_policy) or None,
    )
    if config.arm is Arm.NONE:
        from livecodebench import lcb_optimize
        sys.argv = [sys.argv[0], *remaining]
        return int(lcb_optimize.main() or 0)
    config.validate()

    # Patch the legacy adapter import point and run the existing loop unchanged.
    from livecodebench import lcb_optimize
    import livecodebench.mmh_adapter as mmh_adapter
    set_active_config(config)
    original = mmh_adapter.MMHAdapter
    mmh_adapter.MMHAdapter = LegacyMemoryShim
    try:
        sys.argv = [sys.argv[0], *remaining, "--memory-mode", "mmh"]
        return int(lcb_optimize.main() or 0)
    finally:
        mmh_adapter.MMHAdapter = original


if __name__ == "__main__":
    raise SystemExit(main())
