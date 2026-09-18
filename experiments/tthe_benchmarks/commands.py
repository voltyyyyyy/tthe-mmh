"""Command planning for the TTHE none-vs-MMH benchmark studies."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .spec import Arm, Domain, REPO_ROOT, get_spec
from .streams import StreamSpec, mixed_domain_stream, single_domain_stream


class UnsupportedArm(RuntimeError):
    pass


@dataclass(slots=True)
class RunCommand:
    study: str
    domain: Domain
    arm: Arm
    run_name: str
    argv: list[str]
    cwd: str
    env: dict[str, str] = field(default_factory=dict)
    output_result: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "study": self.study,
            "domain": self.domain.value,
            "arm": self.arm.value,
            "run_name": self.run_name,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "env": dict(self.env),
            "output_result": self.output_result,
        }

    def shell(self) -> str:
        env_prefix = " ".join(f"{key}={value}" for key, value in sorted(self.env.items()))
        command = " ".join(self.argv)
        return f"{env_prefix} {command}".strip()


def _python_module(module: str) -> list[str]:
    return [sys.executable, "-u", "-m", module]


def single_domain_command(
    domain: Domain | str,
    arm: Arm | str,
    *,
    run_name: str,
    batch_size: int = 5,
    group: int = 2,
    max_rounds: int = 3,
    model: str = "deepseek-v4-flash",
    total_budget: float | None = None,
    memory_budget_fraction: float = 0.10,
    initial_harness: str = "bare",
    fresh: bool = True,
) -> RunCommand:
    domain = Domain.coerce(domain)
    arm = arm if isinstance(arm, Arm) else Arm(str(arm).lower())
    spec = get_spec(domain)
    if arm is Arm.MMH and not spec.mmh_adapter:
        raise UnsupportedArm(
            f"MMH adapter is not implemented for {domain.value}; only livecodebench has one in this repo"
        )
    run_dir = f"runs/tthe_bench_{domain.value}_{arm.value}_{run_name}"
    env: dict[str, str] = {}
    if domain is Domain.BIRD:
        env["BIRD_DEV_FILE"] = "mini_dev.json"
    if domain is Domain.LIVECODEBENCH:
        argv = _python_module("experiments.lcb_mmh_online.live_runner")
        argv += ["--arm", arm.value, "--run-name", run_name, "--batch-size", str(batch_size),
                 "--group", str(group), "--max-rounds", str(max_rounds), "--model", model,
                 "--initial-harness", initial_harness]
        if fresh:
            argv.append("--fresh")
        if arm is Arm.MMH:
            if total_budget is None:
                raise ValueError("MMH arm requires total_budget")
            argv += ["--total-budget", str(total_budget),
                     "--memory-budget-fraction", str(memory_budget_fraction)]
        argv += ["--pilot", spec.slice_path]
    elif domain is Domain.BIRD:
        argv = _python_module("text_to_sql.optimize")
        argv += ["--cross-set", spec.slice_path, "--batch-size", str(batch_size),
                 "--group", str(group), "--max-rounds", str(max_rounds), "--model", model,
                 "--run-name", run_name, "--initial-harness", initial_harness]
        if fresh:
            argv.append("--fresh")
    elif domain is Domain.DS1000:
        argv = _python_module("ds1000.ds1000_optimize")
        argv += ["--pilot", spec.slice_path, "--batch-size", str(batch_size),
                 "--group", str(group), "--max-rounds", str(max_rounds), "--model", model,
                 "--run-name", run_name, "--initial-harness", initial_harness]
        if fresh:
            argv.append("--fresh")
    elif domain is Domain.SWE:
        argv = _python_module("swe.swe_optimize")
        argv += ["--pilot", spec.slice_path, "--batch-size", str(batch_size),
                 "--group", str(group), "--max-rounds", str(max_rounds), "--model", model,
                 "--run-name", run_name, "--initial-harness", initial_harness]
        if fresh:
            argv.append("--fresh")
    else:
        raise UnsupportedArm(f"unsupported domain {domain}")

    return RunCommand(
        study="single",
        domain=domain,
        arm=arm,
        run_name=run_name,
        argv=argv,
        cwd=str(REPO_ROOT),
        env=env,
        output_result=f"{run_dir}/result.json",
    )


def single_study_commands(
    domain: Domain | str,
    *,
    run_name: str,
    arms: Sequence[Arm | str] = (Arm.NONE, Arm.MMH),
    **kwargs: Any,
) -> list[RunCommand]:
    return [
        single_domain_command(domain, arm, run_name=run_name, **kwargs)
        for arm in arms
    ]


def mixed_study_plan(
    *,
    batch_size: int = 5,
    repeats: int = 3,
    seed: int = 0,
    run_name: str = "mixed_all",
) -> dict[str, Any]:
    """Return the mixed-stream plan.

    MMH mixed execution is intentionally marked unsupported until each domain
    has a public-only MMH adapter and the domain loops can share memory.
    """
    stream = mixed_domain_stream(batch_size=batch_size, repeats=repeats, seed=seed, name=run_name)
    unsupported = [
        domain.value for domain in stream.domain_order
        if not get_spec(domain).mmh_adapter
    ]
    return {
        "study": "mixed",
        "stream": stream.to_dict(),
        "arms": ["none", "mmh"],
        "none_execution": "requires a cross-domain stream runner (not yet implemented)",
        "mmh_execution": "requires cross-domain shared memory plus MMH adapters",
        "unsupported_mmh_domains": unsupported,
    }


def write_plan(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
