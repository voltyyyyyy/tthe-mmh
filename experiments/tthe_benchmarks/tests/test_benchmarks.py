"""Offline tests for the TTHE benchmark specifications and streams."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.tthe_benchmarks.commands import UnsupportedArm, single_domain_command
from experiments.tthe_benchmarks.report import mcnemar_exact_two_sided, paired_predictions, Prediction
from experiments.tthe_benchmarks.spec import Arm, Domain, arms_for_study, get_spec, validate_slices
from experiments.tthe_benchmarks.streams import mixed_domain_stream, single_domain_stream, summarize_stream


def test_hard_slices_have_paper_counts() -> None:
    report = validate_slices()
    assert report["valid"], report
    assert report["domains"]["bird"]["count"] == 50
    assert report["domains"]["livecodebench"]["count"] == 60
    assert report["domains"]["swe"]["count"] == 40
    assert report["domains"]["ds1000"]["count"] == 50


def test_arms_are_none_and_mmh_only() -> None:
    assert [arm.value for arm in arms_for_study()] == ["none", "mmh"]


def test_single_domain_stream_partitions_slice() -> None:
    stream = single_domain_stream(Domain.LIVECODEBENCH, batch_size=5)
    assert len(stream.batches) == 12
    assert stream.total_tasks == 60
    assert all(len(batch.item_ids) == 5 for batch in stream.batches)
    assert [batch.batch_id for batch in stream.batches] == list(range(12))


def test_mixed_stream_is_domain_blocked_round_robin() -> None:
    stream = mixed_domain_stream(batch_size=5, repeats=2, seed=0)
    assert len(stream.batches) == 2 * (10 + 12 + 8 + 10)
    assert stream.total_tasks == 2 * (50 + 60 + 40 + 50)
    first_four = [batch.domain for batch in stream.batches[:4]]
    assert first_four == [Domain.BIRD, Domain.BIRD, Domain.BIRD, Domain.BIRD] or all(
        batch.domain is Domain.BIRD for batch in stream.batches[:10]
    )
    summary = summarize_stream(stream)
    assert summary["by_domain"]["livecodebench"]["tasks"] == 120
    assert summary["by_domain"]["swe"]["tasks"] == 80
    assert [batch.batch_id for batch in stream.batches] == list(range(len(stream.batches)))


def test_command_planning_lcb_none_and_mmh() -> None:
    none = single_domain_command(Domain.LIVECODEBENCH, Arm.NONE, run_name="x")
    assert "experiments.lcb_mmh_online.live_runner" in none.argv
    assert "--arm" in none.argv and "none" in none.argv
    mmh = single_domain_command(Domain.LIVECODEBENCH, Arm.MMH, run_name="x", total_budget=1000)
    assert "mmh" in mmh.argv
    assert "--total-budget" in mmh.argv
    try:
        single_domain_command(Domain.BIRD, Arm.MMH, run_name="x", total_budget=1000)
    except UnsupportedArm:
        pass
    else:
        raise AssertionError("BIRD MMH should be unsupported until an adapter exists")


def test_report_paired_mcnemar_and_aggregation() -> None:
    none = [Prediction("a", True, "x"), Prediction("b", False, "x"), Prediction("c", True, "x")]
    mmh = [Prediction("a", True, "x"), Prediction("b", True, "x"), Prediction("c", False, "x")]
    paired = paired_predictions(none, mmh)
    assert paired["none_only"] == 1
    assert paired["mmh_only"] == 1
    assert abs(paired["mcnemar_exact_two_sided"] - 1.0) < 1e-12
    assert mcnemar_exact_two_sided(0, 0) == 1.0


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function) and function.__module__ == __name__:
            function()
            print(f"PASS {name}")
