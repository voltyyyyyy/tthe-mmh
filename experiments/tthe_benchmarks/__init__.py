"""Two-study benchmark suite for TTHE none vs MMH."""

from .commands import (
    RunCommand,
    UnsupportedArm,
    mixed_study_plan,
    single_domain_command,
    single_study_commands,
    write_plan,
)
from .prepare import prepare_all, prepare_domain
from .report import aggregate_domain_comparisons, compare_result_files, mcnemar_exact_two_sided
from .spec import Arm, BENCHMARKS, DatasetSpec, Domain, all_specs, arms_for_study, get_spec, validate_slices
from .streams import BatchSpec, StreamSpec, mixed_domain_stream, single_domain_stream, summarize_stream

__all__ = [
    "Arm",
    "BENCHMARKS",
    "BatchSpec",
    "DatasetSpec",
    "Domain",
    "RunCommand",
    "StreamSpec",
    "UnsupportedArm",
    "aggregate_domain_comparisons",
    "all_specs",
    "arms_for_study",
    "compare_result_files",
    "get_spec",
    "mcnemar_exact_two_sided",
    "mixed_domain_stream",
    "mixed_study_plan",
    "prepare_all",
    "prepare_domain",
    "single_domain_command",
    "single_domain_stream",
    "single_study_commands",
    "summarize_stream",
    "validate_slices",
    "write_plan",
]
