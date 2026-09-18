"""Benchmark specifications for the TTHE none-vs-MMH studies.

The hard-slice manifests are already in the repository; the underlying datasets
are not.  This module records exactly what the paper's benchmark suite is and
where each component comes from.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


class Domain(str, Enum):
    BIRD = "bird"
    LIVECODEBENCH = "livecodebench"
    SWE = "swe"
    DS1000 = "ds1000"

    @classmethod
    def coerce(cls, value: "Domain | str") -> "Domain":
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())


class Arm(str, Enum):
    NONE = "none"
    MMH = "mmh"


@dataclass(frozen=True)
class DatasetSpec:
    domain: Domain
    display_name: str
    slice_path: str
    expected_count: int
    slice_format: str
    source_kind: str
    source_id: str
    split: str | None
    context_fields: tuple[str, ...]
    selection: str
    runtime_notes: str
    mmh_adapter: bool = False

    @property
    def path(self) -> Path:
        return REPO_ROOT / self.slice_path

    def load_slice(self) -> list[Any]:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if self.domain is Domain.BIRD:
            items = raw["cross"]
        elif self.domain is Domain.LIVECODEBENCH:
            items = raw["items"]
        else:
            items = raw
        if not isinstance(items, list):
            raise ValueError(f"{self.domain.value}: slice is not a list")
        if len(items) != self.expected_count:
            raise ValueError(
                f"{self.domain.value}: expected {self.expected_count} items, found {len(items)}"
            )
        return items

    def item_ids(self) -> list[str]:
        items = self.load_slice()
        ids: list[str] = []
        for item in items:
            if self.domain is Domain.BIRD:
                ids.append(f"{item[0]}:{item[1]}")
            elif self.domain is Domain.LIVECODEBENCH:
                ids.append(str(item["qid"]))
            else:
                ids.append(str(item))
        return ids

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["domain"] = self.domain.value
        return value


BENCHMARKS: dict[Domain, DatasetSpec] = {
    Domain.BIRD: DatasetSpec(
        domain=Domain.BIRD,
        display_name="Text-to-SQL: BIRD Mini-Dev hard50",
        slice_path="text_to_sql/slices/genuine_hard50.json",
        expected_count=50,
        slice_format="[db_id, mini_dev_index] pairs",
        source_kind="local_bird_root",
        source_id="BIRD Mini-Dev",
        split="mini_dev",
        context_fields=("db_id",),
        selection=(
            "50-question hard slice built from BIRD Mini-Dev by auditing questions the baseline "
            "repeatedly missed; retained cases are unambiguous and substantively hard."
        ),
        runtime_notes=(
            "Requires dataset.bird_root with dev.json/mini_dev.json and dev_databases/<db_id>/<db_id>.sqlite. "
            "Set BIRD_DEV_FILE=mini_dev.json. BIRD data is not vendored in this repo."
        ),
        mmh_adapter=False,
    ),
    Domain.LIVECODEBENCH: DatasetSpec(
        domain=Domain.LIVECODEBENCH,
        display_name="Competitive programming: LiveCodeBench hard60",
        slice_path="livecodebench/slices/hard60.json",
        expected_count=60,
        slice_format="items: [{qid, difficulty}]",
        source_kind="huggingface_hub",
        source_id="livecodebench/code_generation_lite",
        split="test6",
        context_fields=("platform", "difficulty", "contest_date"),
        selection=(
            "60-problem hard slice from the most recent contamination-controlled release window "
            "(problems published after the solver's training cutoff)."
        ),
        runtime_notes=(
            "The loader downloads test6.jsonl from HuggingFace. Public tests are the label-free signal; "
            "hidden tests are measurement-only."
        ),
        mmh_adapter=True,
    ),
    Domain.SWE: DatasetSpec(
        domain=Domain.SWE,
        display_name="Software engineering: SWE-bench Verified hard40",
        slice_path="swe/slices/hard40.json",
        expected_count=40,
        slice_format="instance_id strings",
        source_kind="huggingface",
        source_id="princeton-nlp/SWE-Bench_Verified",
        split="test",
        context_fields=("repo",),
        selection=(
            "40-instance hard slice selected by gold-patch complexity: multi-file or large diffs."
        ),
        runtime_notes=(
            "Needs Docker plus mini-swe-agent and swebench. The baseline is off-the-shelf mini-swe-agent; "
            "TTHE evolves that scaffold."
        ),
        mmh_adapter=False,
    ),
    Domain.DS1000: DatasetSpec(
        domain=Domain.DS1000,
        display_name="Data-science code: DS-1000 hard50",
        slice_path="ds1000/slices/hard50.json",
        expected_count=50,
        slice_format="dataset row indices",
        source_kind="huggingface",
        source_id="xlangai/DS-1000",
        split="test",
        context_fields=("library", "perturbation"),
        selection=(
            "50-problem hard slice selected by reference-solution length."
        ),
        runtime_notes=(
            "Data comes from HuggingFace xlangai/DS-1000; execution needs numpy/pandas/scipy/scikit-learn."
        ),
        mmh_adapter=False,
    ),
}


def get_spec(domain: Domain | str) -> DatasetSpec:
    return BENCHMARKS[Domain.coerce(domain)]


def all_specs() -> list[DatasetSpec]:
    return list(BENCHMARKS.values())


def validate_slices() -> dict[str, Any]:
    """Validate the in-repo hard slices without touching the external datasets."""
    report: dict[str, Any] = {"valid": True, "domains": {}}
    for domain, spec in BENCHMARKS.items():
        try:
            ids = spec.item_ids()
            report["domains"][domain.value] = {
                "path": spec.slice_path,
                "count": len(ids),
                "first": ids[0] if ids else None,
                "last": ids[-1] if ids else None,
                "source_id": spec.source_id,
                "mmh_adapter": spec.mmh_adapter,
            }
        except Exception as exc:  # noqa: BLE001
            report["valid"] = False
            report["domains"][domain.value] = {"error": str(exc)}
    return report


def arms_for_study() -> list[Arm]:
    """The user's final decision: none vs MMH only, no flat arm."""
    return [Arm.NONE, Arm.MMH]
