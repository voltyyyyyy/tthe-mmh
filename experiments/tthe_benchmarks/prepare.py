"""Dataset preparation/verification for the TTHE benchmark suite.

No benchmark data is vendored except the hard-slice manifests.  This module
pulls the external datasets when the program is run in an environment with
network access; in the current sandbox it can only validate local slices and
BIRD roots.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .spec import BENCHMARKS, Domain, DatasetSpec, REPO_ROOT, get_spec


def _manifest_path(data_root: Path) -> Path:
    return data_root / "dataset_manifest.json"


def _write_manifest(data_root: Path, records: dict[str, Any]) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    _manifest_path(data_root).write_text(
        json.dumps(records, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def prepare_livecodebench(data_root: Path, version: str = "test6") -> dict[str, Any]:
    """Download the LiveCodeBench release JSONL used by the loader."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # noqa: BLE001
        return {"domain": "livecodebench", "status": "missing_dependency", "error": str(exc)}
    try:
        path = hf_hub_download(
            "livecodebench/code_generation_lite", f"{version}.jsonl", repo_type="dataset",
        )
    except Exception as exc:  # noqa: BLE001
        return {"domain": "livecodebench", "status": "download_failed", "error": str(exc)}
    return {
        "domain": "livecodebench",
        "status": "ready",
        "source": f"livecodebench/code_generation_lite/{version}.jsonl",
        "cache_path": str(path),
    }


def prepare_hf_dataset(data_root: Path, spec: DatasetSpec) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        return {"domain": spec.domain.value, "status": "missing_dependency", "error": str(exc)}
    try:
        dataset = load_dataset(spec.source_id, split=spec.split)
    except Exception as exc:  # noqa: BLE001
        return {"domain": spec.domain.value, "status": "download_failed", "error": str(exc)}
    return {
        "domain": spec.domain.value,
        "status": "ready",
        "source": spec.source_id,
        "split": spec.split,
        "num_rows": len(dataset),
    }


def prepare_bird(data_root: Path, bird_root: str | Path | None) -> dict[str, Any]:
    if not bird_root:
        return {
            "domain": "bird",
            "status": "needs_root",
            "error": "pass --bird-root pointing at a BIRD Mini-Dev directory; it is not vendored",
        }
    root = Path(bird_root)
    dev_file = root / "mini_dev.json"
    if not dev_file.exists():
        dev_file = root / "dev.json"
    if not dev_file.exists():
        return {
            "domain": "bird",
            "status": "missing_dev",
            "error": f"no mini_dev.json or dev.json under {root}",
        }
    db_dirs = [root / name for name in ("dev_databases", "database", "databases")]
    db_dir = next((path for path in db_dirs if path.is_dir()), None)
    if db_dir is None:
        return {
            "domain": "bird",
            "status": "missing_databases",
            "error": f"no dev_databases/database/databases directory under {root}",
        }
    sqlite_files = list(db_dir.glob("*/*.sqlite"))
    if not sqlite_files:
        return {
            "domain": "bird",
            "status": "missing_sqlite",
            "error": f"no SQLite databases found under {db_dir}",
        }
    return {
        "domain": "bird",
        "status": "ready",
        "source_root": str(root.resolve()),
        "dev_file": dev_file.name,
        "num_sqlite": len(sqlite_files),
    }


def prepare_domain(
    domain: Domain | str,
    data_root: str | Path = "benchmark_data",
    *,
    bird_root: str | Path | None = None,
    version: str = "test6",
) -> dict[str, Any]:
    spec = get_spec(domain)
    root = Path(data_root)
    if spec.domain is Domain.BIRD:
        result = prepare_bird(root, bird_root)
    elif spec.domain is Domain.LIVECODEBENCH:
        result = prepare_livecodebench(root, version=version)
    else:
        result = prepare_hf_dataset(root, spec)
    result["slice_path"] = spec.slice_path
    result["expected_count"] = spec.expected_count
    try:
        result["slice_count"] = len(spec.item_ids())
        result["slice_valid"] = result["slice_count"] == spec.expected_count
    except Exception as exc:  # noqa: BLE001
        result["slice_valid"] = False
        result["slice_error"] = str(exc)
    return result


def prepare_all(
    data_root: str | Path = "benchmark_data",
    *,
    bird_root: str | Path | None = None,
    version: str = "test6",
) -> dict[str, Any]:
    root = Path(data_root)
    records: dict[str, Any] = {}
    for domain in BENCHMARKS:
        records[domain.value] = prepare_domain(
            domain, root, bird_root=bird_root, version=version,
        )
    _write_manifest(root, {"data_root": str(root.resolve()), "datasets": records})
    return records
