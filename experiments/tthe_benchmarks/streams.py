"""Task-stream construction for the single-domain and mixed benchmarks.

The mixed study is domain-blocked round-robin: each batch belongs to one
domain, while the domain changes across the stream.  This preserves each
domain's harness/interface while creating recurring cross-domain contexts.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .spec import BENCHMARKS, Domain, get_spec


@dataclass(slots=True)
class BatchSpec:
    batch_id: int
    domain: Domain
    item_ids: list[str]
    slice_start: int
    slice_end: int
    repeat_index: int

    def to_dict(self) -> dict:
        value = asdict(self)
        value["domain"] = self.domain.value
        return value


@dataclass(slots=True)
class StreamSpec:
    name: str
    batch_size: int
    repeats: int
    seed: int
    domain_order: tuple[Domain, ...]
    batches: list[BatchSpec] = field(default_factory=list)

    @property
    def total_tasks(self) -> int:
        return sum(len(batch.item_ids) for batch in self.batches)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "batch_size": self.batch_size,
            "repeats": self.repeats,
            "seed": self.seed,
            "domain_order": [domain.value for domain in self.domain_order],
            "total_batches": len(self.batches),
            "total_tasks": self.total_tasks,
            "batches": [batch.to_dict() for batch in self.batches],
        }

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _chunks(items: Sequence[str], size: int) -> Iterable[tuple[int, int, list[str]]]:
    for start in range(0, len(items), size):
        chunk = list(items[start:start + size])
        yield start, start + len(chunk), chunk


def single_domain_stream(
    domain: Domain | str,
    *,
    batch_size: int = 5,
    seed: int = 0,
    shuffle: bool = False,
    name: str | None = None,
) -> StreamSpec:
    spec = get_spec(domain)
    items = spec.item_ids()
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)
    batches: list[BatchSpec] = []
    for batch_index, (start, end, chunk) in enumerate(_chunks(items, batch_size)):
        batches.append(BatchSpec(
            batch_id=batch_index, domain=spec.domain, item_ids=chunk,
            slice_start=start, slice_end=end, repeat_index=0,
        ))
    return StreamSpec(
        name=name or f"single_{spec.domain.value}",
        batch_size=batch_size,
        repeats=1,
        seed=seed,
        domain_order=(spec.domain,),
        batches=batches,
    )


def mixed_domain_stream(
    *,
    batch_size: int = 5,
    repeats: int = 3,
    seed: int = 0,
    domain_order: Sequence[Domain | str] = (
        Domain.BIRD, Domain.LIVECODEBENCH, Domain.SWE, Domain.DS1000,
    ),
    shuffle_within_domain: bool = False,
    name: str = "mixed_all",
) -> StreamSpec:
    order = tuple(Domain.coerce(domain) for domain in domain_order)
    if not order:
        raise ValueError("domain_order must not be empty")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    rng = random.Random(seed)
    items_by_domain: dict[Domain, list[str]] = {}
    for domain in order:
        items = get_spec(domain).item_ids()
        if shuffle_within_domain:
            rng.shuffle(items)
        items_by_domain[domain] = items

    batches: list[BatchSpec] = []
    batch_id = 0
    for repeat_index in range(repeats):
        for domain in order:
            items = items_by_domain[domain]
            for start, end, chunk in _chunks(items, batch_size):
                batches.append(BatchSpec(
                    batch_id=batch_id, domain=domain, item_ids=chunk,
                    slice_start=start, slice_end=end, repeat_index=repeat_index,
                ))
                batch_id += 1
    return StreamSpec(
        name=name,
        batch_size=batch_size,
        repeats=repeats,
        seed=seed,
        domain_order=order,
        batches=batches,
    )


def summarize_stream(stream: StreamSpec) -> dict:
    by_domain: dict[str, dict[str, int]] = {}
    for batch in stream.batches:
        record = by_domain.setdefault(batch.domain.value, {"batches": 0, "tasks": 0})
        record["batches"] += 1
        record["tasks"] += len(batch.item_ids)
    return {
        "name": stream.name,
        "total_batches": len(stream.batches),
        "total_tasks": stream.total_tasks,
        "by_domain": by_domain,
        "domain_order": [domain.value for domain in stream.domain_order],
    }
