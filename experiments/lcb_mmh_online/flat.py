"""Flat accumulated-advice control for the online-MMH experiment.

Flat mode appends public-trace-derived intervention summaries chronologically,
truncates to the same proposer memory budget, and has no confidence lifecycle,
semantic retrieval, or promotion.  It leaves the saved memory-validation budget
available to ordinary search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .types import MemoryPackage, ProposalCard, canonical_json, stable_hash, utc_now


class FlatAdviceMemory:
    """Chronological summary list with deterministic budget truncation."""

    def __init__(self, path: str | Path, *, prompt_char_budget: int = 6000) -> None:
        self.path = Path(path)
        self.prompt_char_budget = int(prompt_char_budget)
        self._entries: list[dict[str, Any]] = []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("schema") == 1:
                self._entries = list(data.get("entries", []))
            else:
                raise ValueError(f"unsupported flat memory state: {self.path}")

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps({"schema": 1, "entries": self._entries}, indent=2, sort_keys=True),
                        encoding="utf-8")
        temp.replace(self.path)

    def append_card_summary(self, card: ProposalCard | Mapping[str, Any]) -> dict[str, Any]:
        card_obj = card if isinstance(card, ProposalCard) else ProposalCard.from_mapping(card)
        summary = card_obj.summary()
        summary["entry_id"] = stable_hash({"candidate": card_obj.candidate, "summary": summary})
        summary["created_at"] = utc_now()
        summary["outcome_summaries"] = []
        self._entries.append(summary)
        self._write()
        return summary

    def record_outcome(self, candidate: str, outcome: str, public_summary: Mapping[str, Any] | None = None) -> None:
        for entry in self._entries:
            if entry.get("candidate") == candidate:
                entry.setdefault("outcome_summaries", []).append({
                    "outcome": str(outcome),
                    "public_summary": dict(public_summary or {}),
                    "recorded_at": utc_now(),
                })
                self._write()
                return

    def retrieve_package(self, context: str, features: Mapping[str, Any] | None = None) -> MemoryPackage:
        kept_newest_first: list[dict[str, Any]] = []
        omitted = 0
        for entry in reversed(self._entries):
            candidate = dict(entry)
            candidate.setdefault("caveat", "flat advice: chronological summary, no confidence lifecycle")
            trial = kept_newest_first + [candidate]
            if len(canonical_json(list(reversed(trial)))) > self.prompt_char_budget:
                omitted = len(self._entries) - len(kept_newest_first)
                break
            kept_newest_first.append(candidate)
        selected = list(reversed(kept_newest_first))
        package = MemoryPackage(trusted=selected, tentative=[], failed_interventions=[], omitted=omitted)
        package.estimated_chars = len(canonical_json(package.to_dict()))
        while package.estimated_chars > self.prompt_char_budget and package.trusted:
            package.trusted.pop(0)
            package.omitted += 1
            package.estimated_chars = len(canonical_json(package.to_dict()))
        return package

    def summary(self) -> dict[str, Any]:
        return {"entries": len(self._entries)}
