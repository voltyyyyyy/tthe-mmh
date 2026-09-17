"""Deterministic, labeled walkthrough of the standalone Meta-Memory Harness.

Run with ``python -m meta_memory.demo``.  This is intentionally an offline
teaching fixture: its labels are supplied as validation evidence.  Production
TTHE adaptation instead supplies public executor evidence only.
"""

from __future__ import annotations

import json
from typing import Any

from .engine import MetaMemoryEngine
from .store import SQLiteStore
from .types import Patch, PatchOperation, PatchStatus, Rule, RuleTier, ValidationEvidence


def _event(patch_id: str, *, success: bool, subset: str, round_id: int, recovered: bool | None = None) -> ValidationEvidence:
    return ValidationEvidence(
        patch_id=patch_id,
        success=success,
        subset_id=subset,
        round_id=round_id,
        original_failure_recovered=recovered,
        details={"fixture": "deterministic-labeled-demo"},
    )


def run_demo() -> dict[str, Any]:
    """Run a minimal stream through repair, recurrence, promotion, and drift."""
    store = SQLiteStore(":memory:")
    engine = MetaMemoryEngine(store)
    try:
        repair_rule = Rule(
            rule_id="contrastive-verify",
            phi="identical retrieved labels and low confidence",
            psi="retrieve counterexamples and ask for comparative verification",
            omega="pending",
            initial_confidence=0.9,
            provenance={"demo": True},
        )
        repair = Patch(
            patch_id="repair-contrastive",
            operation=PatchOperation.ADD,
            result_rules=(repair_rule,),
            context=repair_rule.phi,
            judge_confidence=0.9,
            provenance={"failure_case": "rare-class-retrieval"},
        )

        # The first held-out recurrence resolves the pending patch. Later,
        # independent recurrences calibrate its rule rather than re-resolving it.
        engine.update_cycle([repair], 1, [_event(repair.patch_id, success=True, subset="gastro", round_id=1, recovered=True)])
        engine.update_cycle([], 2, [_event(repair.patch_id, success=True, subset="dermatology", round_id=2)])
        engine.update_cycle([], 3, [_event(repair.patch_id, success=True, subset="vascular", round_id=3)])
        promotion = engine.update_cycle([], 4)

        harmful_rule = Rule(
            rule_id="always-first-label",
            phi="any retrieval query",
            psi="return the first retrieved label without verification",
            omega="pending",
            initial_confidence=0.9,
        )
        harmful = Patch(
            patch_id="harmful-shortcut",
            operation=PatchOperation.ADD,
            result_rules=(harmful_rule,),
            context=harmful_rule.phi,
            judge_confidence=0.9,
        )
        engine.update_cycle([harmful], 5, [_event(harmful.patch_id, success=False, subset="shifted-medical", round_id=5)])

        # A distribution shift is observed on the stable rule.  Five independent
        # round observations are accumulated but no automatic destructive edit is
        # made; a future conservative Merge/Delete proposal may use this record.
        for round_id, subset in enumerate(("shift-a", "shift-b", "shift-a", "shift-b", "shift-a"), start=6):
            engine.record_rule_observation("contrastive-verify", False, subset, round_id)

        stable = store.get_rule("contrastive-verify")
        rolled_back = store.get_patch(harmful.patch_id)
        assert stable is not None and stable.tier is RuleTier.STABLE
        assert rolled_back is not None and rolled_back.status is PatchStatus.ROLLED_BACK
        assert store.get_rule("always-first-label") is None

        return {
            "repair_patch_status": store.get_patch(repair.patch_id).status.value,  # type: ignore[union-attr]
            "repair_rule": {
                "tier": stable.tier.value,
                "confidence": stable.confidence,
                "successful_lifespan": stable.successful_lifespan,
                "independent_subsets": sorted(stable.independent_subsets),
            },
            "promoted_rule_ids": [rule.rule_id for rule in promotion["promoted"]],
            "harmful_patch_status": rolled_back.status.value,
            "harmful_rule_present": store.get_rule("always-first-label") is not None,
            "precedent_count": len(store.list_precedents()),
            "distribution_shift_failure_signals": len(stable.provenance.get("stable_failure_history", [])),
        }
    finally:
        store.close()


def main() -> None:
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
