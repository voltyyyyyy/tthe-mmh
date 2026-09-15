"""Offline conformance tests for the standalone Meta-Memory Harness."""

from __future__ import annotations

import math
import tempfile
from contextlib import contextmanager
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meta_memory import (
    MMHConfig,
    MetaMemoryEngine,
    Patch,
    PatchOperation,
    PatchStatus,
    Precedent,
    Rule,
    RuleTier,
    SQLiteStore,
    ValidationEvidence,
)
from meta_memory.adapters import (
    HashEmbeddingProvider,
    ResponseValidationError,
    TokenOverlapMatcher,
    parse_context_match_response,
    parse_fallback_score,
    parse_patch_response,
)
from meta_memory.demo import run_demo


@contextmanager
def assert_raises(error: type[BaseException], match: str | None = None):
    """Tiny stdlib replacement for pytest.raises; keeps this suite dependency-free."""
    try:
        yield
    except error as exc:
        if match is not None:
            assert match in str(exc), f"{match!r} not found in {str(exc)!r}"
    else:
        raise AssertionError(f"expected {error.__name__}")


def _engine(path: str = ":memory:", **config: object) -> MetaMemoryEngine:
    return MetaMemoryEngine(SQLiteStore(path), config=MMHConfig(**config))


def _add_patch(patch_id: str, rule_id: str, *, context: str = "rare diagnosis", confidence: float = 0.9, provenance: dict | None = None) -> Patch:
    return Patch(
        patch_id=patch_id,
        operation=PatchOperation.ADD,
        result_rules=(Rule(rule_id, context, "apply a repair", omega="pending", initial_confidence=confidence),),
        context=context,
        judge_confidence=confidence,
        provenance=provenance or {},
    )


def _evidence(patch_id: str, success: bool, subset: str, round_id: int, *, recovered: bool | None = None) -> ValidationEvidence:
    return ValidationEvidence(patch_id, success, subset, round_id, original_failure_recovered=recovered)


def _put(engine: MetaMemoryEngine, rule: Rule) -> None:
    with engine.store.transaction():
        engine.store.put_rule(rule)


def test_demo_is_offline_deterministic_and_exercises_lifecycle() -> None:
    first = run_demo()
    assert first == run_demo()
    assert first["repair_patch_status"] == "validated"
    assert first["repair_rule"]["tier"] == "stable"
    assert first["repair_rule"]["confidence"] == 0.8
    assert first["harmful_patch_status"] == "rolled_back"
    assert not first["harmful_rule_present"]
    assert first["distribution_shift_failure_signals"] == 5


def test_appendix_response_adapters_are_strict_and_deterministic() -> None:
    patch = parse_patch_response(
        {
            "thinking": "The existing context is too broad.",
            "operation": "Add",
            "target_rule_id": "new",
            "new_rule": {
                "phi": "identical labels and low confidence",
                "psi": "retrieve counterexamples",
                "omega": "pending",
                "confidence": 0.7,
                "lifespan": 0,
            },
            "split_details": [],
            "judge_confidence": 0.8,
        },
        patch_id="from-json",
        context="identical labels and low confidence",
    )
    assert patch.operation is PatchOperation.ADD
    assert patch.result_rules[0].phi == "identical labels and low confidence"
    assert parse_fallback_score("7") == 0.7
    match = parse_context_match_response('{"is_match": true, "confidence": 0.69, "reason": "weak"}')
    assert not match.is_match
    with assert_raises(ResponseValidationError):
        parse_patch_response("not JSON", patch_id="bad", context="context")
    with assert_raises(ResponseValidationError):
        parse_fallback_score("7.0")
    with assert_raises(ResponseValidationError):
        parse_patch_response(
            {
                "operation": "Refine", "target_rule_id": "stable", "new_rule": {"phi": "x", "psi": "y", "omega": "pending", "confidence": 0.5, "lifespan": 0}, "split_details": []
            },
            patch_id="stable-refine", context="x", target_tier="stable",
        )

    embedding = HashEmbeddingProvider().embed("identical labels")
    assert math.isclose(sum(item * item for item in embedding), 1.0)
    assert embedding == HashEmbeddingProvider().embed("identical labels")
    assert TokenOverlapMatcher().match("identical labels", "why are identical labels failing?").is_match


def test_eq5_confidence_and_atomic_promotion_after_independent_evidence() -> None:
    engine = _engine()
    patch = _add_patch("p", "r")
    engine.update_cycle([patch], 1, [_evidence("p", True, "a", 1)])
    engine.update_cycle([], 2, [_evidence("p", True, "b", 2)])
    third = engine.update_cycle([], 3, [_evidence("p", True, "c", 3)])
    assert not third["promoted"]  # age is deliberately separate from lifespan
    promoted = engine.update_cycle([], 4)["promoted"]
    rule = engine.store.get_rule("r")
    assert [item.rule_id for item in promoted] == ["r"]
    assert rule is not None and rule.tier is RuleTier.STABLE
    assert math.isclose(rule.confidence, (1 + 3) / (2 + 3))
    assert rule.successful_lifespan == 3
    assert rule.elapsed_age == 3
    assert rule.independent_subsets == {"a", "b", "c"}


def test_eq11_cold_start_gaussian_precedent_and_conflict_resolution() -> None:
    engine = _engine()
    patch = _add_patch("candidate", "candidate-rule", context="same context", confidence=0.41)
    influence, source = engine.estimate_influence(patch)
    assert (influence, source) == (0.41, "judge")
    embedding = tuple(engine.embedding("same context"))
    with engine.store.transaction():
        for patch_id, outcome in (("history-a", True), ("history-b", True), ("history-c", False)):
            engine.store.add_precedent(Precedent(patch_id, embedding, outcome, PatchOperation.ADD, 1))
    influence, source = engine.estimate_influence(patch)
    assert source == "precedent"
    assert math.isclose(influence, 2 / 3)

    target = Rule("target", "target context", "old")
    _put(engine, target)
    low = Patch("a-low", PatchOperation.REFINE, ("target",), (Rule("low-result", "narrow", "low"),), "target context", 0.8)
    high = Patch("z-high", PatchOperation.REFINE, ("target",), (Rule("high-result", "narrow", "high"),), "target context", 0.9)
    # Comparable precedents make both scores equal. The documented deterministic
    # tie break selects lexical patch ID; make the judge cold-start in a fresh engine.
    conflict_engine = _engine()
    _put(conflict_engine, target)
    winners = conflict_engine.filter_and_resolve([high, low])
    assert [item.patch_id for item in winners] == ["z-high"]
    assert conflict_engine.store.get_patch("a-low").status is PatchStatus.DEFERRED  # type: ignore[union-attr]


def test_all_atomic_operations_and_exact_rollback() -> None:
    engine = _engine()
    original = Rule("original", "broad context", "old instruction", omega="old")
    _put(engine, original)
    refine = Patch("refine", PatchOperation.REFINE, ("original",), (Rule("refined", "narrow context", "new instruction"),), "broad context", 0.9)
    engine.stage_patch(refine, 1)
    engine.validate_patch(_evidence("refine", False, "heldout", 1))
    restored = engine.store.get_rule("original")
    assert restored is not None and restored.to_dict() == original.to_dict()
    assert engine.store.get_rule("refined") is None

    split = Patch(
        "split", PatchOperation.SPLIT, ("original",),
        (Rule("split-a", "regime a", "a"), Rule("split-b", "regime b", "b")), "broad context", 0.9,
    )
    engine.stage_patch(split, 2)
    engine.validate_patch(_evidence("split", True, "heldout", 2))
    assert engine.store.get_rule("original") is None
    assert engine.store.get_rule("split-a") is not None
    assert engine.store.get_rule("split-b") is not None

    delete = Patch("delete", PatchOperation.DELETE, ("split-a",), (), "regime a", 0.9)
    engine.stage_patch(delete, 3)
    engine.validate_patch(_evidence("delete", True, "heldout", 3))
    assert engine.store.get_rule("split-a") is None

    left, right = Rule("left", "same", "left"), Rule("right", "same", "right")
    _put(engine, left)
    _put(engine, right)
    merge = Patch("merge", PatchOperation.MERGE, ("left", "right"), (Rule("merged", "same", "both"),), "same", 0.9)
    engine.stage_patch(merge, 4)
    engine.validate_patch(_evidence("merge", False, "heldout", 4))
    assert engine.store.get_rule("left").to_dict() == left.to_dict()  # type: ignore[union-attr]
    assert engine.store.get_rule("right").to_dict() == right.to_dict()  # type: ignore[union-attr]
    assert engine.store.get_rule("merged") is None


def test_duplicate_evidence_and_reopened_store_do_not_duplicate_transitions() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "memory.sqlite"
        engine = _engine(str(path))
        engine.stage_patch(_add_patch("p", "r"), 1)
        event = _evidence("p", True, "subset", 1)
        engine.validate_patch(event)
        engine.validate_patch(event)
        engine.store.close()
        reopened = _engine(str(path))
        patch = reopened.store.get_patch("p")
        rule = reopened.store.get_rule("r")
        assert patch is not None and patch.status is PatchStatus.VALIDATED and patch.validation_count == 1
        assert rule is not None and rule.successes == 1
        reopened.store.close()


def test_original_failure_recovery_and_stable_tier_protection() -> None:
    engine = _engine()
    patch = _add_patch("recover", "recovery-rule", provenance={"original_failure_required": True})
    engine.update_cycle([patch], 1, [_evidence("recover", True, "a", 1)])
    engine.update_cycle([], 2, [_evidence("recover", True, "b", 2)])
    engine.update_cycle([], 3, [_evidence("recover", True, "c", 3)])
    assert not engine.update_cycle([], 4)["promoted"]
    recovered = engine.update_cycle([], 5, [_evidence("recover", True, "d", 5, recovered=True)])
    assert [rule.rule_id for rule in recovered["promoted"]] == ["recovery-rule"]

    stable = engine.store.get_rule("recovery-rule")
    assert stable is not None and stable.tier is RuleTier.STABLE
    forbidden = Patch("forbidden", PatchOperation.REFINE, (stable.rule_id,), (Rule("bad", "x", "y"),), "x", 0.9)
    with assert_raises(ValueError, match="stable rules only"):
        engine.stage_patch(forbidden, 6)
    for round_id, subset in enumerate(("a", "b", "a", "b", "a"), start=6):
        engine.record_rule_observation(stable.rule_id, False, subset, round_id)
    deletion = Patch("retire", PatchOperation.DELETE, (stable.rule_id,), (), "rare diagnosis", 0.9)
    engine.stage_patch(deletion, 11)
    engine.validate_patch(_evidence("retire", True, "a", 11))
    assert engine.store.get_rule(stable.rule_id) is None


if __name__ == "__main__":
    # Kept executable with the Python standard library because this repository has
    # no test-framework dependency. Pytest can still collect the test_* functions.
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"PASS {name}")
