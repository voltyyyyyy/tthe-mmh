"""Offline checks for the LiveCodeBench public-only MMH boundary."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from livecodebench.mmh_adapter import MMHAdapter, PublicProblem, valid_mmh_proposal_card
from meta_memory import MetaMemoryEngine, PatchStatus, Rule, RuleTier, SQLiteStore


def _problem(qid: str) -> PublicProblem:
    return PublicProblem(
        qid=qid,
        content="find the sum of two integers",
        starter_code="",
        platform="codeforces",
        difficulty="easy",
        public_tests=[{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}],
    )


def _card(candidate: str, parent: str) -> dict:
    return {
        "candidate": candidate,
        "base_candidate": parent,
        "branch_id": 0,
        "generation_round": "b0r0",
        "peer_candidates": [],
        "role": "conservative repair",
        "behavior_changes": [{"change": "retry after a public-test mismatch"}],
        "applied_rule": {"rule_id": "new-retry-rule", "instruction": "retry after mismatch"},
        "memory_patch": {
            "operation": "ADD",
            "target_rule_ids": [],
            "result_rules": [{
                "phi": "find the sum of two integers",
                "psi": "retry after a public-test mismatch",
                "omega": "pending",
                "confidence": 0.9,
            }],
            "judge_confidence": 0.9,
            "rationale": "The public trace showed an unhandled mismatch.",
        },
    }


def test_future_public_validation_reaches_core_without_private_data() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "candidate.py"
        source.write_text("# frozen candidate artifact\n", encoding="utf-8")
        engine = MetaMemoryEngine(SQLiteStore(root / "memory.sqlite"))
        adapter = MMHAdapter(root / "lineage.json", engine=engine)
        card = _card("candidate", "parent")
        assert valid_mmh_proposal_card(
            card, candidate="candidate", parent="parent", branch_id=0,
            generation_round="b0r0", peer_candidates=[],
        )
        adapter.register_candidate(
            candidate="candidate", parent="parent", source_path=source,
            proposal_card=card, batch=0, generation_round="b0r0", origin_problem=_problem("origin"),
        )

        def public_executor(name: str) -> dict:
            passes = 1 if name == "candidate" else 0
            return {
                "n_pass": passes,
                "n_total": 1,
                "results": [{"input": "1 2\n", "expected": "3\n", "stdout": "3\n", "ok": bool(passes)}],
            }

        outcomes = adapter.validate_pending(_problem("future"), "1:public:future", public_executor)
        assert len(outcomes) == 1
        assert outcomes[0]["improved"] is True
        patch = engine.store.get_patch("lcb:candidate")
        assert patch is not None and patch.status is PatchStatus.VALIDATED
        assert engine.store.evidence_for_patch("lcb:candidate")[0].subset_id == "1:public:future"
        adapter.close()


def test_missing_card_judge_confidence_falls_back_to_result_rule_confidence() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "candidate.py"
        source.write_text("# frozen candidate artifact\n", encoding="utf-8")
        engine = MetaMemoryEngine(SQLiteStore(root / "memory.sqlite"))
        adapter = MMHAdapter(root / "lineage.json", engine=engine)
        card = _card("candidate", "parent")
        card["memory_patch"].pop("judge_confidence", None)
        assert valid_mmh_proposal_card(
            card, candidate="candidate", parent="parent", branch_id=0,
            generation_round="b0r0", peer_candidates=[],
        )
        association = adapter.register_candidate(
            candidate="candidate", parent="parent", source_path=source,
            proposal_card=card, batch=0, generation_round="b0r0", origin_problem=_problem("origin"),
        )
        assert association.patch_id == "lcb:candidate"
        adapter.close()


def test_public_validation_records_stable_rule_outcome() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "candidate.py"
        source.write_text("# frozen candidate artifact\n", encoding="utf-8")
        engine = MetaMemoryEngine(SQLiteStore(root / "memory.sqlite"))
        stable = Rule(
            "stable-rule", "find the sum of two integers", "old instruction",
            tier=RuleTier.STABLE, status=PatchStatus.VALIDATED,
        )
        with engine.store.transaction():
            engine.store.put_rule(stable)
        adapter = MMHAdapter(root / "lineage.json", engine=engine)
        card = _card("candidate", "parent")
        card["applied_rule"] = {"rule_id": "stable-rule", "instruction": "old instruction"}
        adapter.register_candidate(
            candidate="candidate", parent="parent", source_path=source,
            proposal_card=card, batch=0, generation_round="b0r0", origin_problem=_problem("origin"),
        )

        def public_executor(name: str) -> dict:
            passes = 0 if name == "candidate" else 1
            return {
                "n_pass": passes,
                "n_total": 1,
                "results": [{"input": "1 2\n", "expected": "3\n", "stdout": "3\n", "ok": bool(passes)}],
            }

        adapter.validate_pending(_problem("future"), "1:public:future", public_executor)
        refreshed = engine.store.get_rule("stable-rule")
        history = refreshed.provenance.get("stable_failure_history", []) if refreshed else []
        assert len(history) == 1
        assert history[0]["success"] is False
        adapter.close()


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"PASS {name}")
