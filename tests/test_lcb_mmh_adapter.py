"""Offline checks for the LiveCodeBench public-only MMH boundary."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from livecodebench.mmh_adapter import MMHAdapter, PublicProblem, valid_mmh_proposal_card
from meta_memory import MetaMemoryEngine, PatchStatus, SQLiteStore


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


if __name__ == "__main__":
    test_future_public_validation_reaches_core_without_private_data()
    print("PASS test_future_public_validation_reaches_core_without_private_data")
