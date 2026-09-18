"""Legacy-compatible shim that lets the existing ``lcb_optimize`` loop drive the
separate online-MMH experiment without changing the default ``none`` path.

The existing loop expects an adapter with ``retrieve``, ``register_candidate``,
``validate_pending(problem, subset_id, evaluator)``, ``finish_round``,
``summary``, and ``close``.  This shim translates those calls to
``OnlineMemory``/``LCBMemoryAdapter``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from .adapter import AdapterError, LCBMemoryAdapter
from .artifacts import ArtifactStore
from .budget import BudgetLedger
from .cache import ObservationCache
from .config import ExperimentConfig, build_budget_ledger, build_memory
from .flat import FlatAdviceMemory
from .memory import OnlineMemory
from .types import ApplicationStatus, Arm


_ACTIVE_CONFIG: ExperimentConfig | None = None


def set_active_config(config: ExperimentConfig) -> None:
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = config


class LegacyMemoryShim:
    """Adapter shim used only by ``experiments.lcb_mmh_online.live_runner``."""

    def __init__(self, state_path: str | Path, engine: Any | None = None) -> None:
        config = _ACTIVE_CONFIG or ExperimentConfig(arm=Arm.MMH)
        self.config = config
        self.base_dir = Path(state_path).parent
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.source_map_path = self.base_dir / "legacy_source_paths.json"
        self.source_map: dict[str, str] = self._load_source_map()
        # The existing optimizer constructs a legacy MetaMemoryEngine even when
        # the shim ignores it.  Close that handle so Windows can clean the run dir.
        if engine is not None and hasattr(engine, "store") and hasattr(engine.store, "close"):
            try:
                engine.store.close()
            except Exception:
                pass
        self.memory = build_memory(config, self.base_dir)
        self.flat: FlatAdviceMemory | None = self.memory if config.arm is Arm.FLAT else None
        self.ledger: BudgetLedger | None = build_budget_ledger(config, self.base_dir)
        self.cache: ObservationCache | None = None
        self.adapter: LCBMemoryAdapter | None = None
        if config.arm is Arm.MMH:
            assert isinstance(self.memory, OnlineMemory)
            self.cache = ObservationCache(self.base_dir / "observation_cache.sqlite")
            self.adapter = LCBMemoryAdapter(
                self.memory,
                ArtifactStore(self.base_dir / "artifacts", run_id=self.base_dir.name),
                execution_config=config.execution_config,
                default_pair_cost=1.0,
                cache=self.cache,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_source_map(self) -> dict[str, str]:
        if not self.source_map_path.exists():
            return {}
        try:
            data = json.loads(self.source_map_path.read_text(encoding="utf-8-sig"))
            return {str(key): str(value) for key, value in data.items()}
        except Exception:
            return {}

    def _save_source_map(self) -> None:
        self.source_map_path.write_text(
            json.dumps(self.source_map, indent=2, sort_keys=True), encoding="utf-8",
        )

    @staticmethod
    def _augment_card(card: Mapping[str, Any], *, candidate: str, parent: str,
                      origin_qid: str, candidate_path: Path, parent_path: Path) -> dict[str, Any]:
        import hashlib
        data = dict(card)
        candidate_payload = candidate_path.read_bytes()
        parent_payload = parent_path.read_bytes()
        data.setdefault("candidate", candidate)
        data.setdefault("parent", parent)
        data.setdefault("parent_sha256", hashlib.sha256(parent_payload).hexdigest())
        data.setdefault("candidate_sha256", hashlib.sha256(candidate_payload).hexdigest())
        data.setdefault("origin_task_ids", [origin_qid])
        data.setdefault("trace_refs", [])
        if not data.get("behavior_change") and isinstance(data.get("behavior_changes"), list) and data["behavior_changes"]:
            data["behavior_change"] = dict(data["behavior_changes"][0])
        applied = data.get("applied_rule") if isinstance(data.get("applied_rule"), Mapping) else {}
        applied_ids = list(data.get("applied_rule_ids", []))
        if not applied_ids and isinstance(applied.get("rule_id"), str):
            applied_ids = [applied["rule_id"]]
        data["applied_rule_ids"] = applied_ids
        if not data.get("new_hypothesis") and not applied_ids:
            data["new_hypothesis"] = "explicit new hypothesis from proposal card"
        if not data.get("rationale"):
            patch = data.get("memory_patch") if isinstance(data.get("memory_patch"), Mapping) else {}
            data["rationale"] = str(patch.get("rationale", "public trace intervention"))
        data.setdefault("expected_effect", data.get("rationale", "improve public tests"))
        data.setdefault("prerequisites", dict(applied.get("scope", {})) if isinstance(applied, Mapping) else {})
        data.setdefault("failure_signature", {})
        return data

    def _eligible_applications(self, batch_id: int) -> list[Any]:
        assert isinstance(self.memory, OnlineMemory)
        eligible = []
        for application in self.memory.store.list_applications():
            if application.status not in (ApplicationStatus.PENDING, ApplicationStatus.SUPPORTED, ApplicationStatus.DEFERRED):
                continue
            if application.created_batch >= batch_id:
                continue
            eligible.append(application)
        return eligible

    # ------------------------------------------------------------------
    # Legacy API
    # ------------------------------------------------------------------
    def register_candidate(self, *, candidate: str, parent: str, source_path: str | Path,
                           proposal_card: Mapping[str, Any], batch: int, generation_round: str,
                           origin_problem: Any = None, **_: Any) -> Any:
        if self.flat is not None:
            from .types import ProposalCard
            data = dict(proposal_card)
            data.setdefault("candidate", candidate)
            data.setdefault("parent", parent)
            data.setdefault("branch_id", 0)
            data.setdefault("generation_round", generation_round)
            data.setdefault("peer_candidates", [])
            data.setdefault("parent_sha256", "")
            data.setdefault("candidate_sha256", "")
            if not data.get("behavior_change") and isinstance(data.get("behavior_changes"), list) and data["behavior_changes"]:
                data["behavior_change"] = dict(data["behavior_changes"][0])
            data.setdefault("behavior_change", {"change": "flat intervention summary"})
            data.setdefault("rationale", "")
            data.setdefault("expected_effect", data.get("rationale") or "public trace intervention")
            data.setdefault("origin_task_ids", [getattr(origin_problem, "qid", "origin")])
            data.setdefault("trace_refs", [])
            applied = data.get("applied_rule") if isinstance(data.get("applied_rule"), Mapping) else {}
            data.setdefault("applied_rule_ids", [applied["rule_id"]] if isinstance(applied.get("rule_id"), str) else [])
            data.setdefault("new_hypothesis", "" if data["applied_rule_ids"] else "flat intervention summary")
            data.setdefault("prerequisites", dict(applied.get("scope", {})) if isinstance(applied, Mapping) else {})
            data.setdefault("failure_signature", {})
            try:
                card_obj = ProposalCard.from_mapping(data)
                self.flat.append_card_summary(card_obj)
            except Exception:
                pass
            return None
        if self.adapter is None:
            raise ValueError("no online adapter configured")
        candidate_path = Path(source_path)
        parent_path = candidate_path.parent / f"{parent}.py"
        if not parent_path.exists():
            raise ValueError(f"parent source not found: {parent_path}")
        self.source_map[str(candidate)] = str(candidate_path.resolve())
        self.source_map[str(parent)] = str(parent_path.resolve())
        self._save_source_map()
        origin_qid = getattr(origin_problem, "qid", "") or (
            (proposal_card.get("origin_task_ids") or [""])[0]
        )
        augmented = self._augment_card(
            proposal_card, candidate=candidate, parent=parent,
            origin_qid=str(origin_qid), candidate_path=candidate_path, parent_path=parent_path,
        )
        try:
            return self.adapter.register_candidate(
                candidate=candidate, parent=parent, candidate_source=candidate_path,
                parent_source=parent_path, proposal_card=augmented, created_batch=int(batch),
                created_round=str(generation_round),
                origin_problems=[SimpleNamespace(qid=str(origin_qid))],
                peer_candidates=list(proposal_card.get("peer_candidates", [])),
                branch_id=proposal_card.get("branch_id"),
            )
        except AdapterError as exc:
            raise ValueError(str(exc)) from exc

    def retrieve(self, problem: Any) -> list[dict[str, Any]]:
        if self.flat is not None:
            package = self.flat.retrieve_package(getattr(problem, "content", ""))
            return list(package.trusted) + list(package.tentative)
        if self.adapter is None:
            return []
        package = self.adapter.retrieve_for_problem(problem, positive_limit=self.config.retrieval_limit)
        return list(package.get("trusted", [])) + list(package.get("tentative", [])) + list(package.get("failed_interventions", []))

    def validate_pending(self, problem: Any, subset_id: str,
                         evaluator: Callable[[str, Any], Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.flat is not None:
            # Flat mode deliberately leaves validation budget to ordinary
            # search; it does not resolve or promote lessons.
            return []
        if self.adapter is None:
            return []
        batch_id = int(str(subset_id).split(":", 1)[0])
        if isinstance(self.memory, OnlineMemory):
            self.memory.retry_deferred()
        eligible = self._eligible_applications(batch_id)
        reservation_id: str | None = None
        if self.ledger is not None and eligible:
            try:
                reservation = self.ledger.reserve(
                    "memory_validation", float(len(eligible)),
                    call_id=f"lcb-validate-b{batch_id}",
                    metadata={"batch_id": batch_id, "eligible": len(eligible)},
                )
                reservation_id = reservation.reservation_id
            except Exception:
                return []
        expected_hashes: dict[str, str] = {}
        expected_refs: dict[str, Any] = {}
        for application in self.memory.store.list_applications():
            expected_hashes[application.candidate] = application.candidate_artifact.sha256
            expected_hashes[application.parent] = application.parent_artifact.sha256
            expected_refs[application.candidate] = application.candidate_artifact
            expected_refs[application.parent] = application.parent_artifact

        def checked_evaluator(name: str, task: Any) -> Mapping[str, Any]:
            import hashlib
            path_value = self.source_map.get(name)
            if not path_value:
                raise AdapterError(f"no frozen source path recorded for harness {name!r}")
            source_path = Path(path_value)
            if not source_path.exists():
                # Persisted validation must not depend on generated candidate
                # files surviving `--fresh`.  The artifact store is the source of
                # truth, so restore the frozen bytes to the module path.
                ref = expected_refs.get(name)
                if ref is None:
                    raise AdapterError(f"frozen source artifact is missing: {source_path}")
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_bytes(self.adapter.artifacts.read_bytes(ref))
            actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
            expected = expected_hashes.get(name)
            if expected is not None and actual != expected:
                raise AdapterError(
                    f"frozen source artifact was overwritten: {source_path} "
                    f"expected={expected} actual={actual}"
                )
            return evaluator(name, task)

        try:
            return self.adapter.validate_pending(
                problem, batch_id=batch_id, round_id=batch_id, evaluator=checked_evaluator,
                ledger=self.ledger, reservation_id=reservation_id,
                cost_estimate=float(len(eligible)),
            )
        except Exception:
            if self.ledger is not None and reservation_id is not None:
                self.ledger.release(reservation_id, reason="validation_failed")
            raise

    def finish_round(self, round_id: int) -> list[dict[str, Any]]:
        if self.adapter is None or self.flat is not None:
            return []
        assert isinstance(self.memory, OnlineMemory)
        self.memory.advance_batch(int(round_id))
        return [rule.to_dict() for rule in self.memory.promote_eligible(int(round_id))]

    def summary(self) -> dict[str, Any]:
        if self.flat is not None:
            return {"mode": "flat", **self.flat.summary()}
        assert isinstance(self.memory, OnlineMemory)
        return {"mode": "mmh", **self.memory.summary()}

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()
        if self.ledger is not None:
            self.ledger.close()
        if isinstance(self.memory, OnlineMemory):
            self.memory.store.close()
