"""Extended proposal-card contract for online-MMH.

A card may apply an existing rule without forcing a new ADD/REFINE memory
patch.  It must declare one attribution unit, the exact parent/child identities,
origin tasks/traces, applicability prerequisites, expected effect, and either
applied rule IDs or an explicit new hypothesis.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .types import ProposalCard


_FORBIDDEN = re.compile(r"(?:private|hidden|gold|answer[_-]?key|is_correct)", re.I)


def _safe_public(value: Any, where: str = "payload") -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if _FORBIDDEN.search(str(key)):
                raise ValueError(f"{where} contains forbidden field {key!r}")
            result[str(key)] = _safe_public(item, f"{where}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_public(item, where) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{where} must be JSON-shaped, got {type(value).__name__}")


def _memory_patch_valid(patch: Mapping[str, Any]) -> bool:
    operation = str(patch.get("operation", "")).upper()
    targets = patch.get("target_rule_ids", [])
    results = patch.get("result_rules", [])
    if operation not in {"ADD", "DELETE", "REFINE", "SPLIT", "MERGE"}:
        return False
    if not isinstance(targets, list) or not all(isinstance(item, str) and item for item in targets):
        return False
    if not isinstance(results, list):
        return False
    if operation == "ADD" and targets:
        return False
    if operation != "ADD" and not targets:
        return False
    if operation == "DELETE" and results:
        return False
    if operation in {"ADD", "REFINE", "MERGE"} and len(results) != 1:
        return False
    if operation == "SPLIT" and len(results) < 2:
        return False
    for item in results:
        if not isinstance(item, Mapping):
            return False
        if not all(isinstance(item.get(key), str) and item[key] for key in ("phi", "psi", "omega")):
            return False
        confidence = item.get("confidence", 0.5)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            return False
    judge_confidence = patch.get("judge_confidence")
    if judge_confidence is not None:
        if isinstance(judge_confidence, bool) or not isinstance(judge_confidence, (int, float)) or not 0 <= judge_confidence <= 1:
            return False
    if not isinstance(patch.get("rationale"), str) or not patch["rationale"].strip():
        return False
    return True


def coerce_card(card: ProposalCard | Mapping[str, Any]) -> ProposalCard:
    return card if isinstance(card, ProposalCard) else ProposalCard.from_mapping(card)


def validate_card(
    card: ProposalCard | Mapping[str, Any],
    *,
    candidate: str,
    parent: str,
    branch_id: int | None = None,
    generation_round: str | None = None,
    peer_candidates: Iterable[str] | None = None,
    parent_sha256: str | None = None,
    candidate_sha256: str | None = None,
    require_memory_patch: bool = False,
) -> tuple[bool, str, ProposalCard | None]:
    """Return ``(valid, reason, parsed_card)``.

    ``memory_patch`` is optional when ``applied_rule_ids`` is non-empty.  This
    preserves the distinction between applying an existing rule and editing
    memory.
    """
    try:
        data = _safe_public(card.to_dict() if isinstance(card, ProposalCard) else card, "proposal_card")
        parsed = coerce_card(data)
    except (TypeError, ValueError, KeyError) as exc:
        return False, f"card is not a public JSON object: {exc}", None

    if parsed.candidate != candidate:
        return False, f"candidate mismatch: {parsed.candidate!r} != {candidate!r}", parsed
    if parsed.parent != parent:
        return False, f"parent mismatch: {parsed.parent!r} != {parent!r}", parsed
    if branch_id is not None and parsed.branch_id != int(branch_id):
        return False, "branch_id mismatch", parsed
    if generation_round is not None and parsed.generation_round != generation_round:
        return False, "generation_round mismatch", parsed
    if peer_candidates is not None and sorted(parsed.peer_candidates) != sorted(str(item) for item in peer_candidates):
        return False, "peer_candidates mismatch", parsed
    if parent_sha256 is not None and parsed.parent_sha256 != parent_sha256:
        return False, "parent source hash mismatch", parsed
    if candidate_sha256 is not None and parsed.candidate_sha256 != candidate_sha256:
        return False, "candidate source hash mismatch", parsed
    if not isinstance(parsed.role, str) or not parsed.role.strip():
        return False, "role must be a non-empty string", parsed
    if not isinstance(parsed.behavior_change, Mapping) or not parsed.behavior_change:
        return False, "exactly one non-empty behavior_change object is required", parsed
    if not isinstance(parsed.rationale, str) or not parsed.rationale.strip():
        return False, "rationale is required", parsed
    if not isinstance(parsed.expected_effect, str) or not parsed.expected_effect.strip():
        return False, "expected_effect is required", parsed
    if not parsed.origin_task_ids or not all(isinstance(item, str) and item for item in parsed.origin_task_ids):
        return False, "origin_task_ids must be a non-empty list of task ids", parsed
    if not parsed.applied_rule_ids and not parsed.new_hypothesis.strip():
        return False, "declare applied_rule_ids or an explicit new_hypothesis", parsed
    if parsed.applied_rule_ids and not all(isinstance(item, str) and item for item in parsed.applied_rule_ids):
        return False, "applied_rule_ids contains an invalid id", parsed
    if parsed.memory_patch is not None and not _memory_patch_valid(parsed.memory_patch):
        return False, "memory_patch is malformed", parsed
    if require_memory_patch and parsed.memory_patch is None:
        return False, "memory_patch is required for this registration path", parsed
    if parsed.override is not None:
        if not isinstance(parsed.override, Mapping):
            return False, "override must be an object", parsed
        if not str(parsed.override.get("reason", "")).strip():
            return False, "override requires a reason", parsed
        if not str(parsed.override.get("changed_condition", "")).strip():
            return False, "override requires a changed_condition", parsed
    return True, "ok", parsed


def card_is_attributable(card: ProposalCard | Mapping[str, Any]) -> bool:
    parsed = coerce_card(card)
    if not parsed.attributable:
        return False
    if not isinstance(parsed.behavior_change, Mapping):
        return False
    # Structural hint only: a single declared change is the attribution unit.
    # This cannot prove causal isolation; evaluators still mark compound or
    # ambiguous diffs non-attributable in the proposal card.
    return True


def cards_from_batch(cards: Iterable[Mapping[str, Any]]) -> list[ProposalCard]:
    return [coerce_card(card) for card in cards]
