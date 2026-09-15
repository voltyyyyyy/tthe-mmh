"""Offline adapters and strict response validators for Meta-Memory Harness.

The core package intentionally knows nothing about model SDKs.  This module is
the narrow boundary at which a caller may turn a prompt response into a typed,
atomic MMH patch.  Invalid model output is rejected before it can mutate the
rule store.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


class ResponseValidationError(ValueError):
    """A model response does not meet an appendix prompt's JSON contract."""


class EmbeddingProvider(Protocol):
    """Small synchronous embedding interface accepted by the MMH engine."""

    def embed(self, text: str) -> list[float]: ...


class ContextMatcher(Protocol):
    """Interface for the appendix's context-routing prompt."""

    def match(self, phi: str, query: str) -> "ContextMatch": ...


@dataclass(frozen=True)
class ContextMatch:
    is_match: bool
    confidence: float
    reason: str = ""


@dataclass(frozen=True)
class Arbitration:
    decision: str
    merged_phi: str | None = None
    merged_psi: str | None = None
    merged_omega: str | None = None
    merged_confidence: float | None = None
    merged_lifespan: int | None = None
    reason: str = ""


_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_OPERATIONS = frozenset({"ADD", "DELETE", "REFINE", "SPLIT", "MERGE"})
_DECISIONS = frozenset({"KEEP_A", "KEEP_B", "MERGE", "REJECT_BOTH"})


def _json_object(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise ResponseValidationError("response must be a JSON object or JSON string")
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ResponseValidationError(f"malformed JSON: {exc.msg}") from exc
    if not isinstance(result, dict):
        raise ResponseValidationError("response must be a JSON object")
    return result


def _string(obj: Mapping[str, Any], name: str, *, nullable: bool = False) -> str | None:
    value = obj.get(name)
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        suffix = " or null" if nullable else ""
        raise ResponseValidationError(f"{name!r} must be a non-empty string{suffix}")
    return value.strip()


def _unit_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ResponseValidationError(f"{name!r} must be a number in [0, 1]")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ResponseValidationError(f"{name!r} must be in [0, 1]")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResponseValidationError(f"{name!r} must be a non-negative integer")
    return value


def _rule_kwargs(payload: Mapping[str, Any], *, required: bool) -> dict[str, Any] | None:
    if not required and payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ResponseValidationError("new_rule must be an object")
    phi = _string(payload, "phi")
    psi = _string(payload, "psi")
    omega = _string(payload, "omega")
    confidence = _unit_float(payload.get("confidence"), "new_rule.confidence")
    lifespan = _nonnegative_int(payload.get("lifespan"), "new_rule.lifespan")
    return {
        "phi": phi,
        "psi": psi,
        "omega": omega,
        "initial_confidence": confidence,
        "provenance": {"appendix_lifespan": lifespan},
    }


def _types() -> Any:
    """Delay core imports so `meta_memory` can safely re-export this module."""
    import meta_memory

    return meta_memory


def parse_patch_response(
    response: str | Mapping[str, Any],
    *,
    patch_id: str,
    context: str,
    target_tier: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Any:
    """Parse Prompt 1 output into one independent core ``Patch``.

    The function rejects extra interpretation such as a half-formed split.  A
    caller should record the rejection; it must not stage a partial patch.
    """
    payload = _json_object(response)
    operation_name = _string(payload, "operation")
    assert operation_name is not None
    operation_name = operation_name.upper()
    if operation_name not in _OPERATIONS:
        raise ResponseValidationError(f"unsupported operation {operation_name!r}")
    target = _string(payload, "target_rule_id")
    assert target is not None
    if operation_name == "ADD":
        if target.lower() != "new":
            raise ResponseValidationError("ADD must target 'new'")
        targets: tuple[str, ...] = ()
    else:
        if target.lower() == "new":
            raise ResponseValidationError(f"{operation_name} requires an existing target_rule_id")
        targets = (target,)

    if target_tier and target_tier.lower() == "stable" and operation_name in {"REFINE", "SPLIT", "ADD"}:
        raise ResponseValidationError(f"{operation_name} is not permitted for a stable target")

    raw_new_rule = payload.get("new_rule")
    result_kwargs: list[dict[str, Any]] = []
    if operation_name != "DELETE":
        result_kwargs.append(_rule_kwargs(raw_new_rule, required=True) or {})
    elif raw_new_rule is not None:
        # Appendix Prompt 1 uses an object with null phi/psi for DELETE; accept
        # that as well as the later literal ``new_rule: null`` shorthand.
        if not isinstance(raw_new_rule, Mapping):
            raise ResponseValidationError("DELETE new_rule must be null or an object")
        if any(raw_new_rule.get(name) not in (None, "") for name in ("phi", "psi")):
            raise ResponseValidationError("DELETE new_rule.phi/psi must be null")

    raw_split = payload.get("split_details", [])
    if operation_name == "SPLIT":
        if not isinstance(raw_split, list) or len(raw_split) < 2:
            raise ResponseValidationError("SPLIT requires at least two complete split_details rules")
        result_kwargs = [_rule_kwargs(item, required=True) or {} for item in raw_split]
    elif raw_split not in (None, []):
        raise ResponseValidationError("split_details is only allowed for SPLIT")

    if operation_name == "MERGE":
        extra_targets: list[str] = []
        merge_with = payload.get("merge_with")
        if isinstance(merge_with, str) and merge_with.strip():
            extra_targets.append(merge_with.strip())
        raw_targets = payload.get("target_rule_ids")
        if isinstance(raw_targets, list):
            extra_targets.extend(
                item.strip() for item in raw_targets
                if isinstance(item, str) and item.strip()
            )
        extra_targets = [item for item in dict.fromkeys(extra_targets) if item != target]
        if not extra_targets:
            raise ResponseValidationError("MERGE requires a second distinct target_rule_id")
        targets = (target, extra_targets[0])

    mm = _types()
    rules = tuple(
        mm.Rule(rule_id=f"{patch_id}:result:{index}", **kwargs)
        for index, kwargs in enumerate(result_kwargs)
    )
    if "judge_confidence" in payload:
        raw_confidence = payload["judge_confidence"]
    elif "confidence" in payload:
        raw_confidence = payload["confidence"]
    else:
        confidence_candidates = [
            kwargs.get("initial_confidence")
            for kwargs in result_kwargs
            if isinstance(kwargs.get("initial_confidence"), (int, float))
            and not isinstance(kwargs.get("initial_confidence"), bool)
        ]
        if isinstance(raw_new_rule, Mapping):
            value = raw_new_rule.get("confidence")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                confidence_candidates.append(float(value))
        raw_confidence = max(confidence_candidates) if confidence_candidates else 0.5
    judge_confidence = _unit_float(raw_confidence, "judge_confidence")
    patch_provenance = dict(provenance or {})
    patch_provenance.update({"thinking": payload.get("thinking", ""), "raw_operation": operation_name})
    return mm.Patch(
        patch_id=patch_id,
        operation=mm.PatchOperation(operation_name),
        target_rule_ids=targets,
        result_rules=rules,
        context=context,
        judge_confidence=judge_confidence,
        provenance=patch_provenance,
    )


def parse_arbitration_response(response: str | Mapping[str, Any]) -> Arbitration:
    """Validate Prompt 2 output without applying its decision."""
    payload = _json_object(response)
    decision = _string(payload, "decision")
    assert decision is not None
    decision = decision.upper()
    if decision not in _DECISIONS:
        raise ResponseValidationError(f"unknown arbitration decision {decision!r}")
    if decision == "MERGE":
        return Arbitration(
            decision=decision,
            merged_phi=_string(payload, "merged_phi"),
            merged_psi=_string(payload, "merged_psi"),
            merged_omega=_string(payload, "merged_omega"),
            merged_confidence=_unit_float(payload.get("merged_confidence"), "merged_confidence"),
            merged_lifespan=_nonnegative_int(payload.get("merged_lifespan"), "merged_lifespan"),
            reason=_string(payload, "reason") or "",
        )
    return Arbitration(decision=decision, reason=_string(payload, "reason") or "")


def parse_context_match_response(response: str | Mapping[str, Any], *, threshold: float = 0.7) -> ContextMatch:
    """Validate Prompt 3 and impose its required 0.7 match threshold."""
    payload = _json_object(response)
    is_match = payload.get("is_match")
    if not isinstance(is_match, bool):
        raise ResponseValidationError("is_match must be a boolean")
    confidence = _unit_float(payload.get("confidence"), "confidence")
    reason = _string(payload, "reason") or ""
    return ContextMatch(is_match=is_match and confidence >= threshold, confidence=confidence, reason=reason)


def parse_promotion_summary(response: str) -> str:
    """Validate Prompt 4's short, plain-text promotion summary."""
    if not isinstance(response, str):
        raise ResponseValidationError("promotion summary must be plain text")
    summary = response.strip().strip('"')
    if not summary or len(summary.split()) > 20 or "\n" in summary:
        raise ResponseValidationError("promotion summary must be one non-empty sentence of at most 20 words")
    return summary


def parse_fallback_score(response: str | int) -> float:
    """Validate Prompt 5's integer 0--10 score and normalize it to [0, 1]."""
    if isinstance(response, str):
        response = response.strip()
        if not re.fullmatch(r"(?:0|[1-9]|10)", response):
            raise ResponseValidationError("fallback score must be an integer from 0 through 10")
        score = int(response)
    elif isinstance(response, int) and not isinstance(response, bool) and 0 <= response <= 10:
        score = response
    else:
        raise ResponseValidationError("fallback score must be an integer from 0 through 10")
    return score / 10.0


@dataclass(frozen=True)
class HashEmbeddingProvider:
    """Deterministic normalized feature hashing for offline tests and demos."""

    dimensions: int = 64

    def __post_init__(self) -> None:
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")

    def embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            values[bucket] += sign
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values] if norm else values


@dataclass(frozen=True)
class TokenOverlapMatcher:
    """Deterministic context matcher used when no configured model is present."""

    threshold: float = 0.7

    def match(self, phi: str, query: str) -> ContextMatch:
        phi_tokens = set(_TOKEN_RE.findall(phi.lower()))
        query_tokens = set(_TOKEN_RE.findall(query.lower()))
        if not phi_tokens:
            return ContextMatch(False, 0.0, "Rule context has no matchable tokens.")
        # Coverage is more faithful than Jaccard for a long user query: all
        # condition tokens should occur, while unrelated query detail is fine.
        confidence = len(phi_tokens & query_tokens) / len(phi_tokens)
        return ContextMatch(confidence >= self.threshold, confidence, "Deterministic token-coverage match.")


class PromptTemplates:
    """Appendix-compatible prompt constructors for configured model adapters."""

    @staticmethod
    def diagnose_failure(*, user_query: str, expected_output: str, actual_output: str, rule: Mapping[str, Any] | None) -> str:
        rule = rule or {}
        return (
            "You are a rigorous AI rule-maintenance expert. Return JSON only. "
            "Choose ADD, DELETE, REFINE, SPLIT, or MERGE; stable rules may only MERGE or DELETE.\n"
            f"Failure query: {user_query}\nExpected: {expected_output}\nActual: {actual_output}\n"
            f"Matched rule: {json.dumps(dict(rule), sort_keys=True)}\n"
            "Required keys: thinking, operation, target_rule_id, new_rule, split_details."
        )

    @staticmethod
    def arbitrate(*, candidate_a: Mapping[str, Any], candidate_b: Mapping[str, Any], target_rule_id: str) -> str:
        return (
            "You are a decision arbitrator. Return JSON only with decision "
            "KEEP_A, KEEP_B, MERGE, or REJECT_BOTH and a reason.\n"
            f"Target: {target_rule_id}\nA: {json.dumps(dict(candidate_a), sort_keys=True)}\n"
            f"B: {json.dumps(dict(candidate_b), sort_keys=True)}"
        )

    @staticmethod
    def route(*, phi: str, user_query: str) -> str:
        return (
            "Return JSON only: is_match (boolean), confidence (0..1), reason. "
            "Treat confidence below 0.7 as no match.\n"
            f"Rule phi: {phi}\nQuery: {user_query}"
        )

    @staticmethod
    def promotion_summary(*, rule: Mapping[str, Any]) -> str:
        return "Write one objective plain-text promotion sentence of at most 20 words.\n" + json.dumps(dict(rule), sort_keys=True)

    @staticmethod
    def fallback_score(*, old_rule: Mapping[str, Any], proposed_rule: Mapping[str, Any]) -> str:
        return (
            "Return only an integer 0 through 10 estimating whether this patch is reasonable.\n"
            f"Old: {json.dumps(dict(old_rule), sort_keys=True)}\nNew: {json.dumps(dict(proposed_rule), sort_keys=True)}"
        )
