"""Proposers: turn execution traces into MMH guideline patches.

Two implementations behind one contract:

  * ``LLMProposer``     -- asks the served model for a typed patch, validated through
                          ``meta_memory.adapters.parse_patch_response`` so malformed
                          output is rejected before it can mutate the store.
  * ``OfflineProposer`` -- deterministic, credential-free.  Maps structural failure
                          signatures to guideline candidates.

The offline proposer exists for three reasons, all of which matter here: it makes the
loop testable without spending GPU hours; it gives a cost-free way to verify the
lifecycle before an overnight run; and it provides a floor the LLM proposer must beat,
so "the model writes good rules" becomes a measurable claim rather than an assumption.

Neither proposer decides *applicability* semantically -- they emit a candidate ``phi``
and the memory layer decides reachability.  Semantic retrieval is a separate concern.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from meta_memory import Patch, PatchOperation, Rule
from meta_memory.adapters import PromptTemplates, ResponseValidationError, parse_patch_response

from .guidelines import Guideline, TaskObservation


@dataclass(frozen=True)
class FailureSignature:
    """Structural features of a failed task, derived without any gold labels.

    ``domain`` and ``task_id`` are used only for *bookkeeping and routing*, never as
    the basis of a guideline's applicability claim.  A guideline keyed on a task id
    would be the "branch on task identity" failure the design forbids.
    """

    domain: str
    task_id: str
    signature: str
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Proposal:
    """A guideline plus the audit trail explaining why it was proposed."""

    guideline: Guideline
    hypothesis: str
    expected_benefit: str
    risks: str
    source_tasks: tuple[str, ...] = ()
    judge_confidence: float | None = None

    def as_provenance(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "expected_benefit": self.expected_benefit,
            "risks": self.risks,
            "source_tasks": list(self.source_tasks),
            "proposer": type(self).__name__,
        }


class Proposer(Protocol):
    def propose(self, *, round_id: int, failures: Sequence[FailureSignature],
                memory: Any) -> list[Proposal]: ...


# --------------------------------------------------------------------- offline

# Signature -> (phi, psi).  Keyed on *structural* symptoms, deliberately not on
# domain or task identity, so a guideline learned on one domain is legitimately
# applicable to another -- which is exactly the transfer the experiment measures.
_SIGNATURE_GUIDELINES: dict[str, tuple[str, str]] = {
    "no_tool_call_before_answer": (
        "the request concerns account, booking, or order state",
        "call a lookup tool and read its result before stating any fact about that state",
    ),
    "premature_stop": (
        "the customer's request has more than one part",
        "confirm every part is resolved before ending the conversation",
    ),
    "ungrounded_claim": (
        "you are about to state a policy, price, or eligibility rule",
        "quote the policy text you retrieved rather than asserting from memory",
    ),
    "repeated_tool_call": (
        "you are about to call a tool with arguments you have already used",
        "reuse the previous observation instead of re-issuing the identical call",
    ),
    "ignored_tool_error": (
        "a tool returns an error or an empty result",
        "acknowledge the failure and try a corrected call rather than proceeding as if it succeeded",
    ),
    "identity_not_verified": (
        "the action changes account state or discloses account details",
        "verify the customer's identity first and say so",
    ),
    "action_order_violation": (
        "a task requires several actions in a fixed order",
        "complete prerequisite actions before dependent ones",
    ),
    # Additional structural signatures used by the seeded sweep, so a run can contain
    # several distinct causes rather than a single repeated one.
    "unconfirmed_mutation": (
        "you are about to perform a state-changing action",
        "state the exact change and get explicit confirmation before performing it",
    ),
    "missing_escalation": (
        "the request is outside the policies you can apply",
        "escalate to a human agent instead of improvising a workaround",
    ),
    "partial_refund_math": (
        "a refund, fee, or compensation amount must be computed",
        "compute the amount from the retrieved figures and show the components",
    ),
    "stale_cache_reuse": (
        "an earlier observation may be out of date after a mutation",
        "re-read the affected record after any change before relying on it",
    ),
}


class OfflineProposer:
    """Deterministic signature-driven proposer. No model, no credentials.

    One guideline per distinct signature, deduplicated, so the patch stream is a
    function of the observed failures rather than of RNG.
    """

    name = "offline"

    def __init__(self, *, max_per_round: int = 2, confidence: float = 0.6) -> None:
        self.max_per_round = max_per_round
        self.confidence = confidence

    @staticmethod
    def _signature_text(signature: str) -> tuple[str, str]:
        """The (phi, psi) pair for a structural signature.

        Exposed so other proposers can reuse the same wording without duplicating it.
        Raises KeyError for an unknown signature rather than inventing a guideline.
        """
        return _SIGNATURE_GUIDELINES[signature]

    def propose(self, *, round_id: int, failures: Sequence[FailureSignature],
                memory: Any) -> list[Proposal]:
        existing = {r.phi for r in memory.store.list_rules()}
        seen: set[str] = set()
        out: list[Proposal] = []
        for failure in failures:
            sig = failure.signature
            if sig in seen or sig not in _SIGNATURE_GUIDELINES:
                continue
            seen.add(sig)
            phi, psi = _SIGNATURE_GUIDELINES[sig]
            if phi in existing:
                continue  # already known; proposing it again would be noise
            out.append(
                Proposal(
                    guideline=Guideline(
                        guideline_id=f"g-{round_id}-{sig}",
                        phi=phi,
                        psi=psi,
                        omega="pending",
                        initial_confidence=self.confidence,
                        provenance={"signature": sig, "source_domain": failure.domain},
                    ),
                    hypothesis=f"failures with signature '{sig}' share an unmet condition",
                    expected_benefit=f"tasks exhibiting '{sig}' should stop failing",
                    risks="guideline may be too broad and disturb tasks that already pass",
                    source_tasks=(failure.task_id,),
                    judge_confidence=self.confidence,
                )
            )
            if len(out) >= self.max_per_round:
                break
        return out


# ------------------------------------------------------------------------- LLM


_LLM_SYSTEM = """You maintain a small set of behavioural guidelines for a customer-service agent.

You are shown failures from recent tasks. Propose AT MOST ONE new guideline, as strict JSON:
{
  "thinking": "<why this failure recurs>",
  "operation": "Add",
  "target_rule_id": "<new>",
  "new_rule": {
    "phi": "<the condition under which the guideline applies, a general situation - NEVER a task id>",
    "psi": "<the instruction the agent should follow>",
    "omega": "pending",
    "confidence": <0..1>,
    "lifespan": 0
  },
  "split_details": [],
  "judge_confidence": <0..1>
}

Hard requirements:
- phi must describe a GENERAL condition, never reference a specific task id, customer, or domain name.
- psi must be actionable in one sentence.
- If the failures share no common cause, reply {"operation": "None"} and nothing else.
- Reply with JSON only."""


class LLMProposer:
    """Asks a served model for one typed patch per round.

    The response is parsed by the packaged validator, so a malformed reply is a
    rejected proposal rather than a mutated store.
    """

    name = "llm"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = 600.0,
        disable_thinking: bool = True,
        max_failures_shown: int = 6,
    ) -> None:
        from openai import OpenAI

        self.model = model or os.environ.get("MMH_PROPOSER_MODEL", "Qwen/Qwen3.8-27B")
        self.temperature = temperature
        self.disable_thinking = disable_thinking
        self.max_failures_shown = max_failures_shown
        self._client = OpenAI(
            base_url=base_url or os.environ.get("MMH_PROPOSER_BASE_URL", "http://127.0.0.1:8100/v1"),
            api_key=api_key or os.environ.get("MMH_PROPOSER_API_KEY", "EMPTY"),
            timeout=timeout,
            max_retries=0,
        )

    def propose(self, *, round_id: int, failures: Sequence[FailureSignature],
                memory: Any) -> list[Proposal]:
        if not failures:
            return []
        prompt = self._build_prompt(round_id, failures, memory)
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[{"role": "system", "content": _LLM_SYSTEM},
                      {"role": "user", "content": prompt}],
            max_tokens=2048,
        )
        # Hard problems otherwise burn the whole budget reasoning before emitting JSON.
        if self.disable_thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        else:
            kwargs["temperature"] = self.temperature
        response = self._client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        return self._to_proposals(text, round_id, failures)

    def _build_prompt(self, round_id: int, failures: Sequence[FailureSignature],
                      memory: Any) -> str:
        lines = [f"Round {round_id} failures:"]
        for f in list(failures)[: self.max_failures_shown]:
            detail = json.dumps(dict(f.evidence), sort_keys=True)[:12000]
            lines.append(f"- [{f.domain}] signature={f.signature} evidence={detail}")
        block = memory.prompt_block()
        if block:
            lines.append("")
            lines.append("Guidelines already in force (do not duplicate these):")
            lines.append(block)
        return "\n".join(lines)

    def _to_proposals(self, text: str, round_id: int,
                      failures: Sequence[FailureSignature]) -> list[Proposal]:
        try:
            payload = json.loads(text.strip()) if text.strip().startswith("{") else None
        except json.JSONDecodeError:
            payload = None
        if payload is None:
            # Fall back to the packaged fenced/embedded-JSON extraction.
            try:
                patch = parse_patch_response(text, patch_id=f"llm-{round_id}", context="")
            except ResponseValidationError:
                return []
        else:
            if str(payload.get("operation", "")).lower() in {"none", ""}:
                return []
            try:
                patch = parse_patch_response(payload, patch_id=f"llm-{round_id}", context="")
            except ResponseValidationError:
                return []
        if patch.operation is not PatchOperation.ADD or not patch.result_rules:
            # Only ADD is auto-staged from a proposal. DELETE/REFINE/SPLIT/MERGE touch
            # existing rules and need the conflict machinery, so they are not applied
            # blind from a single round's proposal.
            return []
        rule: Rule = patch.result_rules[0]
        return [
            Proposal(
                guideline=Guideline(
                    guideline_id=f"g-{round_id}-llm",
                    phi=rule.phi,
                    psi=rule.psi,
                    omega=rule.omega or "pending",
                    initial_confidence=float(rule.initial_confidence),
                    provenance={"proposer": "llm", "round": round_id},
                ),
                hypothesis="model-identified common cause across recent failures",
                expected_benefit="recurrence of the identified pattern should stop failing",
                risks="model may over-generalize from few examples",
                source_tasks=tuple(f.task_id for f in failures),
                judge_confidence=patch.judge_confidence,
            )
        ]


def build_proposer(kind: str = "offline", **kwargs: Any) -> Proposer:
    if kind == "offline":
        return OfflineProposer(**kwargs)
    if kind == "llm":
        return LLMProposer(**kwargs)
    raise ValueError(f"unknown proposer kind: {kind!r}")


__all__ = [
    "FailureSignature",
    "Proposal",
    "Proposer",
    "OfflineProposer",
    "LLMProposer",
    "build_proposer",
    "PromptTemplates",
]
