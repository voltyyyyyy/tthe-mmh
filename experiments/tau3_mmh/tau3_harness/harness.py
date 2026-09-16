"""Guideline-injecting tau3 program harness.

Derived from ``reproduction/qwen_tau3/harnesses/reference_all375/harness.py``
(sha256 ``db0cdc8b783c3a8639c2ef6e904a6cc41991fe8b3d64154ff877aba69f065bc7``), which is
the harness the 375-task run used as its honest all-domain baseline.

The ONLY difference from that baseline is where the guideline block comes from:

    baseline:   system = ctx.system_prompt + "\\n\\n" + GUIDELINES        (a literal)
    this:       system = ctx.system_prompt + "\\n\\n" + load_guidelines()  (from memory)

Everything else -- the 50-step loop, the premature-stop recovery, the per-task seeding
(including the nonnumeric-ID fix), the tool execution path, and the finish reasons -- is
byte-identical, so the two arms differ only in the text of the guideline block. That is
what makes the comparison a test of the memory rather than of the harness.

Environment:
  MMH_GUIDELINES_FILE   path to the rendered guideline block for the current round.
                        Written by the experiment driver before each round. Absent or
                        empty means "no learned guidelines", which falls back to the
                        baseline block rather than degrading the agent.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

GUIDELINES_ENV = "MMH_GUIDELINES_FILE"
DEFAULT_GUIDELINES_PATH = Path(".mmh") / "guidelines.md"

# Byte-identical to reference_all375's GUIDELINES. Used when no learned block exists, so
# the frozen-baseline condition reproduces that harness exactly.
BASELINE_GUIDELINES = """
Customer-service operating rules:
1. Use tools instead of narrating tool use. Never claim a lookup or change
   unless the corresponding tool returned evidence.
2. Look up reservations, passengers, payments, membership, fees, and available
   options before stating facts about them.
3. Attempt the requested action through the system. Explain an inability only
   after the policy or tool result establishes it.
4. Complete the service cycle: identify, inspect, explain, confirm before
   consequential changes, execute, and report the final state to the customer.
5. Handle every part of a multi-part request and adapt to changed instructions.
6. Preserve explicit customer conditions, such as acting only if refundable.
7. Transfer with context: request, attempts, tool results, blockers, and the
   latest customer message.
8. Do not stop while the customer is active. Communicate through
   talk_to_customer until the customer explicitly ends the conversation.
"""


def load_guidelines() -> str:
    """The current round's guideline block, or the baseline block if none was written.

    Falling back (rather than returning "") matters: an empty system block would make the
    agent silently worse than the baseline, and a null result would then be attributed to
    the memory system instead of to a missing file.
    """
    path = os.environ.get(GUIDELINES_ENV)
    if not path:
        return BASELINE_GUIDELINES
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return BASELINE_GUIDELINES
    return text if text else BASELINE_GUIDELINES


def _assistant_message(result: Any) -> dict[str, Any]:
    raw = result.raw["choices"][0]["message"]
    message: dict[str, Any] = {
        "role": "assistant",
        "content": raw.get("content") or "",
    }
    if result.tool_calls:
        message["tool_calls"] = [call.as_openai() for call in result.tool_calls]
    if raw.get("reasoning_content") is not None:
        message["reasoning_content"] = raw["reasoning_content"]
    return message


def _seed_for_task(ctx: Any, step: int) -> int:
    """Preserve historical airline seeds; support qualified nonnumeric IDs.

    Unchanged from the baseline. The numeric branch is what crashed telecom and banking
    before this fix existed, which is why the baseline arm must use this version.
    """
    task_id = str(ctx.task["task_id"])
    try:
        return int(task_id) * 100 + step
    except ValueError:
        domain = str(ctx.task.get("domain", ""))
        digest = hashlib.sha256(f"{domain}:{task_id}:{step}".encode()).digest()
        return int.from_bytes(digest[:4], "big")


async def run(ctx: Any) -> Any:
    messages: list[dict[str, Any]] = [{
        "role": "user",
        "content": "A customer is calling. Greet them with talk_to_customer and resolve their issue.",
    }]
    system = ctx.system_prompt + "\n\n" + load_guidelines()
    forced_continuations = 0
    for step in range(50):
        result = await ctx.call_model(
            messages=messages,
            system=system,
            extra_body={
                "seed": _seed_for_task(ctx, step),
                "top_p": 0.95,
                "top_k": 20,
                "reasoning_effort": "medium",
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "preserve_thinking": True,
                },
            },
        )
        messages.append(_assistant_message(result))
        ctx.log_event("agent_turn", step=step, tool_count=len(result.tool_calls))
        if not result.tool_calls:
            if ctx.customer_ended:
                return ctx.finish(
                    {"status": "customer_ended"},
                    stop_reason="customer_ended",
                    forced_continuations=forced_continuations,
                )
            forced_continuations += 1
            messages.append({
                "role": "user",
                "content": (
                    "The customer has not ended the conversation. Continue helping "
                    "through talk_to_customer or use a database tool. Do not stop yet."
                ),
            })
            ctx.log_event("premature_stop_blocked", step=step)
            continue
        for call in result.tool_calls:
            observation = await ctx.execute_tool(call.name, call.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": observation,
            })
        if ctx.customer_ended:
            return ctx.finish(
                {"status": "customer_ended"},
                stop_reason="customer_ended",
                forced_continuations=forced_continuations,
            )
    return ctx.finish(
        {"status": "max_turns"},
        stop_reason="max_turns",
        forced_continuations=forced_continuations,
    )
