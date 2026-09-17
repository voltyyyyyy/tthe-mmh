"""Run tau3 program harnesses against an OpenAI-compatible model server."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import httpx

from meta_agent.harness_contracts.program import (
    HarnessContext,
    HarnessResult,
    ProgramHarnessError,
    run_program_harness,
)


TAU_USER_TERMINATORS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")


@dataclass(frozen=True)
class NormalizedToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def as_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }


@dataclass(frozen=True)
class TauModelCallResult:
    text: str
    tool_calls: tuple[NormalizedToolCall, ...]
    raw: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgramTaskResult:
    task_id: str
    domain: str
    reward: float
    passed: bool
    gold_reward: float
    num_turns: int
    cost_usd: float | None
    duration_s: float
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    session_id: str | None
    error: str | None = None
    tau2_conversation: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


def _parse_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ProgramHarnessError(f"invalid tool arguments JSON: {value!r}") from exc
    if not isinstance(parsed, dict):
        raise ProgramHarnessError("tool arguments must decode to an object")
    return parsed


def _normalize_tool_calls(message: Mapping[str, Any]) -> tuple[NormalizedToolCall, ...]:
    calls: list[NormalizedToolCall] = []
    for index, raw in enumerate(message.get("tool_calls") or []):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("function"), Mapping):
            raise ProgramHarnessError("invalid OpenAI tool call object")
        function = raw["function"]
        name = str(function.get("name") or "").strip()
        if not name:
            raise ProgramHarnessError("tool call is missing function.name")
        calls.append(NormalizedToolCall(
            id=str(raw.get("id") or f"call_{index}_{uuid.uuid4().hex[:8]}"),
            name=name,
            arguments=_parse_tool_arguments(function.get("arguments")),
        ))
    return tuple(calls)


def _openai_tools(env: Any) -> list[dict[str, Any]]:
    tools = [{
        "type": "function",
        "function": {
            "name": "talk_to_customer",
            "description": "Send a customer-facing message and receive the reply.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    }]
    for tau_tool in env.get_tools():
        schema = dict(tau_tool.openai_schema)
        if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
            tools.append(schema)
        else:
            tools.append({
                "type": "function",
                "function": {
                    "name": tau_tool.name,
                    "description": getattr(tau_tool, "short_desc", tau_tool.name),
                    "parameters": schema.get("function", {}).get("parameters", {}),
                },
            })
    return tools


class TauProgramContext(HarnessContext):
    """Safe tau tools plus direct access to a private chat-completions endpoint."""

    def __init__(self, *, task: Any, model: str, env: Any, user: Any,
                 system_prompt: str, base_url: str, api_key: str,
                 request_timeout_s: float, max_tokens: int,
                 temperature: float) -> None:
        super().__init__(task=task, model=model, cwd="/tmp", timeout=int(request_timeout_s))
        self.env, self.user = env, user
        self.system_prompt = system_prompt
        self.tools = _openai_tools(env)
        self.max_tokens, self.temperature = max_tokens, temperature
        self._base_url, self._api_key = base_url.rstrip("/"), api_key
        self._request_timeout_s = request_timeout_s
        self.messages: list[dict[str, Any]] = []
        self.tool_call_log: list[dict[str, Any]] = []
        self.tau2_trajectory: list[Any] = []
        self.num_turns = self.input_tokens = self.output_tokens = 0
        self._has_user_state = self._customer_ended = False
        self._user_state: Any = None
        self._next_tau_call = 0

    @property
    def customer_ended(self) -> bool:
        return self._customer_ended

    async def call_model(self, *, messages: Sequence[Mapping[str, Any]],
                         system: Optional[str] = None,
                         tools: Optional[Sequence[Mapping[str, Any]]] = None,
                         max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None,
                         extra_body: Optional[Mapping[str, Any]] = None) -> TauModelCallResult:
        request_messages = []
        effective_system = self.system_prompt if system is None else system
        if effective_system:
            request_messages.append({"role": "system", "content": effective_system})
        request_messages.extend(dict(message) for message in messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "tools": [dict(tool) for tool in (tools or self.tools)],
            "tool_choice": "auto",
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if extra_body:
            payload.update(dict(extra_body))
        call_id = f"api_{uuid.uuid4().hex}"
        self.log_event(
            "model_input",
            call_id=call_id,
            endpoint=f"{self._base_url}/chat/completions",
            payload=payload,
        )
        started = time.time()
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            async with httpx.AsyncClient(timeout=self._request_timeout_s) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions", json=payload, headers=headers,
                )
                response.raise_for_status()
        except Exception as exc:
            response_body = ""
            http_status_code = None
            if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                # Preserve the server's structured rejection so a runner can distinguish
                # a native context-budget limit from a transport failure.
                response_body = exc.response.text[:4096]
                http_status_code = exc.response.status_code
            self.log_event(
                "model_error",
                call_id=call_id,
                duration_ms=int((time.time() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
                response_body=response_body,
                http_status_code=http_status_code,
            )
            raise

        raw = response.json()
        choices = raw.get("choices") or []
        if not choices or not isinstance(choices[0], Mapping):
            raise ProgramHarnessError("model response has no choices[0]")
        message = choices[0].get("message") or {}
        if not isinstance(message, Mapping):
            raise ProgramHarnessError("model response has no message object")
        calls = _normalize_tool_calls(message)
        text = message.get("content") if isinstance(message.get("content"), str) else ""
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if not reasoning and text:
            match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
            reasoning = match.group(1).strip() if match else None
        usage = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else {}
        self.num_turns += 1
        self.input_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        self.log_event(
            "raw_model_output",
            call_id=call_id,
            status_code=response.status_code,
            response=raw,
            reasoning=reasoning,
        )
        self.log_event(
            "model_call",
            call_id=call_id,
            duration_ms=int((time.time() - started) * 1000),
            text=text,
            tool_calls=[call.as_openai() for call in calls],
            usage=dict(usage),
        )
        return TauModelCallResult(text=text, tool_calls=calls, raw=dict(raw), usage=dict(usage))

    async def execute_tool(self, name: str, arguments: Mapping[str, Any]) -> str:
        return await (self._talk_to_customer(dict(arguments)) if name == "talk_to_customer"
                      else self._call_environment_tool(name, dict(arguments)))

    async def _talk_to_customer(self, arguments: dict[str, Any]) -> str:
        from tau2.data_model.message import AssistantMessage as TauAssistantMessage
        from tau2.data_model.message import UserMessage as TauUserMessage

        message = str(arguments.get("message") or "").strip()
        if not message:
            raise ProgramHarnessError("talk_to_customer requires a non-empty message")
        tool_call_id = f"customer_{uuid.uuid4().hex}"
        started = time.time()
        self.log_event(
            "action",
            action="talk_to_customer",
            tool_call_id=tool_call_id,
            arguments={"message": message},
        )
        agent_message = TauAssistantMessage(role="assistant", content=message)
        self.messages.append({"role": "assistant", "content": message})
        self.tau2_trajectory.append(agent_message)
        if not self._has_user_state:
            self._user_state = self.user.get_init_state()
            self._has_user_state = True
        try:
            user_message, self._user_state = await asyncio.to_thread(
                self.user.generate_next_message, agent_message, self._user_state,
            )
            user_text = user_message.content if hasattr(user_message, "content") else str(user_message)
            user_text = str(user_text)
            self.messages.append({"role": "user", "content": user_text})
            self.tau2_trajectory.append(TauUserMessage.text(content=user_text))
            self._customer_ended = any(token in user_text for token in TAU_USER_TERMINATORS)
            result = f"Customer: {user_text}"
            duration_ms = int((time.time() - started) * 1000)
            self.tool_call_log.append({
                "id": tool_call_id,
                "tool": "talk_to_customer",
                "args": {"message": message},
                "result": result,
                "error": False,
                "duration_ms": duration_ms,
            })
            self.log_event(
                "customer_message",
                tool_call_id=tool_call_id,
                message=user_text,
                terminal=self._customer_ended,
                duration_ms=duration_ms,
            )
            self.log_event("observation", tool_call_id=tool_call_id, result=result, error=False)
            return result
        except Exception as exc:
            self.log_event(
                "observation",
                tool_call_id=tool_call_id,
                error=True,
                result=f"{type(exc).__name__}: {exc}",
            )
            raise

    async def _call_environment_tool(self, name: str, arguments: dict[str, Any]) -> str:
        from tau2.data_model.message import AssistantMessage as TauAssistantMessage
        from tau2.data_model.message import ToolCall as TauToolCall
        from tau2.data_model.message import ToolMessage as TauToolMessage

        tool_call_id = f"tc_{self._next_tau_call}"
        self._next_tau_call += 1
        started = time.time()
        self.log_event("action", action=name, tool_call_id=tool_call_id, arguments=arguments)
        error = False
        try:
            value = await asyncio.to_thread(self.env.make_tool_call, name, **arguments)
        except Exception as exc:
            value = f"Error: {type(exc).__name__}: {exc}"
            error = True
        result = self.env.to_json_str(value)
        duration_ms = int((time.time() - started) * 1000)
        self.tool_call_log.append({
            "id": tool_call_id,
            "tool": name,
            "args": arguments,
            "result": result,
            "error": error,
            "duration_ms": duration_ms,
        })
        self.tau2_trajectory.append(TauAssistantMessage(
            role="assistant",
            tool_calls=[TauToolCall(
                id=tool_call_id, name=name, arguments=arguments, requestor="assistant",
            )],
        ))
        self.tau2_trajectory.append(TauToolMessage(
            id=tool_call_id,
            role="tool",
            content=result,
            requestor="assistant",
            error=error,
        ))
        self.log_event(
            "tool_result",
            tool_call_id=tool_call_id,
            tool=name,
            result=result,
            error=error,
            duration_ms=duration_ms,
        )
        self.log_event("observation", tool_call_id=tool_call_id, result=result, error=error)
        return result

    def finish(self, final_output: Any, **metadata: Any) -> HarnessResult:
        return HarnessResult(
            final_output=final_output,
            metadata=metadata,
            events=self.events,
            num_turns=self.num_turns,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


def _tau_system_prompt(domain: str, policy: str) -> str:
    return (
        f"You are a customer service agent for the {domain} domain.\n\n"
        "Use talk_to_customer to communicate with the customer. Use database tools "
        "to inspect and modify reservations. Follow the policy exactly, confirm "
        "consequential changes, report outcomes to the customer, and continue until "
        "the customer emits a termination marker.\n\n"
        f"POLICY (follow this exactly):\n{policy}"
    )


async def run_tau_task_program(
    *,
    domain: str,
    task_id: str,
    config_path: str,
    model: str,
    user_model: str | None = None,
    judge_model: str | None = None,
    judge_strategy: str = "binary",
    retrieval_config: str | None = None,
    timeout_s: int = 900,
    base_url: str | None = None,
    request_timeout_s: float = 300.0,
    max_tokens: int = 8192,
    temperature: float = 1.0,
) -> ProgramTaskResult:
    from datetime import datetime

    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.runner import build_environment, build_user, get_tasks
    from tau2.runner.build import _derive_read_log_allowlist

    tasks = get_tasks(domain, task_ids=[str(task_id)])
    if not tasks:
        raise ProgramHarnessError(f"tau task not found: domain={domain!r} id={task_id!r}")
    task = tasks[0]
    env_kwargs: dict[str, Any] = {}
    if domain == "banking_knowledge":
        env_kwargs["task"] = task
        env_kwargs["read_log_allowlist"] = _derive_read_log_allowlist(task)
        if retrieval_config:
            env_kwargs["retrieval_variant"] = retrieval_config
    env = build_environment(domain, env_kwargs=env_kwargs or None)
    if task.initial_state and hasattr(env, "set_state"):
        env.set_state(
            initialization_data=task.initial_state.initialization_data,
            initialization_actions=task.initial_state.initialization_actions,
            message_history=task.initial_state.message_history or [],
        )
    user = build_user("user_simulator", env, task, llm=user_model or model)
    policy = env.get_policy()
    harness_path = Path(config_path)
    if harness_path.is_dir():
        harness_path = harness_path / "harness.py"
    resolved_base_url = (
        base_url
        or os.environ.get("LOCAL_MODEL_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "http://127.0.0.1:8000/v1"
    )
    ctx = TauProgramContext(
        task={"domain": domain, "task_id": str(task_id)},
        model=model,
        env=env,
        user=user,
        system_prompt=_tau_system_prompt(domain, policy),
        base_url=resolved_base_url,
        api_key=os.environ.get("LOCAL_MODEL_API_KEY", "EMPTY"),
        request_timeout_s=request_timeout_s,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    started = time.time()
    session_id = f"program-{task_id}-{uuid.uuid4().hex[:12]}"
    error: str | None = None
    try:
        harness_result = await run_program_harness(harness_path, ctx, timeout=timeout_s)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        ctx.log_event("agent_error", error=error)
        harness_result = ctx.finish({"error": error})
    duration_s = time.time() - started
    initial_messages = (task.initial_state.message_history or []) if task.initial_state else []
    simulation = SimulationRun(
        id=session_id,
        task_id=str(task_id),
        start_time=datetime.fromtimestamp(started).isoformat(),
        end_time=datetime.fromtimestamp(started + duration_s).isoformat(),
        duration=duration_s,
        termination_reason=(
            TerminationReason.AGENT_ERROR if error else TerminationReason.AGENT_STOP
        ),
        messages=list(initial_messages) + ctx.tau2_trajectory,
    )
    reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=domain,
        env_kwargs=env_kwargs,
    )
    reward = float(reward_info.reward)
    ctx.log_event(
        "grading",
        reward=reward,
        reward_info=(
            reward_info.model_dump() if hasattr(reward_info, "model_dump") else str(reward_info)
        ),
    )
    trajectory_dump = [
        message.model_dump() if hasattr(message, "model_dump") else dict(message)
        for message in ctx.tau2_trajectory
    ]
    return ProgramTaskResult(
        task_id=str(task_id),
        domain=domain,
        reward=reward,
        passed=reward > 0,
        gold_reward=reward,
        num_turns=ctx.num_turns,
        cost_usd=harness_result.cost_usd,
        duration_s=duration_s,
        messages=ctx.messages,
        tool_calls=ctx.tool_call_log,
        session_id=session_id,
        error=error,
        tau2_conversation=trajectory_dump,
        events=ctx.events,
    )
