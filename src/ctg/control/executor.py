"""Executor segments: the inner tool-calling loop for one subtask (Sec. 3.2.1, 3.2.3)."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from ctg.config import Budgets, ModelSpec, Role
from ctg.control.memory import render_memory
from ctg.envs.base import Benchmark, Environment, ExecutorProfile
from ctg.graphs.model import Subtask
from ctg.llm.base import LLM, LLMOutputError, Message, ToolCall
from ctg.prompts import render
from ctg.runs.records import Memory, ToolCallRecord

START = "Begin the subtask."
NO_ACTIONS = "No actions were available to this executor in the current state, so the segment ended."
EPISODE_FINISHED = "The environment reported the episode as finished after the last tool call."
NO_SUMMARY = "The executor ended the segment without a summary."
SKIPPED = "Not executed: the tool-call allowance is used up or the episode has ended."
NO_WRAP_UP = "The allowance ran out and the episode ended with this segment, so no summary was asked for."
UNUSABLE_OUTPUT = "The executor's output stayed unusable, so the episode ended during this segment."


@dataclass(frozen=True)
class SegmentResult:
    """H_m: the segment's tool calls with observations, and the executor's closing summary."""

    tool_calls: list[ToolCallRecord]
    summary: str
    failure: str | None = None
    """Set when the executor's output stayed unusable (LLMOutputError), as "<type>: <message>". The summary
    is then UNUSABLE_OUTPUT and the episode ends with this segment."""


def describe_failure(error: LLMOutputError) -> str:
    return f"{type(error).__name__}: {error}"


def segment_allowance(budgets: Budgets, profile: ExecutorProfile, tool_calls: int) -> int:
    """min(C_l, the profile's own cap, C_g − k)."""
    caps = [budgets.local_tool_calls, budgets.global_tool_calls - tool_calls]
    if profile.max_tool_calls is not None:
        caps.append(profile.max_tool_calls)
    return min(caps)


async def run_segment(
    *,
    env: Environment,
    benchmark: Benchmark,
    llm: LLM,
    model: ModelSpec,
    subtask: Subtask,
    profile: ExecutorProfile,
    memory: Memory | None,
    allowance: int,
    wrap_up: bool,
) -> SegmentResult:
    """Run executor α_i on `subtask` until it replies in text, the allowance is spent or the episode ends.

    The first call must call a tool. Each turn offers the profile's tools that are available in the current
    state; a call to any other tool is not executed and is answered with an error, but still spends a tool
    call. If the allowance runs out before a text reply, one tool-free call asks for the summary when
    `wrap_up` is set; otherwise the summary is NO_WRAP_UP. The workflow clears `wrap_up` when the episode
    ends after a segment that spends its allowance, since no router reads that summary.

    An LLMOutputError from any executor call ends the segment with the tool calls made so far and
    `failure` set; every other exception propagates.
    """
    if allowance < 1:
        raise ValueError(f"segment allowance must be positive, got {allowance}")
    system = render(
        "executor",
        benchmark=benchmark.description,
        objective=env.objective,
        subtask_name=subtask.name,
        description=subtask.description,
        success_criteria=subtask.success_criteria,
        memory=render_memory(memory),
        allowance=str(allowance),
    )
    messages: list[Message] = [
        ChatCompletionSystemMessageParam(role="system", content=system),
        ChatCompletionUserMessageParam(role="user", content=START),
    ]
    records: list[ToolCallRecord] = []
    first = True
    while not env.done and len(records) < allowance:
        tools = env.tool_specs(profile.tools)
        if not tools:
            return SegmentResult(records, NO_ACTIONS)
        try:
            turn = await llm.chat(Role.EXECUTOR, model, messages, tools, "required" if first else "auto")
        except LLMOutputError as error:
            return SegmentResult(records, UNUSABLE_OUTPUT, describe_failure(error))
        first = False
        messages.append(turn.message)
        if not turn.tool_calls:
            return SegmentResult(records, _summary(turn.content))
        executed, replies = await execute_tool_calls(
            env, turn.tool_calls, allowance - len(records), offered={spec.name for spec in tools}
        )
        records.extend(executed)
        messages.extend(replies)
    if env.done:
        return SegmentResult(records, EPISODE_FINISHED)
    if not wrap_up:
        return SegmentResult(records, NO_WRAP_UP)
    messages.append(
        ChatCompletionUserMessageParam(
            role="user", content=render("executor_wrapup", allowance=str(allowance))
        )
    )
    try:
        turn = await llm.chat(Role.EXECUTOR, model, messages, env.tool_specs(profile.tools), "none")
    except LLMOutputError as error:
        return SegmentResult(records, UNUSABLE_OUTPUT, describe_failure(error))
    return SegmentResult(records, _summary(turn.content))


async def execute_tool_calls(
    env: Environment, calls: Sequence[ToolCall], limit: int, offered: Collection[str] | None = None
) -> tuple[list[ToolCallRecord], list[Message]]:
    """Execute one turn's calls in order: at most `limit` of them, and none once the episode has ended.

    `offered` is the set of tools given to the agent on this turn; the environment answers a call outside
    it with an error instead of executing it (see Environment.call). None allows every tool.
    Every call id gets a tool message, as the API requires; skipped calls are answered with SKIPPED.
    """
    records: list[ToolCallRecord] = []
    replies: list[Message] = []
    for call in calls:
        if len(records) < limit and not env.done:
            observation = await env.call(call.name, call.arguments, offered)
            records.append(
                ToolCallRecord(
                    tool=call.name,
                    arguments=call.arguments,
                    observation=observation,
                    tool_calls_after=env.tool_calls,
                )
            )
        else:
            observation = SKIPPED
        replies.append(ChatCompletionToolMessageParam(role="tool", tool_call_id=call.id, content=observation))
    return records, replies


def _summary(content: str | None) -> str:
    text = (content or "").strip()
    return text or NO_SUMMARY
