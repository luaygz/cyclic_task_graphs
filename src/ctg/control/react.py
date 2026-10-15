"""ReAct baseline (Sec. 4.1.2): one generalist agent, a required tool call at every step, budget C_g."""

from __future__ import annotations

from collections.abc import Mapping

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from ctg.config import Budgets, ModelSpec, Role
from ctg.control.executor import describe_failure, execute_tool_calls
from ctg.control.workflow import EpisodeResult, environment_end, output_failure
from ctg.envs.base import GENERALIST, Benchmark, Environment
from ctg.llm.base import LLM, LLMOutputError, Message
from ctg.prompts import render
from ctg.runs.records import EndReason, Outcome, ToolCallRecord

START = "Begin."


async def run_react_episode(
    env: Environment,
    benchmark: Benchmark,
    llm: LLM,
    models: Mapping[Role, ModelSpec],
    budgets: Budgets,
) -> EpisodeResult:
    """Act and observe, one tool call per step, until the episode ends or C_g steps have been taken.

    A step whose reply carries no tool call (a provider ignoring `tool_choice="required"`) still counts
    as a step, so the loop is bounded even then. Unusable model output (LLMOutputError) ends the episode as
    an OUTPUT_FAILURE that keeps the steps taken so far; every other exception propagates.
    """
    profile = benchmark.roster[GENERALIST]
    prompt = render(
        "react",
        benchmark=benchmark.description,
        objective=env.objective,
        budget=str(budgets.global_tool_calls),
    )
    messages: list[Message] = [
        ChatCompletionSystemMessageParam(role="system", content=prompt),
        ChatCompletionUserMessageParam(role="user", content=START),
    ]
    steps: list[ToolCallRecord] = []
    for _ in range(budgets.global_tool_calls):
        if environment_end(env, budgets) is not None:
            break
        tools = env.tool_specs(profile.tools)
        if not tools:
            raise RuntimeError("the environment offers the ReAct agent no tools in its current state")
        try:
            turn = await llm.chat(Role.EXECUTOR, models[Role.EXECUTOR], messages, tools, "required")
        except LLMOutputError as error:
            return output_failure(env, [], steps, describe_failure(error))
        messages.append(turn.message)
        executed, replies = await execute_tool_calls(env, turn.tool_calls, limit=1)
        steps.extend(executed)
        messages.extend(replies)
    end_reason = environment_end(env, budgets) or EndReason.BUDGET
    outcome = Outcome(won=env.won, end_reason=end_reason, tool_calls=env.tool_calls, segments=0)
    return EpisodeResult(segments=[], react_steps=steps, outcome=outcome)
