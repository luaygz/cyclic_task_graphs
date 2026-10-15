"""The analyzer+router ρ: one structured call per segment boundary (Sec. 3.2.1, 3.2.3).

The call returns M_{m+1} = UPDATE_ρ(M_m, t_{i_m}, H_m) followed by
i_{m+1} = SELECT_ρ(M_{m+1}, {c_{i_m→j}}_{j∈A_m}), with `next` restricted to the candidate ids by the schema.
The terminal decision ⊥ (END) is shown to the model as GIVE_UP and mapped back on return.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Sequence
from typing import Annotated

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from pydantic import AfterValidator, Field, create_model
from pydantic.config import JsonDict

from ctg.config import Budgets, ModelSpec, Role
from ctg.control.executor import SegmentResult
from ctg.control.memory import render_memory
from ctg.control.policies import Graph
from ctg.envs.base import Benchmark, Environment
from ctg.graphs.model import END, GIVE_UP
from ctg.graphs.schemas import model_id
from ctg.llm.base import LLM, Message
from ctg.prompts import render
from ctg.runs.records import Memory, RouterRecord

START = "Analyze the finished segment, update the memory and choose the next subtask."
NO_TOOL_CALLS = "No tool calls were made in this segment."


class RouterOutput(Memory):
    """Memory fields first (M_{m+1}), then the routing decision. Field order is the generation order."""

    routing_reasoning: list[str]
    next: str


def decision_schema(allowed: Sequence[str]) -> type[RouterOutput]:
    """RouterOutput with `next` constrained to the model-facing ids of `allowed` (enum, also validated)."""
    options = [model_id(target) for target in allowed]

    def check(value: str) -> str:
        if value not in options:
            raise ValueError(f"next must be one of {options}, got {value!r}")
        return value

    enum: JsonDict = {"enum": [*options]}
    choice = Annotated[str, AfterValidator(check), Field(json_schema_extra=enum)]
    return create_model("RouterDecision", __base__=RouterOutput, next=(choice, ...))


async def route(
    *,
    env: Environment,
    benchmark: Benchmark,
    llm: LLM,
    model: ModelSpec,
    budgets: Budgets,
    graph: Graph,
    current: str,
    memory: Memory | None,
    segment: SegmentResult,
    allowed: Sequence[str],
) -> RouterRecord:
    """Update memory from the segment that just finished and choose the next node from `allowed`."""
    subtask = graph.subtask(current)
    prompt = render(
        "router",
        benchmark=benchmark.description,
        objective=env.objective,
        subtask_id=subtask.id,
        subtask_name=subtask.name,
        subtask_description=subtask.description,
        success_criteria=subtask.success_criteria,
        memory=render_memory(memory),
        trace=render_trace(segment),
        tool_calls=str(env.tool_calls),
        budget=str(budgets.global_tool_calls),
        candidates=render_candidates(graph, current, allowed),
    )
    messages: list[Message] = [
        ChatCompletionSystemMessageParam(role="system", content=prompt),
        ChatCompletionUserMessageParam(role="user", content=START),
    ]
    output = await llm.structured(Role.ROUTER, model, messages, decision_schema(allowed))
    return RouterRecord(
        memory=Memory.model_validate(output.model_dump(include=set(Memory.model_fields))),
        routing_reasoning=output.routing_reasoning,
        choice=END if output.next == GIVE_UP else output.next,
    )


def render_trace(segment: SegmentResult) -> str:
    """H_m as the router reads it: numbered tool calls with observations, then the executor's summary."""
    if segment.tool_calls:
        calls = [
            f"{number}. {call.tool}({json.dumps(call.arguments, ensure_ascii=False)})\n"
            + textwrap.indent(f"Observation: {call.observation}", "   ")
            for number, call in enumerate(segment.tool_calls, start=1)
        ]
        trace = "\n".join(calls)
    else:
        trace = NO_TOOL_CALLS
    return f"{trace}\n\nExecutor summary: {segment.summary}"


def render_candidates(graph: Graph, current: str, allowed: Sequence[str]) -> str:
    """Each candidate j ∈ A_m with its description and the criterion c_{current→j}."""
    blocks: list[str] = []
    for target in allowed:
        if target == END:
            head = (
                f"- `{GIVE_UP}`: give up and end the episode now, as a failure, since the objective has not "
                "been reached. Choose it only if the objective can no longer be completed."
            )
        else:
            subtask = graph.subtask(target)
            marker = (
                " (the current subtask; choosing it runs it again to finish or fix what is missing)"
                if target == current
                else ""
            )
            head = f"- `{target}`{marker}: {subtask.name}. {subtask.description}"
        blocks.append(f"{head}\n  Transition criterion: {graph.criteria[current][target]}")
    return "\n".join(blocks)
