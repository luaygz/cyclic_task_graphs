"""Graph-controlled episodes: executor segments alternating with routing (Sec. 3.2.3, 3.2.4, 4.3)."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass

from ctg.config import Budgets, ModelSpec, Role
from ctg.control.executor import describe_failure, run_segment, segment_allowance
from ctg.control.faults import RoutedChoice, inject_fault
from ctg.control.policies import Graph, candidates, is_forced_end, transition_type
from ctg.control.router import route
from ctg.envs.base import Benchmark, Environment
from ctg.graphs.model import END, CyclicGraph
from ctg.llm.base import LLM, LLMOutputError
from ctg.runs.records import EndReason, Memory, Outcome, SegmentRecord, ToolCallRecord, TransitionRecord


@dataclass
class EpisodeResult:
    segments: list[SegmentRecord]
    react_steps: list[ToolCallRecord]
    outcome: Outcome
    failure: str | None = None
    """Set when the episode ended because a model's output stayed unusable (LLMOutputError), as
    "<type>: <message>". The outcome is then an OUTPUT_FAILURE loss and the records hold everything done
    before that call."""


def output_failure(
    env: Environment, segments: list[SegmentRecord], react_steps: list[ToolCallRecord], failure: str
) -> EpisodeResult:
    """The result of an episode cut short by unusable model output."""
    outcome = Outcome(
        won=False, end_reason=EndReason.OUTPUT_FAILURE, tool_calls=env.tool_calls, segments=len(segments)
    )
    return EpisodeResult(segments=segments, react_steps=react_steps, outcome=outcome, failure=failure)


def environment_end(env: Environment, budgets: Budgets) -> EndReason | None:
    """Why the episode is over after the last tool call, or None while it can continue."""
    if env.won:
        return EndReason.WON
    if env.done:
        return EndReason.FINAL_ANSWER
    if env.tool_calls >= budgets.global_tool_calls:
        return EndReason.BUDGET
    return None


async def run_graph_episode(
    graph: Graph,
    env: Environment,
    benchmark: Benchmark,
    llm: LLM,
    models: Mapping[Role, ModelSpec],
    budgets: Budgets,
    perturbation: float,
    rng: random.Random,
) -> EpisodeResult:
    """Run one episode from the graph's entry node with empty memory.

    Each iteration runs a segment, then checks termination, then routes, then applies fault injection
    (cyclic graphs only) and records the transition. At most C_g segments run, since each one normally
    spends at least one tool call; reaching that cap ends the episode with BUDGET.

    An executor that spends its whole allowance gets the closing-summary call only if a router will read
    that summary: not when the allowance is all of the remaining global budget, the segment is the C_g-th,
    or the node's only candidate is ⊥.

    Unusable executor or router output (LLMOutputError) ends the episode as an OUTPUT_FAILURE: the segment
    in progress is kept, without router or transition, and `failure` is set. Every other exception
    propagates.
    """
    if perturbation > 0 and not isinstance(graph, CyclicGraph):
        raise ValueError("fault injection is defined for cyclic graphs only")
    segments: list[SegmentRecord] = []
    current = graph.entry
    memory: Memory | None = None
    end_reason = EndReason.BUDGET
    for index in range(budgets.global_tool_calls):
        subtask = graph.subtask(current)
        profile = benchmark.roster[subtask.executor]
        allowance = segment_allowance(budgets, profile, env.tool_calls)
        allowed = candidates(graph, current)
        last = index + 1 == budgets.global_tool_calls
        result = await run_segment(
            env=env,
            benchmark=benchmark,
            llm=llm,
            model=models[Role.EXECUTOR],
            subtask=subtask,
            profile=profile,
            memory=memory,
            allowance=allowance,
            wrap_up=not (
                last or is_forced_end(allowed) or env.tool_calls + allowance >= budgets.global_tool_calls
            ),
        )
        record = SegmentRecord(
            index=index,
            node=current,
            executor=profile.name,
            tool_calls=result.tool_calls,
            summary=result.summary,
        )
        segments.append(record)
        if result.failure is not None:
            return output_failure(env, segments, [], result.failure)
        stop = environment_end(env, budgets)
        if stop is not None:
            end_reason = stop
            break
        if last:
            break
        if is_forced_end(allowed):
            record.transition = _transition(graph, current, allowed, RoutedChoice(END, END, perturbed=False))
            end_reason = EndReason.TERMINATED
            break
        try:
            router = await route(
                env=env,
                benchmark=benchmark,
                llm=llm,
                model=models[Role.ROUTER],
                budgets=budgets,
                graph=graph,
                current=current,
                memory=memory,
                segment=result,
                allowed=allowed,
            )
        except LLMOutputError as error:
            return output_failure(env, segments, [], describe_failure(error))
        memory = router.memory
        choice = (
            inject_fault(router.choice, graph.ids, perturbation, rng)
            if isinstance(graph, CyclicGraph)
            else RoutedChoice(router.choice, router.choice, perturbed=False)
        )
        record.router = router
        record.transition = _transition(graph, current, allowed, choice)
        if choice.actual == END:
            end_reason = EndReason.TERMINATED
            break
        current = choice.actual
    outcome = Outcome(won=env.won, end_reason=end_reason, tool_calls=env.tool_calls, segments=len(segments))
    return EpisodeResult(segments=segments, react_steps=[], outcome=outcome)


def _transition(
    graph: Graph, source: str, allowed: tuple[str, ...], choice: RoutedChoice
) -> TransitionRecord:
    return TransitionRecord(
        source=source,
        target=choice.actual,
        allowed=list(allowed),
        nominal=choice.nominal,
        perturbed=choice.perturbed,
        type=transition_type(graph, source, choice.actual),
        valid=choice.actual in allowed,
    )
