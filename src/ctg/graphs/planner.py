"""The planner π (Sec. 3.2.2, 3.2.4): two structured calls that build a subtask graph.

Call 1 proposes the subtasks, each assigned to a roster executor (cyclic graphs also name the entry
subtask). Call 2 writes one transition criterion per edge the controller can take: every ordered pair for
a cyclic graph, `DependencyGraph.allowed` for a dependency chain. Spec-Cyc and DepDAG plan from the
instance objective O; Gen-Cyc (no objective) plans a benchmark-generic graph.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ctg.config import ModelSpec, Role
from ctg.envs.base import Benchmark, ExecutorProfile
from ctg.graphs.model import GIVE_UP, CyclicGraph, DependencyGraph, Subtask
from ctg.graphs.schemas import (
    CyclicSubtaskPlan,
    SubtaskDraft,
    SubtaskPlan,
    criteria_schema,
    read_criteria,
    subtask_schema,
)
from ctg.llm.base import LLM, LLMOutputError, Message
from ctg.prompts import render

MAX_ATTEMPTS = 3
"""Each call is retried at most twice, with the validation error added to the conversation."""

OutputT = TypeVar("OutputT", bound=BaseModel)
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class PlannedGraph:
    graph: CyclicGraph | DependencyGraph
    notes: dict[str, Any]
    """Planning fields, both calls' raw outputs and every rejected answer with its error."""


async def plan_cyclic(
    llm: LLM,
    model: ModelSpec,
    benchmark: Benchmark,
    roster: Sequence[ExecutorProfile],
    objective: str | None,
    nshot: Sequence[str] = (),
) -> PlannedGraph:
    """Spec-Cyc for an instance objective; Gen-Cyc when `objective` is None."""
    inputs = _inputs(benchmark, objective, nshot)
    workflow = render("planner_cyclic")
    rejected: list[dict[str, Any]] = []

    def nodes(output: CyclicSubtaskPlan) -> tuple[tuple[Subtask, ...], str]:
        subtasks = _subtasks(output.subtasks)
        return subtasks, _entry(output.entry, subtasks)

    plan, (subtasks, entry) = await _ask(
        llm,
        model,
        "subtasks",
        render("planner_subtasks", workflow=workflow, executors=_roster_json(roster), **inputs),
        subtask_schema(CyclicSubtaskPlan, [profile.name for profile in roster]),
        nodes,
        rejected,
    )
    ids = [subtask.id for subtask in subtasks]

    def graph(output: BaseModel) -> CyclicGraph:
        return CyclicGraph(
            name=plan.name,
            description=plan.description,
            subtasks=subtasks,
            entry=entry,
            criteria=_criteria(output),
        )

    criteria, built = await _ask(
        llm,
        model,
        "criteria",
        _criteria_prompt(workflow, inputs, subtasks, entry),
        criteria_schema(dict.fromkeys(ids, ids)),
        graph,
        rejected,
    )
    return PlannedGraph(built, _notes(plan, criteria, rejected))


async def plan_dependency(
    llm: LLM,
    model: ModelSpec,
    benchmark: Benchmark,
    roster: Sequence[ExecutorProfile],
    objective: str,
    retry: bool,
    nshot: Sequence[str] = (),
) -> PlannedGraph:
    """DepDAG (`retry`) or DepDAG-NoRetry: an ordered subtask chain with criteria on its allowed edges."""
    inputs = _inputs(benchmark, objective, nshot)
    workflow = render("planner_dependency", retry="enabled" if retry else "disabled")
    rejected: list[dict[str, Any]] = []

    def nodes(output: SubtaskPlan) -> tuple[Subtask, ...]:
        return _subtasks(output.subtasks)

    plan, subtasks = await _ask(
        llm,
        model,
        "subtasks",
        render("planner_subtasks", workflow=workflow, executors=_roster_json(roster), **inputs),
        subtask_schema(SubtaskPlan, [profile.name for profile in roster]),
        nodes,
        rejected,
    )
    # The allowed sets depend only on the subtask order and `retry`, so an unvalidated skeleton yields them.
    skeleton = DependencyGraph.model_construct(
        name=plan.name, description=plan.description, subtasks=subtasks, retry=retry, criteria={}
    )

    def graph(output: BaseModel) -> DependencyGraph:
        return DependencyGraph(
            name=plan.name,
            description=plan.description,
            subtasks=subtasks,
            retry=retry,
            criteria=_criteria(output),
        )

    criteria, built = await _ask(
        llm,
        model,
        "criteria",
        _criteria_prompt(workflow, inputs, subtasks, skeleton.entry),
        criteria_schema({source: skeleton.allowed(source) for source in skeleton.ids}),
        graph,
        rejected,
    )
    return PlannedGraph(built, _notes(plan, criteria, rejected))


async def _ask(
    llm: LLM,
    model: ModelSpec,
    call: str,
    prompt: str,
    schema: type[OutputT],
    build: Callable[[OutputT], ResultT],
    rejected: list[dict[str, Any]],
) -> tuple[OutputT, ResultT]:
    """One planner call whose output must pass `build`; retried with the error fed back."""
    messages: list[Message] = [{"role": "user", "content": prompt}]
    for _ in range(MAX_ATTEMPTS):
        output = await llm.structured(Role.PLANNER, model, messages, schema)
        try:
            return output, build(output)
        except ValueError as error:
            reason = _describe(error)
            rejected.append({"call": call, "output": output.model_dump(), "error": reason})
            messages = [
                *messages,
                {"role": "assistant", "content": output.model_dump_json()},
                {"role": "user", "content": render("planner_feedback", error=reason)},
            ]
    raise LLMOutputError(
        f"planner {call} call failed validation {MAX_ATTEMPTS} times: {rejected[-1]['error']}"
    )


def _describe(error: ValueError) -> str:
    if isinstance(error, ValidationError):
        return "; ".join(
            f"{'.'.join(map(str, detail['loc']))}: {detail['msg']}" if detail["loc"] else detail["msg"]
            for detail in error.errors()
        )
    return str(error)


def _inputs(benchmark: Benchmark, objective: str | None, nshot: Sequence[str]) -> dict[str, str]:
    """Template values shared by both calls: B, O (or the generic instruction) and D."""
    summaries = "\n\n".join(
        f"### Summary {number}\n\n{summary.strip()}" for number, summary in enumerate(nshot, start=1)
    )
    return {
        "benchmark": benchmark.description.strip(),
        "objective": render("planner_generic").strip() if objective is None else objective.strip(),
        "nshot": f"\n## Successful trajectories\n\n{summaries}\n" if summaries else "",
    }


def _roster_json(roster: Sequence[ExecutorProfile]) -> str:
    profiles = [
        {"name": profile.name, "description": profile.description, "tools": list(profile.tools)}
        for profile in roster
    ]
    return json.dumps(profiles, indent=2)


def _criteria_prompt(
    workflow: str, inputs: Mapping[str, str], subtasks: Sequence[Subtask], entry: str
) -> str:
    listing = json.dumps([subtask.model_dump() for subtask in subtasks], indent=2)
    return render("planner_criteria", workflow=workflow, entry=entry, subtasks=listing, **inputs)


def _subtasks(drafts: Sequence[SubtaskDraft]) -> tuple[Subtask, ...]:
    if not drafts:
        raise ValueError("`subtasks` is empty; give at least one subtask")
    ids = _unique_ids([draft.name for draft in drafts])
    return tuple(
        Subtask(
            id=subtask_id,
            name=draft.name,
            description=draft.description,
            success_criteria=draft.success_criteria,
            executor=draft.executor,
        )
        for subtask_id, draft in zip(ids, drafts, strict=True)
    )


def _slug(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_") or "subtask"


def _unique_ids(names: Sequence[str]) -> list[str]:
    """Slugs of `names`, suffixed `_2`, `_3`, ... where they would collide with each other or GIVE_UP."""
    ids: list[str] = []
    for name in names:
        base = _slug(name)
        candidate, suffix = base, 2
        while candidate in ids or candidate == GIVE_UP:
            candidate, suffix = f"{base}_{suffix}", suffix + 1
        ids.append(candidate)
    return ids


def _entry(name: str, subtasks: Sequence[Subtask]) -> str:
    """The id of the subtask the planner named as entry, matched by name (case and punctuation ignored)."""
    matches = [subtask.id for subtask in subtasks if _slug(subtask.name) == _slug(name)]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"`entry` {name!r} names several subtasks; give the subtasks distinct names")
    names = ", ".join(repr(subtask.name) for subtask in subtasks)
    raise ValueError(f"`entry` {name!r} is not the name of a subtask; use one of: {names}")


def _criteria(output: BaseModel) -> dict[str, dict[str, str]]:
    criteria = read_criteria(output)
    blank = [
        f"{source} -> {target}"
        for source, row in criteria.items()
        for target, text in row.items()
        if not text.strip()
    ]
    if blank:
        raise ValueError(f"empty criteria for: {', '.join(blank)}")
    return {
        source: {target: text.strip() for target, text in row.items()} for source, row in criteria.items()
    }


def _notes(plan: SubtaskPlan, criteria: BaseModel, rejected: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "plan": plan.plan,
        "potential_blockers": plan.potential_blockers,
        "refined_plan": plan.refined_plan,
        "subtasks_output": plan.model_dump(),
        "criteria_output": criteria.model_dump(),
        "rejected": rejected,
    }
