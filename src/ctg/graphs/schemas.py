"""Structured-output schemas for the planner's two calls (Sec. 3.2.2, 3.2.4).

Both schemas are built per call. The subtask schema restricts each subtask's executor to the roster's
profile names. The criteria schema has one required string per edge the controller can take, nested by
source subtask. Its property names are subtask ids and END, carried as aliases because they need not be
Python identifiers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

from ctg.graphs.model import END, GIVE_UP


class SubtaskDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Short, distinct name.")
    description: str = Field(description="What the subtask must achieve, not how.")
    success_criteria: str = Field(
        description="Conditions, observable in tool outputs, that show it is achieved."
    )
    executor: str


class SubtaskPlan(BaseModel):
    """Call 1 for a dependency workflow: the subtasks in dependency order."""

    model_config = ConfigDict(extra="forbid")

    plan: str = Field(description="Brief plan.")
    potential_blockers: list[str] = Field(description="What could go wrong.")
    refined_plan: str = Field(description="The plan revised to handle the blockers.")
    name: str = Field(description="Name of the graph.")
    description: str = Field(description="One-sentence description of the graph.")
    subtasks: list[SubtaskDraft]


class CyclicSubtaskPlan(SubtaskPlan):
    """Call 1 for a cyclic graph: the subtasks plus the entry subtask."""

    entry: str = Field(description="Name of the subtask to execute first.")


PlanT = TypeVar("PlanT", bound=SubtaskPlan)

_ALIASED = ConfigDict(extra="forbid", serialize_by_alias=True)


def subtask_schema(base: type[PlanT], executors: Sequence[str]) -> type[PlanT]:
    """`base` with every subtask's `executor` restricted to `executors` (a JSON-schema enum)."""
    if not executors:
        raise ValueError("the executor roster is empty")
    names = list(dict.fromkeys(executors))

    def as_enum(schema: dict[str, Any]) -> None:
        # Pydantic renders a one-value Literal as `const`; keep it an enum for strict structured outputs.
        schema.pop("const", None)
        schema["enum"] = names

    executor = (
        cast(Any, Literal)[tuple(names)],
        Field(description="Executor profile that runs the subtask.", json_schema_extra=as_enum),
    )
    draft = create_model(SubtaskDraft.__name__, __base__=SubtaskDraft, executor=executor)
    return create_model(base.__name__, __base__=base, subtasks=(cast(Any, list)[draft], ...))


def criteria_schema(edges: Mapping[str, Sequence[str]]) -> type[BaseModel]:
    """`{source: {destination: criterion}}` with exactly the given edges, all required, in order."""
    rows: dict[str, Any] = {}
    for i, (source, targets) in enumerate(edges.items()):
        fields: dict[str, Any] = {
            f"to_{j}": (str, Field(alias=model_id(target), description=_edge_description(source, target)))
            for j, target in enumerate(targets)
        }
        row = create_model(f"From_{source}", __config__=_ALIASED, **fields)
        rows[f"from_{i}"] = (row, Field(alias=source))
    return create_model("TransitionCriteria", __config__=_ALIASED, **rows)


def model_id(target: str) -> str:
    """The id models see for a destination: END is shown as GIVE_UP."""
    return GIVE_UP if target == END else target


def read_criteria(output: BaseModel) -> dict[str, dict[str, str]]:
    """The criteria of a `criteria_schema` instance, keyed by subtask ids (GIVE_UP mapped back to END)."""
    rows = cast(dict[str, dict[str, str]], output.model_dump(by_alias=True))
    return {
        source: {(END if target == GIVE_UP else target): text for target, text in row.items()}
        for source, row in rows.items()
    }


def _edge_description(source: str, target: str) -> str:
    if target == END:
        return f"When to give up after `{source}`: end the episode now, as a failure."
    if target == source:
        return f"When to run `{source}` again."
    return f"When to move from `{source}` to `{target}`."
