"""Subtask graphs (Sec. 3.2).

A `CyclicGraph` is the complete directed graph over subtasks T with one natural-language criterion
c_{i->j} on every edge, self-loops included. A `DependencyGraph` is the DepDAG workflow (Sec. 3.2.4):
an ordered subtask list whose dependency DAG is the chain t_1 -> t_2 -> ... -> t_n, with criteria only
on the transitions the controller may take (retry, forward, terminate).
"""

from __future__ import annotations

from itertools import pairwise
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

END = "__end__"
"""The terminal decision (the paper's ⊥)."""

GIVE_UP = "give_up"
"""How END is named to models (planner criteria, router choice). A self-describing id: models given the
bare `__end__` read it as "end of this subtask" and pick it when they mean to move on."""


class Subtask(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str
    success_criteria: str
    executor: str
    """Name of the executor profile (tool set) assigned by the planner."""


def _check_unique_ids(subtasks: tuple[Subtask, ...]) -> list[str]:
    ids = [s.id for s in subtasks]
    if not ids:
        raise ValueError("a graph needs at least one subtask")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate subtask ids: {ids}")
    for reserved in (END, GIVE_UP):
        if reserved in ids:
            raise ValueError(f"{reserved!r} is reserved")
    return ids


class CyclicGraph(BaseModel):
    """Complete cyclic subtask graph G = (T, E, C) with E = T x T."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["cyclic"] = "cyclic"
    name: str
    description: str
    subtasks: tuple[Subtask, ...]
    entry: str
    criteria: dict[str, dict[str, str]]
    """criteria[src][dst] = c_{src->dst}, defined for every ordered pair including src == dst."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        ids = _check_unique_ids(self.subtasks)
        if self.entry not in ids:
            raise ValueError(f"entry {self.entry!r} is not a subtask")
        expected = set(ids)
        if set(self.criteria) != expected:
            raise ValueError("criteria must have one row per subtask")
        for src, row in self.criteria.items():
            if set(row) != expected:
                raise ValueError(f"criteria for {src!r} must cover every subtask, including itself")
        return self

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.subtasks)

    def subtask(self, subtask_id: str) -> Subtask:
        return next(s for s in self.subtasks if s.id == subtask_id)


class DependencyGraph(BaseModel):
    """DepDAG workflow: subtasks in dependency order, t_i -> t_{i+1}."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["dependency"] = "dependency"
    name: str
    description: str
    subtasks: tuple[Subtask, ...]
    retry: bool
    """True for DepDAG (same-subtask retry allowed), False for DepDAG-NoRetry."""
    criteria: dict[str, dict[str, str]]
    """criteria[src][dst] for dst in the allowed set of src: successor, END, and src itself if retry."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        ids = _check_unique_ids(self.subtasks)
        if set(self.criteria) != set(ids):
            raise ValueError("criteria must have one row per subtask")
        for src in ids:
            if set(self.criteria[src]) != set(self.allowed(src)):
                raise ValueError(f"criteria for {src!r} must cover exactly {self.allowed(src)}")
        return self

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.subtasks)

    @property
    def entry(self) -> str:
        return self.subtasks[0].id

    def subtask(self, subtask_id: str) -> Subtask:
        return next(s for s in self.subtasks if s.id == subtask_id)

    def successors(self, subtask_id: str) -> tuple[str, ...]:
        """Succ_DAG: the next subtask in the chain, if any."""
        ids = self.ids
        position = ids.index(subtask_id)
        return ids[position + 1 : position + 2]

    def allowed(self, subtask_id: str) -> tuple[str, ...]:
        """A^DAG = Succ_DAG(i) ∪ {i if retry} ∪ {END}, in that order."""
        own = (subtask_id,) if self.retry else ()
        return (*self.successors(subtask_id), *own, END)

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        """Dependency adjacency (prerequisite, dependent)."""
        return tuple(pairwise(self.ids))


TaskGraph = Annotated[CyclicGraph | DependencyGraph, Field(discriminator="kind")]
