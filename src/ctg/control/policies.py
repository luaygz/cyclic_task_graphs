"""Candidate sets A_m and transition typing for each controller (Sec. 3.2.3, 3.2.4)."""

from __future__ import annotations

from collections.abc import Sequence

from ctg.graphs.model import END, CyclicGraph, DependencyGraph
from ctg.runs.records import TransitionType

Graph = CyclicGraph | DependencyGraph


def candidates(graph: Graph, current: str) -> tuple[str, ...]:
    """A_m: every subtask for a cyclic graph; Succ(i) ∪ {i if retry} ∪ {⊥} for a dependency graph."""
    if isinstance(graph, CyclicGraph):
        return graph.ids
    return graph.allowed(current)


def is_forced_end(allowed: Sequence[str]) -> bool:
    """A_m = {⊥}: the controller terminates without consulting the router."""
    return tuple(allowed) == (END,)


def transition_type(graph: Graph, source: str, target: str) -> TransitionType:
    if isinstance(graph, CyclicGraph):
        if target == END:
            raise ValueError("cyclic graphs have no terminal transition")
        return TransitionType.SELF if target == source else TransitionType.CROSS
    if target == END:
        return TransitionType.TERMINAL
    return TransitionType.RETRY if target == source else TransitionType.FORWARD
