"""Graph-size and routing statistics (Appendix E, Tables 21-24).

Per run: subtasks per graph and words per edge criterion, averaged over the test cases that have a graph (a
Gen-Cyc run reuses one graph, so its averages are that graph's; an episode whose planning failed has none),
and self-loops and inter-node transitions per test case, counting every transition an episode took.
Self-loops are cyclic `self` and DepDAG `retry` transitions; inter-node transitions are cyclic `cross`
and DepDAG `forward` transitions. Across runs of one controller and benchmark, the per-run values are
summarized as observed ranges.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from ctg.analysis.load import Run
from ctg.config import Controller
from ctg.graphs.model import CyclicGraph, DependencyGraph
from ctg.runs.records import EpisodeRecord, TransitionType

SELF_LOOP_TYPES = (TransitionType.SELF, TransitionType.RETRY)
INTER_NODE_TYPES = (TransitionType.CROSS, TransitionType.FORWARD)


def criterion_words(graph: CyclicGraph | DependencyGraph) -> float:
    """Mean number of whitespace-separated words per edge criterion."""
    counts = [len(text.split()) for row in graph.criteria.values() for text in row.values()]
    return statistics.fmean(counts)


def _count(episode: EpisodeRecord, types: tuple[TransitionType, ...]) -> int:
    return sum(1 for s in episode.segments if s.transition is not None and s.transition.type in types)


@dataclass(frozen=True)
class RunRouting:
    run: str
    controller: Controller
    variant: str
    test_cases: int
    nodes: float | None
    criterion_words: float | None
    self_loops: float | None
    inter_node: float | None


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def run_routing(run: Run) -> RunRouting:
    if run.controller is Controller.REACT:
        raise ValueError(f"{run.name} is a ReAct run and has no subtask graph")
    episodes = run.valid
    graphs = [e.graph for e in episodes if e.graph is not None]
    return RunRouting(
        run=run.name,
        controller=run.controller,
        variant=run.variant,
        test_cases=len(episodes),
        nodes=_mean([float(len(g.subtasks)) for g in graphs]),
        criterion_words=_mean([criterion_words(g) for g in graphs]),
        self_loops=_mean([float(_count(e, SELF_LOOP_TYPES)) for e in episodes]),
        inter_node=_mean([float(_count(e, INTER_NODE_TYPES)) for e in episodes]),
    )


@dataclass(frozen=True)
class Range:
    low: float
    high: float


def value_range(values: Sequence[float | None]) -> Range | None:
    present = [v for v in values if v is not None]
    return Range(min(present), max(present)) if present else None


@dataclass(frozen=True)
class RoutingSummary:
    controller: Controller
    variant: str
    runs: int
    nodes: Range | None
    criterion_words: Range | None
    self_loops: Range | None
    inter_node: Range | None


def summarize_routing(rows: Sequence[RunRouting]) -> list[RoutingSummary]:
    """One summary per (controller, variant), in order of first appearance."""
    groups: dict[tuple[Controller, str], list[RunRouting]] = {}
    for row in rows:
        groups.setdefault((row.controller, row.variant), []).append(row)
    return [
        RoutingSummary(
            controller=controller,
            variant=variant,
            runs=len(group),
            nodes=value_range([r.nodes for r in group]),
            criterion_words=value_range([r.criterion_words for r in group]),
            self_loops=value_range([r.self_loops for r in group]),
            inter_node=value_range([r.inter_node for r in group]),
        )
        for (controller, variant), group in groups.items()
    ]
