"""Environment-level exploration (A6, Appendix G, Fig. 4): unique environment states visited per episode."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ctg.analysis.load import Run
from ctg.config import Controller
from ctg.runs.records import EpisodeRecord


def canonical_state(state: dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def unique_states(episode: EpisodeRecord) -> int:
    """Distinct snapshots among the initial state and the state after every step."""
    return len({canonical_state(episode.initial_state), *(canonical_state(s.state) for s in episode.steps)})


def unique_states_by_instance(run: Run) -> dict[str, float]:
    """Mean unique states over the repeats of each instance (non-error episodes)."""
    counts: dict[str, list[int]] = defaultdict(list)
    for episode in run.valid:
        counts[episode.instance_id].append(unique_states(episode))
    return {instance: statistics.fmean(values) for instance, values in sorted(counts.items())}


@dataclass(frozen=True)
class InstanceExploration:
    instance_id: str
    cyclic: float
    react: float

    @property
    def difference(self) -> float:
        """Cyclic minus ReAct."""
        return self.cyclic - self.react


@dataclass(frozen=True)
class ExplorationComparison:
    cyclic_run: str
    react_run: str
    instances: tuple[InstanceExploration, ...]
    """Instances evaluated by both runs, ordered by instance id."""
    unmatched: int
    """Instances evaluated by only one of the runs (left out of the comparison)."""

    @property
    def mean_difference(self) -> float | None:
        return statistics.fmean(i.difference for i in self.instances) if self.instances else None


def compare_exploration(cyclic: Run, react: Run) -> ExplorationComparison:
    if react.controller is not Controller.REACT:
        raise ValueError(f"{react.name} is a {react.controller.value} run; the ReAct side needs a ReAct run")
    if cyclic.controller is Controller.REACT:
        raise ValueError(f"{cyclic.name} is a ReAct run; the cyclic side needs a graph-controller run")
    if cyclic.variant != react.variant:
        raise ValueError(f"{cyclic.name} and {react.name} evaluate different benchmark variants")
    ours, theirs = unique_states_by_instance(cyclic), unique_states_by_instance(react)
    shared = sorted(ours.keys() & theirs.keys())
    return ExplorationComparison(
        cyclic_run=cyclic.name,
        react_run=react.name,
        instances=tuple(InstanceExploration(i, cyclic=ours[i], react=theirs[i]) for i in shared),
        unmatched=len(ours.keys() ^ theirs.keys()),
    )
