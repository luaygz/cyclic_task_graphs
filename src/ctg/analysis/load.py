"""Loading run directories for analysis.

Only infrastructure errors (`EndReason.ERROR`) are left out: every analysis excludes them from rates and
reports how many there were. An `OUTPUT_FAILURE` (the model's output stayed unusable) is a loss and counts
like any other lost episode.

A run is complete when every episode its config schedules (`config.repeats` × its selected instances) has a
non-error record. Tables and figures name incomplete runs, since an unfinished run's rates cover fewer
episodes than its configuration promises.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ctg.config import Controller, ExperimentConfig
from ctg.llm.base import LLMCallRecord
from ctg.runs.records import EndReason, EpisodeRecord, RunManifest
from ctg.runs.splits import select_instances
from ctg.runs.store import RunStore

EpisodeKey = tuple[int, str]
"""(repeat, instance id)."""


def is_error(episode: EpisodeRecord) -> bool:
    """An infrastructure failure: excluded from every rate and re-run on resume."""
    return episode.outcome.end_reason is EndReason.ERROR


@dataclass(frozen=True)
class Run:
    path: Path
    manifest: RunManifest
    episodes: tuple[EpisodeRecord, ...]
    """Every stored episode, error episodes included, ordered by repeat and instance id."""
    expected: frozenset[EpisodeKey]
    """Every episode the config schedules: `config.repeats` × the config's selected instances."""
    planning_calls: tuple[LLMCallRecord, ...] = ()
    """Gen-Cyc only: the one-time planning calls stored with the run's graph (not part of any episode)."""

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def config(self) -> ExperimentConfig:
        return self.manifest.config

    @property
    def controller(self) -> Controller:
        return self.manifest.config.controller

    @property
    def variant(self) -> str:
        return self.manifest.config.variant

    @property
    def valid(self) -> tuple[EpisodeRecord, ...]:
        """Episodes that count towards rates (errors excluded)."""
        return tuple(e for e in self.episodes if not is_error(e))

    @property
    def errors(self) -> int:
        return sum(1 for e in self.episodes if is_error(e))

    @property
    def missing(self) -> frozenset[EpisodeKey]:
        """Scheduled episodes without a non-error record: not run yet, still running, or ended in ERROR."""
        return self.expected - {(e.repeat, e.instance_id) for e in self.valid}

    @property
    def finished(self) -> int:
        """Scheduled episodes with a non-error record."""
        return len(self.expected) - len(self.missing)

    @property
    def complete_repeats(self) -> tuple[int, ...]:
        """Scheduled repeats whose every episode has a non-error record, ascending."""
        incomplete = set(self.incomplete_repeats)
        return tuple(sorted({repeat for repeat, _ in self.expected} - incomplete))

    @property
    def incomplete_repeats(self) -> tuple[int, ...]:
        """Scheduled repeats with at least one missing episode, ascending."""
        return tuple(sorted({repeat for repeat, _ in self.missing}))


def expected_episodes(config: ExperimentConfig) -> frozenset[EpisodeKey]:
    """(repeat, instance id) of every episode the config schedules."""
    instances = select_instances(config)
    return frozenset((repeat, instance.id) for repeat in range(config.repeats) for instance in instances)


def load_run(path: Path) -> Run:
    store = RunStore(path)
    manifest = store.manifest()
    if manifest is None:
        raise FileNotFoundError(f"{path} is not a run directory (no manifest.json)")
    stored_graph = store.gen_cyc_graph()
    return Run(
        path=path,
        manifest=manifest,
        episodes=tuple(store.episodes()),
        expected=expected_episodes(manifest.config),
        planning_calls=tuple(stored_graph.llm_calls) if stored_graph is not None else (),
    )


def load_runs(paths: Iterable[Path]) -> list[Run]:
    return [load_run(path) for path in paths]


def error_note(runs: Sequence[Run]) -> str | None:
    """A one-line account of excluded error episodes, or None when there were none."""
    parts = [f"{run.name} ({run.errors})" for run in runs if run.errors]
    if not parts:
        return None
    return f"Error episodes excluded from all rates: {', '.join(parts)}."


def coverage_note(runs: Sequence[Run]) -> str | None:
    """A one-line account of incomplete runs, or None when every run is complete."""
    parts = [
        f"{run.name} ({run.finished}/{len(run.expected)} episodes; "
        f"repeat{'s' if len(run.incomplete_repeats) > 1 else ''} "
        f"{', '.join(map(str, run.incomplete_repeats))} incomplete)"
        for run in runs
        if run.missing
    ]
    if not parts:
        return None
    return f"Incomplete runs (scheduled episodes without a non-error record): {'; '.join(parts)}."


def run_notes(runs: Sequence[Run]) -> tuple[str, ...]:
    """The error and coverage notes that apply to `runs`, in that order."""
    return tuple(note for note in (error_note(runs), coverage_note(runs)) if note is not None)
