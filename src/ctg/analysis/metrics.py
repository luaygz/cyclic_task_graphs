"""Evaluation metrics (Sec. 4.1.3, Appendix B.4).

Spread conventions follow the paper's tables: SR is the mean ± population std of per-repeat success rates
(Table 2), while episode-level quantities pooled across repeats (TC, ATS, AUS, ASR) use the sample std,
reported as 0 for a single episode (Table 9).

SR averages only repeats in which every scheduled episode has a non-error record, so a partly finished
repeat never weighs as much as a full one. A run without any complete repeat reports the rate pooled over
its evaluated episodes, with no std.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ctg.analysis.load import Run, is_error
from ctg.config import Controller
from ctg.runs.records import EpisodeRecord

WILSON_Z = 1.96

Transition = tuple[str, str]


@dataclass(frozen=True)
class MeanStd:
    mean: float
    std: float | None
    """None when unavailable: the pooled SR of a run without any complete repeat."""
    n: int
    """Values averaged: repeats for SR over repeats, episodes for pooled quantities."""


def population_stats(values: Sequence[float]) -> MeanStd | None:
    if not values:
        return None
    return MeanStd(statistics.fmean(values), statistics.pstdev(values), len(values))


def sample_stats(values: Sequence[float]) -> MeanStd | None:
    if not values:
        return None
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return MeanStd(statistics.fmean(values), std, len(values))


def _valid(episodes: Iterable[EpisodeRecord]) -> list[EpisodeRecord]:
    return [e for e in episodes if not is_error(e)]


def success_rates(episodes: Iterable[EpisodeRecord]) -> dict[int, float]:
    """SR in percent for each repeat that has at least one non-error episode."""
    by_repeat: dict[int, list[bool]] = defaultdict(list)
    for episode in _valid(episodes):
        by_repeat[episode.repeat].append(episode.outcome.won)
    return {repeat: 100.0 * sum(wins) / len(wins) for repeat, wins in sorted(by_repeat.items())}


def success_rate(run: Run) -> MeanStd | None:
    """SR (%) as mean ± population std over the run's complete repeats.

    Without a complete repeat: the rate pooled over every non-error episode, std None. None when the run
    has no non-error episode.
    """
    per_repeat = success_rates(run.valid)
    complete = [per_repeat[r] for r in run.complete_repeats if r in per_repeat]
    if complete:
        return population_stats(complete)
    valid = run.valid
    if not valid:
        return None
    return MeanStd(100.0 * sum(e.outcome.won for e in valid) / len(valid), None, len(valid))


def successful_tool_calls(episodes: Iterable[EpisodeRecord]) -> MeanStd | None:
    """TC: tool calls of successful episodes pooled across repeats."""
    return sample_stats([e.outcome.tool_calls for e in _valid(episodes) if e.outcome.won])


def all_tool_calls(episodes: Iterable[EpisodeRecord]) -> MeanStd | None:
    """TC(all) of Table 9: tool calls over every non-error episode."""
    return sample_stats([e.outcome.tool_calls for e in _valid(episodes)])


def success_to_tool_ratio(sr: MeanStd | None, tc: MeanStd | None) -> float | None:
    """STR = SR (percentage points) / TC."""
    if sr is None or tc is None or tc.mean == 0:
        return None
    return sr.mean / tc.mean


@dataclass(frozen=True)
class Visitation:
    ats: MeanStd
    """Total subtask visits (segments) per successful episode."""
    aus: MeanStd
    """Distinct subtasks visited per successful episode."""
    asr: MeanStd
    """Revisits per successful episode, ATS - AUS."""


def visitation(episodes: Iterable[EpisodeRecord]) -> Visitation | None:
    """ATS / AUS / ASR over successful graph-controller episodes; None when there are none."""
    won = [e for e in _valid(episodes) if e.outcome.won and e.controller is not Controller.REACT]
    totals = [float(len(e.segments)) for e in won]
    uniques = [float(len({s.node for s in e.segments})) for e in won]
    ats, aus = sample_stats(totals), sample_stats(uniques)
    asr = sample_stats([t - u for t, u in zip(totals, uniques, strict=True)])
    if ats is None or aus is None or asr is None:
        return None
    return Visitation(ats=ats, aus=aus, asr=asr)


def executed_transitions(episode: EpisodeRecord) -> set[Transition]:
    """Distinct (source, target) pairs the controller actually took (after any fault injection)."""
    return {(s.transition.source, s.transition.target) for s in episode.segments if s.transition is not None}


@dataclass(frozen=True)
class UniqueTransitions:
    all: frozenset[Transition]
    won: frozenset[Transition]

    @property
    def never_won(self) -> frozenset[Transition]:
        return self.all - self.won


def unique_transitions(episodes: Iterable[EpisodeRecord]) -> UniqueTransitions:
    """UT over all episodes, over won episodes, and never in a won episode."""
    seen: set[Transition] = set()
    won: set[Transition] = set()
    for episode in _valid(episodes):
        pairs = executed_transitions(episode)
        seen |= pairs
        if episode.outcome.won:
            won |= pairs
    return UniqueTransitions(all=frozenset(seen), won=frozenset(won))


def wilson_interval(successes: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (Eq. 9), as fractions."""
    if n <= 0:
        raise ValueError("the Wilson interval needs at least one episode")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half_width = z / denominator * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, center - half_width), min(1.0, center + half_width)


@dataclass(frozen=True)
class RunMetrics:
    episodes: int
    """Non-error episodes."""
    errors: int
    wins: int
    sr: MeanStd | None
    tc: MeanStd | None
    tc_all: MeanStd | None
    str_: float | None
    visitation: Visitation | None
    transitions: UniqueTransitions

    @property
    def wilson(self) -> tuple[float, float] | None:
        """Wilson 95% interval (fractions) on wins pooled over repeats."""
        return wilson_interval(self.wins, self.episodes) if self.episodes else None


def run_metrics(run: Run) -> RunMetrics:
    valid = run.valid
    sr = success_rate(run)
    tc = successful_tool_calls(valid)
    return RunMetrics(
        episodes=len(valid),
        errors=run.errors,
        wins=sum(1 for e in valid if e.outcome.won),
        sr=sr,
        tc=tc,
        tc_all=all_tool_calls(valid),
        str_=success_to_tool_ratio(sr, tc),
        visitation=visitation(valid),
        transitions=unique_transitions(valid),
    )


@dataclass(frozen=True)
class Robustness:
    """A nominal run against its fault-injected counterpart (Sec. 4.3, Tables 3, 17, 18)."""

    nominal: RunMetrics
    perturbed: RunMetrics
    srr: float | None
    """SR_fault / SR_nominal."""
    delta_sr: float | None
    """SR_fault - SR_nominal in percentage points."""
    delta_tc: float | None
    """TC_fault - TC_nominal."""
    relative_drop: float | None
    """|SR_fault - SR_nominal| / SR_nominal in percent (the "Rel. Diff." column of Tables 17-18)."""


ROBUSTNESS_IGNORED_FIELDS = frozenset({"name", "perturbation", "seed", "concurrency"})
"""Config fields a nominal and a fault-injected run may differ in."""


def robustness(nominal: Run, perturbed: Run) -> Robustness:
    if not nominal.config.perturbation == 0 < perturbed.config.perturbation:
        raise ValueError(
            f"the nominal run needs perturbation 0 and the perturbed run a positive perturbation; "
            f"{nominal.name} has {nominal.config.perturbation:g}, "
            f"{perturbed.name} has {perturbed.config.perturbation:g}"
        )
    ours = nominal.config.model_dump(exclude=set(ROBUSTNESS_IGNORED_FIELDS))
    theirs = perturbed.config.model_dump(exclude=set(ROBUSTNESS_IGNORED_FIELDS))
    if differing := sorted(key for key in ours if ours[key] != theirs[key]):
        raise ValueError(
            f"{nominal.name} and {perturbed.name} differ in {', '.join(differing)}; "
            "SRR compares one configuration with and without fault injection"
        )
    nom, pert = run_metrics(nominal), run_metrics(perturbed)
    srr = delta_sr = relative_drop = delta_tc = None
    if nom.sr is not None and pert.sr is not None:
        delta_sr = pert.sr.mean - nom.sr.mean
        if nom.sr.mean > 0:
            srr = pert.sr.mean / nom.sr.mean
            relative_drop = 100.0 * abs(delta_sr) / nom.sr.mean
    if nom.tc is not None and pert.tc is not None:
        delta_tc = pert.tc.mean - nom.tc.mean
    return Robustness(
        nominal=nom,
        perturbed=pert,
        srr=srr,
        delta_sr=delta_sr,
        delta_tc=delta_tc,
        relative_drop=relative_drop,
    )
