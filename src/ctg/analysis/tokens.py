"""Token accounting and the shared-win token comparison (Appendix C, Table 12).

The headline total of an episode is input + output tokens over the planner, executor, router and tool roles
(ReAct calls are executor calls). Judge tokens are evaluation overhead and are reported separately.
Per-test-case totals cover an episode's final attempt only. Gen-Cyc's one-time planning calls and the earlier
error attempts of episodes re-run on resume appear in spend but not in the per-test-case totals.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from ctg.analysis.load import Run
from ctg.analysis.metrics import MeanStd, sample_stats
from ctg.config import Role
from ctg.llm.base import LLMCallRecord
from ctg.runs.records import EpisodeRecord

HEADLINE_ROLES = (Role.PLANNER, Role.EXECUTOR, Role.ROUTER, Role.TOOL)


@dataclass(frozen=True)
class RoleUsage:
    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, call: LLMCallRecord) -> RoleUsage:
        return replace(
            self,
            calls=self.calls + 1,
            input_tokens=self.input_tokens + call.input_tokens,
            cached_input_tokens=self.cached_input_tokens + call.cached_input_tokens,
            output_tokens=self.output_tokens + call.output_tokens,
            reasoning_tokens=self.reasoning_tokens + call.reasoning_tokens,
            cost_usd=self.cost_usd + call.cost_usd,
        )


def usage_by_role(calls: Iterable[LLMCallRecord]) -> dict[Role, RoleUsage]:
    usage: dict[Role, RoleUsage] = {}
    for call in calls:
        usage[call.role] = usage.get(call.role, RoleUsage()).add(call)
    return usage


def headline_tokens(calls: Iterable[LLMCallRecord]) -> int:
    return sum(c.input_tokens + c.output_tokens for c in calls if c.role in HEADLINE_ROLES)


def judge_tokens(calls: Iterable[LLMCallRecord]) -> int:
    return sum(c.input_tokens + c.output_tokens for c in calls if c.role is Role.JUDGE)


def episode_tokens(episode: EpisodeRecord) -> int:
    return headline_tokens(episode.llm_calls)


@dataclass(frozen=True)
class RunUsage:
    run: str
    episodes: int
    """Non-error episodes: the test cases behind the per-case averages."""
    tokens_per_case: MeanStd | None
    """Headline tokens per test case."""
    judge_tokens_per_case: MeanStd | None
    by_role: dict[Role, RoleUsage]
    """Every stored call: one-time planning, every episode's final attempt (error episodes included) and the
    earlier error attempts of episodes re-run on resume."""

    @property
    def spend_usd(self) -> float:
        return sum((u.cost_usd for u in self.by_role.values()), 0.0)


def run_usage(run: Run) -> RunUsage:
    every_call = [
        *run.planning_calls,
        *(c for e in run.episodes for attempt in e.previous_attempts for c in attempt.llm_calls),
        *(c for e in run.episodes for c in e.llm_calls),
    ]
    return RunUsage(
        run=run.name,
        episodes=len(run.valid),
        tokens_per_case=sample_stats([float(episode_tokens(e)) for e in run.valid]),
        judge_tokens_per_case=sample_stats([float(judge_tokens(e.llm_calls)) for e in run.valid]),
        by_role=usage_by_role(every_call),
    )


@dataclass(frozen=True)
class TokenScope:
    """Average headline tokens per test case of two runs over one set of won test cases."""

    cases_a: int
    cases_b: int
    avg_a: float | None
    avg_b: float | None

    @property
    def excess(self) -> float | None:
        """B - A."""
        if self.avg_a is None or self.avg_b is None:
            return None
        return self.avg_b - self.avg_a

    @property
    def relative_pct(self) -> float | None:
        """(B - A) / A in percent."""
        if self.excess is None or not self.avg_a:
            return None
        return 100.0 * self.excess / self.avg_a


@dataclass(frozen=True)
class TokenComparison:
    baseline: str
    compare: str
    shared_wins: TokenScope
    """Test cases (instance, repeat) won by both runs: the primary comparison."""
    all_wins: TokenScope
    """Every won test case of each run: secondary context."""


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def compare_tokens(baseline: Run, compare: Run) -> TokenComparison:
    if baseline.variant != compare.variant:
        raise ValueError(f"{baseline.name} and {compare.name} evaluate different benchmark variants")
    wins_a = {(e.instance_id, e.repeat): episode_tokens(e) for e in baseline.valid if e.outcome.won}
    wins_b = {(e.instance_id, e.repeat): episode_tokens(e) for e in compare.valid if e.outcome.won}
    shared = sorted(wins_a.keys() & wins_b.keys())
    return TokenComparison(
        baseline=baseline.name,
        compare=compare.name,
        shared_wins=TokenScope(
            cases_a=len(shared),
            cases_b=len(shared),
            avg_a=_mean([wins_a[key] for key in shared]),
            avg_b=_mean([wins_b[key] for key in shared]),
        ),
        all_wins=TokenScope(
            cases_a=len(wins_a),
            cases_b=len(wins_b),
            avg_a=_mean(list(wins_a.values())),
            avg_b=_mean(list(wins_b.values())),
        ),
    )
