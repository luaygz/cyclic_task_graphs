"""Finance-Agent tool-use diagnostics and trace-derived failure labels (Appendix B.6 and I, Tables 10-11).

Labels are deterministic heuristics over the stored records, not human adjudications. Each failed
(non-error) episode gets one primary label from Appendix I's taxonomy; the first matching rule wins.

A final answer is a non-empty `finance_agent_final_answer` submission (`env_details["answer"]`).

Episodes without a final answer:
1. Executor action-selection drift: an empty answer was submitted, or at least half of the episode's tool
   calls failed (observations starting with "Error: ").
2. Routing loop / unproductive revisitation: at least two segments revisit an already visited subtask and
   the router's memory reports no subtask progress for them.
3. Budget exhaustion before final answer: the episode ended on the tool-call budget.
4. Graph/prompt mismatch: a DepDAG workflow terminated (⊥) without an answer, so the generated chain or
   its criteria never led to one.

Episodes with a final answer judged incorrect:
5. Retrieval/source miss: no document was stored with `finance_agent_parse_html_page`.
6. Evidence grounding mismatch: documents were stored but never read back with a successful
   `finance_agent_retrieve_information` call, so the answer cannot rest on them.
7. Synthesis/calculation error: evidence was read, the judge found at least one correctness criterion
   satisfied and at least one rubric item unsatisfied, i.e. part of the expected content was reached but
   combined or completed wrongly. (A contradiction criterion is satisfied whenever nothing is contradicted,
   so it alone never shows that expected content was reached.)

Everything else is Other / unresolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

from ctg.analysis.load import Run
from ctg.analysis.metrics import MeanStd, all_tool_calls
from ctg.config import BenchmarkName
from ctg.runs.records import EndReason, EpisodeRecord, ToolCallRecord

RETRIEVE_TOOL = "finance_agent_retrieve_information"
TOOL_ERROR_PREFIX = "Error: "
LOOP_MIN_UNPRODUCTIVE_REVISITS = 2
DRIFT_MIN_ERROR_SHARE = 0.5
CORRECTNESS = "correctness"
"""Rubric operator of the criteria an answer must state (the other operator is "contradiction")."""


class FailureLabel(StrEnum):
    BUDGET_EXHAUSTION = "budget_exhaustion"
    ROUTING_LOOP = "routing_loop"
    RETRIEVAL_MISS = "retrieval_miss"
    GROUNDING_MISMATCH = "grounding_mismatch"
    SYNTHESIS_ERROR = "synthesis_error"
    EXECUTOR_DRIFT = "executor_drift"
    GRAPH_MISMATCH = "graph_mismatch"
    OTHER = "other"

    @property
    def display_name(self) -> str:
        return _TITLES[self]


_TITLES = {
    FailureLabel.BUDGET_EXHAUSTION: "Budget exhaustion before final answer",
    FailureLabel.ROUTING_LOOP: "Routing loop / unproductive revisitation",
    FailureLabel.RETRIEVAL_MISS: "Retrieval/source miss",
    FailureLabel.GROUNDING_MISMATCH: "Evidence grounding mismatch",
    FailureLabel.SYNTHESIS_ERROR: "Synthesis/calculation error",
    FailureLabel.EXECUTOR_DRIFT: "Executor action-selection drift",
    FailureLabel.GRAPH_MISMATCH: "Graph/prompt mismatch",
    FailureLabel.OTHER: "Other / unresolved",
}


class _Source(BaseModel):
    url: str
    name: str


class _Verdict(BaseModel):
    operator: str
    satisfied: bool


class FinanceDetails(BaseModel):
    """The parts of the Finance-Agent `env_details` the diagnostics read."""

    answer: str | None = None
    sources: list[_Source] = Field(default_factory=list[_Source])
    verdicts: list[_Verdict] = Field(default_factory=list[_Verdict])
    documents: list[str] = Field(default_factory=list[str])


def finance_details(episode: EpisodeRecord) -> FinanceDetails:
    if episode.benchmark is not BenchmarkName.FINANCE_AGENT:
        raise ValueError(f"{episode.run}/{episode.instance_id} is not a Finance-Agent episode")
    return FinanceDetails.model_validate(episode.env_details)


def has_final_answer(details: FinanceDetails) -> bool:
    return bool(details.answer and details.answer.strip())


def episode_tool_calls(episode: EpisodeRecord) -> list[ToolCallRecord]:
    return [*episode.react_steps, *(c for s in episode.segments for c in s.tool_calls)]


def failed_call(call: ToolCallRecord) -> bool:
    return call.observation.startswith(TOOL_ERROR_PREFIX)


def unproductive_revisits(episode: EpisodeRecord) -> int:
    """Segments that return to an already visited subtask while the router reports no progress."""
    visited: set[str] = set()
    count = 0
    for segment in episode.segments:
        if (
            segment.node in visited
            and segment.router is not None
            and not segment.router.memory.subtask_progress
        ):
            count += 1
        visited.add(segment.node)
    return count


def _drifted(details: FinanceDetails, calls: list[ToolCallRecord]) -> bool:
    if details.answer is not None and not details.answer.strip():
        return True
    return bool(calls) and sum(1 for c in calls if failed_call(c)) >= DRIFT_MIN_ERROR_SHARE * len(calls)


def label_failure(episode: EpisodeRecord) -> FailureLabel:
    """Primary trace-derived failure label of a failed Finance-Agent episode (rules in the module docs)."""
    if episode.outcome.won:
        raise ValueError(f"{episode.run}/{episode.instance_id} was won")
    details = finance_details(episode)
    calls = episode_tool_calls(episode)
    if not has_final_answer(details):
        if _drifted(details, calls):
            return FailureLabel.EXECUTOR_DRIFT
        if unproductive_revisits(episode) >= LOOP_MIN_UNPRODUCTIVE_REVISITS:
            return FailureLabel.ROUTING_LOOP
        if episode.outcome.end_reason is EndReason.BUDGET:
            return FailureLabel.BUDGET_EXHAUSTION
        if episode.outcome.end_reason is EndReason.TERMINATED:
            return FailureLabel.GRAPH_MISMATCH
        return FailureLabel.OTHER
    if not details.documents:
        return FailureLabel.RETRIEVAL_MISS
    if not any(c.tool == RETRIEVE_TOOL and not failed_call(c) for c in calls):
        return FailureLabel.GROUNDING_MISMATCH
    reached = any(v.satisfied for v in details.verdicts if v.operator == CORRECTNESS)
    if reached and not all(v.satisfied for v in details.verdicts):
        return FailureLabel.SYNTHESIS_ERROR
    return FailureLabel.OTHER


@dataclass(frozen=True)
class FinanceDiagnostics:
    run: str
    episodes: int
    """Non-error episodes; every rate is a fraction of these."""
    errors: int
    wins: int
    final_answer_rate: float | None
    source_submission_rate: float | None
    """Final answers submitted with at least one source."""
    budget_exhaustion_rate: float | None
    tool_calls: MeanStd | None
    """Tool calls over all non-error episodes."""
    failures: dict[FailureLabel, int]
    """Primary label counts over failed episodes; every label is present."""

    @property
    def failed(self) -> int:
        return self.episodes - self.wins


def finance_diagnostics(run: Run) -> FinanceDiagnostics:
    if run.config.benchmark is not BenchmarkName.FINANCE_AGENT:
        raise ValueError(f"{run.name} is not a Finance-Agent run")
    episodes = run.valid
    details = [finance_details(e) for e in episodes]
    n = len(episodes)

    def rate(count: int) -> float | None:
        return count / n if n else None

    failures = dict.fromkeys(FailureLabel, 0)
    for episode in episodes:
        if not episode.outcome.won:
            failures[label_failure(episode)] += 1
    return FinanceDiagnostics(
        run=run.name,
        episodes=n,
        errors=run.errors,
        wins=sum(1 for e in episodes if e.outcome.won),
        final_answer_rate=rate(sum(1 for d in details if has_final_answer(d))),
        source_submission_rate=rate(sum(1 for d in details if has_final_answer(d) and d.sources)),
        budget_exhaustion_rate=rate(sum(1 for e in episodes if e.outcome.end_reason is EndReason.BUDGET)),
        tool_calls=all_tool_calls(episodes),
        failures=failures,
    )
