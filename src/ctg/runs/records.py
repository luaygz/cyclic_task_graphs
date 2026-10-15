"""What a run writes to disk. Analysis reads only these models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ctg.config import (
    ROLE_SETTINGS,
    BenchmarkName,
    Budgets,
    Controller,
    ExecutorMode,
    ExperimentConfig,
    ModelSpec,
    Provider,
    ReasoningEffort,
    Role,
)
from ctg.graphs.model import TaskGraph
from ctg.llm.base import LLMCallRecord

SCHEMA_VERSION = 2
"""Version of the manifest and episode record formats. A run directory is read only by the version that
wrote it."""


class TransitionType(StrEnum):
    SELF = "self"
    """Cyclic graph: route back to the current subtask."""
    CROSS = "cross"
    """Cyclic graph: route to a different subtask."""
    RETRY = "retry"
    """DepDAG: repeat the current subtask."""
    FORWARD = "forward"
    """DepDAG: advance to a DAG successor."""
    TERMINAL = "terminal"
    """DepDAG: terminate (⊥)."""


class EndReason(StrEnum):
    WON = "won"
    BUDGET = "budget"
    FINAL_ANSWER = "final_answer"
    """Finance-Agent: a final answer was submitted and judged incorrect."""
    TERMINATED = "terminated"
    """DepDAG: the router chose ⊥ before the goal was reached."""
    OUTPUT_FAILURE = "output_failure"
    """A model's output stayed unusable after the client's retries (truncated, refused, invalid JSON or
    schema, or a plan that kept failing validation). A loss of the method under test: counted in every
    rate and not re-run on resume."""
    ERROR = "error"
    """Infrastructure failure (network, provider outage, spend cap, unusable backend). Excluded from the
    rates and re-run on resume."""


class Inconsistency(BaseModel):
    description: str
    cause: str
    correction_plan: list[str]


class Resource(BaseModel):
    name: str
    quantity: int


class Memory(BaseModel):
    """The router's structured rolling memory M_m (Sec. 3.2.1), passed to the next executor."""

    segment_analysis: list[str]
    inconsistencies: list[Inconsistency]
    held_resources: list[Resource]
    environment_notes: list[str]
    summary: str
    subtask_progress: bool
    failure_analysis: list[str]


class ToolCallRecord(BaseModel):
    tool: str
    arguments: dict[str, Any]
    observation: str
    tool_calls_after: int
    """Global tool-call counter k after this call."""


class StepRecord(BaseModel):
    """One environment step with a canonical snapshot of the resulting state."""

    step: int
    tool: str
    arguments: dict[str, Any]
    observation: str
    state: dict[str, Any]
    """JSON-serializable; two equal snapshots mean the same environment state (used for A6)."""


class RouterRecord(BaseModel):
    memory: Memory
    routing_reasoning: list[str]
    choice: str


class TransitionRecord(BaseModel):
    source: str
    target: str
    allowed: list[str]
    nominal: str
    """The router's choice before fault injection."""
    perturbed: bool
    type: TransitionType
    valid: bool
    """Whether `target` is in `allowed`."""


class SegmentRecord(BaseModel):
    index: int
    node: str
    executor: str
    tool_calls: list[ToolCallRecord]
    summary: str
    """The executor's closing summary of the segment (part of H_m)."""
    router: RouterRecord | None = None
    transition: TransitionRecord | None = None


class Outcome(BaseModel):
    won: bool
    end_reason: EndReason
    tool_calls: int
    segments: int


class PreviousAttempt(BaseModel):
    """An earlier ERROR attempt of the same episode, kept when a resume runs it again."""

    error: str
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime


class EpisodeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    run: str
    benchmark: BenchmarkName
    variant: str
    instance_id: str
    instance: dict[str, Any]
    """Benchmark-specific identity of the instance (goal item, game path, question)."""
    objective: str
    repeat: int
    controller: Controller
    executors: ExecutorMode
    perturbation: float
    graph: TaskGraph | None = None
    planner_notes: dict[str, Any] = Field(default_factory=dict)
    initial_state: dict[str, Any] = Field(default_factory=dict)
    segments: list[SegmentRecord] = Field(default_factory=list)
    react_steps: list[ToolCallRecord] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    env_details: dict[str, Any] = Field(default_factory=dict)
    """Benchmark-specific result details (e.g. Finance final answer and judge verdicts)."""
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    previous_attempts: list[PreviousAttempt] = Field(default_factory=list)
    """Earlier ERROR attempts, oldest first. Their calls count toward spend, not toward per-case tokens."""
    outcome: Outcome
    error: str | None = None
    started_at: datetime
    finished_at: datetime


class RoleModel(BaseModel):
    """What decides a role's model calls: the preset's model fields and the role's decoding settings
    (Table 8). Prices are left out; they change the accounting, not the calls."""

    name: str
    provider: Provider
    reasoning_effort: ReasoningEffort | None
    supports_temperature: bool
    temperature: float | None
    max_tokens: int | None

    @classmethod
    def of(cls, role: Role, model: ModelSpec) -> RoleModel:
        settings = ROLE_SETTINGS[role]
        return cls(
            name=model.name,
            provider=model.provider,
            reasoning_effort=model.reasoning_effort,
            supports_temperature=model.supports_temperature,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )


class RunConditions(BaseModel):
    """What a run's episodes depend on besides its `ExperimentConfig`. A resume requires it unchanged."""

    nshot_sha256: str | None
    """SHA-256 of the n-shot file's bytes; None for runs without n-shot summaries."""
    models: dict[Role, RoleModel]
    """The model of each role the run uses."""
    budgets: Budgets
    """The effective tool-call budgets: the config's, or the benchmark's defaults."""


class RunManifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    name: str
    run_id: str
    """`<name>/<uuid4 hex>`, created with the run directory. Model calls are recorded under it in the spend
    ledger, so the per-run spend cap covers this directory across invocations and nothing else."""
    config: ExperimentConfig
    conditions: RunConditions
    ctg_version: str
    git_commit: str | None
    created_at: datetime
