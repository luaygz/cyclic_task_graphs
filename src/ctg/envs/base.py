"""Benchmark environments: what the controllers and the runner rely on."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ctg.config import BenchmarkName, Budgets, ExperimentConfig, ModelSpec, Role
from ctg.llm.base import LLM, ToolSpec
from ctg.runs.records import StepRecord

GENERALIST = "generalist"
"""Roster name of the executor profile that has every tool."""


@dataclass(frozen=True)
class ExecutorProfile:
    """A tool-using executor: the tool subset U_i it may call and an optional per-segment cap."""

    name: str
    description: str
    tools: tuple[str, ...]
    max_tool_calls: int | None = None
    """Per-segment cap specific to this profile; the effective cap is min(this, C_l)."""


@dataclass(frozen=True)
class InstanceRef:
    """One benchmark instance in a frozen instance list."""

    id: str
    """Stable and filesystem-safe."""
    meta: Mapping[str, Any] = field(default_factory=dict[str, Any])


class Environment(ABC):
    """A single episode of a benchmark instance.

    Every executed tool call counts toward the global budget, including invalid commands, because each
    one is an action the agent spent.
    """

    @property
    @abstractmethod
    def objective(self) -> str:
        """The task text O shown to every agent (for TextCraft this includes the crafting commands)."""

    @property
    @abstractmethod
    def won(self) -> bool: ...

    @property
    @abstractmethod
    def done(self) -> bool:
        """The episode cannot continue: the goal is reached or a terminal action was taken."""

    @property
    @abstractmethod
    def tool_calls(self) -> int:
        """Global tool-call counter k."""

    @property
    @abstractmethod
    def steps(self) -> Sequence[StepRecord]: ...

    @abstractmethod
    def initial_state(self) -> dict[str, Any]:
        """Canonical snapshot of the state before any action (same format as StepRecord.state)."""

    @abstractmethod
    def tool_specs(self, names: Sequence[str]) -> list[ToolSpec]:
        """Specs for the named tools, built from the current state (e.g. admissible-command enums).

        A tool with no valid action in the current state is omitted, so the result may be shorter
        than `names` (possibly empty).
        """

    @abstractmethod
    async def call(
        self, name: str, arguments: Mapping[str, Any], offered: Collection[str] | None = None
    ) -> str:
        """Execute one tool call and return the observation. Increments `tool_calls`.

        `offered` is the set of tools the calling agent was given on this turn (None: every tool). A tool
        outside it is not executed: the call returns
        "Error: tool '<name>' is not available to this executor. Available tools: <sorted offered>." and is
        otherwise handled like an unknown tool: it counts as a tool call and leaves the state unchanged.

        Unknown tools and invalid arguments do not raise: they return an error observation and still
        count as a tool call. Every counted call, executed or not, is recorded as a step numbered by the
        new `tool_calls` value, so `len(steps) == tool_calls`; a call that was not executed records the
        unchanged state. A call after `done` is a programming error and raises RuntimeError. A backend
        that cannot serve the run raises BackendUnavailableError.
        """

    def details(self) -> dict[str, Any]:
        """Benchmark-specific result details stored with the episode."""
        return {}

    async def close(self) -> None:
        return None


class Benchmark(ABC):
    """A benchmark family: description B, tools, executor roster, budgets, instances."""

    name: BenchmarkName

    @property
    @abstractmethod
    def description(self) -> str:
        """Benchmark description B given to the planner (and as context to executors and router)."""

    @property
    @abstractmethod
    def roster(self) -> Mapping[str, ExecutorProfile]:
        """Executor profiles keyed by name. Always contains GENERALIST."""

    def executor_roster(self, specialists: bool) -> list[ExecutorProfile]:
        """Profiles the planner may assign: the generalist alone, or the specialists alone."""
        if not specialists:
            return [self.roster[GENERALIST]]
        return [profile for name, profile in self.roster.items() if name != GENERALIST]

    @abstractmethod
    def budgets(self, config: ExperimentConfig) -> Budgets:
        """Paper defaults (Sec. 4.1.5) for this benchmark setting."""

    @abstractmethod
    def enumerate_instances(self, variant: str) -> list[InstanceRef]:
        """Every instance of a variant, in a deterministic (sorted) order. Used to freeze data/splits."""

    @abstractmethod
    async def make_env(
        self,
        instance: InstanceRef,
        config: ExperimentConfig,
        llm: LLM,
        models: Mapping[Role, ModelSpec],
    ) -> Environment:
        """Create a fresh episode for `instance`. `llm` is scoped to the episode."""
