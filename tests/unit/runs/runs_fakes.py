"""A fake benchmark, planner and controllers for the run tests, and a harness that runs experiments."""

from __future__ import annotations

import logging
import random
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from pydantic import BaseModel

from ctg import cli
from ctg.config import BenchmarkName, Budgets, Controller, ExperimentConfig, ModelSpec, Provider, Role
from ctg.control.workflow import EpisodeResult, output_failure
from ctg.envs.base import (
    GENERALIST,
    Benchmark,
    Environment,
    ExecutorProfile,
    InstanceRef,
)
from ctg.envs.textcraft import recipes
from ctg.errors import BackendUnavailableError
from ctg.graphs.model import END, CyclicGraph, DependencyGraph, Subtask
from ctg.graphs.planner import PlannedGraph
from ctg.llm.base import LLM, AssistantTurn, SpendCapExceededError, ToolSpec
from ctg.llm.client import OpenAIClient
from ctg.llm.fake import FakeLLM, FakeRequest, text_turn
from ctg.llm.pricing import SpendLedger
from ctg.runs import runner, splits
from ctg.runs.records import (
    EndReason,
    EpisodeRecord,
    Memory,
    Outcome,
    RouterRecord,
    RunConditions,
    RunManifest,
    SegmentRecord,
    StepRecord,
    ToolCallRecord,
)
from ctg.runs.runner import DryRunReport, RunReport, run_experiment
from ctg.runs.store import RunStore

SUBTASKS = (
    Subtask(id="gather", name="Gather materials", description="d", success_criteria="s", executor=GENERALIST),
    Subtask(id="craft", name="Craft the goal", description="d", success_criteria="s", executor=GENERALIST),
)


def cyclic_graph(name: str) -> CyclicGraph:
    ids = [subtask.id for subtask in SUBTASKS]
    return CyclicGraph(
        name=name,
        description="two-step crafting",
        subtasks=SUBTASKS,
        entry="gather",
        criteria={source: {target: f"{source}->{target}" for target in ids} for source in ids},
    )


def dependency_graph(retry: bool) -> DependencyGraph:
    allowed = {
        "gather": ["craft", *(["gather"] if retry else []), END],
        "craft": [*(["craft"] if retry else []), END],
    }
    return DependencyGraph(
        name="chain",
        description="gather then craft",
        subtasks=SUBTASKS,
        retry=retry,
        criteria={
            source: {target: f"{source}->{target}" for target in targets}
            for source, targets in allowed.items()
        },
    )


class FakeEnv(Environment):
    """`act` changes nothing, `win` reaches the goal. With `backend_down`, every tool call raises."""

    def __init__(self, instance: InstanceRef, backend_down: bool = False) -> None:
        self.instance = instance
        self.backend_down = backend_down
        self._won = False
        self._steps: list[StepRecord] = []
        self.closed = False

    @property
    def objective(self) -> str:
        return f"Craft {self.instance.id}."

    @property
    def won(self) -> bool:
        return self._won

    @property
    def done(self) -> bool:
        return self._won

    @property
    def tool_calls(self) -> int:
        return len(self._steps)

    @property
    def steps(self) -> Sequence[StepRecord]:
        return self._steps

    def initial_state(self) -> dict[str, Any]:
        return {"inventory": []}

    def tool_specs(self, names: Sequence[str]) -> list[ToolSpec]:
        return []

    async def call(
        self, name: str, arguments: Mapping[str, Any], offered: Collection[str] | None = None
    ) -> str:
        if self._won:
            raise RuntimeError("call after done")
        if offered is not None and name not in offered:
            observation = (
                f"Error: tool {name!r} is not available to this executor. "
                f"Available tools: {', '.join(sorted(offered))}."
            )
        else:
            if self.backend_down:
                raise BackendUnavailableError("search API cannot serve this run (HTTP 402): no credits left")
            self._won = name == "win"
            observation = "goal reached" if self._won else "nothing happens"
        self._steps.append(
            StepRecord(
                step=len(self._steps) + 1,
                tool=name,
                arguments=dict(arguments),
                observation=observation,
                state={"won": self._won},
            )
        )
        return observation

    def details(self) -> dict[str, Any]:
        return {"goal": self.instance.id}

    async def close(self) -> None:
        self.closed = True


class FakeBenchmark(Benchmark):
    def __init__(
        self, count: int = 3, name: BenchmarkName = BenchmarkName.TEXTCRAFT, failing: Sequence[str] = ()
    ) -> None:
        self.name = name
        self.count = count
        self.failing = set(failing)
        self.envs: list[FakeEnv] = []
        self.backend_down = False
        """Environments made while this is set raise BackendUnavailableError on every tool call."""
        self.default_budgets = Budgets(global_tool_calls=10, local_tool_calls=5)

    @property
    def description(self) -> str:
        return "A fake crafting benchmark."

    @property
    def roster(self) -> Mapping[str, ExecutorProfile]:
        return {
            GENERALIST: ExecutorProfile(GENERALIST, "every tool", ("act", "win")),
            "crafter": ExecutorProfile("crafter", "crafts", ("win",)),
        }

    def budgets(self, config: ExperimentConfig) -> Budgets:
        return self.default_budgets

    def enumerate_instances(self, variant: str) -> list[InstanceRef]:
        return [InstanceRef(id=f"item_{index:03d}", meta={"goal": index}) for index in range(self.count)]

    async def make_env(
        self, instance: InstanceRef, config: ExperimentConfig, llm: LLM, models: Mapping[Role, ModelSpec]
    ) -> Environment:
        if instance.id in self.failing:
            raise RuntimeError(f"cannot start {instance.id}")
        env = FakeEnv(instance, self.backend_down)
        self.envs.append(env)
        return env


@dataclass(frozen=True)
class PlannerCall:
    kind: str
    objective: str | None
    nshot: tuple[str, ...]
    roster: tuple[str, ...]
    retry: bool | None = None


@dataclass
class FakePlanner:
    """Plans after one planner call. An objective in `failures` (None: the Gen-Cyc graph) raises its
    exception after that call instead."""

    calls: list[PlannerCall] = field(default_factory=list[PlannerCall])
    failures: dict[str | None, Exception] = field(default_factory=dict[str | None, Exception])

    async def plan_cyclic(
        self,
        llm: LLM,
        model: ModelSpec,
        benchmark: Benchmark,
        roster: Sequence[ExecutorProfile],
        objective: str | None,
        nshot: Sequence[str] = (),
    ) -> PlannedGraph:
        await llm.chat(Role.PLANNER, model, [{"role": "user", "content": "plan"}])
        if objective in self.failures:
            raise self.failures[objective]
        names = tuple(profile.name for profile in roster)
        self.calls.append(PlannerCall("cyclic", objective, tuple(nshot), names))
        return PlannedGraph(cyclic_graph(f"graph {len(self.calls)}"), {"call": len(self.calls)})

    async def plan_dependency(
        self,
        llm: LLM,
        model: ModelSpec,
        benchmark: Benchmark,
        roster: Sequence[ExecutorProfile],
        objective: str,
        retry: bool,
        nshot: Sequence[str] = (),
    ) -> PlannedGraph:
        await llm.chat(Role.PLANNER, model, [{"role": "user", "content": "plan"}])
        if objective in self.failures:
            raise self.failures[objective]
        names = tuple(profile.name for profile in roster)
        self.calls.append(PlannerCall("dependency", objective, tuple(nshot), names, retry))
        return PlannedGraph(dependency_graph(retry), {"call": len(self.calls)})


@dataclass(frozen=True)
class ControlCall:
    instance_id: str
    graph: CyclicGraph | DependencyGraph | None
    perturbation: float | None
    draw: float | None


@dataclass
class FakeControl:
    """Acts once, then wins. After the first act, an instance in `crash_after_act` raises its exception and
    one in `unusable_after_act` makes one more executor call whose output stays unusable, which ends the
    episode as an output failure."""

    crash_after_act: dict[str, Exception] = field(default_factory=dict[str, Exception])
    unusable_after_act: set[str] = field(default_factory=set[str])
    calls: list[ControlCall] = field(default_factory=list[ControlCall])

    async def _play(
        self, env: Environment, llm: LLM, models: Mapping[Role, ModelSpec]
    ) -> tuple[list[ToolCallRecord], str | None]:
        assert isinstance(env, FakeEnv)
        executor = models[Role.EXECUTOR]
        await llm.chat(Role.EXECUTOR, executor, [{"role": "user", "content": env.objective}])
        records: list[ToolCallRecord] = []
        for tool in ("act", "win"):
            observation = await env.call(tool, {})
            records.append(
                ToolCallRecord(
                    tool=tool, arguments={}, observation=observation, tool_calls_after=env.tool_calls
                )
            )
            if env.instance.id in self.crash_after_act:
                raise self.crash_after_act[env.instance.id]
            if env.instance.id in self.unusable_after_act:
                await llm.chat(Role.EXECUTOR, executor, [{"role": "user", "content": observation}])
                return records, "LLMOutputError: unusable executor output 3 times: finish_reason=length"
        return records, None

    async def run_graph_episode(
        self,
        graph: CyclicGraph | DependencyGraph,
        env: Environment,
        benchmark: Benchmark,
        llm: LLM,
        models: Mapping[Role, ModelSpec],
        budgets: Budgets,
        perturbation: float,
        rng: random.Random,
    ) -> EpisodeResult:
        assert isinstance(env, FakeEnv)
        self.calls.append(ControlCall(env.instance.id, graph, perturbation, rng.random()))
        records, failure = await self._play(env, llm, models)
        segment = SegmentRecord(
            index=0, node=graph.entry, executor=GENERALIST, tool_calls=records, summary="done"
        )
        if failure is not None:
            return output_failure(env, [segment], [], failure)
        outcome = Outcome(won=env.won, end_reason=EndReason.WON, tool_calls=env.tool_calls, segments=1)
        return EpisodeResult(segments=[segment], react_steps=[], outcome=outcome)

    async def run_react_episode(
        self,
        env: Environment,
        benchmark: Benchmark,
        llm: LLM,
        models: Mapping[Role, ModelSpec],
        budgets: Budgets,
    ) -> EpisodeResult:
        assert isinstance(env, FakeEnv)
        self.calls.append(ControlCall(env.instance.id, None, None, None))
        records, failure = await self._play(env, llm, models)
        if failure is not None:
            return output_failure(env, [], records, failure)
        outcome = Outcome(won=env.won, end_reason=EndReason.WON, tool_calls=env.tool_calls, segments=0)
        return EpisodeResult(segments=[], react_steps=records, outcome=outcome)


def memory(summary: str) -> Memory:
    return Memory(
        segment_analysis=[],
        inconsistencies=[],
        held_resources=[],
        environment_notes=[],
        summary=summary,
        subtask_progress=True,
        failure_analysis=[],
    )


def run_manifest(config: ExperimentConfig) -> RunManifest:
    return RunManifest(
        name=config.name,
        run_id=f"{config.name}/test",
        config=config,
        conditions=RunConditions(
            nshot_sha256=None, models={}, budgets=Budgets(global_tool_calls=10, local_tool_calls=5)
        ),
        ctg_version="0",
        git_commit=None,
        created_at=datetime.now(UTC),
    )


def episode_record(
    config: ExperimentConfig,
    instance_id: str,
    repeat: int = 0,
    *,
    end_reason: EndReason = EndReason.WON,
    segments: Sequence[SegmentRecord] = (),
    react_steps: Sequence[ToolCallRecord] = (),
    graph: CyclicGraph | DependencyGraph | None = None,
) -> EpisodeRecord:
    now = datetime.now(UTC)
    return EpisodeRecord(
        run=config.name,
        benchmark=config.benchmark,
        variant=config.variant,
        instance_id=instance_id,
        instance={},
        objective=f"Craft {instance_id}.",
        repeat=repeat,
        controller=config.controller,
        executors=config.executors,
        perturbation=config.perturbation,
        graph=graph,
        segments=list(segments),
        react_steps=list(react_steps),
        outcome=Outcome(
            won=end_reason is EndReason.WON,
            end_reason=end_reason,
            tool_calls=len(react_steps),
            segments=len(segments),
        ),
        started_at=now,
        finished_at=now,
    )


def routed_segment(
    index: int, node: str, summary: str, memory_summary: str, call: ToolCallRecord
) -> SegmentRecord:
    return SegmentRecord(
        index=index,
        node=node,
        executor=GENERALIST,
        tool_calls=[call],
        summary=summary,
        router=RouterRecord(memory=memory(memory_summary), routing_reasoning=["ok"], choice=node),
    )


def isolate_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `ctg` invocations in a test from loading `.env` and from leaving their progress logging set up."""
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: True)
    progress = logging.getLogger("ctg.runs")
    monkeypatch.setattr(progress, "handlers", [])
    monkeypatch.setattr(progress, "level", progress.level)
    monkeypatch.setattr(progress, "propagate", progress.propagate)


def write_config(directory: Path, config: ExperimentConfig) -> Path:
    path = directory / f"{config.name}.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json", exclude_none=True)), encoding="utf-8")
    return path


def completion_reply(request: httpx.Request) -> httpx.Response:
    """A chat completion answering "ok" with 1000 prompt and 200 completion tokens."""
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
            ],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
        },
    )


class Harness:
    """Runs experiments on a fake benchmark; planner, controllers, client and recipe check are replaced.

    Runs use `llm` (a FakeLLM) unless `ledger_path` is set: then each run gets a real OpenAIClient served by
    a mocked transport, which prices its calls and enforces the caps against that ledger.
    """

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.benchmark = FakeBenchmark(count=3)
        self.planner = FakePlanner()
        self.control = FakeControl()
        self.llm = FakeLLM(self._respond)
        self.splits_dir = tmp_path / "splits"
        self.runs_dir = tmp_path / "runs"
        self.ledger_path: Path | None = None
        self.cap_at: int | None = None
        """1-based number of the FakeLLM call that hits the spend cap."""
        self.clients: list[dict[str, Any]] = []
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setattr(runner, "get_benchmark", lambda name: self.benchmark)
        monkeypatch.setattr(runner, "plan_cyclic", self.planner.plan_cyclic)
        monkeypatch.setattr(runner, "plan_dependency", self.planner.plan_dependency)
        monkeypatch.setattr(runner, "run_graph_episode", self.control.run_graph_episode)
        monkeypatch.setattr(runner, "run_react_episode", self.control.run_react_episode)
        monkeypatch.setattr(runner, "create_llm", self._create_llm)
        monkeypatch.setattr(recipes, "load_recipe_book", lambda: None)
        splits.freeze(self.benchmark, "textcraft_depth4", self.splits_dir)

    def _create_llm(self, run: str | None = None, run_cap_usd: float | None = None) -> FakeLLM | OpenAIClient:
        self.clients.append({"run": run, "run_cap_usd": run_cap_usd})
        if self.ledger_path is None:
            return self.llm
        return OpenAIClient(
            api_keys={Provider.OPENAI: "sk-test"},
            ledger=SpendLedger(self.ledger_path),
            run=run,
            run_cap_usd=run_cap_usd,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(completion_reply)),
        )

    def _respond(self, request: FakeRequest) -> BaseModel | AssistantTurn:
        if self.cap_at is not None and len(self.llm.requests) >= self.cap_at:
            raise SpendCapExceededError("run cap of $1.00 reached")
        return text_turn("ok")

    def config(self, controller: Controller = Controller.SPEC_CYC, **overrides: Any) -> ExperimentConfig:
        values: dict[str, Any] = {
            "name": "exp",
            "benchmark": "textcraft",
            "textcraft_depth": 4,
            "controller": controller,
            "repeats": 2,
            "concurrency": 2,
        }
        return ExperimentConfig.model_validate(values | overrides)

    async def run(self, config: ExperimentConfig, **kwargs: Any) -> RunReport:
        return await run_experiment(config, self.runs_dir, splits_dir=self.splits_dir, **kwargs)

    async def dry_run(self, config: ExperimentConfig, **kwargs: Any) -> DryRunReport:
        return await run_experiment(config, self.runs_dir, dry_run=True, splits_dir=self.splits_dir, **kwargs)

    def store(self, name: str = "exp") -> RunStore:
        return RunStore(self.runs_dir / name)

    def episode(self, repeat: int, instance_id: str) -> EpisodeRecord:
        record = self.store().episode(repeat, instance_id)
        assert record is not None
        return record
