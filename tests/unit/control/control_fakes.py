"""A small environment, benchmark, graphs and scripted-LLM helpers for the controller tests."""

from __future__ import annotations

import random
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from ctg.config import BenchmarkName, Budgets, ExperimentConfig, ModelSpec, Role
from ctg.control.policies import Graph
from ctg.control.workflow import EpisodeResult, run_graph_episode
from ctg.envs.base import GENERALIST, Benchmark, Environment, ExecutorProfile, InstanceRef
from ctg.graphs.model import END, CyclicGraph, DependencyGraph, Subtask
from ctg.graphs.schemas import model_id
from ctg.llm.base import LLM, AssistantTurn, ToolSpec
from ctg.llm.fake import FakeLLM, FakeRequest
from ctg.runs.records import StepRecord

OBJECTIVE = "OBJECTIVE-TEXT: reach the goal."
DESCRIPTION = "BENCHMARK-DESCRIPTION: a toy benchmark."
MODELS = {role: ModelSpec(name=f"fake-{role.value}") for role in Role}
ALL_TOOLS = ("act", "win", "submit")


class FakeEnv(Environment):
    """`act` records a value, `win` reaches the goal, `submit` ends the episode without winning."""

    def __init__(self, available: Sequence[str] = ALL_TOOLS) -> None:
        self.available = set(available)
        self._won = False
        self._done = False
        self._steps: list[StepRecord] = []

    @property
    def objective(self) -> str:
        return OBJECTIVE

    @property
    def won(self) -> bool:
        return self._won

    @property
    def done(self) -> bool:
        return self._done

    @property
    def tool_calls(self) -> int:
        return len(self._steps)

    @property
    def steps(self) -> Sequence[StepRecord]:
        return self._steps

    def initial_state(self) -> dict[str, Any]:
        return {"steps": 0}

    def tool_specs(self, names: Sequence[str]) -> list[ToolSpec]:
        return [_spec(name) for name in names if name in self.available]

    async def call(
        self, name: str, arguments: Mapping[str, Any], offered: Collection[str] | None = None
    ) -> str:
        if self._done:
            raise RuntimeError("call after done")
        if offered is not None and name not in offered:
            observation = (
                f"Error: tool {name!r} is not available to this executor. "
                f"Available tools: {', '.join(sorted(offered))}."
            )
        elif name == "win":
            self._won = self._done = True
            observation = "Goal reached."
        elif name == "submit":
            self._done = True
            observation = "Answer submitted."
        else:
            observation = f"Observed {arguments.get('value')}."
        step = len(self._steps) + 1
        self._steps.append(
            StepRecord(
                step=step,
                tool=name,
                arguments=dict(arguments),
                observation=observation,
                state={"steps": step},
            )
        )
        return observation


def _spec(name: str) -> ToolSpec:
    properties = {"value": {"type": "string"}} if name == "act" else {}
    return ToolSpec(
        name=name,
        description=f"The {name} tool.",
        parameters={
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    )


CAPPED = ExecutorProfile(name="capped", description="Acts only.", tools=("act",), max_tool_calls=2)


class FakeBenchmark(Benchmark):
    name = BenchmarkName.TEXTCRAFT

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def roster(self) -> Mapping[str, ExecutorProfile]:
        return {
            GENERALIST: ExecutorProfile(name=GENERALIST, description="Every tool.", tools=ALL_TOOLS),
            CAPPED.name: CAPPED,
        }

    def budgets(self, config: ExperimentConfig) -> Budgets:
        return Budgets(global_tool_calls=20, local_tool_calls=3)

    def enumerate_instances(self, variant: str) -> list[InstanceRef]:
        return [InstanceRef(id="only")]

    async def make_env(
        self, instance: InstanceRef, config: ExperimentConfig, llm: LLM, models: Mapping[Role, ModelSpec]
    ) -> Environment:
        return FakeEnv()


BENCHMARK = FakeBenchmark()


def subtask(subtask_id: str, executor: str = GENERALIST) -> Subtask:
    return Subtask(
        id=subtask_id,
        name=f"{subtask_id}-name",
        description=f"{subtask_id}-description",
        success_criteria=f"{subtask_id}-criteria",
        executor=executor,
    )


def cyclic_graph(
    ids: Sequence[str] = ("a", "b", "c"), entry: str = "a", executor: str = GENERALIST
) -> CyclicGraph:
    return CyclicGraph(
        name="cyclic",
        description="A complete graph.",
        subtasks=tuple(subtask(i, executor) for i in ids),
        entry=entry,
        criteria={src: {dst: f"criterion {src}->{dst}" for dst in ids} for src in ids},
    )


def dependency_graph(ids: Sequence[str] = ("a", "b", "c"), retry: bool = True) -> DependencyGraph:
    criteria: dict[str, dict[str, str]] = {}
    for position, src in enumerate(ids):
        allowed = [*ids[position + 1 : position + 2], *([src] if retry else []), END]
        criteria[src] = {dst: f"criterion {src}->{dst}" for dst in allowed}
    return DependencyGraph(
        name="chain",
        description="A dependency chain.",
        subtasks=tuple(subtask(i) for i in ids),
        retry=retry,
        criteria=criteria,
    )


def router_reply(request: FakeRequest, choice: str, summary: str = "rolling summary") -> BaseModel:
    assert request.schema is not None
    return request.schema.model_validate(
        {
            "segment_analysis": ["analysis"],
            "inconsistencies": [],
            "held_resources": [],
            "environment_notes": [],
            "summary": summary,
            "subtask_progress": True,
            "failure_analysis": [],
            "routing_reasoning": ["reasoning"],
            "next": model_id(choice),
        }
    )


def next_enum(request: FakeRequest) -> list[str]:
    assert request.schema is not None
    return request.schema.model_json_schema()["properties"]["next"]["enum"]


def tool_results(request: FakeRequest) -> int:
    """Tool messages so far in this conversation: the executor's position within its segment."""
    return sum(1 for message in request.messages if message["role"] == "tool")


def system_text(request: FakeRequest) -> str:
    content = request.messages[0].get("content")
    assert isinstance(content, str)
    return content


def scripted(
    executor: Callable[[FakeRequest], AssistantTurn], router: Callable[[FakeRequest], str] = lambda _: "a"
) -> FakeLLM:
    """Chat requests go to `executor`; structured (router) requests choose `router(request)`."""

    def handler(request: FakeRequest) -> BaseModel | AssistantTurn:
        if request.kind == "structured":
            return router_reply(request, router(request))
        return executor(request)

    return FakeLLM(handler)


def requests_for(llm: FakeLLM, role: Role) -> list[FakeRequest]:
    return [request for request in llm.requests if request.role is role]


async def run_graph(
    graph: Graph,
    llm: FakeLLM,
    env: FakeEnv | None = None,
    budgets: Budgets | None = None,
    perturbation: float = 0.0,
    rng: random.Random | None = None,
) -> EpisodeResult:
    return await run_graph_episode(
        graph,
        env or FakeEnv(),
        BENCHMARK,
        llm,
        MODELS,
        budgets or Budgets(global_tool_calls=20, local_tool_calls=3),
        perturbation,
        rng or random.Random(0),
    )
