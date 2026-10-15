"""Hand-built run records for the analysis tests."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ctg.analysis.load import EpisodeKey, Run
from ctg.config import (
    BenchmarkName,
    Budgets,
    Controller,
    ExecutorMode,
    ExperimentConfig,
    ModelAssignment,
    Provider,
    Role,
)
from ctg.envs.base import GENERALIST
from ctg.graphs.model import END, CyclicGraph, DependencyGraph, Subtask
from ctg.llm.base import LLMCallRecord
from ctg.runs import splits
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
    TransitionRecord,
    TransitionType,
)
from ctg.runs.store import RunStore

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def split_ids(variant: str, count: int) -> list[str]:
    """The first `count` test-split instance ids of a frozen variant in data/splits/."""
    return [ref.id for ref, split in splits.load(variant) if split == "test"][:count]


_SETTINGS: dict[BenchmarkName, dict[str, Any]] = {
    BenchmarkName.TEXTCRAFT: {"textcraft_depth": 2},
    BenchmarkName.ALFWORLD: {"alfworld_split": "valid_seen"},
    BenchmarkName.FINANCE_AGENT: {},
}


def config(
    name: str = "run",
    controller: Controller = Controller.GEN_CYC,
    benchmark: BenchmarkName = BenchmarkName.TEXTCRAFT,
    models: ModelAssignment | None = None,
    perturbation: float = 0.0,
    instances: Sequence[str] | None = None,
    repeats: int = 3,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        benchmark=benchmark,
        controller=controller,
        models=models or ModelAssignment(),
        perturbation=perturbation,
        instances=tuple(instances) if instances is not None else None,
        repeats=repeats,
        **_SETTINGS[benchmark],
    )


def tool_call(tool: str = "act", observation: str = "ok", k: int = 1, **arguments: Any) -> ToolCallRecord:
    return ToolCallRecord(tool=tool, arguments=arguments, observation=observation, tool_calls_after=k)


def transition(
    source: str,
    target: str,
    type: TransitionType,
    allowed: Sequence[str] | None = None,
    valid: bool = True,
    nominal: str | None = None,
) -> TransitionRecord:
    return TransitionRecord(
        source=source,
        target=target,
        allowed=list(allowed) if allowed is not None else [source, target],
        nominal=nominal or target,
        perturbed=nominal is not None and nominal != target,
        type=type,
        valid=valid,
    )


def segment(
    index: int,
    node: str,
    to: TransitionRecord | None = None,
    progress: bool | None = True,
    calls: Sequence[ToolCallRecord] = (),
) -> SegmentRecord:
    """`progress=None` leaves the segment without a router record (an episode's last segment)."""
    router = None
    if progress is not None:
        memory = Memory(
            segment_analysis=[],
            inconsistencies=[],
            held_resources=[],
            environment_notes=[],
            summary="",
            subtask_progress=progress,
            failure_analysis=[],
        )
        router = RouterRecord(memory=memory, routing_reasoning=[], choice=to.nominal if to else END)
    return SegmentRecord(
        index=index,
        node=node,
        executor=GENERALIST,
        tool_calls=list(calls),
        summary="",
        router=router,
        transition=to,
    )


def path_segments(nodes: Sequence[str], types: Sequence[TransitionType] | None = None) -> list[SegmentRecord]:
    """Segments visiting `nodes` in order, each routed to the next; the last one ends the episode."""
    kinds = list(types) if types is not None else [TransitionType.CROSS] * (len(nodes) - 1)
    segments = [
        segment(i, node, transition(node, nxt, kind))
        for i, (node, nxt, kind) in enumerate(zip(nodes[:-1], nodes[1:], kinds, strict=True))
    ]
    return [*segments, segment(len(nodes) - 1, nodes[-1], progress=None)]


def llm_call(role: Role, input_tokens: int, output_tokens: int, cost: float = 0.0) -> LLMCallRecord:
    return LLMCallRecord(
        role=role,
        model="gpt-4o-mini",
        provider=Provider.OPENAI,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        started_at=T0,
    )


def episode(
    cfg: ExperimentConfig,
    instance_id: str,
    repeat: int = 0,
    won: bool = False,
    tool_calls: int = 1,
    end_reason: EndReason | None = None,
    segments: Sequence[SegmentRecord] = (),
    react_steps: Sequence[ToolCallRecord] = (),
    graph: CyclicGraph | DependencyGraph | None = None,
    llm_calls: Sequence[LLMCallRecord] = (),
    env_details: dict[str, Any] | None = None,
    initial_state: dict[str, Any] | None = None,
    states: Sequence[dict[str, Any]] = (),
) -> EpisodeRecord:
    reason = end_reason or (EndReason.WON if won else EndReason.BUDGET)
    return EpisodeRecord(
        run=cfg.name,
        benchmark=cfg.benchmark,
        variant=cfg.variant,
        instance_id=instance_id,
        instance={},
        objective="objective",
        repeat=repeat,
        controller=cfg.controller,
        executors=ExecutorMode.GENERALIST,
        perturbation=cfg.perturbation,
        graph=graph,
        initial_state=initial_state or {},
        segments=list(segments),
        react_steps=list(react_steps),
        steps=[
            StepRecord(step=i + 1, tool="act", arguments={}, observation="", state=state)
            for i, state in enumerate(states)
        ],
        env_details=env_details or {},
        llm_calls=list(llm_calls),
        outcome=Outcome(won=won, end_reason=reason, tool_calls=tool_calls, segments=len(segments)),
        error="boom" if reason is EndReason.ERROR else None,
        started_at=T0,
        finished_at=T0,
    )


def error_episode(cfg: ExperimentConfig, instance_id: str, repeat: int = 0, **kwargs: Any) -> EpisodeRecord:
    return episode(cfg, instance_id, repeat, end_reason=EndReason.ERROR, **kwargs)


def interrupted_segment(index: int, node: str, calls: Sequence[ToolCallRecord] = ()) -> SegmentRecord:
    """The segment an unusable model output cut short: its tool calls so far, no router, no transition."""
    return segment(index, node, progress=None, calls=calls)


def output_failure(
    cfg: ExperimentConfig,
    instance_id: str,
    repeat: int = 0,
    segments: Sequence[SegmentRecord] = (),
    react_steps: Sequence[ToolCallRecord] = (),
    graph: CyclicGraph | DependencyGraph | None = None,
    env_details: dict[str, Any] | None = None,
) -> EpisodeRecord:
    """An OUTPUT_FAILURE as the runner records it: the trajectory up to the unusable output (a graph
    episode ends on an `interrupted_segment`), with the tool calls counted from that trajectory."""
    calls = len(react_steps) + sum(len(s.tool_calls) for s in segments)
    record = episode(
        cfg,
        instance_id,
        repeat,
        tool_calls=calls,
        end_reason=EndReason.OUTPUT_FAILURE,
        segments=segments,
        react_steps=react_steps,
        graph=graph,
        env_details=env_details,
    )
    error = "LLMOutputError: gpt-4o-mini returned unusable executor output 3 times: finish_reason=length"
    return record.model_copy(update={"error": error})


def planner_failure(
    cfg: ExperimentConfig, instance_id: str, repeat: int = 0, env_details: dict[str, Any] | None = None
) -> EpisodeRecord:
    """An OUTPUT_FAILURE of per-episode planning (Spec-Cyc, DepDAG): no graph, no segments, no tool calls."""
    record = output_failure(cfg, instance_id, repeat, env_details=env_details)
    error = "LLMOutputError: planner graph call failed validation 3 times: unknown subtask 'x'"
    return record.model_copy(update={"error": error})


def manifest(cfg: ExperimentConfig) -> RunManifest:
    return RunManifest(
        name=cfg.name,
        run_id=f"{cfg.name}/test",
        config=cfg,
        conditions=RunConditions(
            nshot_sha256=None, models={}, budgets=Budgets(global_tool_calls=30, local_tool_calls=5)
        ),
        ctg_version="0.1.0",
        git_commit=None,
        created_at=T0,
    )


def make_run(
    cfg: ExperimentConfig,
    episodes: Sequence[EpisodeRecord],
    planning_calls: Sequence[LLMCallRecord] = (),
    expected: Iterable[EpisodeKey] | None = None,
) -> Run:
    """`expected` defaults to the (repeat, instance id) of every given episode: a run scheduled for exactly
    what it stores."""
    return Run(
        path=Path("runs") / cfg.name,
        manifest=manifest(cfg),
        episodes=tuple(episodes),
        expected=frozenset(
            expected if expected is not None else ((e.repeat, e.instance_id) for e in episodes)
        ),
        planning_calls=tuple(planning_calls),
    )


def write_run(root: Path, cfg: ExperimentConfig, episodes: Sequence[EpisodeRecord]) -> Path:
    store = RunStore(root / cfg.name)
    store.write_manifest(manifest(cfg))
    for record in episodes:
        store.write_episode(record)
    return store.root


def subtasks(ids: Sequence[str]) -> tuple[Subtask, ...]:
    return tuple(Subtask(id=i, name=i, description=i, success_criteria=i, executor=GENERALIST) for i in ids)


def cyclic_graph(ids: Sequence[str], words: int = 2) -> CyclicGraph:
    """Every criterion has `words` words."""
    text = " ".join(["word"] * words)
    return CyclicGraph(
        name="g",
        description="g",
        subtasks=subtasks(ids),
        entry=ids[0],
        criteria={source: dict.fromkeys(ids, text) for source in ids},
    )


def dependency_graph(ids: Sequence[str], retry: bool = True) -> DependencyGraph:
    stub = DependencyGraph.model_construct(subtasks=subtasks(ids), retry=retry)
    criteria = {source: dict.fromkeys(stub.allowed(source), "move on now") for source in ids}
    return DependencyGraph(name="d", description="d", subtasks=subtasks(ids), retry=retry, criteria=criteria)
