"""n-shot summaries of successful trajectories (Sec. 4.1.6).

A summary-construction run (Spec-Cyc on the train split) is condensed into one short strategy summary per
won episode. The frozen set is the D the planner receives in the n-shot setting (Eqs. 3-4).
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from ctg.config import BenchmarkName, ModelSpec, Role
from ctg.llm.base import LLM, LLMOutputError, Message
from ctg.prompts import render
from ctg.runs.records import EpisodeRecord, ToolCallRecord
from ctg.runs.store import RunStore, write_json_atomic

OBSERVATION_CHARS = 160
"""Observations are cut to this length in the prompt; the strategy matters, not the raw output."""

CONCURRENCY = 8


class NShotSummary(BaseModel):
    instance_id: str
    text: str


class NShotFile(BaseModel):
    benchmark: BenchmarkName
    variant: str
    source_run: str
    model: str
    created_at: datetime
    summaries: list[NShotSummary]


def load_nshot(path: Path, benchmark: BenchmarkName) -> NShotFile:
    if not path.is_file():
        raise FileNotFoundError(f"n-shot file {path} does not exist; build it with `ctg nshot build`")
    nshot = NShotFile.model_validate_json(path.read_text(encoding="utf-8"))
    if nshot.benchmark is not benchmark:
        raise ValueError(f"n-shot file {path} is for {nshot.benchmark.value}, not {benchmark.value}")
    if not nshot.summaries:
        raise ValueError(f"n-shot file {path} has no summaries")
    return nshot


def _clip(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= OBSERVATION_CHARS else flat[: OBSERVATION_CHARS - 1] + "…"


def _tool_lines(calls: list[ToolCallRecord]) -> str:
    if not calls:
        return "(none)"
    return "\n".join(
        f"{call.tool_calls_after}. {call.tool}({json.dumps(call.arguments, ensure_ascii=False)}) -> "
        f"{_clip(call.observation)}"
        for call in calls
    )


def summary_prompt(record: EpisodeRecord) -> str:
    graph = record.graph
    names = {subtask.id: subtask.name for subtask in graph.subtasks} if graph is not None else {}
    if record.segments:
        labels = [f"{names.get(segment.node, segment.node)} ({segment.node})" for segment in record.segments]
        visited = "\n".join(f"{position}. {label}" for position, label in enumerate(labels, 1))
        segments = "\n".join(
            f"{position}. [{label}] {segment.summary}"
            for position, (label, segment) in enumerate(zip(labels, record.segments, strict=True), 1)
        )
    else:
        visited = "(single agent without subtasks)"
        segments = "(none)"
    routed = [segment.router for segment in record.segments if segment.router is not None]
    memory = routed[-1].memory.summary if routed else "(no router call)"
    calls = [call for segment in record.segments for call in segment.tool_calls] + record.react_steps
    return render(
        "nshot_summary",
        benchmark=record.benchmark.value,
        objective=record.objective,
        visited=visited,
        segments=segments,
        memory=memory,
        tool_calls=_tool_lines(calls),
    )


async def summarize(record: EpisodeRecord, llm: LLM, model: ModelSpec) -> str:
    messages: list[Message] = [{"role": "user", "content": summary_prompt(record)}]
    turn = await llm.chat(Role.NSHOT, model, messages)
    text = (turn.content or "").strip()
    if not text:
        raise LLMOutputError(f"empty n-shot summary for {record.instance_id} (repeat {record.repeat})")
    return text


async def build(run_dir: Path, out_path: Path, model: ModelSpec, llm: LLM) -> NShotFile:
    """Summarize every won episode of a run and write the n-shot file to `out_path`."""
    store = RunStore(run_dir)
    manifest = store.manifest()
    if manifest is None:
        raise FileNotFoundError(f"{run_dir} is not a run directory (no manifest.json)")
    won = [record for record in store.episodes() if record.outcome.won]
    if not won:
        raise ValueError(f"{run_dir} has no won episodes to summarize")

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def one(record: EpisodeRecord) -> str:
        async with semaphore:
            return await summarize(record, llm, model)

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(one(record)) for record in won]
    nshot = NShotFile(
        benchmark=manifest.config.benchmark,
        variant=manifest.config.variant,
        source_run=manifest.name,
        model=model.name,
        created_at=datetime.now(UTC),
        summaries=[
            NShotSummary(instance_id=record.instance_id, text=task.result())
            for record, task in zip(won, tasks, strict=True)
        ],
    )
    write_json_atomic(out_path, nshot.model_dump_json(indent=2))
    return nshot
