"""Four-mode benchmark runner with immutable records and resumable cases."""

from __future__ import annotations

import asyncio
import json
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from luna.actor.graphs import build_case_graph
from luna.actor.model import DecisionModel
from luna.actor.schemas import (
    BenchmarkRunRecord,
    DecisionRequest,
    GraphManifest,
    TaskOutputRecord,
    TestCaseRecord,
    Usage,
)
from luna.benchmarks.registry import get_adapter_class
from luna.config import RunConfig, seed_list_hash
from luna.storage.mongo import NullStore
from luna.storage.redis_bus import NullBus


class BenchmarkRunner:
    def __init__(self, config: RunConfig, model: DecisionModel, store=None, bus=None):
        self.config = config
        self.model = model
        self.store = store or NullStore()
        self.bus = bus or NullBus()

    async def run(
        self,
        seeds: list[int],
        generalized_graph: GraphManifest | None = None,
        resume: str | None = None,
    ) -> BenchmarkRunRecord:
        started_at = datetime.now(timezone.utc)
        resume_run_id = self._resolve_resume(resume)
        completed = self.store.completed_seeds(resume_run_id) if resume_run_id else set()
        pending = [seed for seed in seeds if seed not in completed]
        run_id = f"run.{uuid.uuid4()}"
        if self.config.clear_run_cache:
            self.bus.clear_run_cache(run_id)
        self.bus.publish(
            run_id,
            {"event": "run_started", "requested_cases": len(seeds), "pending_cases": len(pending)},
        )
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def guarded(index_seed: tuple[int, int]):
            index, seed = index_seed
            async with semaphore:
                return await self._execute_case(run_id, seed, index, generalized_graph)

        results = await asyncio.gather(
            *(guarded(pair) for pair in enumerate(pending)), return_exceptions=False
        )
        finished_at = datetime.now(timezone.utc)
        graph_alias = generalized_graph.alias if generalized_graph else None
        graph_hash = generalized_graph.sha256 if generalized_graph else None
        snapshot = self.config.model_dump(mode="json") | {
            "resume_from": resume_run_id,
            "requested_seed_count": len(seeds),
            "executed_seed_count": len(pending),
            "skipped_completed_seed_count": len(completed & set(seeds)),
        }
        record = BenchmarkRunRecord(
            run_id=run_id,
            benchmark=self.config.benchmark,
            split=self.config.split,
            depth=self.config.depth,
            mode=self.config.mode,
            seed_hash=seed_list_hash(seeds),
            seeds=seeds,
            graph_alias=graph_alias,
            graph_hash=graph_hash,
            config_snapshot=snapshot,
            source_commit=self.config.source_commit,
            output_dir=str(self.config.output_dir),
            status="completed",
            started_at=started_at,
            finished_at=finished_at,
        )
        self.store.insert_run(record)
        self.bus.publish(
            run_id,
            {
                "event": "run_completed",
                "won": sum(result.result == "won" for result in results),
                "lost": sum(result.result == "lost" for result in results),
                "crashed": sum(result.result == "crashed" for result in results),
            },
        )
        return record

    def _resolve_resume(self, resume: str | None) -> str | None:
        if resume is None:
            return None
        if resume != "latest":
            return resume
        return self.store.latest_compatible_run(
            self.config.benchmark, self.config.split, self.config.depth, self.config.mode
        )

    async def _execute_case(
        self,
        run_id: str,
        seed: int,
        index: int,
        generalized_graph: GraphManifest | None,
    ) -> TestCaseRecord:
        started_at = datetime.now(timezone.utc)
        case_id = f"case.{uuid.uuid4()}"
        adapter = get_adapter_class(self.config.benchmark)()
        if self.config.benchmark == "finance_agent" and hasattr(self.model, "judge_finance"):
            adapter.set_judge(self.model.judge_finance)
        trajectory: list[dict[str, Any]] = []
        graph = generalized_graph
        result = "lost"
        exception_text = None
        case_dir = self.config.output_dir / run_id / f"case_{index:04d}_seed_{seed}"
        try:
            case_dir.mkdir(parents=True, exist_ok=False)
            await adapter.initialize(seed, index, self.config.depth, case_dir)
            if self.config.mode == "spec-cyc":
                graph = build_case_graph(self.config.benchmark, adapter.task_prompt(), dependency_dag=False)
            elif self.config.mode == "depdag-retry":
                graph = build_case_graph(self.config.benchmark, adapter.task_prompt(), dependency_dag=True)
            await self._execute_steps(run_id, case_id, adapter, graph, trajectory)
            result = "won" if adapter.success else "lost"
        except Exception as exc:  # case failures are persisted and do not abort the batch
            result = "crashed"
            exception_text = f"{type(exc).__name__}: {exc}"
            trajectory.append({"exception": exception_text, "traceback": traceback.format_exc()})
        finished_at = datetime.now(timezone.utc)
        trajectory_path = case_dir / "trajectory.jsonl"
        summary_path = case_dir / "summary.txt"
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(trajectory_path, trajectory)
        _write_json(case_dir / "final_state.json", adapter.state() if result != "crashed" else {})
        (case_dir / "prompt.txt").write_text(
            adapter.task_prompt() if result != "crashed" else "UNAVAILABLE\n", encoding="utf-8"
        )
        summary_path.write_text(
            f"benchmark={self.config.benchmark}\nseed={seed}\nmode={self.config.mode}\n"
            f"result={result}\nsteps={len(trajectory)}\nexception={exception_text or ''}\n",
            encoding="utf-8",
        )
        usage = _trajectory_usage(trajectory)
        evaluation_usage = getattr(adapter, "evaluation_usage", Usage())
        usage = Usage(
            input_tokens=usage.input_tokens + evaluation_usage.input_tokens,
            output_tokens=usage.output_tokens + evaluation_usage.output_tokens,
            total_tokens=usage.total_tokens + evaluation_usage.total_tokens,
        )
        record = TestCaseRecord(
            case_id=case_id,
            run_id=run_id,
            benchmark=self.config.benchmark,
            split=self.config.split,
            depth=self.config.depth,
            seed=seed,
            index=index,
            result=result,
            graph_alias=graph.alias if graph else None,
            graph_hash=graph.sha256 if graph else None,
            num_steps=sum("decision" in item for item in trajectory),
            token_usage=usage,
            exception=exception_text,
            trajectory_path=str(trajectory_path),
            summary_path=str(summary_path),
            started_at=started_at,
            finished_at=finished_at,
        )
        self.store.insert_case(record)
        self.bus.publish(run_id, {"event": "case_completed", "seed": seed, "result": result})
        return record

    async def _execute_steps(
        self,
        run_id: str,
        case_id: str,
        adapter,
        graph: GraphManifest | None,
        trajectory: list[dict[str, Any]],
    ) -> None:
        nodes = graph.subtasks if graph else []
        node_index = 0
        retries: dict[str, int] = {}
        node_calls: dict[str, int] = {}
        for step in range(self.config.tool_limit):
            subtask = nodes[node_index] if nodes else None
            if subtask:
                node_calls[subtask.id] = node_calls.get(subtask.id, 0) + 1
            request = DecisionRequest(
                benchmark=self.config.benchmark,
                mode=self.config.mode,
                task=adapter.task_prompt(),
                state=adapter.state(),
                admissible_actions=adapter.admissible_actions(),
                trajectory=trajectory[-10:],
                subtask=subtask.prompt if subtask else None,
            )
            decision = await self.model.decide(request)
            if decision.done:
                success = await adapter.finalize(decision.final_answer)
                observation = "Terminal answer accepted." if success else "Terminal answer did not pass evaluation."
                terminal = True
            else:
                step_result = await adapter.step(decision.action or "")
                observation, success, terminal = (
                    step_result.observation,
                    step_result.success,
                    step_result.terminal,
                )
            item = {
                "step": step,
                "subtask_id": subtask.id if subtask else None,
                "decision": decision.model_dump(mode="json"),
                "observation": observation,
                "state": adapter.state(),
            }
            trajectory.append(item)
            output = TaskOutputRecord(
                task_output_id=f"task_output.{uuid.uuid4()}",
                run_id=run_id,
                case_id=case_id,
                subtask_id=subtask.id if subtask else None,
                step=step,
                decision=decision.model_dump(mode="json"),
                observation=observation,
                state=adapter.state(),
            )
            self.store.insert_task_output(output)
            self.bus.publish(run_id, {"event": "case_step", "case_id": case_id, "step": step})
            if terminal or adapter.success:
                return
            if not nodes:
                continue
            if graph and graph.graph_type == "dependency_dag":
                failed = _observation_failed(observation)
                if failed:
                    retries[subtask.id] = retries.get(subtask.id, 0) + 1
                    if (
                        retries[subtask.id] <= min(subtask.retry_limit, self.config.retry_limit)
                        and node_calls[subtask.id] < self.config.subtask_tool_limit
                    ):
                        continue
                node_index = min(node_index + 1, len(nodes) - 1)
            else:
                node_index = (node_index + 1) % len(nodes)


def _observation_failed(observation: str) -> bool:
    lowered = observation.lower()
    return lowered.startswith("error:") or "could not" in lowered or "not available" in lowered


def _trajectory_usage(trajectory: list[dict[str, Any]]) -> Usage:
    input_tokens = output_tokens = total_tokens = 0
    for item in trajectory:
        usage = item.get("decision", {}).get("usage", {})
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))
        total_tokens += int(usage.get("total_tokens", 0))
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True, default=str) + "\n" for value in values),
        encoding="utf-8",
    )
