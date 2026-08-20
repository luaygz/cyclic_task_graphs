from pathlib import Path

import pytest

from luna.actor.graphs import build_case_graph
from luna.actor.model import ScriptedModel
from luna.actor.schemas import Decision
from luna.benchmarks.base import BenchmarkAdapter, StepResult
from luna.benchmarks.runner import BenchmarkRunner
from luna.config import ModelSettings, RunConfig
from luna.storage.mongo import NullStore
from luna.storage.redis_bus import NullBus


class FakeAdapter(BenchmarkAdapter):
    benchmark_name = "fake"

    async def initialize(self, seed, index, depth, output_dir=None):
        self.steps = 0
        self._success = False

    def task_prompt(self):
        return "Complete the deterministic mock task."

    def state(self):
        return {"steps": self.steps, "success": self._success}

    def admissible_actions(self):
        return ["advance"]

    async def step(self, action):
        assert action == "advance"
        self.steps += 1
        self._success = self.steps >= 2
        return StepResult(observation="advanced", success=self._success, terminal=self._success)

    async def finalize(self, answer):
        return self._success

    @property
    def success(self):
        return self._success


async def decision(_request):
    return Decision(reasoning=["mock"], action="advance")


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["alfworld", "textcraft", "finance_agent"])
@pytest.mark.parametrize("mode", ["react", "spec-cyc", "gen-cyc", "depdag-retry"])
async def test_mock_end_to_end_all_benchmarks_and_modes(monkeypatch, tmp_path, benchmark, mode):
    monkeypatch.setattr("luna.benchmarks.runner.get_adapter_class", lambda _name: FakeAdapter)
    graph_path = tmp_path / "graph.json" if mode == "gen-cyc" else None
    config = RunConfig(
        benchmark=benchmark,
        mode=mode,
        split="test" if benchmark != "alfworld" else "eval_out_of_distribution",
        depth=2 if benchmark == "textcraft" else None,
        graph_manifest=graph_path,
        output_dir=tmp_path / benchmark / mode,
        tool_limit=4,
        storage_enabled=False,
        model=ModelSettings(provider="fake"),
    )
    graph = build_case_graph(benchmark, "mock task", dependency_dag=False) if mode == "gen-cyc" else None
    store, bus = NullStore(), NullBus()
    record = await BenchmarkRunner(config, ScriptedModel(decision), store, bus).run([0], graph)
    assert record.status == "completed"
    assert len(store.runs) == 1
    assert len(store.cases) == 1
    assert store.cases[0].result == "won"
    assert len(store.outputs) == 2
    assert any(event["event"] == "run_completed" for event in bus.events)
    assert Path(store.cases[0].trajectory_path).is_file()


@pytest.mark.asyncio
async def test_resume_skips_completed_seeds(monkeypatch, tmp_path):
    monkeypatch.setattr("luna.benchmarks.runner.get_adapter_class", lambda _name: FakeAdapter)

    class ResumeStore(NullStore):
        def latest_compatible_run(self, benchmark, split, depth, mode):
            return "run.previous"

        def completed_seeds(self, run_id):
            assert run_id == "run.previous"
            return {0}

    config = RunConfig(
        benchmark="textcraft", mode="react", split="test", depth=2,
        output_dir=tmp_path / "resume", tool_limit=4, storage_enabled=False,
        model=ModelSettings(provider="fake"),
    )
    store = ResumeStore()
    record = await BenchmarkRunner(config, ScriptedModel(decision), store, NullBus()).run(
        [0, 1], resume="latest"
    )
    assert [case.seed for case in store.cases] == [1]
    assert record.config_snapshot["resume_from"] == "run.previous"
    assert record.config_snapshot["skipped_completed_seed_count"] == 1


@pytest.mark.asyncio
async def test_case_failure_is_persisted_without_aborting_run(monkeypatch, tmp_path):
    class FailingAdapter(FakeAdapter):
        async def initialize(self, seed, index, depth, output_dir=None):
            raise RuntimeError("controlled failure")

    monkeypatch.setattr("luna.benchmarks.runner.get_adapter_class", lambda _name: FailingAdapter)
    config = RunConfig(
        benchmark="textcraft", mode="react", split="test", depth=2,
        output_dir=tmp_path / "failure", tool_limit=2, storage_enabled=False,
        model=ModelSettings(provider="fake"),
    )
    store = NullStore()
    record = await BenchmarkRunner(config, ScriptedModel(decision), store, NullBus()).run([0])
    assert record.status == "completed"
    assert store.cases[0].result == "crashed"
    assert store.cases[0].exception == "RuntimeError: controlled failure"
    assert Path(store.cases[0].summary_path).is_file()
