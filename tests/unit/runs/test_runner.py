"""Runner behavior: records, resume, errors, limits, Gen-Cyc reuse, run stops, setup checks and dry runs."""

from __future__ import annotations

import importlib.util
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from runs_fakes import FakeBenchmark, Harness

from ctg.config import (
    ROLE_SETTINGS,
    BenchmarkName,
    Budgets,
    Controller,
    ExperimentConfig,
    ModelAssignment,
    Provider,
    Role,
    RoleSettings,
    load_model_presets,
)
from ctg.errors import BackendUnavailableError
from ctg.graphs.model import DependencyGraph
from ctg.llm.base import LLMCallRecord, LLMOutputError
from ctg.llm.pricing import SpendLedger
from ctg.runs import runner, splits
from ctg.runs.nshot import NShotFile, NShotSummary
from ctg.runs.records import SCHEMA_VERSION, EndReason, PreviousAttempt, RunManifest
from ctg.runs.runner import RunSetupError, episode_rng

ALL_KEYS = [(repeat, f"item_{index:03d}") for repeat in range(2) for index in range(3)]


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    return Harness(tmp_path, monkeypatch)


def write_nshot(path: Path, texts: list[str]) -> Path:
    nshot = NShotFile(
        benchmark=BenchmarkName.TEXTCRAFT,
        variant="textcraft_depth3",
        source_run="train",
        model="gpt-4o-mini",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        summaries=[NShotSummary(instance_id=f"i{index}", text=text) for index, text in enumerate(texts)],
    )
    path.write_text(nshot.model_dump_json(), encoding="utf-8")
    return path


def manifest_of(harness: Harness, name: str = "exp") -> RunManifest:
    manifest = harness.store(name).manifest()
    assert manifest is not None
    return manifest


async def test_records_every_episode_and_resume_skips_completed(harness: Harness) -> None:
    config = harness.config()

    report = await harness.run(config, max_cost_usd=2.5)

    assert (report.total, report.scheduled, report.finished, report.won, report.errors) == (6, 6, 6, 6, 0)
    store = harness.store()
    assert sorted(store.completed()) == ALL_KEYS
    manifest = manifest_of(harness)
    assert manifest.config == config
    assert harness.clients == [{"run": manifest.run_id, "run_cap_usd": 2.5}]
    record = harness.episode(1, "item_002")
    assert record.objective == "Craft item_002."
    assert record.instance == {"goal": 2}
    assert record.initial_state == {"inventory": []}
    assert [step.tool for step in record.steps] == ["act", "win"]
    assert record.env_details == {"goal": "item_002"}
    assert record.graph is not None and record.planner_notes
    assert record.previous_attempts == []
    # Each episode's sink holds exactly its own calls, even with concurrent episodes.
    assert all([call.role for call in r.llm_calls] == [Role.PLANNER, Role.EXECUTOR] for r in store.episodes())
    assert all(env.closed for env in harness.benchmark.envs)
    assert not list(store.root.rglob("*.tmp"))

    requests = len(harness.llm.requests)
    again = await harness.run(config)

    assert (again.completed_before, again.scheduled, again.finished) == (6, 0, 0)
    assert len(harness.llm.requests) == requests
    assert len(harness.benchmark.envs) == 6
    assert harness.clients[1]["run"] == manifest.run_id


async def test_exception_is_recorded_as_error_with_traceback_and_partial_trajectory(harness: Harness) -> None:
    harness.control.crash_after_act["item_001"] = ValueError("controller crashed")

    report = await harness.run(harness.config(repeats=1))

    assert (report.finished, report.won, report.output_failures, report.errors) == (3, 2, 0, 1)
    record = harness.episode(0, "item_001")
    assert record.outcome.end_reason is EndReason.ERROR
    assert not record.outcome.won
    assert record.outcome.tool_calls == 1
    assert record.error is not None and "ValueError: controller crashed" in record.error
    assert [step.tool for step in record.steps] == ["act"]
    assert record.graph is not None
    assert all(env.closed for env in harness.benchmark.envs)


@pytest.mark.parametrize("controller", [Controller.SPEC_CYC, Controller.REACT])
async def test_unusable_model_output_is_a_loss_that_keeps_its_trajectory_and_is_not_rerun(
    harness: Harness, controller: Controller
) -> None:
    harness.control.unusable_after_act.add("item_001")
    config = harness.config(controller, repeats=1)

    report = await harness.run(config)

    assert (report.finished, report.won, report.output_failures, report.errors) == (3, 2, 1, 0)
    assert report.stop_reason is None
    record = harness.episode(0, "item_001")
    recorded_calls = [call for segment in record.segments for call in segment.tool_calls] + record.react_steps
    assert record.outcome.end_reason is EndReason.OUTPUT_FAILURE and not record.outcome.won
    assert record.outcome.tool_calls == 1 == len(recorded_calls)
    assert [call.tool for call in recorded_calls] == [step.tool for step in record.steps] == ["act"]
    assert record.error == "LLMOutputError: unusable executor output 3 times: finish_reason=length"
    assert record.env_details == {"goal": "item_001"}
    assert [call.role for call in record.llm_calls][-2:] == [Role.EXECUTOR, Role.EXECUTOR]
    assert harness.store().completed() == {(0, "item_000"), (0, "item_001"), (0, "item_002")}

    requests = len(harness.llm.requests)
    resumed = await harness.run(config)

    assert (resumed.completed_before, resumed.scheduled) == (3, 0)
    assert len(harness.llm.requests) == requests


@pytest.mark.parametrize("controller", [Controller.SPEC_CYC, Controller.DEPDAG])
async def test_a_plan_that_keeps_failing_is_a_loss_of_that_episode(
    harness: Harness, controller: Controller
) -> None:
    harness.planner.failures["Craft item_002."] = LLMOutputError("plan failed validation 3 times")

    report = await harness.run(harness.config(controller, repeats=1))

    assert (report.finished, report.won, report.output_failures, report.errors) == (3, 2, 1, 0)
    assert report.stop_reason is None
    record = harness.episode(0, "item_002")
    assert record.outcome.end_reason is EndReason.OUTPUT_FAILURE and not record.outcome.won
    assert record.graph is None and record.steps == []
    assert [call.role for call in record.llm_calls] == [Role.PLANNER]
    assert (0, "item_002") in harness.store().completed()


async def test_errored_episodes_are_rerun_on_resume(harness: Harness) -> None:
    harness.benchmark.failing = {"item_002"}
    config = harness.config(repeats=1)

    first = await harness.run(config)

    assert (first.finished, first.errors) == (3, 1)
    failed = harness.episode(0, "item_002")
    assert failed.error is not None and "cannot start item_002" in failed.error
    assert harness.store().completed() == {(0, "item_000"), (0, "item_001")}

    harness.benchmark.failing = set()
    second = await harness.run(config)

    assert (second.completed_before, second.scheduled, second.won, second.errors) == (2, 1, 1, 0)
    assert harness.episode(0, "item_002").outcome.won


async def test_a_rerun_keeps_every_earlier_errored_attempt(harness: Harness) -> None:
    harness.control.crash_after_act["item_001"] = ValueError("controller crashed")
    config = harness.config(repeats=1)
    await harness.run(config)
    first = harness.episode(0, "item_001")
    await harness.run(config)
    second = harness.episode(0, "item_001")
    del harness.control.crash_after_act["item_001"]

    await harness.run(config)

    final = harness.episode(0, "item_001")
    assert final.outcome.won and final.error is None
    assert first.error is not None and second.error is not None
    first_attempt = PreviousAttempt(
        error=first.error,
        llm_calls=first.llm_calls,
        started_at=first.started_at,
        finished_at=first.finished_at,
    )
    second_attempt = PreviousAttempt(
        error=second.error,
        llm_calls=second.llm_calls,
        started_at=second.started_at,
        finished_at=second.finished_at,
    )
    assert second.previous_attempts == [first_attempt]
    assert final.previous_attempts == [first_attempt, second_attempt]
    assert harness.episode(0, "item_000").previous_attempts == []
    # The records hold every model call made, so spend computed from them is complete.
    stored_calls = sum(
        len(record.llm_calls) + sum(len(attempt.llm_calls) for attempt in record.previous_attempts)
        for record in harness.store().episodes()
    )
    assert stored_calls == len(harness.llm.requests) == 3 * 2 + 2 * 2


async def test_limit_runs_the_first_pending_episodes_repeat_major(harness: Harness) -> None:
    config = harness.config()

    first = await harness.run(config, limit=4)

    assert first.scheduled == 4
    assert sorted(harness.store().completed()) == ALL_KEYS[:4]

    second = await harness.run(config, limit=1)

    assert second.scheduled == 1
    assert sorted(harness.store().completed()) == ALL_KEYS[:5]
    with pytest.raises(ValueError, match="limit"):
        await harness.run(config, limit=0)


async def test_gen_cyc_graph_is_planned_once_per_run_and_reused_on_resume(harness: Harness) -> None:
    config = harness.config(Controller.GEN_CYC)

    await harness.run(config, limit=2)
    await harness.run(config)

    assert [call.objective for call in harness.planner.calls] == [None]
    store = harness.store()
    stored = store.gen_cyc_graph()
    assert stored is not None
    assert [call.role for call in stored.llm_calls] == [Role.PLANNER]
    records = list(store.episodes())
    assert len(records) == 6
    assert all(record.graph == stored.graph for record in records)
    assert all(Role.PLANNER not in {call.role for call in record.llm_calls} for record in records)


async def test_spec_cyc_plans_every_episode_from_its_objective_with_nshot(
    harness: Harness, tmp_path: Path
) -> None:
    nshot_path = write_nshot(tmp_path / "nshot.json", ["First.", "Second."])
    config = harness.config(nshot=nshot_path, executors="specialist", repeats=1)

    await harness.run(config)

    calls = harness.planner.calls
    assert sorted(call.objective or "" for call in calls) == [f"Craft item_00{i}." for i in range(3)]
    assert all(call.nshot == ("First.", "Second.") for call in calls)
    assert all(call.roster == ("crafter",) for call in calls)


@pytest.mark.parametrize(
    ("controller", "retry"), [(Controller.DEPDAG, True), (Controller.DEPDAG_NORETRY, False)]
)
async def test_dependency_controllers_plan_with_their_retry_flag(
    harness: Harness, controller: Controller, retry: bool
) -> None:
    await harness.run(harness.config(controller, repeats=1))

    assert [call.retry for call in harness.planner.calls] == [retry] * 3
    graphs = [record.graph for record in harness.store().episodes()]
    assert all(isinstance(graph, DependencyGraph) and graph.retry is retry for graph in graphs)


async def test_react_runs_without_planning(harness: Harness) -> None:
    await harness.run(harness.config(Controller.REACT, repeats=1))

    assert harness.planner.calls == []
    records = list(harness.store().episodes())
    assert all(r.graph is None and r.segments == [] and len(r.react_steps) == 2 for r in records)


async def test_episode_rng_depends_only_on_seed_repeat_and_instance(harness: Harness) -> None:
    await harness.run(harness.config(perturbation=0.5, seed=7, concurrency=3))

    expected = {episode_rng(7, repeat, instance).random() for repeat, instance in ALL_KEYS}
    assert len(expected) == 6
    assert {call.draw for call in harness.control.calls} == expected
    assert {call.perturbation for call in harness.control.calls} == {0.5}


async def test_spend_cap_stops_starting_new_episodes(harness: Harness) -> None:
    harness.cap_at = 3
    config = harness.config(Controller.REACT, concurrency=1)

    report = await harness.run(config, max_cost_usd=1.0)

    assert (report.finished, report.won, report.errors, report.not_started) == (3, 2, 1, 3)
    assert report.stop_reason is not None and "run cap of $1.00 reached" in report.stop_reason
    assert report.halts_sweep
    assert len(harness.benchmark.envs) == 3
    capped = [r for r in harness.store().episodes() if r.outcome.end_reason is EndReason.ERROR]
    assert len(capped) == 1 and capped[0].error is not None and "SpendCapExceededError" in capped[0].error

    harness.cap_at = None
    resumed = await harness.run(config)

    assert (resumed.completed_before, resumed.scheduled, resumed.won) == (2, 4, 4)


async def test_the_spend_cap_covers_this_run_directory_across_invocations_and_nothing_else(
    harness: Harness, tmp_path: Path
) -> None:
    harness.ledger_path = tmp_path / "spend.jsonl"
    ledger = SpendLedger(harness.ledger_path)
    # $10 already spent by the same experiment name: under the name itself and in another run directory.
    for run in ("exp", "exp/0123456789abcdef"):
        seed = LLMCallRecord(
            role=Role.EXECUTOR,
            model="gpt-4o-mini",
            provider=Provider.OPENAI,
            cost_usd=5.0,
            started_at=datetime.now(UTC),
        )
        ledger.append(seed, run)
    config = harness.config(concurrency=1)

    first = await harness.run(config, limit=2, max_cost_usd=1.0)

    assert (first.won, first.stop_reason) == (2, None)
    run_id = manifest_of(harness).run_id
    spent = SpendLedger(harness.ledger_path).total(run=run_id)
    assert spent > 0 and spent == pytest.approx(first.spent_usd)

    capped = await harness.run(config, max_cost_usd=spent)

    assert (capped.finished, capped.won, capped.errors, capped.not_started) == (1, 0, 1, 3)
    assert capped.stop_reason is not None and "spend cap reached" in capped.stop_reason

    shutil.rmtree(harness.store().root)
    fresh = await harness.run(config, max_cost_usd=1.0)

    assert (fresh.won, fresh.stop_reason) == (6, None)
    assert manifest_of(harness).run_id != run_id


async def test_spend_cap_while_planning_gen_cyc_starts_no_episode(harness: Harness) -> None:
    harness.cap_at = 1

    report = await harness.run(harness.config(Controller.GEN_CYC))

    assert (report.finished, report.not_started) == (0, 6)
    assert report.stop_reason is not None and "Gen-Cyc" in report.stop_reason
    assert report.halts_sweep
    assert harness.store().gen_cyc_graph() is None
    assert harness.benchmark.envs == []


async def test_a_failed_gen_cyc_planning_stops_only_this_run_and_its_cost_is_reported(
    harness: Harness, tmp_path: Path
) -> None:
    harness.ledger_path = tmp_path / "spend.jsonl"
    harness.planner.failures[None] = LLMOutputError("planner output failed validation 3 times")
    config = harness.config(Controller.GEN_CYC)

    report = await harness.run(config)

    assert report.stop_reason == (
        "planning the Gen-Cyc graph failed: LLMOutputError: planner output failed validation 3 times"
    )
    assert not report.halts_sweep
    assert (report.finished, report.not_started) == (0, 6)
    spent = SpendLedger(harness.ledger_path).total(run=manifest_of(harness).run_id)
    assert spent > 0 and report.spent_usd == pytest.approx(spent)
    assert harness.store().gen_cyc_graph() is None
    assert harness.benchmark.envs == []

    del harness.planner.failures[None]
    resumed = await harness.run(config)

    assert (resumed.scheduled, resumed.won, resumed.stop_reason) == (6, 6, None)


async def test_an_unavailable_backend_while_planning_gen_cyc_stops_the_run(harness: Harness) -> None:
    harness.planner.failures[None] = BackendUnavailableError("OpenAI rejected the API key (HTTP 401)")

    report = await harness.run(harness.config(Controller.GEN_CYC))

    assert report.stop_reason == (
        "backend unavailable while planning the Gen-Cyc graph: OpenAI rejected the API key (HTTP 401)"
    )
    assert report.halts_sweep
    assert (report.finished, report.not_started) == (0, 6)
    assert harness.benchmark.envs == []


async def test_an_unavailable_backend_stops_starting_new_episodes(harness: Harness) -> None:
    harness.benchmark.backend_down = True
    config = harness.config(Controller.REACT, concurrency=1)

    report = await harness.run(config)

    assert (report.finished, report.won, report.errors, report.not_started) == (1, 0, 1, 5)
    assert report.stop_reason is not None
    assert report.stop_reason.startswith("backend unavailable during r0/")
    assert "search API cannot serve this run (HTTP 402): no credits left" in report.stop_reason
    assert report.halts_sweep
    assert len(harness.benchmark.envs) == 1 and harness.benchmark.envs[0].closed
    [failed] = harness.store().episodes()
    assert failed.outcome.end_reason is EndReason.ERROR and failed.steps == []
    assert failed.error is not None and "BackendUnavailableError" in failed.error

    harness.benchmark.backend_down = False
    resumed = await harness.run(config)

    assert (resumed.completed_before, resumed.scheduled, resumed.won) == (0, 6, 6)
    assert resumed.stop_reason is None


async def test_a_second_process_on_the_same_run_directory_is_refused(harness: Harness) -> None:
    config = harness.config()

    with harness.store().lock(), pytest.raises(RunSetupError, match="in use by another `ctg run` process"):
        await harness.run(config)

    assert harness.clients == [] and harness.llm.requests == []
    assert harness.store().manifest() is None
    report = await harness.run(config)
    assert report.won == 6


async def test_dry_run_makes_no_calls_and_counts_pending_episodes(harness: Harness) -> None:
    config = harness.config()
    await harness.run(config, limit=2)
    requests, envs, clients = len(harness.llm.requests), len(harness.benchmark.envs), len(harness.clients)

    report = await harness.dry_run(config, limit=3)
    full = await harness.dry_run(config)

    assert (report.total, report.completed, report.scheduled) == (6, 2, 3)
    assert report.problems == ()
    assert report.estimate is not None and report.estimate.episodes == 3 and report.estimate.usd > 0
    assert full.estimate is not None and full.estimate.usd == pytest.approx(report.estimate.usd * 4 / 3)
    assert (len(harness.llm.requests), len(harness.benchmark.envs), len(harness.clients)) == (
        requests,
        envs,
        clients,
    )


async def test_dry_run_of_a_new_experiment_writes_nothing(harness: Harness) -> None:
    report = await harness.dry_run(harness.config(Controller.GEN_CYC, name="fresh"))

    assert (report.total, report.completed, report.scheduled) == (6, 0, 6)
    assert any("planned once" in note for note in report.notes)
    assert not (harness.runs_dir / "fresh").exists()


async def test_dry_run_reports_every_problem_and_a_real_run_refuses_to_start(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = harness.config(
        textcraft_depth=3,
        nshot=tmp_path / "missing.json",
        models=ModelAssignment(executor="gemini-2.5-pro"),
    )

    report = await harness.dry_run(config)

    assert len(report.problems) == 3
    assert any("ctg splits freeze textcraft_depth3" in problem for problem in report.problems)
    assert any("missing.json" in problem for problem in report.problems)
    assert any("OPENROUTER_API_KEY" in problem for problem in report.problems)
    with pytest.raises(RunSetupError, match="OPENROUTER_API_KEY"):
        await harness.run(config)
    assert harness.clients == []
    assert not (harness.runs_dir / "exp").exists()


async def test_a_blank_api_key_counts_as_missing(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", " \t")

    report = await harness.dry_run(harness.config())

    assert report.problems == ("missing environment variable OPENAI_API_KEY",)


async def test_alfworld_setup_requires_the_alfworld_extra(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    absent: set[str] = {"textworld", "fast_downward"}
    find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None) -> Any:
        if name in runner.ALFWORLD_MODULES:
            return None if name in absent else object()
        return find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    config = ExperimentConfig(
        name="alf",
        benchmark=BenchmarkName.ALFWORLD,
        alfworld_split="valid_unseen",
        controller=Controller.REACT,
    )

    missing = await harness.dry_run(config)
    absent.clear()
    installed = await harness.dry_run(config)

    assert (
        "ALFWorld needs the alfworld extra (textworld, fast_downward not installed); "
        "run `uv sync --extra alfworld`"
    ) in missing.problems
    assert not any("alfworld extra" in problem for problem in installed.problems)


async def test_dry_run_reports_unknown_presets(harness: Harness) -> None:
    report = await harness.dry_run(harness.config(models=ModelAssignment(router="no-such-model")))

    assert len(report.problems) == 1 and "no-such-model" in report.problems[0]
    assert report.estimate is None


async def test_dry_run_estimate_leaves_out_unpriced_models(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    report = await harness.dry_run(harness.config(models=ModelAssignment(executor="gemini-2.5-pro")))

    assert report.problems == ()
    assert report.estimate is not None
    assert report.estimate.unpriced == ("executor (google/gemini-2.5-pro)",)
    assert report.estimate.usd > 0


async def test_dry_run_checks_finance_keys_for_the_chosen_search_backend(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("SEC_API_KEY", "SEC_USER_AGENT", "SERPAPI_API_KEY", "EXA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    harness.benchmark = FakeBenchmark(count=20, name=BenchmarkName.FINANCE_AGENT)
    splits.freeze(harness.benchmark, "finance_agent", harness.splits_dir)
    config = ExperimentConfig(
        name="fin", benchmark=BenchmarkName.FINANCE_AGENT, controller=Controller.REACT, web_search="exa"
    )

    report = await harness.dry_run(config)

    assert report.total == 3 * 5
    missing = sorted(problem.rsplit(" ", 1)[-1] for problem in report.problems)
    assert missing == ["EXA_API_KEY", "SEC_API_KEY", "SEC_USER_AGENT"]


async def test_resuming_with_a_changed_config_is_rejected(harness: Harness) -> None:
    await harness.run(harness.config(), limit=1)

    with pytest.raises(RunSetupError, match="perturbation"):
        await harness.run(harness.config(perturbation=0.5))
    resumed = await harness.run(harness.config(concurrency=1), limit=1)
    assert resumed.finished == 1


async def test_resuming_after_the_nshot_file_changed_is_rejected(harness: Harness, tmp_path: Path) -> None:
    nshot_path = write_nshot(tmp_path / "nshot.json", ["Old."])
    config = harness.config(nshot=nshot_path, repeats=1)
    await harness.run(config, limit=1)
    write_nshot(nshot_path, ["New A.", "New B."])

    report = await harness.dry_run(config)

    assert len(report.problems) == 1
    assert f"changed: contents of the n-shot file {nshot_path})" in report.problems[0]
    with pytest.raises(RunSetupError, match="contents of the n-shot file"):
        await harness.run(config)
    assert [call.nshot for call in harness.planner.calls] == [("Old.",)]

    write_nshot(nshot_path, ["Old."])
    resumed = await harness.run(config)

    assert resumed.won == 2
    assert [call.nshot for call in harness.planner.calls] == [("Old.",)] * 3


async def test_resuming_after_a_model_decoding_or_budget_change_is_rejected(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = harness.config(Controller.REACT)
    await harness.run(config, limit=1)
    presets = load_model_presets()
    edited = presets["gpt-4o-mini"].model_copy(update={"reasoning_effort": "low"})
    monkeypatch.setattr(runner, "load_model_presets", lambda: presets | {"gpt-4o-mini": edited})

    with pytest.raises(RunSetupError, match=r"changed: executor model settings \(preset 'gpt-4o-mini'\)\)"):
        await harness.run(config)

    monkeypatch.setattr(runner, "load_model_presets", lambda: presets)
    with monkeypatch.context() as patch:
        patch.setitem(ROLE_SETTINGS, Role.EXECUTOR, RoleSettings(temperature=0.7, max_tokens=2048))
        with pytest.raises(RunSetupError, match=r"changed: executor model settings"):
            await harness.run(config)
    harness.benchmark.default_budgets = Budgets(global_tool_calls=20, local_tool_calls=5)

    with pytest.raises(RunSetupError, match=r"changed: budgets\)"):
        await harness.run(config)

    harness.benchmark.default_budgets = Budgets(global_tool_calls=10, local_tool_calls=5)
    resumed = await harness.run(config)
    assert (resumed.completed_before, resumed.won) == (1, 5)


async def test_a_price_edit_does_not_block_resume(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    config = harness.config(Controller.REACT)
    await harness.run(config, limit=1)
    presets = load_model_presets()
    repriced = presets["gpt-4o-mini"].model_copy(
        update={"input_usd_per_mtok": 0.2, "cached_input_usd_per_mtok": 0.1, "output_usd_per_mtok": 0.8}
    )
    monkeypatch.setattr(runner, "load_model_presets", lambda: presets | {"gpt-4o-mini": repriced})

    assert (await harness.dry_run(config)).problems == ()
    resumed = await harness.run(config)

    assert (resumed.completed_before, resumed.won) == (1, 5)


async def test_a_run_directory_of_another_schema_version_is_a_setup_problem(harness: Harness) -> None:
    config = harness.config(repeats=1)
    await harness.run(config, limit=1)
    store = harness.store()
    episode_path = store.episode_path(0, "item_000")
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode_path.write_text(json.dumps(episode | {"schema_version": 3}), encoding="utf-8")

    newer = await harness.dry_run(config)

    assert newer.problems == (
        f"{episode_path} was written by schema version 3; this code reads {SCHEMA_VERSION}. "
        "Start a new run directory: use a new experiment name or remove the old one",
    )

    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    del manifest["run_id"], manifest["conditions"]
    store.manifest_path.write_text(json.dumps(manifest | {"schema_version": 1}), encoding="utf-8")

    older = await harness.dry_run(config)

    assert older.problems == (
        f"{store.manifest_path} was written by schema version 1; this code reads {SCHEMA_VERSION}. "
        "Start a new run directory: use a new experiment name or remove the old one",
    )
    with pytest.raises(RunSetupError, match="written by schema version 1"):
        await harness.run(config)
    assert len(harness.llm.requests) == 2
