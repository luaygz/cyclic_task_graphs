"""n-shot builder: one summary per won episode, built from the trajectory, in the shared file format."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import BaseModel
from runs_fakes import cyclic_graph, episode_record, isolate_cli, routed_segment, run_manifest
from typer.testing import CliRunner

from ctg import cli
from ctg.config import BenchmarkName, Controller, ExperimentConfig, ModelSpec, Role
from ctg.llm.base import AssistantTurn, SpendCapExceededError
from ctg.llm.fake import FakeLLM, FakeRequest, text_turn
from ctg.runs import nshot
from ctg.runs.records import EndReason, ToolCallRecord
from ctg.runs.store import RunStore

CONFIG = ExperimentConfig(
    name="tc2_train", benchmark="textcraft", textcraft_depth=2, split="train", controller=Controller.SPEC_CYC
)
MODEL = ModelSpec(name="gpt-4o-mini")


def _call(tool: str, argument: str, observation: str, k: int) -> ToolCallRecord:
    return ToolCallRecord(
        tool=tool, arguments={"command": argument}, observation=observation, tool_calls_after=k
    )


def _run_dir(tmp_path: Path) -> Path:
    store = RunStore(tmp_path / "tc2_train")
    store.write_manifest(run_manifest(CONFIG))
    graph = cyclic_graph("crafting")
    won = [
        routed_segment(
            0, "gather", "Fetched 2 oak logs.", "Logs in hand.", _call("get", "get 2 oak logs", "Got", 1)
        ),
        routed_segment(
            1,
            "craft",
            "Crafted planks.",
            "Planks crafted, goal done.",
            _call("craft", "craft planks", "x" * 400, 2),
        ),
    ]
    store.write_episode(episode_record(CONFIG, "item_001", graph=graph, segments=won))
    store.write_episode(episode_record(CONFIG, "item_002", end_reason=EndReason.BUDGET, graph=graph))
    store.write_episode(episode_record(CONFIG, "item_003", end_reason=EndReason.ERROR))
    store.write_episode(episode_record(CONFIG, "item_001", 1, graph=graph, segments=won[:1]))
    return store.root


async def test_build_summarizes_each_won_episode_into_the_shared_format(tmp_path: Path) -> None:
    prompts: list[str] = []

    def respond(request: FakeRequest) -> BaseModel | AssistantTurn:
        assert request.role is Role.NSHOT
        prompts.append(request.text)
        return text_turn(f"  Strategy {len(prompts)}.  ")

    llm = FakeLLM(respond)
    out = tmp_path / "nshot" / "textcraft_depth2.json"

    result = await nshot.build(_run_dir(tmp_path), out, MODEL, llm)

    raw = json.loads(out.read_text(encoding="utf-8"))
    assert set(raw) == {"benchmark", "variant", "source_run", "model", "created_at", "summaries"}
    assert (raw["benchmark"], raw["variant"], raw["source_run"], raw["model"]) == (
        "textcraft",
        "textcraft_depth2",
        "tc2_train",
        "gpt-4o-mini",
    )
    assert [set(summary) for summary in raw["summaries"]] == [{"instance_id", "text"}] * 2
    assert [summary.instance_id for summary in result.summaries] == ["item_001", "item_001"]
    assert sorted(summary.text for summary in result.summaries) == ["Strategy 1.", "Strategy 2."]
    full = next(prompt for prompt in prompts if "Crafted planks." in prompt)
    for expected in (
        "Craft item_001.",
        "1. Gather materials (gather)\n2. Craft the goal (craft)",
        "Fetched 2 oak logs.",
        "Planks crafted, goal done.",
        'get({"command": "get 2 oak logs"}) -> Got',
    ):
        assert expected in full
    assert "x" * (nshot.OBSERVATION_CHARS + 1) not in full
    assert nshot.load_nshot(out, BenchmarkName.TEXTCRAFT) == result


async def test_build_without_won_episodes_writes_nothing(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "lost")
    store.write_manifest(run_manifest(CONFIG.model_copy(update={"name": "lost"})))
    store.write_episode(episode_record(CONFIG, "item_002", end_reason=EndReason.BUDGET))
    out = tmp_path / "out.json"

    with pytest.raises(ValueError, match="no won episodes"):
        await nshot.build(store.root, out, MODEL, FakeLLM(lambda request: text_turn("unused")))
    assert not out.exists()


async def test_load_rejects_a_file_for_another_benchmark(tmp_path: Path) -> None:
    out = tmp_path / "nshot.json"
    await nshot.build(_run_dir(tmp_path), out, MODEL, FakeLLM(lambda request: text_turn("Plan, then craft.")))

    with pytest.raises(ValueError, match="is for textcraft"):
        nshot.load_nshot(out, BenchmarkName.ALFWORLD)


def _spend_cap(request: FakeRequest) -> BaseModel | AssistantTurn:
    raise SpendCapExceededError("run cap of $0.01 reached")


def _empty_summary(request: FakeRequest) -> BaseModel | AssistantTurn:
    return text_turn(" ")


@pytest.mark.parametrize(
    ("respond", "message"),
    [
        (
            _spend_cap,
            "Spend cap reached; no n-shot file written: SpendCapExceededError: run cap of $0.01 reached",
        ),
        (
            _empty_summary,
            "n-shot build failed; no n-shot file written: LLMOutputError: empty n-shot summary for item_001",
        ),
    ],
)
def test_build_command_reports_a_failed_summary_and_writes_no_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    respond: Callable[[FakeRequest], BaseModel | AssistantTurn],
    message: str,
) -> None:
    isolate_cli(monkeypatch)
    monkeypatch.setattr(cli, "create_llm", lambda run=None, run_cap_usd=None: FakeLLM(respond))
    out = tmp_path / "nshot.json"

    result = CliRunner().invoke(cli.app, ["nshot", "build", str(_run_dir(tmp_path)), "--out", str(out)])

    assert result.exit_code == 1
    assert message in result.output
    assert "Traceback" not in result.output
    assert not out.exists()
