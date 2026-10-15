"""`ctg run` over several configs: which stops end the sweep and which end only their own run."""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
from runs_fakes import Harness, isolate_cli, write_config
from typer.testing import CliRunner

from ctg import cli
from ctg.config import Controller
from ctg.llm.base import LLMOutputError
from ctg.runs.runner import run_experiment


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    harness = Harness(tmp_path, monkeypatch)
    isolate_cli(monkeypatch)
    monkeypatch.setattr(
        cli, "run_experiment", functools.partial(run_experiment, splits_dir=harness.splits_dir)
    )
    return harness


def invoke_run(harness: Harness, *configs: Path) -> tuple[int, str]:
    result = CliRunner().invoke(cli.app, ["run", *map(str, configs), "--runs-dir", str(harness.runs_dir)])
    return result.exit_code, result.output


def test_a_failed_gen_cyc_planning_is_reported_and_the_remaining_configs_still_run(
    harness: Harness, tmp_path: Path
) -> None:
    harness.planner.failures[None] = LLMOutputError("planner output failed validation 3 times")
    gen_cyc = write_config(tmp_path, harness.config(Controller.GEN_CYC, name="gen"))
    react = write_config(tmp_path, harness.config(Controller.REACT, name="react"))

    exit_code, output = invoke_run(harness, gen_cyc, react)

    assert exit_code == 1
    assert (
        "STOPPED: planning the Gen-Cyc graph failed: LLMOutputError: planner output failed validation 3 times"
    ) in output
    assert "6 episodes were not started" in output
    assert "Not starting the remaining configs" not in output
    assert "react: 6/6 episodes finished (6 won, 0 errors)" in output
    assert len(harness.store("react").completed()) == 6
    assert list(harness.store("gen").episodes()) == []


def test_an_unavailable_backend_ends_the_sweep(harness: Harness, tmp_path: Path) -> None:
    harness.benchmark.backend_down = True
    first = write_config(tmp_path, harness.config(Controller.REACT, name="first", concurrency=1))
    second = write_config(tmp_path, harness.config(Controller.REACT, name="second"))

    exit_code, output = invoke_run(harness, first, second)

    assert exit_code == 1
    assert "STOPPED: backend unavailable during r0/item_000" in output
    assert f"Not starting the remaining configs: {second}" in output
    assert not (harness.runs_dir / "second").exists()


def test_a_run_directory_in_use_is_a_setup_error_and_the_sweep_goes_on(
    harness: Harness, tmp_path: Path
) -> None:
    busy = write_config(tmp_path, harness.config(Controller.REACT, name="busy"))
    free = write_config(tmp_path, harness.config(Controller.REACT, name="free"))

    with harness.store("busy").lock():
        exit_code, output = invoke_run(harness, busy, free)

    assert exit_code == 1
    assert "busy: cannot start" in output and "is in use by another `ctg run` process" in output
    assert len(harness.store("free").completed()) == 6
    assert list(harness.store("busy").episodes()) == []
