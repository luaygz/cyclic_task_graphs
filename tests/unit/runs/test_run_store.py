"""Run directory storage: atomic writes, file modes, the run lock and which episodes count as complete."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest
from runs_fakes import episode_record

from ctg.config import Controller, ExperimentConfig
from ctg.runs import store as store_module
from ctg.runs.records import EndReason
from ctg.runs.store import RunInUseError, RunStore, write_json_atomic

CONFIG = ExperimentConfig(name="exp", benchmark="textcraft", textcraft_depth=2, controller=Controller.REACT)


@pytest.fixture
def umask() -> Iterator[None]:
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


def test_atomic_write_keeps_the_old_file_when_the_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "nested" / "file.json"
    write_json_atomic(path, '{"version": 1}')

    def failing_replace(src: Path, dst: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        write_json_atomic(path, '{"version": 2}')

    assert path.read_text(encoding="utf-8") == '{"version": 1}'
    assert os.listdir(path.parent) == ["file.json"]


@pytest.mark.usefixtures("umask")
def test_atomic_write_creates_files_readable_as_the_umask_allows(tmp_path: Path) -> None:
    path = tmp_path / "run" / "manifest.json"

    write_json_atomic(path, "{}")
    write_json_atomic(path, '{"again": true}')
    os.umask(0o077)
    private = tmp_path / "private.json"
    write_json_atomic(private, "{}")

    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert stat.S_IMODE(private.stat().st_mode) == 0o600


def test_the_run_lock_admits_one_holder_at_a_time(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "exp")

    with store.lock():
        with pytest.raises(RunInUseError, match="in use by another"), RunStore(tmp_path / "exp").lock():
            pass
        with RunStore(tmp_path / "other").lock():
            pass

    with store.lock():
        assert store.manifest() is None


def test_completed_excludes_only_errored_episodes(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "exp")
    store.write_episode(episode_record(CONFIG, "a", 0))
    store.write_episode(episode_record(CONFIG, "b", 0, end_reason=EndReason.BUDGET))
    store.write_episode(episode_record(CONFIG, "c", 0, end_reason=EndReason.ERROR))
    store.write_episode(episode_record(CONFIG, "c", 1))
    store.write_episode(episode_record(CONFIG, "d", 0, end_reason=EndReason.OUTPUT_FAILURE))

    assert store.completed() == {(0, "a"), (0, "b"), (1, "c"), (0, "d")}

    store.write_episode(episode_record(CONFIG, "c", 0))

    assert (0, "c") in store.completed()


def test_episodes_are_ordered_by_numeric_repeat(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "exp")
    for repeat in (10, 2, 0):
        store.write_episode(episode_record(CONFIG, "a", repeat))

    assert [record.repeat for record in store.episodes()] == [0, 2, 10]
    assert store.episode_path(2, "a") == tmp_path / "exp" / "episodes" / "r2" / "a.json"
