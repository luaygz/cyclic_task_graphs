"""The game-process lifecycle and the tool interface, played against a scripted stand-in for the game process.

The stand-in replaces the interpreter that `GameProcess` launches, so these tests need neither the alfworld
extra nor the ALFWorld data.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest

from ctg.envs.alfworld import games
from ctg.envs.alfworld.env import ALFWorldEnv
from ctg.envs.alfworld.games import GameProcess, games_dir
from ctg.envs.alfworld.tools import EXPLORE, MANAGE_INVENTORY, SELECT_COMMAND

GAME_PATH = "valid_unseen/fake-task/trial_1/game.tw-pddl"
ADMISSIBLE = ["go to bed 1", "help", "inventory", "look"]

# Speaks the game-process protocol: one JSON state per line, one JSON command per line on stdin.
FAKE_GAME = """
import json, os, sys, time

spec = json.loads(open(sys.argv[3], encoding="utf-8").read())
with open(spec["pid_file"], "w") as f:
    f.write(str(os.getpid()))

def send(observation):
    state = {"observation": observation, "won": False, "admissible": spec["admissible"], "held": []}
    sys.stdout.write(json.dumps({"state": state}) + "\\n")
    sys.stdout.flush()

if spec["stall"] == "start":
    time.sleep(600)
send("-= Welcome to TextWorld, ALFRED! =-\\n\\nYou are in a room.\\n\\nYour task is to: look at a bed.")
for line in sys.stdin:
    if spec["stall"] == "command":
        time.sleep(600)
    command = json.loads(line)
    with open(spec["commands_file"], "a") as f:
        f.write(command + "\\n")
    send(f"You {command}.")
with open(spec["stopped_file"], "w") as f:
    f.write("stdin closed")
"""


@dataclass(frozen=True)
class FakeGame:
    root: Path
    """Data root holding the game file and the files the stand-in writes."""

    @property
    def pid(self) -> int:
        return int((self.root / "pid").read_text())

    @property
    def commands(self) -> list[str]:
        log = self.root / "commands"
        return log.read_text().splitlines() if log.exists() else []

    @property
    def stopped_on_stdin_close(self) -> bool:
        return (self.root / "stopped").exists()

    def running(self) -> bool:
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        return True


@pytest.fixture
def fake_interpreter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    script = tmp_path / "python"
    script.write_text(f"#!{sys.executable}\n{FAKE_GAME}")
    script.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(script))
    yield tmp_path
    pid_file = tmp_path / "pid"
    if pid_file.exists():
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(pid_file.read_text()), signal.SIGKILL)


def fake_game(
    root: Path, admissible: list[str] = ADMISSIBLE, stall: Literal["start", "command"] | None = None
) -> FakeGame:
    game_file = games_dir(root) / GAME_PATH
    game_file.parent.mkdir(parents=True)
    spec: dict[str, Any] = {
        "admissible": admissible,
        "stall": stall,
        "pid_file": str(root / "pid"),
        "commands_file": str(root / "commands"),
        "stopped_file": str(root / "stopped"),
    }
    game_file.write_text(json.dumps(spec))
    return FakeGame(root)


def call(command: str) -> dict[str, Any]:
    return {"reasoning": ["test"], "command": command}


async def test_a_failing_episode_setup_stops_the_game_process(fake_interpreter: Path) -> None:
    game = fake_game(fake_interpreter, admissible=["fly to the moon", "look"])

    with pytest.raises(ValueError, match="unrecognized ALFWorld command"):
        await ALFWorldEnv.start(GAME_PATH, root=game.root)

    assert game.stopped_on_stdin_close
    assert not game.running()


async def test_a_game_that_never_loads_is_killed(fake_interpreter: Path) -> None:
    game = fake_game(fake_interpreter, stall="start")

    with pytest.raises(RuntimeError, match=r"no reply within 0\.5 s and was killed"):
        await asyncio.wait_for(ALFWorldEnv.start(GAME_PATH, root=game.root, reply_timeout_s=0.5), 10)

    assert not game.running()


async def test_a_game_that_stops_answering_is_killed_and_reported(fake_interpreter: Path) -> None:
    game = fake_game(fake_interpreter, stall="command")
    env = await ALFWorldEnv.start(GAME_PATH, root=game.root, reply_timeout_s=0.5)
    try:
        with pytest.raises(RuntimeError, match=r"no reply within 0\.5 s and was killed"):
            await asyncio.wait_for(env.call(SELECT_COMMAND, call("look")), 10)
        assert not game.running()
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            await env.call(SELECT_COMMAND, call("look"))
    finally:
        await env.close()


async def test_close_stops_a_game_that_exits_on_its_own_just_before_the_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExitsBeforeTheKill:
        """A child that ignores its stdin closing and exits by itself right as `close` gives up waiting:
        asyncio's `kill` then finds no process and raises ProcessLookupError."""

        def __init__(self) -> None:
            self.stdin = SimpleNamespace(close=lambda: None)
            self.stdout = asyncio.StreamReader()
            self.returncode: int | None = None
            self._exited = asyncio.Event()

        async def wait(self) -> int:
            await self._exited.wait()
            assert self.returncode is not None
            return self.returncode

        def kill(self) -> None:
            self.returncode = 0
            self._exited.set()
            raise ProcessLookupError

    monkeypatch.setattr(games, "CLOSE_TIMEOUT_S", 0.05)
    process = ExitsBeforeTheKill()
    game = GameProcess(cast(asyncio.subprocess.Process, process))

    await game.close()
    await game.close()

    assert process.returncode == 0


async def test_a_tool_not_offered_to_the_executor_never_reaches_the_game(fake_interpreter: Path) -> None:
    game = fake_game(fake_interpreter)
    env = await ALFWorldEnv.start(GAME_PATH, root=game.root)
    try:
        observation = await env.call(SELECT_COMMAND, call("go to bed 1"), offered={MANAGE_INVENTORY, EXPLORE})

        assert observation == (
            f"Error: tool {SELECT_COMMAND!r} is not available to this executor. "
            f"Available tools: {EXPLORE}, {MANAGE_INVENTORY}."
        )
        assert env.tool_calls == 1
        [refused] = env.steps
        assert (refused.step, refused.tool, refused.observation) == (1, SELECT_COMMAND, observation)
        assert refused.state == env.initial_state()
        assert game.commands == []

        assert await env.call(EXPLORE, call("go to bed 1"), offered={EXPLORE}) == "You go to bed 1."
        assert await env.call(SELECT_COMMAND, call("look")) == "You look."
        assert env.tool_calls == 3
        assert [(step.step, step.tool) for step in env.steps] == [
            (1, SELECT_COMMAND),
            (2, EXPLORE),
            (3, SELECT_COMMAND),
        ]
        assert game.commands == ["go to bed 1", "look"]
    finally:
        await env.close()
    assert game.stopped_on_stdin_close
