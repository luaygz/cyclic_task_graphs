"""ALFWorld game files and the process that plays one game.

This module imports only the standard library; TextWorld and ALFWorld load inside the game process.
TextWorld's PDDL loader (fast_downward's translator) swaps `sys.stdout` and mutates module globals, so two
games cannot load concurrently in one process. Each episode therefore plays its game in its own process,
`python -m ctg.envs.alfworld.games <game file>`, which exchanges JSON lines with the episode over stdin and
stdout.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TextIO

DATA_ENV_VAR = "ALFWORLD_DATA"
DEFAULT_DATA_DIR = "~/.cache/alfworld"
GAMES_SUBDIR = "json_2.1.1"
GAME_FILE = "game.tw-pddl"

Split = Literal["train", "valid_seen", "valid_unseen"]

TASK_TYPES = frozenset(
    {
        "pick_and_place_simple",
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
    }
)
"""ALFWorld's task types 1-6, the ones its TextWorld environment plays."""

REPLY_TIMEOUT_S = 120.0
"""Default longest wait for the game process to load the game or answer a command."""
CLOSE_TIMEOUT_S = 10.0
"""Longest wait for the game process to exit once its stdin is closed; after that it is killed."""

_MAX_MESSAGE_BYTES = 1 << 20


def data_dir() -> Path:
    """The ALFWorld data directory: `$ALFWORLD_DATA`, or ~/.cache/alfworld."""
    return Path(os.environ.get(DATA_ENV_VAR) or DEFAULT_DATA_DIR).expanduser()


def games_dir(root: Path | None = None) -> Path:
    return (root or data_dir()) / GAMES_SUBDIR


@dataclass(frozen=True)
class GameFile:
    path: str
    """Path of `game.tw-pddl` relative to the games directory: `<split>/<task>/<trial>/game.tw-pddl`."""
    task_type: str

    @property
    def id(self) -> str:
        """Filesystem-safe instance id: the trial directory's relative path with `/` replaced by `__`."""
        return self.path.removesuffix(f"/{GAME_FILE}").replace("/", "__")


def list_games(split: Split, root: Path | None = None) -> list[GameFile]:
    """The games of a split that ALFWorld's TextWorld environment plays, sorted by path.

    Same filters as `AlfredTWEnv.collect_game_files`: no movable-receptacle or sliced-object trajectories,
    task types 1-6 only, and a game file marked solvable.
    """
    base = games_dir(root)
    split_dir = base / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"ALFWorld split not found: {split_dir} (set {DATA_ENV_VAR})")
    games: list[GameFile] = []
    for traj_file in split_dir.rglob("traj_data.json"):
        trial = traj_file.parent
        relative = trial.relative_to(base).as_posix()
        if "movable" in relative or "Sliced" in relative:
            continue
        task_type = json.loads(traj_file.read_text(encoding="utf-8"))["task_type"]
        if task_type not in TASK_TYPES:
            continue
        game_file = trial / GAME_FILE
        if not game_file.is_file():
            continue
        if not json.loads(game_file.read_text(encoding="utf-8")).get("solvable", False):
            continue
        games.append(GameFile(path=f"{relative}/{GAME_FILE}", task_type=task_type))
    return sorted(games, key=lambda game: game.path)


@dataclass(frozen=True)
class GameState:
    observation: str
    won: bool
    admissible: tuple[str, ...]
    """Admissible commands exactly as TextWorld lists them (sorted, including `help`)."""
    held: tuple[str, ...]
    """Objects the agent is holding."""


def _send(replies: TextIO, message: dict[str, object]) -> None:
    replies.write(json.dumps(message) + "\n")
    replies.flush()


def _serve(game_file: str) -> None:
    """Game-process loop: send the initial state, then answer each command on stdin with the next state.

    Replies go to a private copy of stdout; fd 1 is pointed at stderr so nothing TextWorld prints can corrupt
    the protocol. The loop ends when stdin closes.
    """
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        request = textworld.EnvInfos(won=True, admissible_commands=True, facts=True)
        env_id = textworld.gym.register_game(
            game_file,
            request,
            max_episode_steps=0,  # no step limit: the controllers enforce the tool-call budget
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
        )
        env = textworld.gym.make(env_id)
        observation, infos = env.reset()
        while True:
            held = sorted(fact.arguments[1].name for fact in infos["facts"] if fact.name == "holds")
            state = GameState(
                observation, bool(infos["won"]), tuple(infos["admissible_commands"]), tuple(held)
            )
            _send(replies, {"state": asdict(state)})
            line = sys.stdin.readline()
            if not line:
                return
            observation, _, _, infos = env.step(json.loads(line))
    except BrokenPipeError:
        return
    except Exception:
        _send(replies, {"error": traceback.format_exc()})


class GameProcess:
    """One ALFWorld game played by TextWorld in a dedicated process. Requests must not overlap.

    A process that sends no reply within the reply timeout is killed and the request raises RuntimeError.
    """

    def __init__(self, process: asyncio.subprocess.Process, reply_timeout_s: float = REPLY_TIMEOUT_S) -> None:
        assert process.stdin is not None and process.stdout is not None
        self._process = process
        self._stdin = process.stdin
        self._stdout = process.stdout
        self._reply_timeout_s = reply_timeout_s

    @classmethod
    async def start(
        cls, game_file: Path, reply_timeout_s: float = REPLY_TIMEOUT_S
    ) -> tuple[GameProcess, GameState]:
        """Launch the game and return it with its initial state."""
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            __name__,
            str(game_file),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            limit=_MAX_MESSAGE_BYTES,
        )
        game = cls(process, reply_timeout_s)
        try:
            return game, await game._receive()
        except BaseException:
            await game.close()
            raise

    async def step(self, command: str) -> GameState:
        with contextlib.suppress(ConnectionError):  # a dead process is reported by `_receive`
            self._stdin.write(json.dumps(command).encode() + b"\n")
            await self._stdin.drain()
        return await self._receive()

    async def _receive(self) -> GameState:
        try:
            line = await asyncio.wait_for(self._stdout.readline(), self._reply_timeout_s)
        except TimeoutError:
            await self._kill()
            raise RuntimeError(
                f"the ALFWorld game process sent no reply within {self._reply_timeout_s:g} s and was killed"
            ) from None
        if not line:
            code = await self._process.wait()
            raise RuntimeError(f"the ALFWorld game process exited unexpectedly (exit code {code})")
        message = json.loads(line)
        if "error" in message:
            raise RuntimeError(f"the ALFWorld game process failed:\n{message['error']}")
        state = message["state"]
        return GameState(state["observation"], state["won"], tuple(state["admissible"]), tuple(state["held"]))

    async def close(self) -> None:
        """Stop the game process. Safe to call more than once."""
        if self._process.returncode is not None:
            return
        with contextlib.suppress(OSError):
            self._stdin.close()
        try:
            await asyncio.wait_for(self._process.wait(), CLOSE_TIMEOUT_S)
        except TimeoutError:
            await self._kill()

    async def _kill(self) -> None:
        with contextlib.suppress(ProcessLookupError):  # the process exited on its own meanwhile
            self._process.kill()
        await self._process.wait()


if __name__ == "__main__":
    _serve(sys.argv[1])
