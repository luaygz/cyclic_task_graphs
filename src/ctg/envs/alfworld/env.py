"""One ALFWorld episode behind the tool interface of Appendix B.1.2."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ctg.envs.alfworld.games import REPLY_TIMEOUT_S, GameProcess, GameState, games_dir
from ctg.envs.alfworld.tools import TOOL_NAMES, command_sets, offered_commands, tool_spec
from ctg.envs.base import Environment
from ctg.llm.base import ToolSpec
from ctg.runs.records import StepRecord

_BANNER = "-= Welcome to TextWorld, ALFRED! =-"
_TASK_PREFIX = "Your task is to: "


def _state_snapshot(state: GameState) -> dict[str, Any]:
    """Canonical state: the offered admissible commands, which encode the location and the held object."""
    return {"admissible": offered_commands(state.admissible)}


class ALFWorldEnv(Environment):
    """Commands are validated against the current state: an inadmissible one costs a tool call but never
    reaches the game."""

    def __init__(self, game: GameProcess, path: str, initial: GameState) -> None:
        self._game = game
        self._path = path
        self._initial = initial
        self._state = initial
        self._commands = command_sets(initial.admissible, initial.held)
        self._objective = initial.observation.removeprefix(_BANNER).strip()
        self._task = next(
            (
                line.removeprefix(_TASK_PREFIX)
                for line in self._objective.splitlines()
                if line.startswith(_TASK_PREFIX)
            ),
            "",
        )
        self._tool_calls = 0
        self._steps: list[StepRecord] = []

    @classmethod
    async def start(
        cls, path: str, root: Path | None = None, reply_timeout_s: float = REPLY_TIMEOUT_S
    ) -> ALFWorldEnv:
        """Launch the game at `path` (relative to the games directory, see `GameFile.path`).

        `reply_timeout_s` bounds every wait for the game process (see `GameProcess`).
        """
        game, initial = await GameProcess.start(games_dir(root) / path, reply_timeout_s)
        try:
            return cls(game, path, initial)
        except BaseException:
            await game.close()
            raise

    @property
    def objective(self) -> str:
        """The initial observation without TextWorld's banner: the room overview and the task sentence."""
        return self._objective

    @property
    def won(self) -> bool:
        return self._state.won

    @property
    def done(self) -> bool:
        return self._state.won

    @property
    def tool_calls(self) -> int:
        return self._tool_calls

    @property
    def steps(self) -> Sequence[StepRecord]:
        return self._steps

    def initial_state(self) -> dict[str, Any]:
        return _state_snapshot(self._initial)

    def tool_specs(self, names: Sequence[str]) -> list[ToolSpec]:
        unknown = [name for name in names if name not in self._commands]
        if unknown:
            raise ValueError(f"unknown ALFWorld tools: {unknown}")
        return [tool_spec(name, self._commands[name]) for name in names if self._commands[name]]

    async def call(
        self, name: str, arguments: Mapping[str, Any], offered: Collection[str] | None = None
    ) -> str:
        """Rejected calls (a tool not offered, an unknown tool, a missing or inadmissible command) count
        as tool calls and are recorded as steps, but never reach the game: the state stays unchanged."""
        if self.done:
            raise RuntimeError("the ALFWorld episode is over")
        self._tool_calls += 1
        observation = self._rejection(name, arguments, offered)
        if observation is None:
            self._state = await self._game.step(cast(str, arguments["command"]))
            self._commands = command_sets(self._state.admissible, self._state.held)
            observation = self._state.observation
        self._steps.append(
            StepRecord(
                step=self._tool_calls,
                tool=name,
                arguments=dict(arguments),
                observation=observation,
                state=_state_snapshot(self._state),
            )
        )
        return observation

    def _rejection(
        self, name: str, arguments: Mapping[str, Any], offered: Collection[str] | None
    ) -> str | None:
        """The error observation for a call that must not reach the game, or None to execute it."""
        if offered is not None and name not in offered:
            return (
                f"Error: tool {name!r} is not available to this executor. "
                f"Available tools: {', '.join(sorted(offered))}."
            )
        commands = self._commands.get(name)
        if commands is None:
            return f"Error: unknown tool {name!r}. ALFWorld tools: {', '.join(TOOL_NAMES)}."
        command = arguments.get("command")
        if not isinstance(command, str):
            return f"Error: {name} needs a `command` string."
        if command not in commands:
            return (
                f"Error: {command!r} is not a command {name} can execute in the current state. "
                "Nothing happened; choose one of the listed commands."
            )
        return None

    def details(self) -> dict[str, Any]:
        return {"game": self._path, "task": self._task}

    async def close(self) -> None:
        await self._game.close()
