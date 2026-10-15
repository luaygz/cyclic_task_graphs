"""ALFWorld episodes driven by ALFWorld's own handcoded expert.

The expert plays each game in-process first; the recorded commands are then replayed through the tools.
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Any

import pytest

from ctg.config import BenchmarkName, Controller, ExperimentConfig
from ctg.envs.alfworld.benchmark import ALFWorldBenchmark
from ctg.envs.alfworld.env import ALFWorldEnv
from ctg.envs.alfworld.games import GameFile, games_dir, list_games
from ctg.envs.alfworld.tools import (
    EXPLORE,
    MANAGE_INVENTORY,
    MANIPULATE_OBJECT,
    PARTITIONED_TOOLS,
    SELECT_COMMAND,
    TOOL_NAMES,
)
from ctg.envs.base import InstanceRef
from ctg.llm.fake import FakeLLM, FakeRequest

pytestmark = [
    pytest.mark.alfworld,
    pytest.mark.skipif(not games_dir().is_dir(), reason="ALFWorld data not downloaded"),
]

_EXPERT_STEP_LIMIT = 150


@dataclass(frozen=True)
class ExpertRun:
    game: GameFile
    commands: list[str]
    observations: list[str]


def _play_expert(game: GameFile) -> ExpertRun:
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler,
        AlfredExpert,
        AlfredExpertType,
        AlfredInfos,
    )

    random.seed(0)
    request = textworld.EnvInfos(won=True, admissible_commands=True, extras=["expert_plan"])
    env_id = textworld.gym.register_game(
        str(games_dir() / game.path),
        request,
        max_episode_steps=0,
        wrappers=[
            AlfredDemangler(shuffle=False),
            AlfredInfos,
            AlfredExpert(expert_type=AlfredExpertType.HANDCODED),
        ],
    )
    env = textworld.gym.make(env_id)
    _, infos = env.reset()
    commands: list[str] = []
    observations: list[str] = []
    for _ in range(_EXPERT_STEP_LIMIT):
        if infos["won"]:
            break
        command = infos["extra.expert_plan"][0]
        observation, _, _, infos = env.step(command)
        if command != "help":  # the expert's random fallback; it changes nothing and no tool offers it
            commands.append(command)
            observations.append(observation)
    env.close()
    assert infos["won"], f"the expert did not solve {game.path}"
    return ExpertRun(game, commands, observations)


@pytest.fixture(scope="module")
def expert_runs() -> list[ExpertRun]:
    """Expert solutions of eight valid_unseen games covering all six task types."""
    games = list_games("valid_unseen")
    by_type: dict[str, list[GameFile]] = {}
    for game in games:
        by_type.setdefault(game.task_type, []).append(game)
    extra = [by_type["pick_and_place_simple"][1], by_type["pick_clean_then_place_in_recep"][1]]
    picked = [group[0] for group in by_type.values()] + extra
    return [_play_expert(game) for game in picked]


def _enum(env: ALFWorldEnv, tool: str) -> list[str]:
    specs = env.tool_specs([tool])
    return specs[0].parameters["properties"]["command"]["enum"] if specs else []


def _call(command: str) -> dict[str, Any]:
    return {"reasoning": ["follow the expert"], "command": command}


async def test_expert_commands_win_through_select_command(expert_runs: list[ExpertRun]) -> None:
    for run in expert_runs[:3]:
        env = await ALFWorldEnv.start(run.game.path)
        try:
            assert env.objective.endswith(f"Your task is to: {env.details()['task']}")
            assert env.initial_state() == {"admissible": _enum(env, SELECT_COMMAND)}
            for command, expected in zip(run.commands, run.observations, strict=True):
                assert not env.won
                assert await env.call(SELECT_COMMAND, _call(command)) == expected
            assert env.won
            assert env.done
            assert env.tool_calls == len(env.steps) == len(run.commands)
            assert env.details()["game"] == run.game.path
            with pytest.raises(RuntimeError):
                await env.call(SELECT_COMMAND, _call(run.commands[-1]))
        finally:
            await env.close()


async def test_partitioned_tools_cover_every_admissible_command(expert_runs: list[ExpertRun]) -> None:
    """Every state on the expert's path: each offered command belongs to exactly one partitioned tool, so the
    expert also wins using the partitioned tools alone."""
    used: set[str] = set()
    for run in expert_runs[:6]:
        env = await ALFWorldEnv.start(run.game.path)
        try:
            for command in run.commands:
                offered = _enum(env, SELECT_COMMAND)
                assert "help" not in offered
                partitions = {tool: _enum(env, tool) for tool in PARTITIONED_TOOLS}
                assert sorted(c for commands in partitions.values() for c in commands) == sorted(offered)
                (tool,) = [tool for tool, commands in partitions.items() if command in commands]
                used.add(tool)
                await env.call(tool, _call(command))
            assert env.won
        finally:
            await env.close()
    assert used == set(PARTITIONED_TOOLS)


async def test_tools_without_commands_are_omitted(expert_runs: list[ExpertRun]) -> None:
    run = next(run for run in expert_runs if run.game.task_type == "look_at_obj_in_light")
    env = await ALFWorldEnv.start(run.game.path)
    try:
        assert [spec.name for spec in env.tool_specs(TOOL_NAMES)] == [
            SELECT_COMMAND,
            MANAGE_INVENTORY,
            EXPLORE,
        ]
        assert env.tool_specs([MANIPULATE_OBJECT]) == []
        for command in run.commands[:-1]:
            await env.call(SELECT_COMMAND, _call(command))
        assert run.commands[-1].startswith("use ")
        assert run.commands[-1] in _enum(env, MANIPULATE_OBJECT)
        with pytest.raises(ValueError, match="unknown"):
            env.tool_specs(["alfworld_fly"])
    finally:
        await env.close()


async def test_rejected_commands_cost_a_call_and_a_step_but_do_not_step_the_game(
    expert_runs: list[ExpertRun],
) -> None:
    run = expert_runs[0]
    env = await ALFWorldEnv.start(run.game.path)
    try:
        before = _enum(env, SELECT_COMMAND)
        rejected = [
            (SELECT_COMMAND, _call("take unicorn 1 from desk 1")),
            (MANIPULATE_OBJECT, _call(before[0])),  # admissible, but not a manipulation
            (SELECT_COMMAND, {"reasoning": []}),
            ("alfworld_fly", _call(before[0])),
        ]
        for count, (tool, arguments) in enumerate(rejected, start=1):
            observation = await env.call(tool, arguments)
            assert observation.startswith("Error:")
            assert env.tool_calls == len(env.steps) == count
            step = env.steps[-1]
            assert (step.step, step.tool, step.observation) == (count, tool, observation)
            assert step.state == env.initial_state()
            assert _enum(env, SELECT_COMMAND) == before
        assert await env.call(SELECT_COMMAND, _call(run.commands[0])) == run.observations[0]
        assert env.tool_calls == len(env.steps) == len(rejected) + 1
        assert [step.step for step in env.steps] == [1, 2, 3, 4, 5]
    finally:
        await env.close()


async def test_eight_concurrent_episodes(expert_runs: list[ExpertRun]) -> None:
    assert len(expert_runs) == 8
    benchmark = ALFWorldBenchmark()
    config = ExperimentConfig(
        name="concurrency",
        benchmark=BenchmarkName.ALFWORLD,
        alfworld_split="valid_unseen",
        controller=Controller.REACT,
    )

    def no_llm(request: FakeRequest) -> Any:
        raise AssertionError("ALFWorld episodes make no LLM calls")

    async def play(run: ExpertRun) -> list[str]:
        instance = InstanceRef(id=run.game.id, meta={"path": run.game.path, "task_type": run.game.task_type})
        env = await benchmark.make_env(instance, config, FakeLLM(no_llm), {})
        try:
            observations = [await env.call(SELECT_COMMAND, _call(command)) for command in run.commands]
            assert env.won
            return observations
        finally:
            await env.close()

    results = await asyncio.gather(*(play(run) for run in expert_runs))

    assert results == [run.observations for run in expert_runs]
