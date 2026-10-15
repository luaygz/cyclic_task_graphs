import hashlib
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pytest

from ctg.envs.textcraft.benchmark import GLOBAL_TOOL_CALLS
from ctg.envs.textcraft.env import MAX_DISTRACTORS, TextCraftEnv, build_task
from ctg.envs.textcraft.recipes import display_name, load_recipe_book
from ctg.envs.textcraft.tools import CRAFT, GET_ITEM, INVENTORY, SELECT_COMMAND
from ctg.runs import splits

pytestmark = pytest.mark.textcraft_recipes

ALL_GOALS = [instance.id for depth in (2, 3, 4) for instance, _ in splits.load(f"textcraft_depth{depth}")]
"""Every benchmark goal, from the frozen splits (the benchmark's goal order)."""


def _env(goal: str) -> TextCraftEnv:
    return TextCraftEnv(load_recipe_book(), goal)


async def _run(env: TextCraftEnv, command: str) -> str:
    return await env.call(SELECT_COMMAND, {"reasoning": [], "command": command})


@dataclass(frozen=True)
class _Shown:
    line: str
    count: int
    inputs: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class _Objective:
    recipes: dict[str, list[_Shown]]
    categories: dict[str, list[str]]
    goal: str

    @property
    def lines(self) -> set[str]:
        return {shown.line for shown in (s for recipes in self.recipes.values() for s in recipes)}


def _parse_objective(objective: str) -> _Objective:
    """Read the instance from its text alone."""
    blocks = objective.split("\n\n")
    commands = blocks[0].splitlines()
    assert commands[0] == "Crafting commands:"
    recipes: dict[str, list[_Shown]] = defaultdict(list)
    for line in commands[1:]:
        match = re.fullmatch(r"craft (\d+) (.+?) using (.+)", line)
        assert match is not None, line
        inputs: list[tuple[int, str]] = []
        for part in match[3].split(", "):
            amount = re.fullmatch(r"(\d+) (.+)", part)
            assert amount is not None, line
            inputs.append((int(amount[1]), amount[2]))
        recipes[match[2]].append(_Shown(line, int(match[1]), tuple(inputs)))
    categories: dict[str, list[str]] = {}
    for block in blocks[1:-1]:
        for line in block.splitlines()[1:]:
            tag, members = line.split(": ", 1)
            categories[tag] = members.split(", ")
    goal = re.fullmatch(r"Goal: craft (.+)\.", blocks[-1])
    assert goal is not None
    return _Objective(dict(recipes), categories, goal[1])


class _Solver:
    """Reaches the goal using only the instance's text: listed crafting commands, and `get` for the rest.

    The generalist runs every command through textcraft_select_command; the specialists run `get` through
    textcraft_get_item and crafts through textcraft_craft. Each command must fit the schema of its tool.
    """

    def __init__(self, env: TextCraftEnv, specialists: bool) -> None:
        self.env = env
        self.specialists = specialists
        self.shown = _parse_objective(env.objective)
        get_spec, craft_spec, select_spec = env.tool_specs([GET_ITEM, CRAFT, SELECT_COMMAND])
        self.items: list[str] = get_spec.parameters["properties"]["item_name"]["enum"]
        self.commands: list[str] = craft_spec.parameters["properties"]["crafting_command"]["enum"]
        self.pattern: str = select_spec.parameters["properties"]["command"]["pattern"]
        self.used: list[str] = []

    def held(self, name: str) -> int:
        inventory = {display_name(item): n for item, n in self.env.inventory.items()}
        if name in self.shown.categories:
            return sum(inventory.get(member, 0) for member in self.shown.categories[name])
        return inventory.get(name, 0)

    def depth(self, name: str) -> int:
        if name in self.shown.categories:
            return min(self.depth(member) for member in self.shown.categories[name])
        return min((self.recipe_depth(shown) for shown in self.shown.recipes.get(name, [])), default=0)

    def recipe_depth(self, shown: _Shown) -> int:
        return 1 + max(self.depth(name) for _, name in shown.inputs)

    async def act(self, command: str) -> None:
        self.used.append(command)
        if not self.specialists:
            assert re.fullmatch(self.pattern, command), command
            observation = await _run(self.env, command)
        elif command.startswith("get "):
            _, count, item = command.split(" ", 2)
            assert item in self.items, command
            observation = await self.env.call(
                GET_ITEM, {"reasoning": [], "item_name": item, "num_total_items_needed": int(count)}
            )
        else:
            assert command in self.commands, command
            observation = await self.env.call(CRAFT, {"reasoning": [], "crafting_command": command})
        assert observation.startswith(("Got ", "Crafted ")), (command, observation)

    async def ensure(self, name: str, total: int) -> None:
        missing = total - self.held(name)
        if missing <= 0:
            return
        if name in self.shown.categories:
            member = min(self.shown.categories[name], key=self.depth)
            await self.ensure(member, self.held(member) + missing)
            return
        if name not in self.shown.recipes:
            await self.act(f"get {missing} {name}")
            return
        recipe = min(self.shown.recipes[name], key=self.recipe_depth)
        times = math.ceil(missing / recipe.count)
        inputs = sorted(recipe.inputs, key=lambda entry: -self.depth(entry[1]))
        for _ in range(10):
            if all(self.held(ingredient) >= n * times for n, ingredient in inputs):
                break
            for n, ingredient in inputs:
                await self.ensure(ingredient, n * times)
        else:
            pytest.fail(f"could not gather the ingredients of {recipe.line}")
        for _ in range(times):
            await self.act(recipe.line)


@pytest.mark.parametrize("specialists", [False, True], ids=["generalist", "specialists"])
@pytest.mark.parametrize("goal", ALL_GOALS)
async def test_every_goal_is_solvable_with_its_displayed_commands(goal: str, specialists: bool) -> None:
    env = _env(goal)
    solver = _Solver(env, specialists)

    await solver.ensure(solver.shown.goal, 1)

    assert env.won and env.done
    assert solver.shown.goal == display_name(goal)
    crafts = [command for command in solver.used if command.startswith("craft ")]
    assert set(crafts) <= solver.shown.lines
    assert env.tool_calls == len(solver.used) <= GLOBAL_TOOL_CALLS[load_recipe_book().depth(goal)]


@pytest.mark.parametrize("goal", ALL_GOALS)
def test_objective_shows_the_needed_recipes_and_few_distractors(goal: str) -> None:
    book = load_recipe_book()
    task = build_task(book, goal)
    needed = set(book.required_recipes(goal))

    assert needed <= set(task.recipes)
    assert len(task.recipes) - len(needed) <= MAX_DISTRACTORS
    assert len(set(task.recipes)) == len(task.recipes)
    assert task.objective.endswith(f"\n\nGoal: craft {display_name(goal)}.")
    assert "_" not in task.objective
    assert "minecraft" not in task.objective
    assert "Ingredient(" not in task.objective


def test_objectives_are_deterministic_across_processes() -> None:
    script = (
        "import hashlib\n"
        "from ctg.envs.textcraft.env import build_task\n"
        "from ctg.envs.textcraft.recipes import load_recipe_book\n"
        "book = load_recipe_book()\n"
        "text = '\\n'.join(build_task(book, g).objective for d in (2, 3, 4) for g in book.goals(d))\n"
        "print(hashlib.sha256(text.encode()).hexdigest())\n"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for seed in ("1", "2")
    }
    local = "\n".join(_env(goal).objective for goal in ALL_GOALS)

    assert digests == {hashlib.sha256(local.encode()).hexdigest()}


async def test_get_finds_nothing_for_craftable_category_and_unknown_items() -> None:
    env = _env("dark_oak_sign")

    assert await _run(env, "get 2 dark oak planks") == "Could not find dark oak planks"
    assert await _run(env, "get 2 planks") == "Could not find planks"
    assert await _run(env, "get 1 unobtainium") == "Could not find unobtainium"
    assert await _run(env, "get 0 dark oak logs") == (
        "Could not get dark oak logs: the number of items must be at least 1."
    )
    assert (
        await env.call(GET_ITEM, {"reasoning": [], "item_name": "stick", "num_total_items_needed": 4})
        == "Could not find stick"
    )
    assert env.inventory == {}

    assert await _run(env, "get 3 dark oak logs") == "Got 3 dark oak logs"
    assert (
        await env.call(
            GET_ITEM, {"reasoning": ["need more"], "item_name": "dark oak logs", "num_total_items_needed": 2}
        )
        == "Got 2 dark oak logs"
    )
    assert env.inventory == {"dark_oak_logs": 5}


async def test_get_accepts_any_raw_item_of_the_recipe_book() -> None:
    env = _env("andesite")
    assert "oak logs" not in env.task.items

    assert await _run(env, "get 1 oak logs") == "Got 1 oak logs"
    # Only a storage-block recipe, which the data rules drop, makes snow: it is raw.
    assert (
        await env.call(GET_ITEM, {"reasoning": [], "item_name": "snow", "num_total_items_needed": 3})
        == "Got 3 snow"
    )
    # Craftable in the recipe book, though no listed command makes it here.
    assert await _run(env, "get 1 stick") == "Could not find stick"
    assert env.inventory == {"oak_logs": 1, "snow": 3}


async def test_gatherer_may_request_intermediates_and_finds_nothing() -> None:
    env = _env("polished_granite")
    items = env.tool_specs([GET_ITEM])[0].parameters["properties"]["item_name"]["enum"]
    assert {"diorite", "granite", "quartz"} <= set(items)
    await env.call(GET_ITEM, {"reasoning": [], "item_name": "quartz", "num_total_items_needed": 4})
    calls = env.tool_calls

    observation = await env.call(
        GET_ITEM, {"reasoning": [], "item_name": "diorite", "num_total_items_needed": 4}
    )

    assert observation == "Could not find diorite"
    assert env.tool_calls == calls + 1
    assert env.inventory == env.steps[-1].state == {"quartz": 4}


async def test_craft_consumes_inputs_and_adds_the_recipe_output() -> None:
    env = _env("dark_oak_sign")
    await _run(env, "get 2 dark oak logs")

    assert await _run(env, "craft 4 dark oak planks using 1 dark oak logs") == (
        "Crafted 4 minecraft:dark_oak_planks"
    )
    assert env.inventory == {"dark_oak_logs": 1, "dark_oak_planks": 4}
    # A tag ingredient written as the tag takes any member.
    assert await _run(env, "craft 4 stick using 2 planks") == "Crafted 4 minecraft:stick"
    assert env.inventory == {"dark_oak_logs": 1, "dark_oak_planks": 2, "stick": 4}
    # ... or as one specific member.
    assert await env.call(
        CRAFT, {"reasoning": [], "crafting_command": "craft 4 stick using 2 dark oak planks"}
    ) == ("Crafted 4 minecraft:stick")
    assert env.inventory == {"dark_oak_logs": 1, "stick": 8}
    assert not env.won


async def test_tag_ingredients_draw_from_several_members() -> None:
    env = _env("dark_oak_sign")
    await _run(env, "get 1 oak logs")
    await _run(env, "get 1 birch logs")
    await _run(env, "craft 4 oak planks using 1 oak logs")
    await _run(env, "craft 4 birch planks using 1 birch logs")
    for _ in range(3):
        await _run(env, "craft 4 stick using 2 planks")

    assert env.inventory == {"oak_planks": 2, "stick": 12}


async def test_failed_crafts_change_nothing() -> None:
    env = _env("dark_oak_sign")
    await _run(env, "get 1 dark oak logs")
    await _run(env, "craft 4 dark oak planks using 1 dark oak logs")
    before = env.inventory

    # Written ingredients the inventory lacks are reported before the command is matched to a recipe.
    assert await _run(env, "craft 3 dark oak sign using 6 dark oak planks, 1 stick") == (
        "Could not find enough items to craft minecraft:dark_oak_sign"
    )
    assert await _run(env, "craft 1 diamond sword using 2 diamond, 1 stick") == (
        "Could not find enough items to craft minecraft:diamond_sword"
    )
    # Held ingredients, but no listed crafting command matches.
    assert await _run(env, "craft 1 stick using 2 planks") == (
        "Could not find a valid recipe for "
        "ItemTagWithCount(item_tag=ItemTag(tag=None, item_id='minecraft:stick'), count=1)"
    )
    assert await _run(env, "craft 4 stick using 3 dark oak planks") == (
        "Could not find a valid recipe for "
        "ItemTagWithCount(item_tag=ItemTag(tag=None, item_id='minecraft:stick'), count=4)"
    )
    assert await _run(env, "craft 4 planks using 1 dark oak planks") == (
        "Could not find a valid recipe for "
        "ItemTagWithCount(item_tag=ItemTag(tag='minecraft:planks', item_id=None), count=4)"
    )
    assert env.inventory == before


async def test_goal_crafted_with_reordered_ingredients_ends_the_episode() -> None:
    env = _env("dark_oak_sign")
    await _run(env, "get 2 dark oak logs")
    await _run(env, "craft 4 dark oak planks using 1 dark oak logs")
    await _run(env, "craft 4 dark oak planks using 1 dark oak logs")
    await _run(env, "get 2 bamboo")
    await _run(env, "craft 1 stick using 2 bamboo")

    assert (
        await _run(env, "Craft 3 Dark_Oak_Sign using 1 stick, 6 dark oak planks.")
        == "Crafted 3 minecraft:dark_oak_sign"
    )
    assert env.won and env.done
    assert env.details() == {"goal": "dark_oak_sign", "inventory": {"dark_oak_planks": 2, "dark_oak_sign": 3}}
    with pytest.raises(RuntimeError):
        await env.call(INVENTORY, {})


@pytest.mark.parametrize(
    ("name", "arguments", "observation"),
    [
        ("textcraft_teleport", {}, "Unknown tool textcraft_teleport."),
        (SELECT_COMMAND, {"reasoning": [], "command": "dance"}, "Unknown command."),
        (SELECT_COMMAND, {"reasoning": [], "command": "get some logs"}, "Invalid get command."),
        (SELECT_COMMAND, {"reasoning": [], "command": "craft a sign"}, "Invalid crafting command."),
        (
            SELECT_COMMAND,
            {"command": "inventory"},
            "Invalid call to textcraft_select_command: missing reasoning.",
        ),
        (INVENTORY, {"verbose": True}, "Invalid call to textcraft_inventory: unexpected verbose."),
        (CRAFT, {"reasoning": [], "crafting_command": "get 1 bamboo"}, "Invalid crafting command."),
        (
            GET_ITEM,
            {"reasoning": [], "item_name": "bamboo", "num_total_items_needed": True},
            "Invalid call to textcraft_get_item: num_total_items_needed must be an integer of at least 1.",
        ),
        (
            GET_ITEM,
            {"reasoning": "get bamboo", "item_name": "bamboo", "num_total_items_needed": 1},
            "Invalid call to textcraft_get_item: reasoning must be a list of strings.",
        ),
    ],
)
async def test_invalid_calls_count_as_tool_calls(
    name: str, arguments: dict[str, Any], observation: str
) -> None:
    env = _env("dark_oak_sign")

    result = await env.call(name, arguments)

    assert result.startswith(observation)
    assert env.tool_calls == 1
    assert [(step.step, step.tool, step.arguments, step.observation, step.state) for step in env.steps] == [
        (1, name, arguments, result, {})
    ]


async def test_a_tool_outside_the_offered_set_is_refused_and_counted() -> None:
    env = _env("polished_granite")
    gatherer = {GET_ITEM, INVENTORY}
    for item in ("quartz", "cobblestone"):
        await env.call(GET_ITEM, {"reasoning": [], "item_name": item, "num_total_items_needed": 2}, gatherer)
    held = env.inventory
    arguments = {"reasoning": [], "crafting_command": "craft 2 diorite using 2 quartz, 2 cobblestone"}

    observation = await env.call(CRAFT, arguments, gatherer)

    assert observation == (
        "Error: tool 'textcraft_craft' is not available to this executor. "
        "Available tools: textcraft_get_item, textcraft_inventory."
    )
    assert env.tool_calls == 3
    assert env.inventory == held == {"quartz": 2, "cobblestone": 2}
    step = env.steps[-1]
    assert (step.step, step.tool, step.arguments, step.observation, step.state) == (
        3,
        CRAFT,
        arguments,
        observation,
        held,
    )
    assert await env.call(CRAFT, arguments, {CRAFT, INVENTORY}) == "Crafted 2 minecraft:diorite"


async def test_steps_record_the_inventory_after_each_call() -> None:
    env = _env("dark_oak_sign")
    assert env.initial_state() == {}

    await _run(env, "get 1 dark oak logs")
    await env.call(INVENTORY, {})
    await _run(env, "craft 4 dark oak planks using 1 dark oak logs")

    assert [step.state for step in env.steps] == [
        {"dark_oak_logs": 1},
        {"dark_oak_logs": 1},
        {"dark_oak_planks": 4},
    ]
    assert env.steps[1].observation == "Inventory: [dark oak logs] (1) "
    assert env.tool_calls == 3


async def test_inventory_lists_items_in_the_order_they_entered_it() -> None:
    env = _env("dark_oak_sign")
    assert await env.call(INVENTORY, {}) == "Inventory: You are not carrying anything."

    await _run(env, "get 1 dark oak logs")
    await _run(env, "get 2 bamboo")
    assert await env.call(INVENTORY, {}) == "Inventory: [dark oak logs] (1) [bamboo] (2) "

    await _run(env, "craft 4 dark oak planks using 1 dark oak logs")
    await _run(env, "get 1 dark oak logs")
    assert (
        await _run(env, "inventory") == "Inventory: [bamboo] (2) [dark oak planks] (4) [dark oak logs] (1) "
    )


@pytest.mark.parametrize("goal", ["dark_oak_sign", "painting", "polished_granite", "lectern"])
async def test_tool_enums_match_the_instance(goal: str) -> None:
    book = load_recipe_book()
    env = _env(goal)
    shown = _parse_objective(env.objective)
    get_spec, craft_spec, select_spec = env.tool_specs([GET_ITEM, CRAFT, SELECT_COMMAND])
    items = get_spec.parameters["properties"]["item_name"]["enum"]
    commands = craft_spec.parameters["properties"]["crafting_command"]["enum"]
    pattern = select_spec.parameters["properties"]["command"]["pattern"]
    named = set(shown.recipes) | {name for s in shown.recipes.values() for r in s for _, name in r.inputs}
    members = {member for category in shown.categories.values() for member in category}

    assert commands == sorted(shown.lines)
    for command in commands:
        observation = await env.call(CRAFT, {"reasoning": [], "crafting_command": command})
        output = command.split(" using ")[0].split(" ", 2)[2]
        assert observation == f"Could not find enough items to craft minecraft:{output.replace(' ', '_')}"

    assert items == sorted(named)
    assert all(re.fullmatch(pattern, f"get 12 {name}") for name in named)
    assert not any(re.fullmatch(pattern, f"get 1 {name}") for name in members - named)
    for name in sorted(named | members):
        observation = await env.call(
            GET_ITEM, {"reasoning": [], "item_name": name, "num_total_items_needed": 1}
        )
        known = book.lookup(name)
        raw = known is not None and not known.is_tag and not book.is_craftable(known.id)
        assert observation == (f"Got 1 {name}" if raw else f"Could not find {name}")
