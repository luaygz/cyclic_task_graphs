"""A TextCraft episode: the instance's crafting commands, the inventory and the command interpreter.

Observations of `get`, `craft` and `inventory` are ADaPT's (TextCraft env.py), word for word, including the
`minecraft:` ids ADaPT prints for crafts.
"""

from __future__ import annotations

import random
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ctg.envs.base import Environment
from ctg.envs.textcraft import tools
from ctg.envs.textcraft.recipes import Ingredient, Recipe, RecipeBook, display_name, normalize_name
from ctg.llm.base import ToolSpec
from ctg.runs.records import StepRecord

MAX_DISTRACTORS = 10

_GET = re.compile(r"get (\d+) (.+)")
_CRAFT = re.compile(r"craft (\d+) (.+?) using (.+)")
_AMOUNT = re.compile(r"(\d+) (.+)")

_GET_USAGE = "Invalid get command. Use: get <number> <item>."
_CRAFT_USAGE = "Invalid crafting command. Use: craft <number> <item> using <number> <ingredient>, ..."
_COMMAND_USAGE = "Unknown command. Use: get <number> <item>, craft <crafting command>, or inventory."


@dataclass(frozen=True)
class TextCraftTask:
    """What an instance shows: its crafting commands (in display order) and its goal."""

    goal: str
    recipes: tuple[Recipe, ...]
    items: tuple[str, ...]
    """Display names of every item and category the crafting commands name (outputs and ingredients,
    distractors included), sorted: the `get` targets the tool schemas offer."""
    objective: str


def build_task(book: RecipeBook, goal: str) -> TextCraftTask:
    """Every recipe needed for `goal` plus up to 10 distractors, chosen and shuffled from the goal id.

    Distractors are recipes for other items that use an ingredient of the needed recipes.
    """
    if not book.is_craftable(goal):
        raise ValueError(f"{goal!r} is not a craftable TextCraft item")
    needed = book.required_recipes(goal)
    needed_set = set(needed)
    used = {ingredient for recipe in needed for ingredient, _ in recipe.inputs}
    candidates = [
        recipe
        for recipe in book.recipes
        if recipe not in needed_set and any(ingredient in used for ingredient, _ in recipe.inputs)
    ]
    rng = random.Random(goal)
    recipes = [*needed, *rng.sample(candidates, min(MAX_DISTRACTORS, len(candidates)))]
    rng.shuffle(recipes)

    ingredients = {ingredient for recipe in recipes for ingredient, _ in recipe.inputs}
    categories = sorted((i.id for i in ingredients if i.is_tag), key=display_name)
    items = {display_name(recipe.output) for recipe in recipes} | {i.name for i in ingredients}

    lines = ["Crafting commands:", *(recipe.command for recipe in recipes)]
    if categories:
        lines += ["", "Ingredient categories (any item of a category can be used for it):"]
        lines += [
            f"{display_name(tag)}: {', '.join(sorted(display_name(m) for m in book.members(tag)))}"
            for tag in categories
        ]
    lines += ["", f"Goal: craft {display_name(goal)}."]
    return TextCraftTask(goal, tuple(recipes), tuple(sorted(items)), "\n".join(lines))


class TextCraftEnv(Environment):
    """One TextCraft episode. The inventory starts empty; the episode is won when it holds the goal.

    `get` accepts any item of the recipe book that no recipe makes; `craft` runs the instance's listed
    crafting commands.
    """

    def __init__(self, book: RecipeBook, goal: str) -> None:
        self._book = book
        self._task = build_task(book, goal)
        self._recipes_by_name: dict[str, list[Recipe]] = {}
        for recipe in self._task.recipes:
            self._recipes_by_name.setdefault(display_name(recipe.output), []).append(recipe)
        self._inventory: dict[str, int] = {}
        """Item id -> quantity (positive), in the order the items entered the inventory."""
        self._steps: list[StepRecord] = []

    @property
    def task(self) -> TextCraftTask:
        return self._task

    @property
    def objective(self) -> str:
        return self._task.objective

    @property
    def won(self) -> bool:
        return self._inventory.get(self._task.goal, 0) > 0

    @property
    def done(self) -> bool:
        return self.won

    @property
    def tool_calls(self) -> int:
        return len(self._steps)

    @property
    def steps(self) -> Sequence[StepRecord]:
        return self._steps

    @property
    def inventory(self) -> dict[str, int]:
        """Item id -> quantity, sorted by id."""
        return dict(sorted(self._inventory.items()))

    def initial_state(self) -> dict[str, Any]:
        return {}

    def tool_specs(self, names: Sequence[str]) -> list[ToolSpec]:
        return [self._tool_spec(name) for name in names]

    async def call(
        self, name: str, arguments: Mapping[str, Any], offered: Collection[str] | None = None
    ) -> str:
        if self.done:
            raise RuntimeError("the TextCraft episode is over")
        if offered is not None and name not in offered:
            observation = (
                f"Error: tool {name!r} is not available to this executor. "
                f"Available tools: {', '.join(sorted(offered))}."
            )
        else:
            try:
                observation = self._perform(tools.parse_call(name, arguments))
            except tools.ToolCallError as error:
                observation = str(error)
        self._steps.append(
            StepRecord(
                step=len(self._steps) + 1,
                tool=name,
                arguments=dict(arguments),
                observation=observation,
                state=self.inventory,
            )
        )
        return observation

    def details(self) -> dict[str, Any]:
        return {"goal": self._task.goal, "inventory": self.inventory}

    def _tool_spec(self, name: str) -> ToolSpec:
        if name == tools.INVENTORY:
            return tools.inventory_spec()
        if name == tools.GET_ITEM:
            return tools.get_item_spec(self._task.items)
        if name == tools.CRAFT:
            return tools.craft_spec(sorted(recipe.command for recipe in self._task.recipes))
        if name == tools.SELECT_COMMAND:
            return tools.select_command_spec(self._task.items)
        raise ValueError(f"unknown TextCraft tool {name!r}")

    def _perform(self, action: tools.Action) -> str:
        match action:
            case tools.Inventory():
                return self._describe_inventory()
            case tools.GetItem(item_name, count):
                return self._get(count, normalize_name(item_name))
            case tools.Craft(command):
                return self._craft(_normalize_command(command))
            case tools.SelectCommand(command):
                return self._execute(_normalize_command(command))

    def _execute(self, command: str) -> str:
        if command == "inventory":
            return self._describe_inventory()
        if command == "get" or command.startswith("get "):
            match = _GET.fullmatch(command)
            return self._get(int(match[1]), match[2]) if match else _GET_USAGE
        if command == "craft" or command.startswith("craft "):
            return self._craft(command)
        return _COMMAND_USAGE

    def _describe_inventory(self) -> str:
        if not self._inventory:
            return "Inventory: You are not carrying anything."
        return "Inventory: " + "".join(
            f"[{display_name(item)}] ({n}) " for item, n in self._inventory.items()
        )

    def _get(self, count: int, name: str) -> str:
        if count < 1:
            return f"Could not get {name}: the number of items must be at least 1."
        known = self._book.lookup(name)
        if known is None or known.is_tag or self._book.is_craftable(known.id):
            return f"Could not find {name}"
        self._inventory[known.id] = self._inventory.get(known.id, 0) + count
        return f"Got {count} {name}"

    def _craft(self, command: str) -> str:
        """ADaPT's order: the written ingredients must be held before the command is matched to a recipe."""
        parsed = _parse_craft(command)
        if parsed is None:
            return _CRAFT_USAGE
        count, name, amounts = parsed
        tag, item_id = self._adapt_item_tag(name)
        if any(self._held(ingredient) < n for n, ingredient in amounts):
            return f"Could not find enough items to craft {item_id}"
        for recipe in self._recipes_by_name.get(name, []):
            sources = self._match(recipe, count, amounts)
            if sources is not None:
                break
        else:
            return (
                "Could not find a valid recipe for "
                f"ItemTagWithCount(item_tag=ItemTag(tag={tag!r}, item_id={item_id!r}), count={count})"
            )
        remaining = self._consume(sources)
        if remaining is None:
            return f"Could not find enough items to craft {item_id}"
        remaining[recipe.output] = remaining.get(recipe.output, 0) + recipe.count
        self._inventory = remaining
        return f"Crafted {recipe.count} minecraft:{recipe.output}"

    def _adapt_item_tag(self, name: str) -> tuple[str | None, str | None]:
        """ADaPT's (tag, item_id) for a written item name: its `minecraft:` id, as the tag for a category."""
        identifier = "minecraft:" + name.replace(" ", "_")
        known = self._book.lookup(name)
        return (identifier, None) if known is not None and known.is_tag else (None, identifier)

    def _held(self, name: str) -> int:
        """How many of the written item, or of a written category's members, the inventory holds."""
        known = self._book.lookup(name)
        if known is None:
            return 0
        if known.is_tag:
            return sum(self._inventory.get(member, 0) for member in self._book.members(known.id))
        return self._inventory.get(known.id, 0)

    def _match(
        self, recipe: Recipe, count: int, amounts: Sequence[tuple[int, str]]
    ) -> list[tuple[Ingredient, int]] | None:
        """Where each ingredient of `recipe` comes from, if the command states this recipe.

        A tag ingredient may be written as the tag (any member is used) or as one of its members.
        """
        if count != recipe.count or len(amounts) != len(recipe.inputs):
            return None
        unmatched = list(amounts)
        sources: list[tuple[Ingredient, int] | None] = []
        for ingredient, n in recipe.inputs:
            source = None
            if (n, ingredient.name) in unmatched:
                unmatched.remove((n, ingredient.name))
                source = (ingredient, n)
            sources.append(source)
        for index, (ingredient, n) in enumerate(recipe.inputs):
            if sources[index] is not None or not ingredient.is_tag:
                continue
            for amount in unmatched:
                member = self._book.lookup(amount[1])
                if amount[0] == n and member is not None and member.id in self._book.members(ingredient.id):
                    unmatched.remove(amount)
                    sources[index] = (member, n)
                    break
        if unmatched:
            return None
        return [source for source in sources if source is not None]

    def _consume(self, sources: Sequence[tuple[Ingredient, int]]) -> dict[str, int] | None:
        """The inventory after removing `sources` (specific items first, then tags from any member), or
        None if it does not hold them all."""
        inventory = dict(self._inventory)
        for source, n in sorted(sources, key=lambda entry: entry[0].is_tag):
            pool = sorted(self._book.members(source.id)) if source.is_tag else [source.id]
            needed = n
            for item in pool:
                taken = min(needed, inventory.get(item, 0))
                needed -= taken
                inventory[item] = inventory.get(item, 0) - taken
            if needed:
                return None
        return {item: n for item, n in inventory.items() if n > 0}


def _normalize_command(command: str) -> str:
    return normalize_name(command).rstrip(".").rstrip()


def _parse_craft(command: str) -> tuple[int, str, list[tuple[int, str]]] | None:
    match = _CRAFT.fullmatch(command)
    if match is None:
        return None
    amounts: list[tuple[int, str]] = []
    for part in match[3].split(","):
        amount = _AMOUNT.fullmatch(part.strip())
        if amount is None:
            return None
        amounts.append((int(amount[1]), amount[2]))
    return int(match[1]), match[2], amounts
