"""TextCraft environment adapted from the MIT-licensed ADaPT release."""

from __future__ import annotations

import random
import re
from itertools import product
from importlib import resources
from pathlib import Path

from luna.benchmarks.textcraft.crafting_tree import CraftingTree
from luna.benchmarks.textcraft.utils import (
    ActionFailed,
    ItemTag,
    ItemTagWithCount,
    Recipe,
    item_id_to_str,
)


class TextCraftEnvironment:
    def __init__(self, data_dir: Path | None = None):
        if data_dir is None:
            resource = resources.files("luna.benchmarks.textcraft") / "data"
            with resources.as_file(resource) as resolved:
                self.crafting_tree = CraftingTree(str(resolved))
        else:
            self.crafting_tree = CraftingTree(str(data_dir))
        self.inventory: dict[str, int] = {}
        self.goal = ""
        self.recipe_commands: list[str] = []

    def expanded_recipe_commands(self) -> list[str]:
        """Expand tag ingredients to concrete variants used elsewhere in the task."""
        mentioned = self._mentioned_items(self.recipe_commands)
        expanded: list[str] = []
        for command in self.recipe_commands:
            if not command.startswith("craft ") or " using " not in command:
                expanded.append(command)
                continue
            output, ingredient_text = command.split(" using ", 1)
            choices: list[list[str]] = []
            for ingredient in ingredient_text.split(", "):
                match = re.fullmatch(r"([0-9]+) (.*)", ingredient.strip())
                if match is None:
                    choices.append([ingredient.strip()])
                    continue
                count, display_name = match.groups()
                item_id = "minecraft:" + display_name.replace(" ", "_")
                variants = [display_name]
                if self.crafting_tree.is_tag(item_id):
                    concrete = sorted(
                        item.replace("minecraft:", "").replace("_", " ")
                        for item in self.crafting_tree.get_items_with_tags(item_id)
                        if item.replace("minecraft:", "").replace("_", " ") in mentioned
                    )
                    if concrete:
                        variants = concrete
                choices.append([f"{count} {variant}" for variant in variants])
            expanded.extend(
                f"{output} using {', '.join(combination)}"
                for combination in product(*choices)
            )
        return sorted(set(expanded))

    def reset(self, seed: int, min_depth: int, max_depth: int) -> str:
        self.inventory = {}
        rng = random.Random(seed)
        candidates = list(self.crafting_tree.item_recipes_min_depth(min_depth, max_depth))
        if not candidates:
            raise ValueError(f"no TextCraft recipes exist at depth {min_depth}..{max_depth}")
        self.goal, _ = sorted(candidates, key=lambda item: -item[1])[seed % len(candidates)]
        recipes, distractors = self.crafting_tree.create_recipe_set(self.goal)
        gold = {recipe.recipe_str for recipe in recipes}
        distractor_commands = sorted(
            {recipe.recipe_str for recipe in distractors if recipe.recipe_str not in gold}
        )
        commands = sorted(gold) + rng.sample(distractor_commands, min(10, len(distractor_commands)))
        rng.shuffle(commands)
        self.recipe_commands = commands
        return f"Crafting commands:\n{chr(10).join(commands)}\n\nGoal: craft {item_id_to_str(self.goal)}."

    def step(self, action: str) -> tuple[str, bool]:
        try:
            if action == "inventory":
                if not self.inventory:
                    return "Inventory: You are not carrying anything.", False
                contents = " ".join(
                    f"[{item_id_to_str(item)}] ({amount})" for item, amount in self.inventory.items()
                )
                return f"Inventory: {contents}", False
            craft_match = re.fullmatch(r"craft (.*) using (.*)", action)
            if craft_match:
                recipe = self._extract_recipe(craft_match.group(1), craft_match.group(2))
                if not self._has_items(recipe.input_items):
                    raise ActionFailed(f"Could not find enough items to craft {recipe.output_item.item_tag.name}")
                result = self.crafting_tree.craft(recipe)
                if result is None:
                    raise ActionFailed(f"Could not find a valid recipe for {recipe.output_item.item_tag.name}")
                self._remove_items(recipe.input_items)
                self._add_item(result.item_tag, result.count)
                won = result.item_tag.item_id == self.goal
                return f"Crafted {result.count} {item_id_to_str(result.item_tag.item_id)}", won
            get_match = re.fullmatch(r"get ([0-9]+) (.*)", action)
            if get_match:
                amount, item_text = int(get_match.group(1)), get_match.group(2)
                item = self._item_str_to_obj(item_text)
                if (
                    self.crafting_tree.is_craftable(item.name)
                    or self.crafting_tree.is_tag(item.item_id)
                    or item.item_id is None
                    or not self.crafting_tree.is_valid_item(item.item_id)
                ):
                    raise ActionFailed(f"Could not find {item_text}")
                self._add_item(item, amount)
                return f"Got {amount} {item_text}", False
            raise ActionFailed(f"Could not execute {action}")
        except ActionFailed as exc:
            return str(exc), False

    def _extract_recipe(self, output_text: str, ingredients_text: str) -> Recipe:
        output_match = re.fullmatch(r"([0-9]+) (.*)", output_text)
        if output_match:
            output = ItemTagWithCount(self._item_str_to_obj(output_match.group(2)), int(output_match.group(1)))
        else:
            output = ItemTagWithCount(self._item_str_to_obj(output_text), 1)
        inputs: list[ItemTagWithCount] = []
        for ingredient in ingredients_text.split(","):
            match = re.fullmatch(r"([0-9]+) (.*)", ingredient.strip())
            if match is None:
                raise ActionFailed(f"Wrong item format: {ingredient.strip()}")
            inputs.append(ItemTagWithCount(self._item_str_to_obj(match.group(2)), int(match.group(1))))
        return Recipe(input_items=inputs, output_item=output)

    @staticmethod
    def _mentioned_items(commands: list[str]) -> set[str]:
        mentioned: set[str] = set()
        for command in commands:
            if not command.startswith("craft ") or " using " not in command:
                continue
            output, ingredients = command.split(" using ", 1)
            output_words = output.split()
            if len(output_words) >= 3:
                mentioned.add(" ".join(output_words[2:]))
            for ingredient in ingredients.split(", "):
                words = ingredient.strip().split()
                if len(words) >= 2:
                    mentioned.add(" ".join(words[1:]))
        return mentioned

    def _item_str_to_obj(self, item: str) -> ItemTag:
        item_id = "minecraft:" + item.replace(" ", "_")
        return ItemTag(tag=item_id) if self.crafting_tree.is_tag(item_id) else ItemTag(item_id=item_id)

    def _has_items(self, items: list[ItemTagWithCount]) -> bool:
        return all(
            item.item_tag.item_id in self.inventory
            and self.inventory[item.item_tag.item_id] >= item.count
            for item in items
        )

    def _add_item(self, item: ItemTag, amount: int) -> None:
        assert item.item_id is not None
        self.inventory[item.item_id] = self.inventory.get(item.item_id, 0) + amount

    def _remove_items(self, items: list[ItemTagWithCount]) -> None:
        for item in items:
            assert item.item_tag.item_id is not None
            self.inventory[item.item_tag.item_id] -= item.count
            if self.inventory[item.item_tag.item_id] == 0:
                del self.inventory[item.item_tag.item_id]
