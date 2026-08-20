"""Deterministic TextCraft oracle used only for environment validation."""

from __future__ import annotations

import math

from luna.benchmarks.textcraft.environment import TextCraftEnvironment
from luna.benchmarks.textcraft.utils import ItemTagWithCount, Recipe, item_id_to_str


def solve_actions(environment: TextCraftEnvironment) -> list[str]:
    actions: list[str] = []
    _plan_item(environment, environment.goal, 1, actions, set())
    return actions


def _plan_item(
    environment: TextCraftEnvironment,
    item_name: str,
    amount: int,
    actions: list[str],
    visiting: set[str],
) -> None:
    tree = environment.crafting_tree
    if item_name in visiting:
        raise ValueError(f"cycle while solving {item_name}")
    recipes = tree.itemid_recipes.get(item_name) or tree.tag_recipes.get(item_name)
    if not recipes:
        actions.append(f"get {amount} {item_id_to_str(item_name)}")
        return
    recipe: Recipe = min(
        recipes,
        key=lambda candidate: max(
            (tree.get_min_depth(item.item_tag.name) for item in candidate.input_items), default=0
        ),
    )
    batches = math.ceil(amount / recipe.output_item.count)
    concrete_inputs: list[ItemTagWithCount] = []
    next_visiting = visiting | {item_name}
    for item in recipe.input_items:
        concrete_name = _concrete_item(environment, item.item_tag.name)
        required = item.count * batches
        _plan_item(environment, concrete_name, required, actions, next_visiting)
        concrete_tag = environment._item_str_to_obj(item_id_to_str(concrete_name))
        concrete_inputs.append(ItemTagWithCount(concrete_tag, required))
    output_count = recipe.output_item.count * batches
    inputs = ", ".join(
        f"{item.count} {item_id_to_str(item.item_tag.item_id)}" for item in concrete_inputs
    )
    actions.append(f"craft {output_count} {item_id_to_str(recipe.output_item.item_tag.item_id)} using {inputs}")


def _concrete_item(environment: TextCraftEnvironment, name: str) -> str:
    tree = environment.crafting_tree
    if name in tree.itemid_recipes or name in tree.itemid_set:
        return name
    recipes = tree.tag_recipes.get(name, [])
    if recipes:
        return min(
            recipes, key=lambda recipe: tree.get_min_depth(recipe.output_item.item_tag.item_id)
        ).output_item.item_tag.item_id
    candidates = sorted(tree.get_items_with_tags(name))
    if not candidates:
        return name
    return min(candidates, key=tree.get_min_depth)

