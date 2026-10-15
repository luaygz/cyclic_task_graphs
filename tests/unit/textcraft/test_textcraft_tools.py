import re
from typing import Any

import pytest

from ctg.envs.textcraft.env import TextCraftEnv
from ctg.envs.textcraft.recipes import load_recipe_book
from ctg.envs.textcraft.tools import CRAFT, GET_ITEM, INVENTORY, SELECT_COMMAND, TOOL_NAMES

pytestmark = pytest.mark.textcraft_recipes

_KEYWORDS = {
    "object": {"type", "description", "properties", "required", "additionalProperties"},
    "array": {"type", "description", "items"},
    "string": {"type", "description", "enum", "pattern"},
    "integer": {"type", "description", "minimum"},
}


def _assert_strict(schema: dict[str, Any], path: str) -> None:
    """OpenAI strict mode: closed objects, every property required, supported keywords only."""
    kind = schema["type"]
    assert set(schema) <= _KEYWORDS[kind], path
    if kind == "object":
        assert schema["additionalProperties"] is False, path
        assert schema["required"] == list(schema["properties"]), path
        for key, value in schema["properties"].items():
            _assert_strict(value, f"{path}.{key}")
    elif kind == "array":
        _assert_strict(schema["items"], f"{path}[]")
    elif kind == "string" and "enum" in schema:
        enum = schema["enum"]
        assert enum and all(isinstance(value, str) for value in enum) and len(set(enum)) == len(enum), path
    elif kind == "string" and "pattern" in schema:
        re.compile(schema["pattern"])


@pytest.mark.parametrize("goal", ["dark_oak_sign", "painting", "hopper_minecart"])
def test_tool_schemas_are_strict(goal: str) -> None:
    specs = TextCraftEnv(load_recipe_book(), goal).tool_specs(TOOL_NAMES)

    assert [spec.name for spec in specs] == list(TOOL_NAMES)
    for spec in specs:
        _assert_strict(spec.parameters, spec.name)
    parameters = {spec.name: list(spec.parameters["properties"]) for spec in specs}
    assert parameters == {
        INVENTORY: [],
        GET_ITEM: ["reasoning", "item_name", "num_total_items_needed"],
        CRAFT: ["reasoning", "crafting_command"],
        SELECT_COMMAND: ["reasoning", "command"],
    }


def test_command_pattern_admits_every_command_form_and_only_named_get_targets() -> None:
    env = TextCraftEnv(load_recipe_book(), "lectern")
    pattern = env.tool_specs([SELECT_COMMAND])[0].parameters["properties"]["command"]["pattern"]
    accepted = [
        "inventory",
        "get 4 oak logs",
        "get 1 lectern",
        "get 2 planks",
        *(recipe.command for recipe in env.task.recipes),
    ]
    rejected = [
        "Inventory",
        "get oak logs",
        "get 4 bamboo",
        "get 4 oak logs please",
        "craft oak planks",
        "craft 4 oak planks",
        "look around",
        "",
    ]

    assert all(re.fullmatch(pattern, command) for command in accepted)
    assert not any(re.fullmatch(pattern, command) for command in rejected)
