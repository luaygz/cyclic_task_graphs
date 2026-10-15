"""TextCraft tools (paper Appendix B.1.1): strict schemas and validation of call arguments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from ctg.llm.base import ToolSpec

INVENTORY = "textcraft_inventory"
GET_ITEM = "textcraft_get_item"
CRAFT = "textcraft_craft"
SELECT_COMMAND = "textcraft_select_command"
TOOL_NAMES = (INVENTORY, GET_ITEM, CRAFT, SELECT_COMMAND)


@dataclass(frozen=True)
class Inventory:
    pass


@dataclass(frozen=True)
class GetItem:
    item_name: str
    count: int


@dataclass(frozen=True)
class Craft:
    command: str


@dataclass(frozen=True)
class SelectCommand:
    command: str


Action = Inventory | GetItem | Craft | SelectCommand


class ToolCallError(ValueError):
    """A call that names an unknown tool or has invalid arguments. The message is the observation."""


def inventory_spec() -> ToolSpec:
    return ToolSpec(
        name=INVENTORY,
        description=(
            "Check the current contents of the agent's inventory. "
            "Returns every item in the inventory with its quantity."
        ),
        parameters=_object({}),
    )


def get_item_spec(items: Sequence[str]) -> ToolSpec:
    """`items`: display names of every item and category the instance's crafting commands name."""
    return ToolSpec(
        name=GET_ITEM,
        description=(
            "Acquire items from the environment. Not all items can be obtained directly: some must be "
            "crafted. Returns the result of the acquisition attempt."
        ),
        parameters=_object(
            {
                "reasoning": _reasoning("which item to get"),
                "item_name": {
                    "type": "string",
                    "enum": list(items),
                    "description": "The name of the item to acquire, constrained to obtainable items.",
                },
                "num_total_items_needed": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "The quantity of items to acquire (at least 1).",
                },
            }
        ),
    )


def craft_spec(commands: Sequence[str]) -> ToolSpec:
    """`commands`: the instance's crafting commands."""
    return ToolSpec(
        name=CRAFT,
        description=(
            "Craft items using materials in the inventory according to available recipes. Returns the "
            "result of the crafting action, including success or failure with an explanation."
        ),
        parameters=_object(
            {
                "reasoning": _reasoning("which item to craft"),
                "crafting_command": {
                    "type": "string",
                    "enum": list(commands),
                    "description": (
                        'The crafting command in the format "craft N item using M ingredient1, ...", '
                        "constrained to valid recipes."
                    ),
                },
            }
        ),
    )


def select_command_spec(items: Sequence[str]) -> ToolSpec:
    """`items`: the `get` targets, as for `get_item_spec`."""
    return ToolSpec(
        name=SELECT_COMMAND,
        description=(
            "A unified command interface combining all TextCraft operations. "
            "Returns the observation after executing the command."
        ),
        parameters=_object(
            {
                "reasoning": _reasoning("which command to execute"),
                "command": {
                    "type": "string",
                    "pattern": command_pattern(items),
                    "description": (
                        "One of: `get <number> <item>` to acquire items from the environment; "
                        "`craft <crafting_command>` to craft items using inventory materials, where the "
                        "command is one of the listed crafting commands, e.g. "
                        "`craft 4 oak planks using 1 oak logs`; `inventory` to check inventory contents."
                    ),
                },
            }
        ),
    )


def command_pattern(items: Sequence[str]) -> str:
    """The `command` pattern of textcraft_select_command: `inventory`, `get <number> <item>` for one of
    `items`, or `craft <number> <item> using <number> <ingredient>, ...`.

    Item names are Minecraft identifiers with spaces for underscores (`[a-z0-9 ]`), so they need no escaping.
    """
    return rf"^(inventory|get [0-9]+ ({'|'.join(items)})|craft [0-9]+ .+ using [0-9]+ .+)$"


def parse_call(name: str, arguments: Mapping[str, Any]) -> Action:
    """Validate a call against its schema. Raises ToolCallError with the observation to return."""
    if name == INVENTORY:
        _check_keys(name, arguments, ())
        return Inventory()
    if name == GET_ITEM:
        _check_keys(name, arguments, ("reasoning", "item_name", "num_total_items_needed"))
        count = arguments["num_total_items_needed"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ToolCallError(
                f"Invalid call to {name}: num_total_items_needed must be an integer of at least 1."
            )
        return GetItem(_string(name, arguments, "item_name"), count)
    if name == CRAFT:
        _check_keys(name, arguments, ("reasoning", "crafting_command"))
        return Craft(_string(name, arguments, "crafting_command"))
    if name == SELECT_COMMAND:
        _check_keys(name, arguments, ("reasoning", "command"))
        return SelectCommand(_string(name, arguments, "command"))
    raise ToolCallError(f"Unknown tool {name}. Available tools: {', '.join(TOOL_NAMES)}.")


def _object(properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _reasoning(subject: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string"},
        "description": f"Step-by-step reasoning assessing {subject}.",
    }


def _check_keys(name: str, arguments: Mapping[str, Any], expected: Sequence[str]) -> None:
    missing = [key for key in expected if key not in arguments]
    unexpected = [key for key in arguments if key not in expected]
    if missing or unexpected:
        problems = [f"missing {key}" for key in missing] + [f"unexpected {key}" for key in unexpected]
        raise ToolCallError(f"Invalid call to {name}: {', '.join(problems)}.")
    if "reasoning" in expected:
        reasoning = arguments["reasoning"]
        if not isinstance(reasoning, list) or not all(
            isinstance(step, str) for step in cast(list[Any], reasoning)
        ):
            raise ToolCallError(f"Invalid call to {name}: reasoning must be a list of strings.")


def _string(name: str, arguments: Mapping[str, Any], key: str) -> str:
    value = arguments[key]
    if not isinstance(value, str):
        raise ToolCallError(f"Invalid call to {name}: {key} must be a string.")
    return value
