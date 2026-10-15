"""ALFWorld tools (Appendix B.1.2). Each tool's `command` enum holds the commands admissible right now.

`alfworld_select_command` offers every admissible command. The three partitioned tools split the same set by
command verb; `examine` belongs to inventory management for a held object and to exploration otherwise.
`help` is never offered.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass

from ctg.llm.base import ToolSpec

SELECT_COMMAND = "alfworld_select_command"
MANAGE_INVENTORY = "alfworld_manage_inventory"
EXPLORE = "alfworld_explore"
MANIPULATE_OBJECT = "alfworld_manipulate_object"
PARTITIONED_TOOLS = (MANAGE_INVENTORY, EXPLORE, MANIPULATE_OBJECT)
TOOL_NAMES = (SELECT_COMMAND, *PARTITIONED_TOOLS)

_HIDDEN_COMMANDS = frozenset({"help"})

_VERB_TOOLS = {
    "inventory": MANAGE_INVENTORY,
    "take": MANAGE_INVENTORY,
    "move": MANAGE_INVENTORY,
    "look": EXPLORE,
    "go": EXPLORE,
    "open": EXPLORE,
    "close": EXPLORE,
    "use": MANIPULATE_OBJECT,
    "heat": MANIPULATE_OBJECT,
    "cool": MANIPULATE_OBJECT,
    "clean": MANIPULATE_OBJECT,
    "slice": MANIPULATE_OBJECT,
}


@dataclass(frozen=True)
class _ToolText:
    description: str
    reasoning: str
    command: str


_TOOL_TEXT = {
    SELECT_COMMAND: _ToolText(
        description="Select and execute a command from the list of currently admissible commands.",
        reasoning="Step-by-step reasoning assessing which command to execute.",
        command="The command to execute, constrained to the set of admissible commands.",
    ),
    MANAGE_INVENTORY: _ToolText(
        description=(
            "Manage the agent's inventory by checking contents, taking items from receptacles, "
            "or placing items."
        ),
        reasoning="Step-by-step reasoning assessing which inventory command to execute.",
        command=(
            "One of: `inventory` (check current inventory contents); "
            "`take <object> from <receptacle>` (take an item from a receptacle); "
            "`move <object> to <receptacle>` (place the held item in or on a receptacle); "
            "`examine <inventory_item>` (examine the item in inventory). "
            "Constrained to the commands admissible in the current state."
        ),
    ),
    EXPLORE: _ToolText(
        description="Navigate and explore the environment to discover objects and locations.",
        reasoning="Step-by-step reasoning about which navigational command to execute.",
        command=(
            "One of: `look` (look around the current location); "
            "`go to <receptacle>` (move to a specific piece of furniture or receptacle); "
            "`examine <receptacle>` (get detailed information about a receptacle or an object in view); "
            "`open <receptacle>` (open a closed receptacle); "
            "`close <receptacle>` (close an open receptacle). "
            "Constrained to the commands admissible in the current state."
        ),
    ),
    MANIPULATE_OBJECT: _ToolText(
        description="Perform physical manipulations on objects using tools or appliances.",
        reasoning="Step-by-step reasoning assessing which manipulation to perform.",
        command=(
            "One of: `use <object>` (use an object here, e.g. turn on a desk lamp); "
            "`heat <object> with <appliance>` (heat the held item, e.g. with a microwave); "
            "`cool <object> with <appliance>` (cool the held item, e.g. with a fridge); "
            "`clean <object> with <receptacle>` (clean the held item, e.g. in a sink basin); "
            "`slice <object> with <tool>` (cut an item with a held tool, e.g. a knife). "
            "Constrained to the commands admissible in the current state."
        ),
    ),
}


def offered_commands(admissible: Iterable[str]) -> list[str]:
    """The admissible commands an agent may issue (everything but `help`), sorted."""
    return sorted(command for command in admissible if command not in _HIDDEN_COMMANDS)


def partition_tool(command: str, held: Collection[str]) -> str:
    """The partitioned tool that offers `command`. Raises ValueError for a verb ALFWorld does not emit."""
    verb, _, target = command.partition(" ")
    if verb == "examine":
        return MANAGE_INVENTORY if target in held else EXPLORE
    try:
        return _VERB_TOOLS[verb]
    except KeyError:
        raise ValueError(f"unrecognized ALFWorld command: {command!r}") from None


def command_sets(admissible: Iterable[str], held: Collection[str]) -> dict[str, tuple[str, ...]]:
    """Commands each tool offers in a state, keyed by tool name (empty tuple: nothing to offer)."""
    offered = offered_commands(admissible)
    partitions: dict[str, list[str]] = {name: [] for name in PARTITIONED_TOOLS}
    for command in offered:
        partitions[partition_tool(command, held)].append(command)
    return {SELECT_COMMAND: tuple(offered)} | {name: tuple(commands) for name, commands in partitions.items()}


def tool_spec(name: str, commands: Collection[str]) -> ToolSpec:
    """The strict function-tool spec for `name` with `commands` as its enum (must be non-empty)."""
    if not commands:
        raise ValueError(f"{name} has no commands to offer; omit the tool instead")
    text = _TOOL_TEXT[name]
    return ToolSpec(
        name=name,
        description=text.description,
        parameters={
            "type": "object",
            "properties": {
                "reasoning": {"type": "array", "items": {"type": "string"}, "description": text.reasoning},
                "command": {"type": "string", "enum": list(commands), "description": text.command},
            },
            "required": ["reasoning", "command"],
            "additionalProperties": False,
        },
    )
