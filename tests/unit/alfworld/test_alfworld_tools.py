import pytest

from ctg.envs.alfworld.tools import (
    EXPLORE,
    MANAGE_INVENTORY,
    MANIPULATE_OBJECT,
    PARTITIONED_TOOLS,
    SELECT_COMMAND,
    command_sets,
    partition_tool,
    tool_spec,
)

ADMISSIBLE = (
    "clean mug 1 with sinkbasin 1",
    "close cabinet 1",
    "examine cabinet 1",
    "examine mug 1",
    "examine spoon 2",
    "go to countertop 1",
    "help",
    "inventory",
    "look",
    "move mug 1 to cabinet 1",
    "use desklamp 1",
)


def test_examine_belongs_to_inventory_only_for_the_held_object() -> None:
    assert partition_tool("examine mug 1", held=("mug 1",)) == MANAGE_INVENTORY
    assert partition_tool("examine mug 1", held=()) == EXPLORE
    assert partition_tool("examine mug 10", held=("mug 1",)) == EXPLORE
    assert partition_tool("examine cabinet 1", held=("mug 1",)) == EXPLORE


def test_unknown_verb_is_rejected() -> None:
    with pytest.raises(ValueError, match="unrecognized"):
        partition_tool("dance with mug 1", held=())


def test_partitions_split_the_offered_commands_without_help() -> None:
    sets = command_sets(ADMISSIBLE, held=("mug 1",))
    offered = set(ADMISSIBLE) - {"help"}
    assert set(sets[SELECT_COMMAND]) == offered
    partitioned = [command for name in PARTITIONED_TOOLS for command in sets[name]]
    assert sorted(partitioned) == sorted(offered)
    assert sets[MANAGE_INVENTORY] == ("examine mug 1", "inventory", "move mug 1 to cabinet 1")
    assert sets[EXPLORE] == (
        "close cabinet 1",
        "examine cabinet 1",
        "examine spoon 2",
        "go to countertop 1",
        "look",
    )
    assert sets[MANIPULATE_OBJECT] == ("clean mug 1 with sinkbasin 1", "use desklamp 1")


def test_help_alone_offers_nothing() -> None:
    assert all(commands == () for commands in command_sets(("help",), held=()).values())


def test_tool_spec_is_a_strict_schema_over_the_commands() -> None:
    spec = tool_spec(EXPLORE, ("go to countertop 1", "look"))
    parameters = spec.parameters
    assert spec.name == EXPLORE
    assert parameters["additionalProperties"] is False
    assert sorted(parameters["required"]) == sorted(parameters["properties"])
    assert parameters["properties"]["command"]["enum"] == ["go to countertop 1", "look"]
    reasoning = parameters["properties"]["reasoning"]
    assert (reasoning["type"], reasoning["items"]) == ("array", {"type": "string"})


def test_tool_spec_refuses_an_empty_enum() -> None:
    with pytest.raises(ValueError, match="omit"):
        tool_spec(MANIPULATE_OBJECT, ())
