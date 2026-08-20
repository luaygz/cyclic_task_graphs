import pytest

from luna.benchmarks.textcraft.adapter import TextCraftAdapter


@pytest.mark.asyncio
async def test_programmatic_textcraft_solution():
    adapter = TextCraftAdapter()
    await adapter.initialize(seed=0, index=0, depth=2)
    actions = adapter.oracle_actions()
    assert actions
    for action in actions:
        await adapter.step(action)
    assert adapter.success


def test_recipe_loading_does_not_mutate_recipe_index():
    from luna.benchmarks.textcraft.environment import TextCraftEnvironment

    environment = TextCraftEnvironment()
    before = {key: len(value) for key, value in environment.crafting_tree.itemid_recipes.items()}
    environment.reset(seed=0, min_depth=2, max_depth=2)
    after = {key: len(value) for key, value in environment.crafting_tree.itemid_recipes.items()}
    assert before == after


def test_generic_recipe_commands_expand_to_mentioned_concrete_items():
    from luna.benchmarks.textcraft.environment import TextCraftEnvironment

    environment = TextCraftEnvironment()
    environment.recipe_commands = [
        "craft 4 oak planks using 1 oak log",
        "craft 4 stick using 2 planks",
    ]
    assert "craft 4 stick using 2 oak planks" in environment.expanded_recipe_commands()
