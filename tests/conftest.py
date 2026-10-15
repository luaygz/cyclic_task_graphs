"""Tests marked `textcraft_recipes` need the real TextCraft recipe files and skip until they are fetched."""

import pytest

from ctg.envs.textcraft.recipes import missing_recipes, recipes_dir


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    marked = [item for item in items if item.get_closest_marker("textcraft_recipes") is not None]
    if marked and missing_recipes():
        skip = pytest.mark.skip(
            reason=f"TextCraft recipes not fetched to {recipes_dir()}; run `ctg data fetch textcraft`"
        )
        for item in marked:
            item.add_marker(skip)
