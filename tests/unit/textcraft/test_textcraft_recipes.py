import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ctg.envs.textcraft.benchmark import TextCraftBenchmark
from ctg.envs.textcraft.recipes import RECIPES_ENV_VAR, Ingredient, load_recipe_book, missing_recipes


@pytest.mark.textcraft_recipes
@pytest.mark.parametrize(("depth", "expected"), [(2, 291), (3, 117), (4, 11)])
def test_instance_counts_match_the_paper(depth: int, expected: int) -> None:
    instances = TextCraftBenchmark().enumerate_instances(f"textcraft_depth{depth}")
    book = load_recipe_book()

    assert len(instances) == expected
    assert [instance.id for instance in instances] == sorted(instance.id for instance in instances)
    for instance in instances:
        assert instance.meta == {"goal": instance.id, "depth": depth}
        assert book.depth(instance.id) == depth
        assert ":" not in instance.id


@pytest.mark.textcraft_recipes
def test_depth_is_the_cheapest_recipe_over_the_deepest_ingredient() -> None:
    book = load_recipe_book()
    # polished granite <- granite <- diorite <- quartz + cobblestone (the paper's TextCraft-3 example)
    assert [book.depth(item) for item in ("quartz", "diorite", "granite", "polished_granite")] == [0, 1, 2, 3]
    # stick: 2 planks (depth 2) or 2 bamboo (depth 1); the sign takes the cheaper stick
    assert book.depth("stick") == 1
    assert book.depth("dark_oak_sign") == 2


@pytest.mark.textcraft_recipes
def test_cycles_in_the_real_data_are_broken_by_file_order() -> None:
    book = load_recipe_book()
    # iron_ingot_from_nuggets.json sorts before iron_nugget.json; the iron block unpacking recipe is dropped
    assert [recipe.inputs for recipe in book.recipes_for("iron_ingot")] == [((Ingredient("iron_nugget"), 9),)]
    assert not book.is_craftable("iron_nugget")


def _write(directory: Path, name: str, recipe: dict[str, Any]) -> None:
    (directory / f"{name}.json").write_text(json.dumps(recipe))


def _lock(directory: Path) -> Path:
    """Lock every recipe file now in `directory`, writing the lock file next to them."""
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.glob("*.json"))
    }
    lock = directory / "recipes.lock.json"
    lock.write_text(
        json.dumps({"repository": "owner/assets", "commit": "0" * 40, "path": "recipes", "files": files})
    )
    return lock


def _shapeless(
    output: str, *ingredients: dict[str, Any], group: str | None = None, count: int = 1
) -> dict[str, Any]:
    recipe: dict[str, Any] = {
        "type": "minecraft:crafting_shapeless",
        "ingredients": list(ingredients),
        "result": {"item": f"minecraft:{output}", "count": count},
    }
    if group is not None:
        recipe["group"] = group
    return recipe


def _item(name: str) -> dict[str, str]:
    return {"item": f"minecraft:{name}"}


def _tag(name: str) -> dict[str, str]:
    return {"tag": f"minecraft:{name}"}


def test_loader_rules(tmp_path: Path) -> None:
    # A cycle between gizmo and widget: the recipe read first (a_*) wins.
    _write(tmp_path, "a_widget", _shapeless("widget", _item("gizmo"), _item("gizmo")))
    _write(tmp_path, "b_gizmo", _shapeless("gizmo", _item("widget")))
    # Storage-block unpacking is dropped even without a cycle.
    _write(tmp_path, "c_nugget", _shapeless("nugget", _item("nugget_block"), count=9))
    # Groups define tags; a tag no group defines is a raw item.
    _write(tmp_path, "d_red_cloth", _shapeless("red_cloth", _item("thread"), group="cloth", count=2))
    _write(tmp_path, "e_blue_cloth", _shapeless("blue_cloth", _item("thread"), _item("dye"), group="cloth"))
    _write(tmp_path, "f_flag", _shapeless("flag", _tag("cloth"), _tag("cloth"), _tag("poles")))
    # Shaped recipes count each key symbol in the pattern; a multi-item slot takes its first item.
    shaped = {
        "type": "minecraft:crafting_shaped",
        "pattern": ["#X#", " # "],
        "key": {"#": _item("flag"), "X": [_item("thread"), _item("dye")]},
        "result": {"item": "minecraft:banner_stand"},
    }
    _write(tmp_path, "g_banner_stand", shaped)
    lock = _lock(tmp_path)
    # Files the lock does not name are ignored.
    _write(tmp_path, "z_flag", _shapeless("flag", _item("thread")))

    book = load_recipe_book(tmp_path, lock)

    assert [recipe.command for recipe in book.recipes] == [
        "craft 1 widget using 2 gizmo",
        "craft 2 red cloth using 1 thread",
        "craft 1 blue cloth using 1 thread, 1 dye",
        "craft 1 flag using 2 cloth, 1 poles",
        "craft 1 banner stand using 3 flag, 1 thread",
    ]
    assert not book.is_craftable("gizmo")
    # Items of dropped recipes stay items of the book; no kept recipe makes them, so they are raw.
    assert not book.is_craftable("nugget")
    assert book.lookup("nugget") == Ingredient("nugget")
    assert book.depth("nugget") == 0
    assert book.members("cloth") == {"red_cloth", "blue_cloth"}
    assert book.lookup("cloth") == Ingredient("cloth", is_tag=True)
    assert book.lookup("poles") == Ingredient("poles")
    assert book.depth("flag") == 2
    assert book.depth("banner_stand") == 3
    assert book.goals(2) == ["flag"]


def test_missing_locked_files_point_to_the_fetch_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "a_widget", _shapeless("widget", _item("gizmo")))
    _write(tmp_path, "b_gizmo", _shapeless("gizmo", _item("thread")))
    lock = _lock(tmp_path)
    (tmp_path / "b_gizmo.json").unlink()
    monkeypatch.setenv(RECIPES_ENV_VAR, str(tmp_path))

    assert missing_recipes(lock_path=lock) == ["b_gizmo.json"]
    with pytest.raises(FileNotFoundError) as raised:
        load_recipe_book(lock_path=lock)
    assert str(tmp_path) in str(raised.value)
    assert "b_gizmo.json" in str(raised.value)
    assert "run `ctg data fetch textcraft`" in str(raised.value)


def test_altered_locked_files_point_to_the_fetch_command(tmp_path: Path) -> None:
    _write(tmp_path, "a_widget", _shapeless("widget", _item("gizmo")))
    _write(tmp_path, "b_gizmo", _shapeless("gizmo", _item("thread")))
    lock = _lock(tmp_path)
    _write(tmp_path, "b_gizmo", _shapeless("gizmo", _item("thread"), count=64))

    with pytest.raises(ValueError) as raised:
        load_recipe_book(tmp_path, lock)
    assert str(tmp_path) in str(raised.value)
    assert "b_gizmo.json" in str(raised.value)
    assert "a_widget.json" not in str(raised.value)
    assert "run `ctg data fetch textcraft`" in str(raised.value)
