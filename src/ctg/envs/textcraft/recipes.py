"""Minecraft crafting data for TextCraft: where the recipe files come from, items, tags, recipes and depth.

The recipe files are Minecraft: Java Edition recipe JSON, which ADaPT also downloads rather than ships.
`data/textcraft/recipes.lock.json` pins their source (a GitHub repository, commit and path) and names every
file the benchmark reads together with its SHA-256 digest. `ctg data fetch textcraft` downloads them into
`$TEXTCRAFT_RECIPES`, or ~/.cache/ctg/textcraft/recipes.

The data rules are ADaPT's, which define the benchmark's goal sets (291 / 117 / 11 goals at depth 2 / 3 / 4):

- Only shaped and shapeless crafting recipes are used. A slot that accepts several items takes the first.
- Recipes whose only ingredient is a `*_block` item are dropped (ADaPT's guard against storage-block
  unpacking cycles such as iron block -> 9 iron ingots).
- A recipe's `group` makes its output a member of the tag of that name. A tag ingredient that no
  recipe group defines denotes a raw item of that name (e.g. `oak_logs`).
- Recipes are read in file-name order; a recipe whose output is already required, transitively, by one
  of its ingredients is dropped, so the kept recipes are acyclic.
- Every item a recipe file names (output or ingredient) is an item of the book, including the items of
  dropped recipes; an item no kept recipe makes is raw.

Identifiers carry no `minecraft:` namespace. Display names replace underscores with spaces.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, cast

import httpx
from pydantic import BaseModel

from ctg.config import DATA_DIR

LOCK_PATH = DATA_DIR / "textcraft" / "recipes.lock.json"
RECIPES_ENV_VAR = "TEXTCRAFT_RECIPES"
DEFAULT_RECIPES_DIR = "~/.cache/ctg/textcraft/recipes"
FETCH_CONCURRENCY = 16
FETCH_TIMEOUT_S = 30.0

_NAMESPACE = "minecraft:"
_SHAPED = "minecraft:crafting_shaped"
_SHAPELESS = "minecraft:crafting_shapeless"


def display_name(identifier: str) -> str:
    """`dark_oak_planks` -> `dark oak planks`."""
    return identifier.replace("_", " ")


def normalize_name(text: str) -> str:
    """Canonical form of a written name: lower case, underscores as spaces, single spaces."""
    return " ".join(text.lower().replace("_", " ").split())


@dataclass(frozen=True)
class Ingredient:
    """An item, or a tag that any of its member items satisfies."""

    id: str
    is_tag: bool = False

    @property
    def name(self) -> str:
        return display_name(self.id)


@dataclass(frozen=True)
class Recipe:
    output: str
    count: int
    inputs: tuple[tuple[Ingredient, int], ...]
    """Ingredients with the number of each consumed, in data order."""

    @property
    def command(self) -> str:
        """The crafting command shown to agents, e.g. `craft 4 oak planks using 1 oak logs`."""
        ingredients = ", ".join(f"{n} {ingredient.name}" for ingredient, n in self.inputs)
        return f"craft {self.count} {display_name(self.output)} using {ingredients}"


class RecipeBook:
    """An acyclic recipe set with its tags and the minimum crafting depth of every item.

    Items without a recipe are raw (depth 0). A recipe's depth is 1 + the largest depth of its
    ingredients, a tag's depth is the smallest depth of its members, and an item's depth is the
    smallest depth of its recipes.
    """

    def __init__(
        self, recipes: Iterable[Recipe], tags: Mapping[str, Iterable[str]], items: Iterable[str] = ()
    ) -> None:
        """`items`: item ids the book holds besides the outputs, ingredients and tag members of `recipes`."""
        self._recipes = tuple(recipes)
        self._tags = {tag: frozenset(members) for tag, members in tags.items()}
        by_output: dict[str, list[Recipe]] = {}
        for recipe in self._recipes:
            by_output.setdefault(recipe.output, []).append(recipe)
        self._by_output = {item: tuple(recipes) for item, recipes in by_output.items()}

        known = set(self._by_output) | set(items)
        for tag, members in self._tags.items():
            if not members:
                raise ValueError(f"tag {tag} has no members")
            known |= members
        for recipe in self._recipes:
            for ingredient, _ in recipe.inputs:
                if ingredient.is_tag and ingredient.id not in self._tags:
                    raise ValueError(f"recipe for {recipe.output} uses undefined tag {ingredient.id}")
                if not ingredient.is_tag:
                    known.add(ingredient.id)

        self._by_name: dict[str, Ingredient] = {}
        for ingredient in [
            *(Ingredient(item) for item in known),
            *(Ingredient(tag, True) for tag in self._tags),
        ]:
            if ingredient.name in self._by_name:
                raise ValueError(f"display name {ingredient.name!r} is ambiguous")
            self._by_name[ingredient.name] = ingredient

        self._depth: dict[str, int] = {}
        for item in sorted(known):
            self._item_depth(item, ())

    @property
    def recipes(self) -> tuple[Recipe, ...]:
        return self._recipes

    def recipes_for(self, item: str) -> tuple[Recipe, ...]:
        return self._by_output.get(item, ())

    def is_craftable(self, item: str) -> bool:
        return item in self._by_output

    def members(self, tag: str) -> frozenset[str]:
        return self._tags[tag]

    def lookup(self, name: str) -> Ingredient | None:
        """The item or tag with this display name (compared after `normalize_name`)."""
        return self._by_name.get(normalize_name(name))

    def depth(self, item: str) -> int:
        return self._depth[item]

    def goals(self, depth: int) -> list[str]:
        """Craftable items whose minimum crafting depth is exactly `depth`, sorted by id."""
        return sorted(item for item in self._by_output if self._depth[item] == depth)

    def required_recipes(self, goal: str) -> tuple[Recipe, ...]:
        """Every recipe of the goal and, recursively, of every ingredient (all members of a tag)."""
        found: dict[Recipe, None] = {}
        seen: set[str] = set()

        def visit(item: str) -> None:
            if item in seen:
                return
            seen.add(item)
            for recipe in self.recipes_for(item):
                found[recipe] = None
                for ingredient, _ in recipe.inputs:
                    for member in sorted(self._tags[ingredient.id]) if ingredient.is_tag else [ingredient.id]:
                        visit(member)

        visit(goal)
        return tuple(found)

    def _item_depth(self, item: str, path: tuple[str, ...]) -> int:
        if item in self._depth:
            return self._depth[item]
        if item in path:
            raise ValueError(f"recipes contain a cycle: {' -> '.join((*path, item))}")
        path = (*path, item)
        recipes = self.recipes_for(item)
        depth = (
            min(
                1 + max(self._ingredient_depth(ingredient, path) for ingredient, _ in r.inputs)
                for r in recipes
            )
            if recipes
            else 0
        )
        self._depth[item] = depth
        return depth

    def _ingredient_depth(self, ingredient: Ingredient, path: tuple[str, ...]) -> int:
        if ingredient.is_tag:
            return min(self._item_depth(member, path) for member in sorted(self._tags[ingredient.id]))
        return self._item_depth(ingredient.id, path)


def recipes_dir() -> Path:
    """The TextCraft recipe directory: `$TEXTCRAFT_RECIPES`, or ~/.cache/ctg/textcraft/recipes."""
    return Path(os.environ.get(RECIPES_ENV_VAR) or DEFAULT_RECIPES_DIR).expanduser()


class RecipeLock(BaseModel):
    """The pinned source of the recipe files and the SHA-256 digest of each file, by file name."""

    repository: str
    """GitHub `owner/name`."""
    commit: str
    path: str
    """Directory of the recipe files inside the repository."""
    files: dict[str, str]

    def url(self, name: str) -> str:
        return f"https://raw.githubusercontent.com/{self.repository}/{self.commit}/{self.path}/{name}"


def load_lock(path: Path = LOCK_PATH) -> RecipeLock:
    return RecipeLock.model_validate_json(path.read_text(encoding="utf-8"))


def missing_recipes(directory: Path | None = None, lock_path: Path = LOCK_PATH) -> list[str]:
    """Names of the locked recipe files absent from `directory` (default: `recipes_dir()`), sorted."""
    root = directory or recipes_dir()
    return [name for name in sorted(load_lock(lock_path).files) if not (root / name).is_file()]


def load_recipe_book(directory: Path | None = None, lock_path: Path = LOCK_PATH) -> RecipeBook:
    """Build the TextCraft recipe book from the locked files in `directory` (default: `recipes_dir()`).

    Files are read in sorted name order; files the lock does not name are ignored. The book is cached per
    directory and lock file. Raises FileNotFoundError if a locked file is missing and ValueError if a
    file's SHA-256 digest differs from the lock's; both messages name `ctg data fetch textcraft`.
    """
    return _load_recipe_book(directory or recipes_dir(), lock_path)


@cache
def _load_recipe_book(directory: Path, lock_path: Path) -> RecipeBook:
    missing = missing_recipes(directory, lock_path)
    if missing:
        raise FileNotFoundError(
            f"TextCraft recipe files are missing from {directory} ({len(missing)} files, e.g. {missing[0]}); "
            "run `ctg data fetch textcraft`"
        )
    lock = load_lock(lock_path)
    contents = {name: (directory / name).read_bytes() for name in sorted(lock.files)}
    altered = [
        name for name, data in contents.items() if hashlib.sha256(data).hexdigest() != lock.files[name]
    ]
    if altered:
        raise ValueError(
            f"TextCraft recipe files in {directory} differ from the lock {lock_path} "
            f"({len(altered)} files, e.g. {altered[0]}); run `ctg data fetch textcraft`"
        )
    return _build([_parse(name, data) for name, data in contents.items()])


class RecipeFetchError(RuntimeError):
    """A recipe file could not be downloaded, or its SHA-256 digest differs from the lock."""


@dataclass(frozen=True)
class FetchSummary:
    directory: Path
    downloaded: int
    present: int
    """Files that were already in place with the locked digest."""


async def fetch_recipes(
    http: httpx.AsyncClient, directory: Path | None = None, lock_path: Path = LOCK_PATH
) -> FetchSummary:
    """Download the locked recipe files into `directory` (default: `recipes_dir()`), verifying each digest.

    Files already present with the locked digest are kept; every other locked file is downloaded and
    replaced atomically. The first failed download or digest mismatch cancels the remaining downloads and
    raises `RecipeFetchError`; files written before it stay, so a rerun resumes.
    """
    root = directory or recipes_dir()
    lock = load_lock(lock_path)
    pending = [name for name, digest in sorted(lock.files.items()) if _sha256(root / name) != digest]
    root.mkdir(parents=True, exist_ok=True)
    slots = asyncio.Semaphore(FETCH_CONCURRENCY)

    async def download(name: str) -> None:
        url = lock.url(name)
        async with slots:
            try:
                response = await http.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise RecipeFetchError(f"{url}: {str(exc) or type(exc).__name__}") from exc
        digest = hashlib.sha256(response.content).hexdigest()
        if digest != lock.files[name]:
            raise RecipeFetchError(f"{url}: SHA-256 {digest} does not match the lock's {lock.files[name]}")
        _write_atomic(root / name, response.content)

    try:
        async with asyncio.TaskGroup() as group:
            for name in pending:
                group.create_task(download(name))
    except* RecipeFetchError as errors:
        raise RecipeFetchError("; ".join(str(error) for error in errors.exceptions)) from None
    return FetchSummary(root, downloaded=len(pending), present=len(lock.files) - len(pending))


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _write_atomic(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class _RawRecipe:
    output: str
    count: int
    group: str | None
    inputs: tuple[tuple[Ingredient, int], ...]


def _build(raw: Sequence[_RawRecipe]) -> RecipeBook:
    usable = [recipe for recipe in raw if not _unpacks_block(recipe)]
    groups: dict[str, set[str]] = {}
    for recipe in usable:
        if recipe.group is not None and recipe.group != recipe.output:
            groups.setdefault(recipe.group, set()).add(recipe.output)
    referenced = {ingredient.id for recipe in usable for ingredient, _ in recipe.inputs if ingredient.is_tag}
    tags = {tag: frozenset(groups[tag]) for tag in sorted(referenced) if tag in groups}

    kept: dict[Recipe, None] = {}
    by_output: dict[str, list[Recipe]] = {}
    for raw_recipe in usable:
        inputs = tuple(
            (Ingredient(ingredient.id, ingredient.is_tag and ingredient.id in tags), n)
            for ingredient, n in raw_recipe.inputs
        )
        recipe = Recipe(raw_recipe.output, raw_recipe.count, inputs)
        if recipe in kept or _requires(inputs, recipe.output, by_output, tags):
            continue
        kept[recipe] = None
        by_output.setdefault(recipe.output, []).append(recipe)
    named = {recipe.output for recipe in raw}
    named |= {ingredient.id for recipe in raw for ingredient, _ in recipe.inputs if not ingredient.is_tag}
    return RecipeBook(kept, tags, named)


def _unpacks_block(recipe: _RawRecipe) -> bool:
    if len(recipe.inputs) != 1:
        return False
    ingredient = recipe.inputs[0][0]
    return not ingredient.is_tag and ingredient.id.endswith("_block")


def _requires(
    inputs: Iterable[tuple[Ingredient, int]],
    item: str,
    by_output: Mapping[str, Sequence[Recipe]],
    tags: Mapping[str, frozenset[str]],
) -> bool:
    """Whether crafting any of `inputs` with the recipes so far can need `item`."""
    stack = [ingredient for ingredient, _ in inputs]
    seen: set[Ingredient] = set()
    while stack:
        ingredient = stack.pop()
        if ingredient in seen:
            continue
        seen.add(ingredient)
        if ingredient.is_tag:
            stack.extend(Ingredient(member) for member in tags[ingredient.id])
        elif ingredient.id == item:
            return True
        else:
            stack.extend(i for recipe in by_output.get(ingredient.id, ()) for i, _ in recipe.inputs)
    return False


def _parse(name: str, data: bytes) -> _RawRecipe:
    recipe = cast(dict[str, Any], json.loads(data))
    counts: dict[Ingredient, int] = {}
    if recipe["type"] == _SHAPED:
        pattern = cast(list[str], recipe["pattern"])
        for symbol, spec in cast(dict[str, Any], recipe["key"]).items():
            ingredient = _ingredient(spec, name)
            counts[ingredient] = counts.get(ingredient, 0) + sum(row.count(symbol) for row in pattern)
    elif recipe["type"] == _SHAPELESS:
        for spec in cast(list[Any], recipe["ingredients"]):
            ingredient = _ingredient(spec, name)
            counts[ingredient] = counts.get(ingredient, 0) + 1
    else:
        raise ValueError(f"{name}: unsupported recipe type {recipe['type']!r}")
    result = cast(dict[str, Any], recipe["result"])
    return _RawRecipe(
        output=_strip_namespace(result["item"]),
        count=int(result.get("count", 1)),
        group=cast(str | None, recipe.get("group")),
        inputs=tuple(counts.items()),
    )


def _ingredient(spec: Any, name: str) -> Ingredient:
    if isinstance(spec, list):
        spec = cast(list[Any], spec)[0]
    if isinstance(spec, dict):
        fields = cast(dict[str, str], spec)
        if "item" in fields:
            return Ingredient(_strip_namespace(fields["item"]))
        if "tag" in fields:
            return Ingredient(_strip_namespace(fields["tag"]), is_tag=True)
    raise ValueError(f"{name}: unsupported ingredient {spec!r}")


def _strip_namespace(identifier: str) -> str:
    return identifier.removeprefix(_NAMESPACE)
