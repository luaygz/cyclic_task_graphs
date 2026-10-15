"""`fetch_recipes` against a mocked upstream: verified downloads, reruns, digest mismatches, HTTP errors."""

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from ctg.envs.textcraft.recipes import RecipeFetchError, fetch_recipes, load_recipe_book

BASE_URL = "https://raw.githubusercontent.com/owner/assets/0123abc/data/minecraft/recipes"
FILES = {
    "planks.json": json.dumps(
        {
            "type": "minecraft:crafting_shapeless",
            "ingredients": [{"item": "minecraft:log"}],
            "result": {"item": "minecraft:planks", "count": 4},
        }
    ).encode(),
    "stick.json": json.dumps(
        {
            "type": "minecraft:crafting_shaped",
            "pattern": ["#", "#"],
            "key": {"#": {"item": "minecraft:planks"}},
            "result": {"item": "minecraft:stick", "count": 4},
        }
    ).encode(),
}


class Upstream:
    """Serves `files` by name and records every requested URL."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.requested: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requested.append(url)
        name = url.removeprefix(f"{BASE_URL}/")
        if name not in self.files:
            return httpx.Response(404)
        return httpx.Response(200, content=self.files[name])

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))


def _lock(tmp_path: Path) -> Path:
    files = {name: hashlib.sha256(body).hexdigest() for name, body in FILES.items()}
    lock = tmp_path / "recipes.lock.json"
    lock.write_text(
        json.dumps(
            {
                "repository": "owner/assets",
                "commit": "0123abc",
                "path": "data/minecraft/recipes",
                "files": files,
            }
        )
    )
    return lock


async def test_fetch_verifies_downloads_and_reruns_fetch_only_absent_or_altered_files(tmp_path: Path) -> None:
    lock = _lock(tmp_path)
    target = tmp_path / "recipes"
    upstream = Upstream(FILES)

    async with upstream.client() as http:
        first = await fetch_recipes(http, target, lock)
        assert sorted(upstream.requested) == [f"{BASE_URL}/planks.json", f"{BASE_URL}/stick.json"]
        assert {path.name: path.read_bytes() for path in target.iterdir()} == FILES
        assert [recipe.command for recipe in load_recipe_book(target, lock).recipes] == [
            "craft 4 planks using 1 log",
            "craft 4 stick using 2 planks",
        ]

        upstream.requested.clear()
        second = await fetch_recipes(http, target, lock)
        assert upstream.requested == []

        (target / "stick.json").write_bytes(b"{}")
        third = await fetch_recipes(http, target, lock)
        assert upstream.requested == [f"{BASE_URL}/stick.json"]

    assert [(run.downloaded, run.present) for run in (first, second, third)] == [(2, 0), (0, 2), (1, 1)]
    assert first.directory == target
    assert {path.name: path.read_bytes() for path in target.iterdir()} == FILES


@pytest.mark.parametrize(
    ("served", "error"),
    [
        ({**FILES, "stick.json": b"tampered"}, r"stick\.json: SHA-256 [0-9a-f]{64} does not match the lock"),
        ({"planks.json": FILES["planks.json"]}, r"stick\.json: .*404 Not Found"),
    ],
)
async def test_fetch_fails_on_a_digest_mismatch_or_an_http_error(
    tmp_path: Path, served: dict[str, bytes], error: str
) -> None:
    lock = _lock(tmp_path)
    target = tmp_path / "recipes"

    async with Upstream(served).client() as http:
        with pytest.raises(RecipeFetchError, match=error):
            await fetch_recipes(http, target, lock)

    assert not (target / "stick.json").exists()
    assert all(path.name in FILES for path in target.iterdir())
