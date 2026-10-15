import json
import re
from pathlib import Path

import pytest

from ctg.envs.alfworld.benchmark import ALFWorldBenchmark
from ctg.envs.alfworld.games import GAME_FILE, TASK_TYPES, data_dir, games_dir, list_games


def _trial(root: Path, relative: str, task_type: str, solvable: bool | None = True) -> None:
    trial = root / "json_2.1.1" / relative
    trial.mkdir(parents=True)
    (trial / "traj_data.json").write_text(json.dumps({"task_type": task_type}))
    if solvable is not None:
        (trial / GAME_FILE).write_text(json.dumps({"solvable": solvable}))


def test_list_games_applies_alfworlds_filters_and_sorts(tmp_path: Path) -> None:
    _trial(
        tmp_path,
        "valid_seen/pick_heat_then_place_in_recep-Mug-None-Cabinet-5/trial_2",
        "pick_heat_then_place_in_recep",
    )
    _trial(tmp_path, "valid_seen/look_at_obj_in_light-Book-None-DeskLamp-301/trial_1", "look_at_obj_in_light")
    _trial(
        tmp_path,
        "valid_seen/pick_and_place_with_movable_recep-Pen-Mug-Desk-3/trial_1",
        "pick_and_place_simple",
    )
    _trial(
        tmp_path,
        "valid_seen/pick_cool_then_place_in_recep-AppleSliced-None-Pot-7/trial_1",
        "pick_cool_then_place_in_recep",
    )
    _trial(tmp_path, "valid_seen/pick_and_place_simple-Pen-None-Desk-3/trial_1", "some_new_task_type")
    _trial(
        tmp_path,
        "valid_seen/pick_and_place_simple-Pen-None-Desk-3/trial_2",
        "pick_and_place_simple",
        solvable=None,
    )
    _trial(
        tmp_path,
        "valid_seen/pick_and_place_simple-Pen-None-Desk-3/trial_3",
        "pick_and_place_simple",
        solvable=False,
    )
    _trial(tmp_path, "valid_unseen/pick_and_place_simple-Pen-None-Desk-3/trial_4", "pick_and_place_simple")

    games = list_games("valid_seen", root=tmp_path)

    assert [game.path for game in games] == [
        f"valid_seen/look_at_obj_in_light-Book-None-DeskLamp-301/trial_1/{GAME_FILE}",
        f"valid_seen/pick_heat_then_place_in_recep-Mug-None-Cabinet-5/trial_2/{GAME_FILE}",
    ]
    assert games[0].id == "valid_seen__look_at_obj_in_light-Book-None-DeskLamp-301__trial_1"
    assert games[1].task_type == "pick_heat_then_place_in_recep"


def test_missing_split_names_the_data_variable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ALFWORLD_DATA"):
        list_games("train", root=tmp_path)


def test_data_dir_follows_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALFWORLD_DATA", str(tmp_path))
    assert data_dir() == tmp_path
    monkeypatch.delenv("ALFWORLD_DATA")
    assert data_dir() == Path.home() / ".cache" / "alfworld"


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="alfworld_valid_unseen"):
        ALFWorldBenchmark().enumerate_instances("alfworld_test")


@pytest.mark.alfworld
@pytest.mark.skipif(not games_dir().is_dir(), reason="ALFWorld data not downloaded")
@pytest.mark.parametrize(
    ("variant", "count"),
    [("alfworld_train", 3553), ("alfworld_valid_seen", 140), ("alfworld_valid_unseen", 134)],
)
def test_split_sizes_match_the_paper(variant: str, count: int) -> None:
    instances = ALFWorldBenchmark().enumerate_instances(variant)

    assert len(instances) == count
    paths = [instance.meta["path"] for instance in instances]
    assert paths == sorted(paths)
    assert len({instance.id for instance in instances}) == count
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+", instance.id) for instance in instances)
    assert {instance.meta["task_type"] for instance in instances} == TASK_TYPES
