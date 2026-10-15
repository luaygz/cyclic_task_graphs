"""Frozen splits: Table 4 sizes, the Appendix B.2.5 shuffle, determinism and instance selection."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from runs_fakes import FakeBenchmark

from ctg.config import BenchmarkName, Controller, ExperimentConfig
from ctg.runs import splits


@pytest.mark.parametrize(
    ("variant", "name", "count", "train"),
    [
        ("textcraft_depth2", BenchmarkName.TEXTCRAFT, 291, 88),
        ("textcraft_depth3", BenchmarkName.TEXTCRAFT, 117, 35),
        ("textcraft_depth4", BenchmarkName.TEXTCRAFT, 11, 0),
        ("finance_agent", BenchmarkName.FINANCE_AGENT, 50, 15),
    ],
)
def test_shuffled_variants_follow_table_4_and_the_paper_recipe(
    tmp_path: Path, variant: str, name: BenchmarkName, count: int, train: int
) -> None:
    benchmark = FakeBenchmark(count=count, name=name)

    frozen = splits.freeze(benchmark, variant, tmp_path)

    # Appendix B.2.5, verbatim: random.seed(42); random.shuffle(seeds); train_seeds = seeds[:train_size]
    state = random.getstate()
    random.seed(42)
    seeds = list(range(count))
    random.shuffle(seeds)
    random.setstate(state)
    assert {index for index, entry in enumerate(frozen.instances) if entry.split == "train"} == set(
        seeds[:train]
    )
    assert (frozen.count("train"), frozen.count("test")) == (train, count - train)
    assert [entry.id for entry in frozen.instances] == [
        ref.id for ref in benchmark.enumerate_instances(variant)
    ]
    on_disk = json.loads(splits.split_path(variant, tmp_path).read_text(encoding="utf-8"))
    assert set(on_disk) == {"benchmark", "variant", "seed", "instances"}
    assert (on_disk["benchmark"], on_disk["variant"], on_disk["seed"]) == (name.value, variant, 42)
    assert on_disk["instances"][0] == {
        "id": "item_000",
        "split": frozen.instances[0].split,
        "meta": {"goal": 0},
    }


def test_freezing_twice_gives_identical_files(tmp_path: Path) -> None:
    benchmark = FakeBenchmark(count=117)
    splits.freeze(benchmark, "textcraft_depth3", tmp_path / "a")
    splits.freeze(benchmark, "textcraft_depth3", tmp_path / "b")

    first = splits.split_path("textcraft_depth3", tmp_path / "a").read_bytes()
    assert first == splits.split_path("textcraft_depth3", tmp_path / "b").read_bytes()


@pytest.mark.parametrize(
    ("variant", "expected"),
    [("alfworld_train", "train"), ("alfworld_valid_seen", "test"), ("alfworld_valid_unseen", "test")],
)
def test_alfworld_uses_the_benchmark_partitions(tmp_path: Path, variant: str, expected: str) -> None:
    frozen = splits.freeze(FakeBenchmark(count=7, name=BenchmarkName.ALFWORLD), variant, tmp_path)

    assert {entry.split for entry in frozen.instances} == {expected}
    assert [split for _, split in splits.load(variant, tmp_path)] == [expected] * 7


def test_freeze_rejects_too_few_instances_and_foreign_variants(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train size 88"):
        splits.freeze(FakeBenchmark(count=50), "textcraft_depth2", tmp_path)
    with pytest.raises(ValueError, match="does not belong"):
        splits.freeze(FakeBenchmark(count=50), "finance_agent", tmp_path)
    assert not list(tmp_path.iterdir())


def _config(**overrides: object) -> ExperimentConfig:
    values: dict[str, object] = {
        "name": "x",
        "benchmark": "textcraft",
        "textcraft_depth": 3,
        "controller": Controller.REACT,
    }
    return ExperimentConfig.model_validate(values | overrides)


def test_select_instances_filters_by_split_then_by_listed_ids(tmp_path: Path) -> None:
    frozen = splits.freeze(FakeBenchmark(count=117), "textcraft_depth3", tmp_path)
    test_ids = [entry.id for entry in frozen.instances if entry.split == "test"]
    train_ids = [entry.id for entry in frozen.instances if entry.split == "train"]

    assert [ref.id for ref in splits.select_instances(_config(), tmp_path)] == test_ids
    assert [ref.id for ref in splits.select_instances(_config(split="train"), tmp_path)] == train_ids
    chosen = (test_ids[5], test_ids[1])
    assert [ref.id for ref in splits.select_instances(_config(instances=chosen), tmp_path)] == [
        test_ids[1],
        test_ids[5],
    ]
    with pytest.raises(ValueError, match=train_ids[0]):
        splits.select_instances(_config(instances=(train_ids[0],)), tmp_path)


def test_select_instances_rejects_an_empty_selection_and_a_missing_file(tmp_path: Path) -> None:
    splits.freeze(FakeBenchmark(count=11), "textcraft_depth4", tmp_path)

    with pytest.raises(ValueError, match="no train instances"):
        splits.select_instances(_config(textcraft_depth=4, split="train"), tmp_path)
    with pytest.raises(FileNotFoundError, match="ctg splits freeze textcraft_depth3"):
        splits.select_instances(_config(), tmp_path)
