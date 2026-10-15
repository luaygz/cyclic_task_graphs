"""Frozen train/test instance lists in data/splits/<variant>.json (paper Appendix B.2).

Instances keep the benchmark's deterministic order. TextCraft and Finance-Agent take the first `train_size`
indices of a seed-42 shuffle as `train`; ALFWorld follows the benchmark's own partitions.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ctg.config import DATA_DIR, BenchmarkName, ExperimentConfig
from ctg.envs.base import Benchmark, InstanceRef
from ctg.runs.store import write_json_atomic

SPLITS_DIR = DATA_DIR / "splits"
SPLIT_SEED = 42

Split = Literal["train", "test"]

VARIANTS: dict[str, BenchmarkName] = {
    "textcraft_depth2": BenchmarkName.TEXTCRAFT,
    "textcraft_depth3": BenchmarkName.TEXTCRAFT,
    "textcraft_depth4": BenchmarkName.TEXTCRAFT,
    "alfworld_train": BenchmarkName.ALFWORLD,
    "alfworld_valid_seen": BenchmarkName.ALFWORLD,
    "alfworld_valid_unseen": BenchmarkName.ALFWORLD,
    "finance_agent": BenchmarkName.FINANCE_AGENT,
}

TRAIN_SIZES: dict[str, int] = {
    "textcraft_depth2": 88,
    "textcraft_depth3": 35,
    "textcraft_depth4": 0,
    "finance_agent": 15,
}
"""Table 4 train sizes for the shuffled variants."""

FIXED_SPLITS: dict[str, Split] = {
    "alfworld_train": "train",
    "alfworld_valid_seen": "test",
    "alfworld_valid_unseen": "test",
}
"""ALFWorld uses the benchmark partitions: train games build n-shot summaries, valid games are evaluated."""

PAPER_TOTALS: dict[str, int] = {
    "textcraft_depth2": 291,
    "textcraft_depth3": 117,
    "textcraft_depth4": 11,
    "alfworld_train": 3553,
    "alfworld_valid_seen": 140,
    "alfworld_valid_unseen": 134,
    "finance_agent": 50,
}
"""Table 4 instance counts, for reporting deviations when freezing."""


class SplitEntry(BaseModel):
    id: str
    split: Split
    meta: dict[str, Any] = Field(default_factory=dict)


class SplitFile(BaseModel):
    benchmark: BenchmarkName
    variant: str
    seed: int
    instances: list[SplitEntry]

    def count(self, split: Split) -> int:
        return sum(entry.split == split for entry in self.instances)


def split_path(variant: str, splits_dir: Path = SPLITS_DIR) -> Path:
    return splits_dir / f"{variant}.json"


def assign_splits(variant: str, count: int) -> list[Split]:
    """Split label of each instance index, in enumeration order."""
    if variant in FIXED_SPLITS:
        return [FIXED_SPLITS[variant]] * count
    train_size = TRAIN_SIZES[variant]
    if train_size > count:
        raise ValueError(f"{variant}: train size {train_size} exceeds the {count} available instances")
    indices = list(range(count))
    random.Random(SPLIT_SEED).shuffle(indices)
    train = set(indices[:train_size])
    return ["train" if index in train else "test" for index in range(count)]


def freeze(benchmark: Benchmark, variant: str, splits_dir: Path = SPLITS_DIR) -> SplitFile:
    """Enumerate a variant's instances, assign splits and write the frozen list."""
    if VARIANTS.get(variant) is not benchmark.name:
        raise ValueError(f"variant {variant!r} does not belong to benchmark {benchmark.name.value}")
    instances = benchmark.enumerate_instances(variant)
    ids = [instance.id for instance in instances]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{variant}: instance ids are not unique")
    labels = assign_splits(variant, len(instances))
    frozen = SplitFile(
        benchmark=benchmark.name,
        variant=variant,
        seed=SPLIT_SEED,
        instances=[
            SplitEntry(id=instance.id, split=label, meta=dict(instance.meta))
            for instance, label in zip(instances, labels, strict=True)
        ],
    )
    write_json_atomic(split_path(variant, splits_dir), frozen.model_dump_json(indent=2))
    return frozen


def load(variant: str, splits_dir: Path = SPLITS_DIR) -> list[tuple[InstanceRef, Split]]:
    path = split_path(variant, splits_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no frozen split for {variant} at {path}; run `ctg splits freeze {variant}`")
    frozen = SplitFile.model_validate_json(path.read_text(encoding="utf-8"))
    if frozen.variant != variant:
        raise ValueError(f"{path} holds variant {frozen.variant!r}, expected {variant!r}")
    return [(InstanceRef(id=entry.id, meta=entry.meta), entry.split) for entry in frozen.instances]


def select_instances(config: ExperimentConfig, splits_dir: Path = SPLITS_DIR) -> list[InstanceRef]:
    """The experiment's instances: its split of the frozen list, narrowed to `config.instances` if given."""
    in_split = [instance for instance, split in load(config.variant, splits_dir) if split == config.split]
    if config.instances is not None:
        wanted = set(config.instances)
        unknown = wanted - {instance.id for instance in in_split}
        if unknown:
            raise ValueError(
                f"instances not in the {config.split} split of {config.variant}: {sorted(unknown)}"
            )
        in_split = [instance for instance in in_split if instance.id in wanted]
    if not in_split:
        raise ValueError(f"{config.variant} has no {config.split} instances")
    return in_split
