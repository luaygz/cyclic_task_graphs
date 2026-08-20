#!/usr/bin/env python3
"""Regenerate the exact deterministic public seed manifests."""

from __future__ import annotations

import json
import random
from pathlib import Path

SOURCE_COMMIT = "153bfc03716be20757e7d7b4480b80c0bcc3025d"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs" / "splits"


def shuffled(total: int) -> list[int]:
    values = list(range(total))
    random.Random(42).shuffle(values)
    return values


def write(name: str, benchmark: str, split: str, seeds: list[int], depth: int | None = None) -> None:
    document = {
        "schema_version": 1,
        "benchmark": benchmark,
        "split": split,
        "depth": depth,
        "shuffle_seed": 42 if benchmark != "alfworld" and depth != 4 else None,
        "seeds": seeds,
        "source_commit": SOURCE_COMMIT,
    }
    (OUTPUT / name).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for depth, total, train_size in ((2, 291, 88), (3, 117, 35)):
        values = shuffled(total)
        write(f"textcraft_depth{depth}_train.json", "textcraft", "train", values[:train_size], depth)
        write(f"textcraft_depth{depth}_test.json", "textcraft", "test", values[train_size:], depth)
    write("textcraft_depth4_test.json", "textcraft", "test", list(range(11)), 4)
    write(
        "alfworld_eval_out_of_distribution.json",
        "alfworld",
        "eval_out_of_distribution",
        list(range(134)),
    )
    finance = shuffled(50)
    write("finance_agent_train.json", "finance_agent", "train", finance[:15])
    write("finance_agent_test.json", "finance_agent", "test", finance[15:])


if __name__ == "__main__":
    main()

