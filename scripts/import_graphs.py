#!/usr/bin/env python3
"""Idempotently import versioned local graph manifests into MongoDB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from luna.actor.graphs import load_graph
from luna.config import ServiceSettings
from luna.storage.mongo import MongoStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted((ROOT / "configs" / "graphs").glob("*.json"))
    settings = ServiceSettings.from_env()
    store = MongoStore(settings.mongodb_uri, settings.mongodb_db)
    store.ping()
    store.ensure_indexes()
    for path in paths:
        graph = load_graph(path)
        store.upsert_graph(graph)
        print(f"{graph.alias}@{graph.version} {graph.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

