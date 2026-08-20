#!/usr/bin/env python3
"""Export one MongoDB graph and verify its normalized round trip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from luna.actor.graphs import load_graph, normalized_graph_json
from luna.actor.schemas import GraphManifest
from luna.config import ServiceSettings
from luna.storage.mongo import MongoStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    settings = ServiceSettings.from_env()
    store = MongoStore(settings.mongodb_uri, settings.mongodb_db)
    document = store.graph_document(args.alias, args.version)
    if document is None:
        raise SystemExit(f"graph not found: {args.alias}@{args.version}")
    expected_hash = document.pop("sha256")
    graph = GraphManifest.model_validate(document)
    if graph.sha256 != expected_hash:
        raise SystemExit("stored graph document does not match its recorded hash")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(normalized_graph_json(graph), encoding="utf-8")
    round_trip = load_graph(args.output)
    if round_trip.sha256 != expected_hash:
        raise SystemExit("export round trip changed the normalized graph")
    print(f"exported {args.alias}@{args.version} {expected_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

