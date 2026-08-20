#!/usr/bin/env python3
"""Idempotently upsert only public benchmark definitions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from luna.config import ServiceSettings
from luna.storage.definitions import definitions
from luna.storage.mongo import MongoStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-model", default="NOT_RECORDED")
    args = parser.parse_args()
    settings = ServiceSettings.from_env()
    store = MongoStore(settings.mongodb_uri, settings.mongodb_db)
    store.ping()
    store.ensure_indexes()
    before = store.counts()["definitions"]
    for definition in definitions():
        if definition["kind"] == "llm":
            definition = definition | {"provider_model": args.provider_model}
        store.upsert_definition(definition)
    after = store.counts()["definitions"]
    print(f"definitions: before={before} after={after} expected={len(definitions())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

