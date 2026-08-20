#!/usr/bin/env python3
"""Verify service connectivity, definition references, and collection counts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from luna.config import ServiceSettings
from luna.storage.mongo import MongoStore, required_definition_aliases
from luna.storage.redis_bus import RedisBus


def main() -> int:
    settings = ServiceSettings.from_env()
    store = MongoStore(settings.mongodb_uri, settings.mongodb_db)
    bus = RedisBus(settings.redis_url)
    store.ping()
    bus.ping()
    store.ensure_indexes()
    missing = set(required_definition_aliases()) - store.definition_aliases()
    if missing:
        raise SystemExit(f"missing required definitions: {sorted(missing)}")
    definitions = list(store.db.definitions.find({}, {"_id": 0}))
    aliases = {item["alias"] for item in definitions}
    broken = []
    for item in definitions:
        for tool in item.get("tools", []):
            if tool not in aliases:
                broken.append((item["alias"], tool))
    if broken:
        raise SystemExit(f"broken definition references: {broken}")
    print(store.counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

