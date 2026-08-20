#!/usr/bin/env python3
"""Export only owner-approved paper runs into sanitized normalized files."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from luna.config import ServiceSettings
from luna.storage.mongo import MongoStore


def main() -> int:
    manifest_path = ROOT / "results" / "paper_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        approved = [row for row in csv.DictReader(handle) if row["publication_status"] == "approved"]
    settings = ServiceSettings.from_env()
    store = MongoStore(settings.mongodb_uri, settings.mongodb_db)
    store.ping()
    output = ROOT / "results" / "generated"
    output.mkdir(parents=True, exist_ok=True)
    aggregates = []
    case_rows = []
    for item in approved:
        run = store.db.benchmark_runs.find_one({"run_id": item["run_id"]}, {"_id": 0})
        if run is None:
            raise SystemExit(f"approved run missing from MongoDB: {item['run_id']}")
        cases = list(store.db.test_cases.find({"run_id": item["run_id"]}, {"_id": 0}))
        counts = Counter(case["result"] for case in cases)
        aggregates.append(
            {
                "run_id": item["run_id"],
                "benchmark": run["benchmark"],
                "split": run["split"],
                "depth": run.get("depth") if run.get("depth") is not None else "NOT_RECORDED",
                "mode": run["mode"],
                "cases": len(cases),
                "won": counts["won"],
                "lost": counts["lost"],
                "crashed": counts["crashed"],
                "success_rate": counts["won"] / len(cases) if cases else 0.0,
            }
        )
        for case in cases:
            case_rows.append(
                {
                    key: case.get(key, "NOT_RECORDED")
                    for key in (
                        "case_id", "run_id", "benchmark", "split", "depth", "seed", "index",
                        "result", "graph_alias", "graph_hash", "num_steps", "token_usage", "exception",
                    )
                }
            )
    fieldnames = [
        "run_id", "benchmark", "split", "depth", "mode", "cases", "won", "lost",
        "crashed", "success_rate",
    ]
    with (output / "aggregates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregates)
    (output / "aggregates.json").write_text(json.dumps(aggregates, indent=2, default=str) + "\n", encoding="utf-8")
    (output / "case_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in case_rows),
        encoding="utf-8",
    )
    print(f"exported {len(aggregates)} approved runs and {len(case_rows)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

