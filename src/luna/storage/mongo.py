"""Minimal MongoDB persistence with immutable run/case/output inserts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from luna.actor.schemas import BenchmarkRunRecord, GraphManifest, TaskOutputRecord, TestCaseRecord


class MongoStore:
    def __init__(self, uri: str, database: str):
        try:
            from pymongo import ASCENDING, MongoClient
        except ImportError as exc:
            raise RuntimeError("Install persistence support with: pip install '.[storage]'") from exc
        self.client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self.db = self.client[database]
        self._ascending = ASCENDING

    def ping(self) -> None:
        self.client.admin.command("ping")

    def ensure_indexes(self) -> None:
        self.db.benchmark_runs.create_index("run_id", unique=True)
        self.db.test_cases.create_index("case_id", unique=True)
        self.db.test_cases.create_index([("run_id", self._ascending), ("seed", self._ascending)])
        self.db.task_outputs.create_index("task_output_id", unique=True)
        self.db.definitions.create_index([("kind", self._ascending), ("alias", self._ascending)], unique=True)
        self.db.graphs.create_index([("alias", self._ascending), ("version", self._ascending)], unique=True)

    def insert_run(self, record: BenchmarkRunRecord) -> None:
        self.db.benchmark_runs.insert_one(record.model_dump(mode="json"))

    def insert_case(self, record: TestCaseRecord) -> None:
        self.db.test_cases.insert_one(record.model_dump(mode="json"))

    def insert_task_output(self, record: TaskOutputRecord) -> None:
        self.db.task_outputs.insert_one(record.model_dump(mode="json"))

    def upsert_definition(self, definition: dict[str, Any]) -> None:
        immutable = {**definition, "updated_at": datetime.now(timezone.utc).isoformat()}
        self.db.definitions.update_one(
            {"kind": definition["kind"], "alias": definition["alias"]},
            {"$set": immutable, "$setOnInsert": {"created_at": immutable["updated_at"]}},
            upsert=True,
        )

    def definition_aliases(self) -> set[str]:
        return {document["alias"] for document in self.db.definitions.find({}, {"alias": 1})}

    def upsert_graph(self, graph: GraphManifest) -> None:
        document = graph.normalized() | {"sha256": graph.sha256}
        existing = self.db.graphs.find_one({"alias": graph.alias, "version": graph.version})
        if existing is not None and existing.get("sha256") != graph.sha256:
            raise ValueError(
                f"graph {graph.alias}@{graph.version} already exists with a different hash; bump its version"
            )
        self.db.graphs.update_one(
            {"alias": graph.alias, "version": graph.version},
            {"$setOnInsert": document},
            upsert=True,
        )

    def get_graph(self, alias: str, version: str) -> GraphManifest | None:
        document = self.db.graphs.find_one({"alias": alias, "version": version}, {"_id": 0, "sha256": 0})
        return GraphManifest.model_validate(document) if document else None

    def graph_document(self, alias: str, version: str) -> dict[str, Any] | None:
        return self.db.graphs.find_one({"alias": alias, "version": version}, {"_id": 0})

    def completed_seeds(self, run_id: str) -> set[int]:
        return {
            int(document["seed"])
            for document in self.db.test_cases.find(
                {"run_id": run_id, "result": {"$in": ["won", "lost"]}}, {"seed": 1}
            )
        }

    def latest_compatible_run(self, benchmark: str, split: str, depth: int | None, mode: str) -> str | None:
        document = self.db.benchmark_runs.find_one(
            {"benchmark": benchmark, "split": split, "depth": depth, "mode": mode},
            sort=[("started_at", -1)],
        )
        return document["run_id"] if document else None

    def counts(self) -> dict[str, int]:
        return {
            name: self.db[name].count_documents({})
            for name in ("definitions", "graphs", "benchmark_runs", "test_cases", "task_outputs")
        }


class NullStore:
    """No-service test store implementing the MongoStore interface."""

    def __init__(self):
        self.runs: list[BenchmarkRunRecord] = []
        self.cases: list[TestCaseRecord] = []
        self.outputs: list[TaskOutputRecord] = []

    def ping(self) -> None: pass
    def ensure_indexes(self) -> None: pass
    def insert_run(self, record: BenchmarkRunRecord) -> None: self.runs.append(record)
    def insert_case(self, record: TestCaseRecord) -> None: self.cases.append(record)
    def insert_task_output(self, record: TaskOutputRecord) -> None: self.outputs.append(record)
    def definition_aliases(self) -> set[str]: return set(required_definition_aliases())
    def get_graph(self, alias: str, version: str) -> None: return None
    def completed_seeds(self, run_id: str) -> set[int]: return set()
    def latest_compatible_run(self, benchmark: str, split: str, depth: int | None, mode: str) -> None: return None


def required_definition_aliases() -> tuple[str, ...]:
    return (
        "llm.executor", "llm.planner", "llm.router", "llm.judge",
        "agent.alfworld_select_command", "agent.textcraft_select_command",
        "agent.finance_agent_select_command", "agent.summarizer",
    )

