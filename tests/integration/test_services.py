import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from luna.actor.graphs import load_graph
from luna.actor.schemas import BenchmarkRunRecord, TaskOutputRecord, TestCaseRecord as CaseRecord
from luna.config import ServiceSettings
from luna.storage.definitions import definitions
from luna.storage.mongo import MongoStore
from luna.storage.redis_bus import RedisBus


pytestmark = pytest.mark.skipif(
    os.getenv("LUNA_RUN_SERVICE_TESTS") != "1",
    reason="set LUNA_RUN_SERVICE_TESTS=1 with compose services running",
)


def test_bootstrap_is_idempotent_and_services_work():
    from pymongo.errors import DuplicateKeyError

    settings = ServiceSettings.from_env()
    database = f"{settings.mongodb_db}_integration_{uuid.uuid4().hex}"
    store = MongoStore(settings.mongodb_uri, database)
    bus = RedisBus(settings.redis_url)
    run_id = f"integration-{uuid.uuid4().hex}"
    other_run_id = f"integration-{uuid.uuid4().hex}"
    try:
        store.ping()
        bus.ping()
        store.ensure_indexes()
        for definition in definitions():
            store.upsert_definition(definition)
        first = store.counts()["definitions"]
        for definition in definitions():
            store.upsert_definition(definition)
        assert store.counts()["definitions"] == first == len(definitions())

        aliases = store.definition_aliases()
        for definition in definitions():
            assert set(definition.get("tools", [])) <= aliases

        graph_paths = sorted(Path("configs/graphs").glob("*.json"))
        for path in graph_paths:
            graph = load_graph(path)
            store.upsert_graph(graph)
            store.upsert_graph(graph)
            assert store.get_graph(graph.alias, graph.version).sha256 == graph.sha256
        assert store.counts()["graphs"] == len(graph_paths)

        now = datetime.now(timezone.utc)
        run = BenchmarkRunRecord(
            run_id=run_id, benchmark="textcraft", split="test", depth=2, mode="react",
            seed_hash="0" * 64, seeds=[0], graph_alias=None, graph_hash=None,
            config_snapshot={"integration": True}, source_commit="test", output_dir="test",
            status="completed", started_at=now, finished_at=now,
        )
        case = CaseRecord(
            case_id=f"case-{uuid.uuid4().hex}", run_id=run_id, benchmark="textcraft",
            split="test", depth=2, seed=0, index=0, result="won", graph_alias=None,
            graph_hash=None, num_steps=1, token_usage={}, started_at=now, finished_at=now,
        )
        output = TaskOutputRecord(
            task_output_id=f"output-{uuid.uuid4().hex}", run_id=run_id,
            case_id=case.case_id, subtask_id=None, step=0, decision={}, observation="ok", state={},
        )
        store.insert_run(run)
        store.insert_case(case)
        store.insert_task_output(output)
        with pytest.raises(DuplicateKeyError):
            store.insert_run(run)

        pubsub = bus.client.pubsub()
        pubsub.subscribe(f"luna:run:{run_id}:events")
        assert pubsub.get_message(timeout=2)["type"] == "subscribe"
        bus.publish(run_id, {"event": "ping"})
        message = pubsub.get_message(timeout=2)
        assert message and '"event": "ping"' in message["data"]
        pubsub.close()

        bus.client.set(f"luna:run:{run_id}:cache:item", "one")
        bus.client.set(f"luna:run:{other_run_id}:cache:item", "two")
        assert bus.clear_run_cache(run_id) == 1
        assert bus.client.get(f"luna:run:{other_run_id}:cache:item") == "two"
    finally:
        bus.client.delete(f"luna:run:{other_run_id}:cache:item")
        store.client.drop_database(database)
