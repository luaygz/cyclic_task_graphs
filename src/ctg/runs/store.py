"""On-disk layout of a run.

    runs/<name>/manifest.json                       RunManifest
    runs/<name>/graph.json                          Gen-Cyc graph planned once per run (StoredGraph)
    runs/<name>/episodes/r<repeat>/<instance>.json  EpisodeRecord
    runs/<name>/.lock                               held (flock) by the process running the experiment

Every write goes to a temporary file in the target directory and is renamed into place, so a reader or a
resumed run never sees a partial file.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, Field, ValidationError

from ctg.graphs.model import CyclicGraph
from ctg.llm.base import LLMCallRecord
from ctg.runs.records import SCHEMA_VERSION, EndReason, EpisodeRecord, RunManifest

VersionedT = TypeVar("VersionedT", EpisodeRecord, RunManifest)


class RunInUseError(RuntimeError):
    """Another process holds the run directory's lock."""


class SchemaVersionError(ValueError):
    """A run file was written in another schema version than `SCHEMA_VERSION`."""


class StoredGraph(BaseModel):
    """The Gen-Cyc graph of a run with the planner notes and the calls that produced it."""

    graph: CyclicGraph
    notes: dict[str, Any] = Field(default_factory=dict)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    created_at: datetime


def write_json_atomic(path: Path, text: str) -> None:
    """Replace `path` with `text` in one rename. The file is created with mode 0666 minus the umask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def graph_path(self) -> Path:
        return self.root / "graph.json"

    @property
    def episodes_dir(self) -> Path:
        return self.root / "episodes"

    @property
    def lock_path(self) -> Path:
        return self.root / ".lock"

    def episode_path(self, repeat: int, instance_id: str) -> Path:
        return self.episodes_dir / f"r{repeat}" / f"{instance_id}.json"

    @contextmanager
    def lock(self) -> Generator[None]:
        """Hold an exclusive advisory lock on the run directory (created if missing) until the block exits.

        Raises `RunInUseError` at once if another holder has it. The lock is released when its file
        descriptor closes, so a crashed process never leaves the directory locked.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RunInUseError(f"{self.root} is in use by another `ctg run` process") from exc
            yield
        finally:
            os.close(fd)

    def episode(self, repeat: int, instance_id: str) -> EpisodeRecord | None:
        path = self.episode_path(repeat, instance_id)
        if not path.is_file():
            return None
        return _load(EpisodeRecord, path)

    def manifest(self) -> RunManifest | None:
        if not self.manifest_path.is_file():
            return None
        return _load(RunManifest, self.manifest_path)

    def write_manifest(self, manifest: RunManifest) -> None:
        write_json_atomic(self.manifest_path, manifest.model_dump_json(indent=2))

    def gen_cyc_graph(self) -> StoredGraph | None:
        if not self.graph_path.is_file():
            return None
        return StoredGraph.model_validate_json(self.graph_path.read_text(encoding="utf-8"))

    def write_gen_cyc_graph(self, stored: StoredGraph) -> None:
        write_json_atomic(self.graph_path, stored.model_dump_json(indent=2))

    def write_episode(self, record: EpisodeRecord) -> Path:
        path = self.episode_path(record.repeat, record.instance_id)
        write_json_atomic(path, record.model_dump_json(indent=2))
        return path

    def episodes(self) -> Iterator[EpisodeRecord]:
        """Every stored episode, ordered by repeat and instance id."""
        if not self.episodes_dir.is_dir():
            return
        repeat_dirs = [d for d in self.episodes_dir.iterdir() if d.is_dir() and d.name.startswith("r")]
        for repeat_dir in sorted(repeat_dirs, key=lambda d: int(d.name[1:])):
            for path in sorted(repeat_dir.glob("*.json")):
                yield _load(EpisodeRecord, path)

    def completed(self) -> set[tuple[int, str]]:
        """(repeat, instance id) of episodes that need no rerun. Errored episodes are not complete."""
        return {
            (record.repeat, record.instance_id)
            for record in self.episodes()
            if record.outcome.end_reason is not EndReason.ERROR
        }


def _load(model: type[VersionedT], path: Path) -> VersionedT:
    """Validate a manifest or episode record. A file of another schema version raises SchemaVersionError,
    also when its contents no longer validate; a file without a version raises the ValidationError."""
    text = path.read_text(encoding="utf-8")
    try:
        loaded = model.model_validate_json(text)
    except ValidationError:
        version = _stored_version(text)
        if version is not None:
            _check_version(path, version)
        raise
    _check_version(path, loaded.schema_version)
    return loaded


def _stored_version(text: str) -> object:
    """The `schema_version` value of a JSON object, or None if `text` holds none."""
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data.get("schema_version") if isinstance(data, dict) else None


def _check_version(path: Path, version: object) -> None:
    if version != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{path} was written by schema version {version}; this code reads {SCHEMA_VERSION}. "
            "Start a new run directory: use a new experiment name or remove the old one"
        )
