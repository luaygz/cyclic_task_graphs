"""Stable schemas shared by model adapters, graphs, and persistence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class CriterionEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    reasoning: str
    criterion: str
    operator: Literal["correctness", "contradiction"]
    passed: bool


class RubricEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    overall_reasoning: str
    criteria_evaluations: list[CriterionEvaluation]
    overall_passed: bool
    usage: Usage = Field(default_factory=Usage)


class Decision(BaseModel):
    """One model-selected environment action or terminal answer."""

    reasoning: list[str] = Field(default_factory=list)
    action: str | None = None
    final_answer: str | None = None
    done: bool = False
    usage: Usage = Field(default_factory=Usage)

    @model_validator(mode="after")
    def validate_choice(self) -> "Decision":
        if self.done and self.final_answer is None:
            raise ValueError("terminal decisions require final_answer")
        if not self.done and self.action is None:
            raise ValueError("non-terminal decisions require action")
        return self


class DecisionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    benchmark: str
    mode: str
    task: str
    state: dict[str, Any]
    admissible_actions: list[str]
    trajectory: list[dict[str, Any]]
    subtask: str | None = None


class Subtask(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    agent_alias: str
    tool_aliases: list[str]
    prompt: str
    success_criterion: str
    dependencies: list[str] = Field(default_factory=list)
    retry_limit: int = Field(default=0, ge=0)


class GraphEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    condition: str = "success"


class GraphManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    alias: str
    version: str
    benchmark: str
    graph_type: Literal["cyclic", "dependency_dag"]
    entrypoint: str
    subtasks: list[Subtask]
    edges: list[GraphEdge]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> "GraphManifest":
        ids = {node.id for node in self.subtasks}
        if len(ids) != len(self.subtasks):
            raise ValueError("subtask ids must be unique")
        if self.entrypoint not in ids:
            raise ValueError("entrypoint must reference a subtask")
        for node in self.subtasks:
            missing = set(node.dependencies) - ids
            if missing:
                raise ValueError(f"{node.id} has missing dependencies: {sorted(missing)}")
        for edge in self.edges:
            if edge.source not in ids or edge.target not in ids:
                raise ValueError(f"edge references unknown node: {edge}")
        if self.graph_type == "dependency_dag":
            _assert_acyclic(ids, self.edges)
        return self

    def normalized(self) -> dict[str, Any]:
        data = self.model_dump()
        data["subtasks"] = sorted(data["subtasks"], key=lambda item: item["id"])
        data["edges"] = sorted(
            data["edges"], key=lambda item: (item["source"], item["target"], item["condition"])
        )
        return data

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.normalized(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TaskOutputRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_output_id: str
    run_id: str
    case_id: str
    subtask_id: str | None
    step: int
    decision: dict[str, Any]
    observation: str
    state: dict[str, Any]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TestCaseRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    run_id: str
    benchmark: str
    split: str
    depth: int | None
    seed: int
    index: int
    result: Literal["won", "lost", "crashed"]
    graph_alias: str | None
    graph_hash: str | None
    num_steps: int
    token_usage: Usage
    exception: str | None = None
    trajectory_path: str | None = None
    summary_path: str | None = None
    started_at: datetime
    finished_at: datetime


class BenchmarkRunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    benchmark: str
    split: str
    depth: int | None
    mode: str
    seed_hash: str
    seeds: list[int]
    graph_alias: str | None
    graph_hash: str | None
    config_snapshot: dict[str, Any]
    source_commit: str
    output_dir: str
    status: Literal["running", "completed", "failed"] = "running"
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None


def _assert_acyclic(ids: set[str], edges: list[GraphEdge]) -> None:
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in ids}
    indegree = {node_id: 0 for node_id in ids}
    for edge in edges:
        outgoing[edge.source].append(edge.target)
        indegree[edge.target] += 1
    ready = [node_id for node_id, count in indegree.items() if count == 0]
    visited = 0
    while ready:
        node_id = ready.pop()
        visited += 1
        for target in outgoing[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(ids):
        raise ValueError("dependency_dag manifests cannot contain cycles")

