"""Typed, side-effect-free configuration loading."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

BenchmarkName = Literal["alfworld", "textcraft", "finance_agent"]
ExecutionMode = Literal["react", "spec-cyc", "gen-cyc", "depdag-retry"]
SOURCE_COMMIT = "153bfc03716be20757e7d7b4480b80c0bcc3025d"


def repository_root(start: Path | None = None) -> Path:
    """Find a checkout root without creating files or reading environment state."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    return current


class ServiceSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    mongodb_uri: str = "mongodb://127.0.0.1:27017"
    mongodb_db: str = "luna_benchmarks"
    redis_url: str = "redis://127.0.0.1:6379/0"

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        return cls(
            mongodb_uri=os.getenv("LUNA_MONGODB_URI", cls.model_fields["mongodb_uri"].default),
            mongodb_db=os.getenv("LUNA_MONGODB_DB", cls.model_fields["mongodb_db"].default),
            redis_url=os.getenv("LUNA_REDIS_URL", cls.model_fields["redis_url"].default),
        )


class ModelSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Literal["openai", "fake"] = "openai"
    executor_alias: str = "executor"
    executor_model: str = "gpt-4o-mini"
    planner_alias: str = "planner"
    planner_model: str = "gpt-4o-mini"
    router_alias: str = "router"
    router_model: str = "gpt-4o-mini"
    judge_alias: str = "judge"
    judge_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    reasoning_effort: str | None = None


class RoutingSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    prioritized: bool = False
    weights_file: str | None = None
    weights_factor: float = 0.3
    routing_temperature: float = 0.0
    position_temperature: float = 1.0
    top_k: int = 3
    transition_top_k: int = 0
    drop_lowest_percentile_edges: float = 10.0


class RunConfig(BaseModel):
    """An immutable snapshot of every behavior-affecting run setting."""

    model_config = ConfigDict(frozen=True)

    benchmark: BenchmarkName
    mode: ExecutionMode
    split: str = "test"
    depth: int | None = None
    max_cases: int | None = Field(default=None, ge=1)
    concurrency: int = Field(default=1, ge=1)
    graph_manifest: Path | None = None
    output_dir: Path = Path("outputs")
    include_n_shot: bool = False
    tool_limit: int = Field(default=20, ge=1)
    subtask_tool_limit: int = Field(default=5, ge=1)
    retry_limit: int = Field(default=2, ge=0)
    perturbation_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    storage_enabled: bool = True
    clear_run_cache: bool = True
    model: ModelSettings = Field(default_factory=ModelSettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    source_commit: str = SOURCE_COMMIT

    @model_validator(mode="after")
    def validate_benchmark_settings(self) -> "RunConfig":
        if self.benchmark == "textcraft" and self.depth not in {2, 3, 4}:
            raise ValueError("TextCraft requires depth 2, 3, or 4")
        if self.benchmark != "textcraft" and self.depth is not None:
            raise ValueError("depth is only valid for TextCraft")
        if self.mode == "gen-cyc" and self.graph_manifest is None:
            raise ValueError("gen-cyc requires graph_manifest")
        if self.mode != "gen-cyc" and self.graph_manifest is not None:
            raise ValueError("graph_manifest is only valid with gen-cyc")
        return self

    def resolved(self, root: Path) -> "RunConfig":
        values = self.model_dump()
        values["output_dir"] = _resolve_path(root, self.output_dir)
        if self.graph_manifest is not None:
            values["graph_manifest"] = _resolve_path(root, self.graph_manifest)
        return RunConfig.model_validate(values)


class SeedManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    benchmark: BenchmarkName
    split: str
    depth: int | None = None
    shuffle_seed: int | None = None
    seeds: list[int]
    source_commit: str = SOURCE_COMMIT

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (root / value).resolve()


def load_run_config(path: Path, overrides: dict[str, Any] | None = None) -> RunConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return RunConfig.model_validate(raw)


def load_seed_manifest(path: Path) -> SeedManifest:
    with path.open("r", encoding="utf-8") as handle:
        return SeedManifest.model_validate(json.load(handle))


def parse_seeds(value: str) -> list[int]:
    """Parse comma-separated integers and inclusive ranges such as ``0,3-5``."""
    result: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token[1:]:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"descending seed range is not supported: {token}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(token))
    return list(dict.fromkeys(result))


def seed_list_hash(seeds: list[int]) -> str:
    canonical = json.dumps(seeds, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

