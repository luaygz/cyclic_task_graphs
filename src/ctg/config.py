"""Experiment configuration, model presets, roles and budgets.

Defaults follow the paper: budgets from Sec. 4.1.5 and decoding settings from Table 8.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CONFIGS_DIR = REPO_ROOT / "configs"
MODELS_FILE = CONFIGS_DIR / "models.yaml"


class Provider(StrEnum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"


class Role(StrEnum):
    """Who is making an LLM call. Used for decoding settings and token accounting."""

    PLANNER = "planner"
    EXECUTOR = "executor"
    ROUTER = "router"
    JUDGE = "judge"
    TOOL = "tool"
    NSHOT = "nshot"


class Controller(StrEnum):
    REACT = "react"
    DEPDAG = "depdag"
    DEPDAG_NORETRY = "depdag_noretry"
    SPEC_CYC = "spec_cyc"
    GEN_CYC = "gen_cyc"

    @property
    def is_cyclic(self) -> bool:
        return self in (Controller.SPEC_CYC, Controller.GEN_CYC)

    @property
    def is_dependency(self) -> bool:
        return self in (Controller.DEPDAG, Controller.DEPDAG_NORETRY)


class BenchmarkName(StrEnum):
    TEXTCRAFT = "textcraft"
    ALFWORLD = "alfworld"
    FINANCE_AGENT = "finance_agent"


class ExecutorMode(StrEnum):
    GENERALIST = "generalist"
    SPECIALIST = "specialist"


ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class ModelSpec(BaseModel):
    """A concrete model on a provider. Prices are USD per million tokens.

    OpenRouter reports the exact cost of each call, so OpenRouter presets need no prices.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    provider: Provider = Provider.OPENAI
    reasoning_effort: ReasoningEffort | None = None
    supports_temperature: bool = True
    input_usd_per_mtok: float | None = None
    cached_input_usd_per_mtok: float | None = None
    output_usd_per_mtok: float | None = None


class RoleSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float | None
    max_tokens: int | None


# Table 8. `None` means provider default.
ROLE_SETTINGS: dict[Role, RoleSettings] = {
    Role.PLANNER: RoleSettings(temperature=1.0, max_tokens=16384),
    Role.EXECUTOR: RoleSettings(temperature=0.0, max_tokens=2048),
    Role.ROUTER: RoleSettings(temperature=0.0, max_tokens=4096),
    Role.JUDGE: RoleSettings(temperature=None, max_tokens=None),
    Role.TOOL: RoleSettings(temperature=None, max_tokens=None),
    Role.NSHOT: RoleSettings(temperature=0.0, max_tokens=1024),
}


class Budgets(BaseModel):
    """Tool-call budgets: global per episode (C_g) and local per segment (C_l)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    global_tool_calls: int = Field(gt=0)
    local_tool_calls: int = Field(gt=0)


class ModelAssignment(BaseModel):
    """Model preset names (keys of configs/models.yaml) for each role."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    planner: str = "gpt-4o-mini"
    executor: str = "gpt-4o-mini"
    router: str = "gpt-4o-mini"
    judge: str = "gpt-5"
    tool: str = "gpt-5-mini-minimal"
    nshot: str = "gpt-4o-mini"


class ExperimentConfig(BaseModel):
    """One experiment cell: a benchmark setting, a controller and its model/tool configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    benchmark: BenchmarkName
    textcraft_depth: Literal[2, 3, 4] | None = None
    alfworld_split: Literal["train", "valid_seen", "valid_unseen"] | None = None
    split: Literal["train", "test"] = "test"
    instances: tuple[str, ...] | None = None
    controller: Controller
    executors: ExecutorMode = ExecutorMode.GENERALIST
    models: ModelAssignment = ModelAssignment()
    nshot: Path | None = None
    perturbation: float = Field(default=0.0, ge=0.0, le=1.0)
    repeats: int = Field(default=3, ge=1)
    seed: int = 0
    concurrency: int = Field(default=8, ge=1)
    web_search: Literal["serpapi", "exa"] = "serpapi"
    budgets: Budgets | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.benchmark is BenchmarkName.TEXTCRAFT and self.textcraft_depth is None:
            raise ValueError("textcraft experiments need textcraft_depth")
        if self.benchmark is not BenchmarkName.TEXTCRAFT and self.textcraft_depth is not None:
            raise ValueError("textcraft_depth only applies to textcraft")
        if self.benchmark is BenchmarkName.ALFWORLD and self.alfworld_split is None:
            raise ValueError("alfworld experiments need alfworld_split")
        if self.benchmark is not BenchmarkName.ALFWORLD and self.alfworld_split is not None:
            raise ValueError("alfworld_split only applies to alfworld")
        if self.perturbation > 0 and not self.controller.is_cyclic:
            raise ValueError("fault injection is defined for cyclic controllers only")
        if self.controller is Controller.REACT and self.nshot is not None:
            raise ValueError("ReAct does not use n-shot summaries")
        if self.controller is Controller.REACT and self.executors is ExecutorMode.SPECIALIST:
            raise ValueError("ReAct is a single generalist agent")
        return self

    @property
    def variant(self) -> str:
        """Name of the frozen instance list in data/splits/ for this benchmark setting."""
        match self.benchmark:
            case BenchmarkName.TEXTCRAFT:
                return f"textcraft_depth{self.textcraft_depth}"
            case BenchmarkName.ALFWORLD:
                return f"alfworld_{self.alfworld_split}"
            case BenchmarkName.FINANCE_AGENT:
                return "finance_agent"

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        with path.open() as f:
            return cls.model_validate(yaml.safe_load(f))


def load_model_presets(path: Path = MODELS_FILE) -> dict[str, ModelSpec]:
    with path.open() as f:
        raw: dict[str, dict[str, object]] = yaml.safe_load(f)
    return {key: ModelSpec.model_validate(value) for key, value in raw.items()}


def resolve_models(assignment: ModelAssignment, presets: dict[str, ModelSpec]) -> dict[Role, ModelSpec]:
    resolved: dict[Role, ModelSpec] = {}
    for role in Role:
        key = getattr(assignment, role.value)
        if key not in presets:
            raise KeyError(f"model preset {key!r} for role {role.value} is not defined in {MODELS_FILE.name}")
        resolved[role] = presets[key]
    return resolved
