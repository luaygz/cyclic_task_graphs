"""Small environment protocol shared by all public benchmark adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StepResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    observation: str
    success: bool
    terminal: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkAdapter(ABC):
    benchmark_name: str

    @abstractmethod
    async def initialize(
        self, seed: int, index: int, depth: int | None, output_dir: Path | None = None
    ) -> None: ...

    @abstractmethod
    def task_prompt(self) -> str: ...

    @abstractmethod
    def state(self) -> dict[str, Any]: ...

    @abstractmethod
    def admissible_actions(self) -> list[str]: ...

    @abstractmethod
    async def step(self, action: str) -> StepResult: ...

    @abstractmethod
    async def finalize(self, answer: str | None) -> bool: ...

    @property
    @abstractmethod
    def success(self) -> bool: ...

