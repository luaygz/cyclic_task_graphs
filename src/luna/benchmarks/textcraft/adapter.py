"""Public TextCraft benchmark adapter."""

from __future__ import annotations

from pathlib import Path

from luna.benchmarks.base import BenchmarkAdapter, StepResult
from luna.benchmarks.textcraft.environment import TextCraftEnvironment
from luna.benchmarks.textcraft.solver import solve_actions
from luna.benchmarks.textcraft.utils import item_id_to_str


class TextCraftAdapter(BenchmarkAdapter):
    benchmark_name = "textcraft"

    def __init__(self):
        self.environment: TextCraftEnvironment | None = None
        self.seed = 0
        self.depth = 2
        self._task = ""
        self._success = False
        self.steps = 0

    async def initialize(
        self, seed: int, index: int, depth: int | None, output_dir: Path | None = None
    ) -> None:
        if depth not in {2, 3, 4}:
            raise ValueError("TextCraft depth must be 2, 3, or 4")
        self.seed, self.depth, self.steps, self._success = seed, depth, 0, False
        self.environment = TextCraftEnvironment()
        self._task = self.environment.reset(seed=seed, min_depth=depth, max_depth=depth)

    def task_prompt(self) -> str:
        return self._task

    def state(self) -> dict:
        env = self._require_environment()
        return {
            "goal": f"craft {item_id_to_str(env.goal)}",
            "inventory": {item_id_to_str(key): value for key, value in env.inventory.items()},
            "success": self._success,
            "steps": self.steps,
        }

    def admissible_actions(self) -> list[str]:
        env = self._require_environment()
        return ["inventory", "get <count> <raw item>", *env.expanded_recipe_commands()]

    async def step(self, action: str) -> StepResult:
        observation, won = self._require_environment().step(action.strip())
        self.steps += 1
        self._success = self._success or won
        return StepResult(observation=observation, success=self._success, terminal=self._success)

    async def finalize(self, answer: str | None) -> bool:
        return self._success

    @property
    def success(self) -> bool:
        return self._success

    def oracle_actions(self) -> list[str]:
        return solve_actions(self._require_environment())

    def _require_environment(self) -> TextCraftEnvironment:
        if self.environment is None:
            raise RuntimeError("TextCraft adapter is not initialized")
        return self.environment
