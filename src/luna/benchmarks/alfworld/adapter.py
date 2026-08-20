"""Deterministic ALFWorld OOD adapter with actionable data preflight errors."""

from __future__ import annotations

import os
import re
from importlib import resources
from pathlib import Path

import yaml

from luna.benchmarks.base import BenchmarkAdapter, StepResult


class ALFWorldAdapter(BenchmarkAdapter):
    benchmark_name = "alfworld"

    def __init__(self):
        self.environment = None
        self.info: dict = {}
        self.observation = ""
        self.task = ""
        self._success = False
        self.steps = 0

    async def initialize(
        self, seed: int, index: int, depth: int | None, output_dir: Path | None = None
    ) -> None:
        data_dir = require_alfworld_data()
        if seed < 0 or seed >= 134:
            raise ValueError("ALFWorld eval_out_of_distribution seeds must be in 0..133")
        try:
            from luna.benchmarks.alfworld.alfred_tw_env import AlfredTWEnv
        except ImportError as exc:
            raise RuntimeError("Install ALFWorld support with: pip install '.[alfworld]'") from exc

        config_resource = resources.files("luna.benchmarks.alfworld") / "base_config.yaml"
        with resources.as_file(config_resource) as config_path:
            with config_path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
        config["general"]["use_cuda"] = os.getenv("LUNA_USE_CUDA", "false").lower() == "true"
        os.environ["ALFWORLD_DATA"] = str(data_dir)
        wrapper = AlfredTWEnv(config, train_eval="eval_out_of_distribution", game_index=seed)
        await wrapper.collect_game_files()
        self.environment = await wrapper.init_env(batch_size=1)
        welcome, self.info = self.environment.reset()
        text = welcome[0]
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        self.observation = lines[1] if len(lines) > 1 else lines[0]
        task_line = next((line for line in lines if line.startswith("Your task is to:")), lines[-1])
        self.task = re.sub(r"^Your task is to:\s*", "", task_line)
        self._success, self.steps = False, 0

    def task_prompt(self) -> str:
        return (
            "You are playing ALFWorld. Use only admissible commands and preserve exact numbered "
            f"object names. Task: {self.task}"
        )

    def state(self) -> dict:
        return {
            "task": self.task,
            "observation": self.observation,
            "admissible_commands": self.admissible_actions(),
            "success": self._success,
            "steps": self.steps,
        }

    def admissible_actions(self) -> list[str]:
        if self.environment is None:
            raise RuntimeError("ALFWorld adapter is not initialized")
        return [command for command in self.info["admissible_commands"][0] if command != "help"]

    async def step(self, action: str) -> StepResult:
        if self.environment is None:
            raise RuntimeError("ALFWorld adapter is not initialized")
        if action not in self.admissible_actions():
            return StepResult(
                observation="The required command is not available in the current state.",
                success=self._success,
                terminal=False,
            )
        observations, _, successes, info = self.environment.step([action])
        self.observation, self.info = observations[0], info
        self._success = bool(successes[0])
        self.steps += 1
        return StepResult(
            observation=self.observation,
            success=self._success,
            terminal=self._success,
        )

    async def finalize(self, answer: str | None) -> bool:
        return self._success

    @property
    def success(self) -> bool:
        return self._success


def require_alfworld_data() -> Path:
    configured = os.getenv("ALFWORLD_DATA")
    if not configured:
        raise RuntimeError(
            "ALFWORLD_DATA is not set. Install the optional dependency with "
            "`pip install 'luna-benchmarks[alfworld]'`, run `alfworld-download`, then set ALFWORLD_DATA "
            "to the directory containing json_2.1.1/ and logic/."
        )
    path = Path(configured).expanduser().resolve()
    required = [
        path / "json_2.1.1" / "valid_unseen",
        path / "logic" / "alfred.pddl",
        path / "logic" / "alfred.twl2",
    ]
    missing = [str(item.relative_to(path)) for item in required if not item.exists()]
    if missing:
        raise RuntimeError(
            f"ALFWORLD_DATA={path} is incomplete; missing: {', '.join(missing)}. "
            "Run `alfworld-download` using the official ALFWorld package."
        )
    return path

