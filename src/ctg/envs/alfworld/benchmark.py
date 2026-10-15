"""ALFWorld (Appendix B.1.2, B.2.3): household tasks played through TextWorld."""

from __future__ import annotations

from collections.abc import Mapping

from ctg.config import BenchmarkName, Budgets, ExperimentConfig, ModelSpec, Role
from ctg.envs.alfworld.env import ALFWorldEnv
from ctg.envs.alfworld.games import Split, list_games
from ctg.envs.alfworld.tools import EXPLORE, MANAGE_INVENTORY, MANIPULATE_OBJECT, SELECT_COMMAND
from ctg.envs.base import GENERALIST, Benchmark, Environment, ExecutorProfile, InstanceRef
from ctg.llm.base import LLM
from ctg.prompts import render

_VARIANT_SPLITS: dict[str, Split] = {
    "alfworld_train": "train",
    "alfworld_valid_seen": "valid_seen",
    "alfworld_valid_unseen": "valid_unseen",
}

_ROSTER = {
    GENERALIST: ExecutorProfile(
        name=GENERALIST,
        description="Executes any admissible command: navigation, inventory and object manipulation.",
        tools=(SELECT_COMMAND,),
    ),
    "explorer": ExecutorProfile(
        name="explorer",
        description="Navigates and inspects the room: look, go to, examine, open and close receptacles.",
        tools=(EXPLORE,),
    ),
    "inventory_manager": ExecutorProfile(
        name="inventory_manager",
        description="Checks the inventory, takes objects from receptacles and places the held object.",
        tools=(MANAGE_INVENTORY,),
    ),
    "object_manipulator": ExecutorProfile(
        name="object_manipulator",
        description="Uses, heats, cools, cleans and slices objects with appliances and tools.",
        tools=(MANIPULATE_OBJECT,),
    ),
}

_BUDGETS = Budgets(global_tool_calls=75, local_tool_calls=15)


class ALFWorldBenchmark(Benchmark):
    name = BenchmarkName.ALFWORLD

    @property
    def description(self) -> str:
        return render("benchmarks/alfworld")

    @property
    def roster(self) -> Mapping[str, ExecutorProfile]:
        return _ROSTER

    def budgets(self, config: ExperimentConfig) -> Budgets:
        return _BUDGETS

    def enumerate_instances(self, variant: str) -> list[InstanceRef]:
        split = _VARIANT_SPLITS.get(variant)
        if split is None:
            raise ValueError(
                f"unknown ALFWorld variant {variant!r}; expected one of {sorted(_VARIANT_SPLITS)}"
            )
        return [
            InstanceRef(id=game.id, meta={"path": game.path, "task_type": game.task_type})
            for game in list_games(split)
        ]

    async def make_env(
        self,
        instance: InstanceRef,
        config: ExperimentConfig,
        llm: LLM,
        models: Mapping[Role, ModelSpec],
    ) -> Environment:
        return await ALFWorldEnv.start(instance.meta["path"])
