"""TextCraft (ADaPT): Minecraft-recipe crafting at depths 2, 3 and 4 (paper Appendix B.1.1, B.2.2)."""

from __future__ import annotations

import re
from collections.abc import Mapping

from ctg.config import BenchmarkName, Budgets, ExperimentConfig, ModelSpec, Role
from ctg.envs.base import GENERALIST, Benchmark, Environment, ExecutorProfile, InstanceRef
from ctg.envs.textcraft import tools
from ctg.envs.textcraft.env import TextCraftEnv
from ctg.envs.textcraft.recipes import load_recipe_book
from ctg.llm.base import LLM
from ctg.prompts import render

GLOBAL_TOOL_CALLS = {2: 30, 3: 50, 4: 100}
"""C_g per depth (Sec. 4.1.5)."""
LOCAL_TOOL_CALLS = 5
"""C_l (Sec. 4.1.5)."""

_VARIANT = re.compile(r"textcraft_depth([234])")

_ROSTER = {
    GENERALIST: ExecutorProfile(
        name=GENERALIST,
        description=(
            "Runs any TextCraft command (get raw items, craft with a listed crafting command, check the "
            "inventory) through textcraft_select_command."
        ),
        tools=(tools.SELECT_COMMAND,),
    ),
    "gatherer": ExecutorProfile(
        name="gatherer",
        description=(
            "Obtains raw items with textcraft_get_item and checks the inventory with textcraft_inventory. "
            "Cannot craft."
        ),
        tools=(tools.GET_ITEM, tools.INVENTORY),
    ),
    "crafter": ExecutorProfile(
        name="crafter",
        description=(
            "Runs listed crafting commands with textcraft_craft and checks the inventory with "
            "textcraft_inventory. Cannot obtain raw items."
        ),
        tools=(tools.CRAFT, tools.INVENTORY),
    ),
}


class TextCraftBenchmark(Benchmark):
    name = BenchmarkName.TEXTCRAFT

    @property
    def description(self) -> str:
        return render("benchmarks/textcraft")

    @property
    def roster(self) -> Mapping[str, ExecutorProfile]:
        return _ROSTER

    def budgets(self, config: ExperimentConfig) -> Budgets:
        if config.textcraft_depth is None:
            raise ValueError("TextCraft budgets need textcraft_depth")
        return Budgets(
            global_tool_calls=GLOBAL_TOOL_CALLS[config.textcraft_depth], local_tool_calls=LOCAL_TOOL_CALLS
        )

    def enumerate_instances(self, variant: str) -> list[InstanceRef]:
        match = _VARIANT.fullmatch(variant)
        if match is None:
            raise ValueError(f"unknown TextCraft variant {variant!r}")
        depth = int(match[1])
        return [
            InstanceRef(id=goal, meta={"goal": goal, "depth": depth})
            for goal in load_recipe_book().goals(depth)
        ]

    async def make_env(
        self,
        instance: InstanceRef,
        config: ExperimentConfig,
        llm: LLM,
        models: Mapping[Role, ModelSpec],
    ) -> Environment:
        return TextCraftEnv(load_recipe_book(), instance.id)
