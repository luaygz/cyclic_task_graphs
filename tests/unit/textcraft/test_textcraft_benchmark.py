from typing import Literal

import pytest

from ctg.config import BenchmarkName, Budgets, Controller, ExperimentConfig
from ctg.envs.base import GENERALIST, InstanceRef
from ctg.envs.textcraft.benchmark import TextCraftBenchmark
from ctg.envs.textcraft.tools import CRAFT, GET_ITEM, INVENTORY, SELECT_COMMAND
from ctg.llm.fake import FakeLLM


def _config(depth: Literal[2, 3, 4]) -> ExperimentConfig:
    return ExperimentConfig(
        name="textcraft-test",
        benchmark=BenchmarkName.TEXTCRAFT,
        textcraft_depth=depth,
        controller=Controller.REACT,
    )


def test_roster_splits_get_and_craft_between_specialists() -> None:
    benchmark = TextCraftBenchmark()

    assert [(p.name, p.tools, p.max_tool_calls) for p in benchmark.executor_roster(specialists=False)] == [
        (GENERALIST, (SELECT_COMMAND,), None)
    ]
    assert [(p.name, p.tools, p.max_tool_calls) for p in benchmark.executor_roster(specialists=True)] == [
        ("gatherer", (GET_ITEM, INVENTORY), None),
        ("crafter", (CRAFT, INVENTORY), None),
    ]


@pytest.mark.parametrize(("depth", "global_calls"), [(2, 30), (3, 50), (4, 100)])
def test_budgets_follow_section_4_1_5(depth: Literal[2, 3, 4], global_calls: int) -> None:
    assert TextCraftBenchmark().budgets(_config(depth)) == Budgets(
        global_tool_calls=global_calls, local_tool_calls=5
    )


@pytest.mark.parametrize("variant", ["textcraft_depth1", "textcraft_depth5", "textcraft", "alfworld_train"])
def test_unknown_variants_are_rejected(variant: str) -> None:
    with pytest.raises(ValueError, match="unknown TextCraft variant"):
        TextCraftBenchmark().enumerate_instances(variant)


@pytest.mark.textcraft_recipes
async def test_make_env_starts_the_instance_goal() -> None:
    benchmark = TextCraftBenchmark()
    instance = InstanceRef(id="polished_granite", meta={"goal": "polished_granite", "depth": 3})

    env = await benchmark.make_env(
        instance, _config(3), FakeLLM(lambda request: pytest.fail("no LLM calls")), {}
    )

    assert env.objective.endswith("Goal: craft polished granite.")
    assert env.tool_calls == 0 and not env.done
    assert "$" not in benchmark.description
