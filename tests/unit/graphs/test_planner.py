from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from ctg.config import BenchmarkName, Budgets, ExperimentConfig, ModelSpec, Role
from ctg.envs.base import GENERALIST, Benchmark, Environment, ExecutorProfile, InstanceRef
from ctg.graphs.model import END, GIVE_UP, CyclicGraph, DependencyGraph
from ctg.graphs.planner import MAX_ATTEMPTS, plan_cyclic, plan_dependency
from ctg.graphs.schemas import SubtaskPlan
from ctg.llm.base import LLM, LLMOutputError
from ctg.llm.fake import FakeLLM, FakeRequest
from ctg.prompts import render

MODEL = ModelSpec(name="planner-model")
OBJECTIVE = "Craft 1 polished granite. Crafting commands: craft 1 granite using 1 diorite, 1 quartz."
GENERALIST_PROFILE = ExecutorProfile(GENERALIST, "Has every tool.", ("textcraft_select_command",))
GATHERER = ExecutorProfile("gatherer", "Gets raw items.", ("textcraft_get_item", "textcraft_inventory"))
CRAFTER = ExecutorProfile("crafter", "Crafts items.", ("textcraft_craft", "textcraft_inventory"))


class FakeBenchmark(Benchmark):
    name = BenchmarkName.TEXTCRAFT

    @property
    def description(self) -> str:
        return "A text world where items are gathered and crafted from recipes."

    @property
    def roster(self) -> Mapping[str, ExecutorProfile]:
        return {p.name: p for p in (GENERALIST_PROFILE, GATHERER, CRAFTER)}

    def budgets(self, config: ExperimentConfig) -> Budgets:
        raise AssertionError("not used by the planner")

    def enumerate_instances(self, variant: str) -> list[InstanceRef]:
        raise AssertionError("not used by the planner")

    async def make_env(
        self, instance: InstanceRef, config: ExperimentConfig, llm: LLM, models: Mapping[Role, ModelSpec]
    ) -> Environment:
        raise AssertionError("not used by the planner")


BENCHMARK = FakeBenchmark()

Edges = dict[str, list[str]]


def plan_answer(names: Sequence[str], entry: str | None = None, executor: str = GENERALIST) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "plan": "Gather, then craft.",
        "potential_blockers": ["Intermediate items cannot be gathered."],
        "refined_plan": "Craft intermediates before the target.",
        "name": "Crafting graph",
        "description": "Obtain ingredients and craft the target.",
        "subtasks": [
            {
                "name": name,
                "description": f"Achieve {name}.",
                "success_criteria": f"{name} is done.",
                "executor": executor,
            }
            for name in names
        ],
    }
    if entry is not None:
        answer["entry"] = entry
    return answer


def edges_of(schema: type[BaseModel]) -> Edges:
    """The edges a criteria schema asks for, read from the JSON schema the model sees."""
    json_schema = schema.model_json_schema()
    rows = {
        source: json_schema["$defs"][prop["$ref"].rsplit("/", 1)[-1]]
        for source, prop in json_schema["properties"].items()
    }
    return {source: list(row["properties"]) for source, row in rows.items()}


def write_criteria(edges: Edges) -> dict[str, dict[str, str]]:
    """The scripted planner's answer: one criterion per edge, keyed by the ids the schema shows."""
    return {
        source: {target: f"{source} -> {target}" for target in targets} for source, targets in edges.items()
    }


def describe_edges(edges: Edges) -> dict[str, dict[str, str]]:
    """The criteria the graph ends up with for that answer: keyed by graph ids (GIVE_UP back to END)."""
    return {
        source: {(END if target == GIVE_UP else target): text for target, text in row.items()}
        for source, row in write_criteria(edges).items()
    }


@dataclass
class Script:
    """Answers subtask calls from `plans` and criteria calls from `criteria`, in order."""

    plans: list[dict[str, Any]]
    criteria: list[Callable[[Edges], dict[str, dict[str, str]]]] = field(default_factory=list)

    def __call__(self, request: FakeRequest) -> BaseModel:
        assert request.kind == "structured" and request.role is Role.PLANNER and request.schema is not None
        if issubclass(request.schema, SubtaskPlan):
            return request.schema.model_validate(self.plans.pop(0))
        answer = self.criteria.pop(0) if self.criteria else write_criteria
        return request.schema.model_validate(answer(edges_of(request.schema)))


def criteria_requests(llm: FakeLLM) -> list[FakeRequest]:
    return [r for r in llm.requests if r.schema is not None and not issubclass(r.schema, SubtaskPlan)]


async def test_spec_cyc_builds_a_complete_graph_starting_at_the_named_entry() -> None:
    names = ["Find recipe", "Gather materials", "Craft target"]
    llm = FakeLLM(Script([plan_answer(names, entry="gather Materials")]))

    planned = await plan_cyclic(llm, MODEL, BENCHMARK, [GENERALIST_PROFILE], OBJECTIVE)

    graph = planned.graph
    ids = ["find_recipe", "gather_materials", "craft_target"]
    assert isinstance(graph, CyclicGraph)
    assert list(graph.ids) == ids
    assert graph.entry == "gather_materials"
    assert graph.criteria == {src: {dst: f"{src} -> {dst}" for dst in ids} for src in ids}
    assert [s.executor for s in graph.subtasks] == [GENERALIST] * 3

    subtasks_request, criteria_request = llm.requests
    assert criteria_request.schema is not None
    assert edges_of(criteria_request.schema) == {src: ids for src in ids}
    assert "Execution starts at `gather_materials`." in criteria_request.text
    assert all(OBJECTIVE in request.text for request in (subtasks_request, criteria_request))

    notes = planned.notes
    assert notes["refined_plan"] == "Craft intermediates before the target."
    assert notes["potential_blockers"] == ["Intermediate items cannot be gathered."]
    assert notes["subtasks_output"]["entry"] == "gather Materials"
    assert notes["criteria_output"]["craft_target"]["find_recipe"] == "craft_target -> find_recipe"
    assert notes["rejected"] == []


async def test_gen_cyc_prompts_carry_the_generic_instruction_instead_of_an_objective() -> None:
    generic = render("planner_generic").strip()
    names = ["Inspect state", "Acquire ingredients", "Craft"]
    gen_llm = FakeLLM(Script([plan_answer(names, entry="Inspect state")]))
    spec_llm = FakeLLM(Script([plan_answer(names, entry="Inspect state")]))

    await plan_cyclic(gen_llm, MODEL, BENCHMARK, [GENERALIST_PROFILE], None)
    await plan_cyclic(spec_llm, MODEL, BENCHMARK, [GENERALIST_PROFILE], OBJECTIVE)

    assert len(gen_llm.requests) == len(spec_llm.requests) == 2
    for gen_request, spec_request in zip(gen_llm.requests, spec_llm.requests, strict=True):
        assert generic in gen_request.text
        assert OBJECTIVE not in gen_request.text
        assert OBJECTIVE in spec_request.text
        assert generic not in spec_request.text
        assert BENCHMARK.description in gen_request.text


@pytest.mark.parametrize(
    ("retry", "expected"),
    [
        (True, {"a": ["b", "a", GIVE_UP], "b": ["c", "b", GIVE_UP], "c": ["c", GIVE_UP]}),
        (False, {"a": ["b", GIVE_UP], "b": ["c", GIVE_UP], "c": [GIVE_UP]}),
    ],
)
async def test_dependency_criteria_schema_covers_exactly_the_allowed_edges(
    retry: bool, expected: Edges
) -> None:
    llm = FakeLLM(Script([plan_answer(["A", "B", "C"])]))

    planned = await plan_dependency(llm, MODEL, BENCHMARK, [GENERALIST_PROFILE], OBJECTIVE, retry=retry)

    graph = planned.graph
    assert isinstance(graph, DependencyGraph)
    assert graph.retry is retry
    assert graph.entry == "a"
    assert graph.criteria == describe_edges(expected)

    subtasks_request, criteria_request = llm.requests
    assert subtasks_request.schema is not None and "entry" not in subtasks_request.schema.model_fields
    assert criteria_request.schema is not None
    json_schema = criteria_request.schema.model_json_schema()
    assert json_schema["required"] == list(json_schema["properties"]) == list(expected)
    assert json_schema["additionalProperties"] is False
    for source, targets in expected.items():
        row = json_schema["$defs"][json_schema["properties"][source]["$ref"].rsplit("/", 1)[-1]]
        assert row["required"] == list(row["properties"]) == targets
        assert row["additionalProperties"] is False
        assert all(prop["type"] == "string" for prop in row["properties"].values())


@pytest.mark.parametrize("retry", [True, False])
async def test_both_dependency_planner_calls_state_whether_retry_is_enabled(retry: bool) -> None:
    llm = FakeLLM(Script([plan_answer(["A", "B", "C"])]))

    await plan_dependency(llm, MODEL, BENCHMARK, [GENERALIST_PROFILE], OBJECTIVE, retry=retry)

    setting, other = ("enabled", "disabled") if retry else ("disabled", "enabled")
    assert len(llm.requests) == 2
    for request in llm.requests:
        assert f"Retry is {setting} in this workflow." in request.text
        assert f"Retry is {other}" not in request.text


@pytest.mark.parametrize(
    ("roster", "outsider"),
    [([GATHERER, CRAFTER], GENERALIST), ([GENERALIST_PROFILE], "gatherer")],
)
async def test_subtask_schema_offers_only_the_roster_executors(
    roster: list[ExecutorProfile], outsider: str
) -> None:
    chosen = roster[-1].name
    llm = FakeLLM(Script([plan_answer(["Work"], entry="Work", executor=chosen)]))

    planned = await plan_cyclic(llm, MODEL, BENCHMARK, roster, OBJECTIVE)

    schema = llm.requests[0].schema
    assert schema is not None
    json_schema = schema.model_json_schema()
    draft = json_schema["$defs"][json_schema["properties"]["subtasks"]["items"]["$ref"].rsplit("/", 1)[-1]]
    executor = draft["properties"]["executor"]
    assert executor["enum"] == [profile.name for profile in roster]
    assert "const" not in executor
    assert all(f'"{tool}"' in llm.requests[0].text for profile in roster for tool in profile.tools)
    assert planned.graph.subtasks[0].executor == chosen
    with pytest.raises(ValueError, match="executor"):
        schema.model_validate(plan_answer(["Work"], entry="Work", executor=outsider))


async def test_subtask_ids_are_slugs_made_unique() -> None:
    names = ["Gather Materials", "gather materials!", "Craft: Oak Planks (x4)", "Crème brûlée", "???"]
    names += ["a", "A", "a 2"]
    llm = FakeLLM(Script([plan_answer(names)]))

    planned = await plan_dependency(llm, MODEL, BENCHMARK, [GENERALIST_PROFILE], OBJECTIVE, retry=True)

    ids = ["gather_materials", "gather_materials_2", "craft_oak_planks_x4", "creme_brulee", "subtask"]
    ids += ["a", "a_2", "a_2_2"]
    assert list(planned.graph.ids) == ids
    assert [s.name for s in planned.graph.subtasks] == names
    criteria_schema = llm.requests[1].schema
    assert criteria_schema is not None
    assert list(edges_of(criteria_schema)) == ids


@pytest.mark.parametrize(
    ("names", "entry", "complaint"),
    [
        (["Gather", "Craft"], "Smelt", "is not the name of a subtask; use one of: 'Gather', 'Craft'"),
        (["Gather", "gather", "Craft"], "Gather", "give the subtasks distinct names"),
    ],
)
async def test_invalid_subtasks_answer_is_retried_with_the_error(
    names: list[str], entry: str, complaint: str
) -> None:
    llm = FakeLLM(Script([plan_answer(names, entry=entry), plan_answer(["Gather", "Craft"], entry="Craft")]))

    planned = await plan_cyclic(llm, MODEL, BENCHMARK, [GENERALIST_PROFILE], OBJECTIVE)

    assert planned.graph.entry == "craft"
    first, retried, _criteria = llm.requests
    assert retried.messages[0] == first.messages[0]
    assistant, feedback = retried.messages[1:]
    assert assistant["role"] == "assistant" and f'"entry":"{entry}"' in str(assistant.get("content"))
    assert feedback["role"] == "user" and complaint in str(feedback.get("content"))
    assert [(r["call"], complaint in r["error"]) for r in planned.notes["rejected"]] == [("subtasks", True)]


async def test_invalid_criteria_answer_retries_only_the_criteria_call() -> None:
    def blank_edge(edges: Edges) -> dict[str, dict[str, str]]:
        criteria = write_criteria(edges)
        criteria["b"]["a"] = "  "
        return criteria

    llm = FakeLLM(Script([plan_answer(["A", "B"], entry="A")], criteria=[blank_edge, write_criteria]))

    planned = await plan_cyclic(llm, MODEL, BENCHMARK, [GENERALIST_PROFILE], OBJECTIVE)

    assert [r.schema is not None and issubclass(r.schema, SubtaskPlan) for r in llm.requests] == [
        True,
        False,
        False,
    ]
    feedback = criteria_requests(llm)[1].messages[-1]
    assert "empty criteria for: b -> a" in str(feedback.get("content"))
    assert planned.graph.criteria["b"]["a"] == "b -> a"
    assert [r["call"] for r in planned.notes["rejected"]] == ["criteria"]


async def test_planner_raises_after_the_last_retry_fails() -> None:
    llm = FakeLLM(Script([plan_answer([])] * MAX_ATTEMPTS))

    with pytest.raises(LLMOutputError, match="at least one subtask"):
        await plan_dependency(llm, MODEL, BENCHMARK, [GENERALIST_PROFILE], OBJECTIVE, retry=True)

    assert len(llm.requests) == MAX_ATTEMPTS == 3
    assert [m["role"] for m in llm.requests[-1].messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]


async def test_nshot_summaries_reach_both_calls() -> None:
    summaries = (
        "SUMMARY-ONE: gathered logs, crafted planks.",
        "SUMMARY-TWO: crafted diorite before granite.",
    )
    with_nshot = FakeLLM(Script([plan_answer(["A"], entry="A")]))
    without = FakeLLM(Script([plan_answer(["A"], entry="A")]))

    await plan_cyclic(with_nshot, MODEL, BENCHMARK, [GENERALIST_PROFILE], None, nshot=summaries)
    await plan_cyclic(without, MODEL, BENCHMARK, [GENERALIST_PROFILE], None)

    assert len(with_nshot.requests) == 2
    for request in with_nshot.requests:
        assert all(summary in request.text for summary in summaries)
    assert not any("Successful trajectories" in request.text for request in without.requests)
