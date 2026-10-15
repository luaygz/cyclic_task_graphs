import pytest
from control_fakes import (
    BENCHMARK,
    DESCRIPTION,
    MODELS,
    OBJECTIVE,
    FakeEnv,
    dependency_graph,
    next_enum,
    router_reply,
)
from pydantic import BaseModel, ValidationError

from ctg.config import Budgets, Role
from ctg.control.executor import SegmentResult
from ctg.control.memory import NO_MEMORY
from ctg.control.policies import candidates
from ctg.control.router import route
from ctg.graphs.model import END, GIVE_UP
from ctg.llm.fake import FakeLLM, FakeRequest
from ctg.runs.records import Memory, RouterRecord, ToolCallRecord

SEGMENT = SegmentResult(
    tool_calls=[
        ToolCallRecord(
            tool="act", arguments={"value": "probe"}, observation="OBS-FROM-SEGMENT", tool_calls_after=1
        )
    ],
    summary="EXECUTOR-SUMMARY",
)


async def run(llm: FakeLLM, memory: Memory | None = None) -> RouterRecord:
    graph = dependency_graph(("a", "b", "c"), retry=True)
    return await route(
        env=FakeEnv(),
        benchmark=BENCHMARK,
        llm=llm,
        model=MODELS[Role.ROUTER],
        budgets=Budgets(global_tool_calls=20, local_tool_calls=3),
        graph=graph,
        current="a",
        memory=memory,
        segment=SEGMENT,
        allowed=candidates(graph, "a"),
    )


async def test_router_output_is_restricted_to_the_candidates_and_split_into_memory_and_choice() -> None:
    def handler(request: FakeRequest) -> BaseModel:
        return router_reply(request, "b", summary="UPDATED-SUMMARY")

    llm = FakeLLM(handler)
    record = await run(llm)

    (request,) = llm.requests
    assert request.role is Role.ROUTER
    assert next_enum(request) == ["b", "a", GIVE_UP]
    assert record.choice == "b"
    assert record.memory.summary == "UPDATED-SUMMARY"
    assert record.routing_reasoning == ["reasoning"]


async def test_router_shows_termination_as_give_up_and_records_it_as_end() -> None:
    llm = FakeLLM(lambda request: router_reply(request, END))
    record = await run(llm)

    assert record.choice == END
    text = llm.requests[0].text
    assert f"`{GIVE_UP}`" in text
    assert f"`{END}`" not in text


async def test_router_prompt_has_the_segment_trace_candidates_and_criteria() -> None:
    llm = FakeLLM(lambda request: router_reply(request, "a"))
    await run(llm)

    text = llm.requests[0].text
    for expected in (
        DESCRIPTION,
        OBJECTIVE,
        "a-description",
        "a-criteria",
        NO_MEMORY,
        '"value": "probe"',
        "OBS-FROM-SEGMENT",
        "EXECUTOR-SUMMARY",
        "b-description",
        "criterion a->b",
        "criterion a->a",
        f"criterion a->{END}",
        "0 of 20",
    ):
        assert expected in text
    assert "c-description" not in text


async def test_a_choice_outside_the_candidates_fails_validation() -> None:
    llm = FakeLLM(lambda request: router_reply(request, "c"))

    with pytest.raises(ValidationError):
        await run(llm)
