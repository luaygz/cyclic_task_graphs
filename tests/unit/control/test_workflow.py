import random
from collections.abc import Callable, Sequence

import pytest
from control_fakes import (
    CAPPED,
    cyclic_graph,
    dependency_graph,
    next_enum,
    requests_for,
    router_reply,
    run_graph,
    scripted,
    system_text,
    tool_results,
)
from pydantic import BaseModel

from ctg.config import Budgets, Role
from ctg.control.executor import NO_WRAP_UP, UNUSABLE_OUTPUT
from ctg.control.memory import NO_MEMORY
from ctg.control.policies import Graph
from ctg.control.workflow import EpisodeResult
from ctg.errors import BackendUnavailableError
from ctg.graphs.model import END, GIVE_UP
from ctg.llm.base import AssistantTurn, LLMOutputError, SpendCapExceededError
from ctg.llm.fake import FakeLLM, FakeRequest, text_turn, tool_turn
from ctg.runs.records import EndReason, Outcome, TransitionRecord, TransitionType

UNUSABLE = LLMOutputError("fake model returned unusable output 3 times: empty reply")


def act_once(request: FakeRequest) -> AssistantTurn:
    """One `act` call per segment, then a text summary."""
    if tool_results(request) == 0:
        return tool_turn("act", {"value": "x"})
    return text_turn("acted once")


def always_act(request: FakeRequest) -> AssistantTurn:
    if request.tool_choice == "none":
        return text_turn("out of calls")
    return tool_turn("act", {"value": "x"})


def in_order(choices: Sequence[str]) -> Callable[[FakeRequest], str]:
    remaining = list(choices)
    return lambda _: remaining.pop(0)


def act_from_third_segment() -> Callable[[FakeRequest], AssistantTurn]:
    """Replies in text in the first two segments, then acts like `always_act`."""
    segments = 0

    def executor(request: FakeRequest) -> AssistantTurn:
        nonlocal segments
        if request.tool_choice == "required":
            segments += 1
        return always_act(request) if segments >= 3 else text_turn("nothing to do yet")

    return executor


def transitions(result: EpisodeResult) -> list[TransitionRecord]:
    return [segment.transition for segment in result.segments if segment.transition is not None]


async def test_reaching_the_goal_ends_with_won_and_records_the_last_segment_unrouted() -> None:
    def executor(request: FakeRequest) -> AssistantTurn:
        return tool_turn("win", {}) if "b-description" in system_text(request) else act_once(request)

    llm = scripted(executor, router=lambda _: "b")
    result = await run_graph(cyclic_graph(("a", "b", "c")), llm)

    assert result.outcome.won
    assert result.outcome.end_reason is EndReason.WON
    assert (result.outcome.segments, result.outcome.tool_calls) == (2, 2)
    first, last = result.segments
    assert first.router is not None and first.router.choice == "b"
    assert first.transition == TransitionRecord(
        source="a",
        target="b",
        allowed=["a", "b", "c"],
        nominal="b",
        perturbed=False,
        type=TransitionType.CROSS,
        valid=True,
    )
    assert (last.index, last.node, last.router, last.transition) == (1, "b", None, None)
    assert len(requests_for(llm, Role.ROUTER)) == 1


async def test_ending_the_episode_without_a_win_is_a_final_answer() -> None:
    llm = scripted(lambda _: tool_turn("submit", {}))
    result = await run_graph(cyclic_graph(), llm)

    assert not result.outcome.won
    assert result.outcome.end_reason is EndReason.FINAL_ANSWER
    assert len(result.segments) == 1
    assert requests_for(llm, Role.ROUTER) == []


async def test_global_budget_shrinks_the_last_segment_and_ends_without_routing() -> None:
    llm = scripted(always_act, router=lambda _: "a")
    result = await run_graph(cyclic_graph(), llm, budgets=Budgets(global_tool_calls=4, local_tool_calls=3))

    assert result.outcome.end_reason is EndReason.BUDGET
    assert result.outcome.tool_calls == 4
    assert [len(segment.tool_calls) for segment in result.segments] == [3, 1]
    assert [t.type for t in transitions(result)] == [TransitionType.SELF]
    assert result.segments[-1].router is None
    assert len(requests_for(llm, Role.ROUTER)) == 1


async def test_profile_cap_applies_to_every_segment_of_its_subtasks() -> None:
    llm = scripted(always_act, router=lambda _: "b")
    graph = cyclic_graph(("a", "b"), executor=CAPPED.name)
    result = await run_graph(graph, llm, budgets=Budgets(global_tool_calls=4, local_tool_calls=3))

    assert [(s.executor, len(s.tool_calls)) for s in result.segments] == [(CAPPED.name, 2), (CAPPED.name, 2)]


async def test_segment_cap_ends_an_episode_whose_executor_never_acts() -> None:
    llm = scripted(lambda _: text_turn("nothing to do"), router=lambda _: "a")
    result = await run_graph(cyclic_graph(), llm, budgets=Budgets(global_tool_calls=3, local_tool_calls=3))

    assert result.outcome.end_reason is EndReason.BUDGET
    assert (result.outcome.segments, result.outcome.tool_calls) == (3, 0)
    assert len(requests_for(llm, Role.ROUTER)) == 2
    assert result.segments[-1].transition is None


@pytest.mark.parametrize(
    ("graph", "budgets", "executor", "choices", "end_reason"),
    [
        pytest.param(
            cyclic_graph(),
            Budgets(global_tool_calls=4, local_tool_calls=3),
            lambda: always_act,
            ["required", "auto", "auto", "none", "required"],
            EndReason.BUDGET,
            id="global-budget-spent",
        ),
        pytest.param(
            dependency_graph(("a", "b"), retry=False),
            Budgets(global_tool_calls=20, local_tool_calls=3),
            lambda: always_act,
            ["required", "auto", "auto", "none", "required", "auto", "auto"],
            EndReason.TERMINATED,
            id="no-retry-sink",
        ),
        pytest.param(
            cyclic_graph(),
            Budgets(global_tool_calls=3, local_tool_calls=1),
            act_from_third_segment,
            ["required", "required", "required"],
            EndReason.BUDGET,
            id="segment-cap",
        ),
    ],
)
async def test_a_spent_segment_that_ends_the_episode_asks_for_no_summary(
    graph: Graph,
    budgets: Budgets,
    executor: Callable[[], Callable[[FakeRequest], AssistantTurn]],
    choices: list[str],
    end_reason: EndReason,
) -> None:
    llm = scripted(executor(), router=lambda _: "b")
    result = await run_graph(graph, llm, budgets=budgets)

    assert result.outcome.end_reason is end_reason
    assert [r.tool_choice for r in requests_for(llm, Role.EXECUTOR)] == choices
    *routed, last = result.segments
    assert (last.router, last.summary) == (None, NO_WRAP_UP)
    assert all(segment.router is not None and segment.summary != NO_WRAP_UP for segment in routed)


async def test_router_sees_every_subtask_and_the_trace_of_the_segment_that_just_finished() -> None:
    segment = 0

    def executor(request: FakeRequest) -> AssistantTurn:
        nonlocal segment
        if request.tool_choice == "required":
            segment += 1
            return tool_turn("act", {"value": f"segment-{segment}"})
        return text_turn(f"summary of segment {segment}")

    llm = scripted(executor, router=in_order(["b", "b", "a"]))
    result = await run_graph(
        cyclic_graph(("a", "b", "c")), llm, budgets=Budgets(global_tool_calls=4, local_tool_calls=3)
    )

    routings = requests_for(llm, Role.ROUTER)
    assert len(routings) == 3
    for number, request in enumerate(routings, start=1):
        assert next_enum(request) == ["a", "b", "c"]
        assert f"Observed segment-{number}." in request.text
        assert f"summary of segment {number}" in request.text
        assert all(f"segment-{other}" not in request.text for other in range(1, 5) if other != number)
    assert [(t.source, t.target, t.type) for t in transitions(result)] == [
        ("a", "b", TransitionType.CROSS),
        ("b", "b", TransitionType.SELF),
        ("b", "a", TransitionType.CROSS),
    ]


async def test_router_memory_feeds_the_next_executor_and_the_next_routing() -> None:
    routings = 0

    def handler(request: FakeRequest) -> BaseModel | AssistantTurn:
        nonlocal routings
        if request.kind == "structured":
            routings += 1
            return router_reply(request, "b", summary=f"MEMORY-{routings}")
        return act_once(request)

    llm = FakeLLM(handler)
    await run_graph(cyclic_graph(), llm, budgets=Budgets(global_tool_calls=3, local_tool_calls=3))

    executors = [r for r in requests_for(llm, Role.EXECUTOR) if r.tool_choice == "required"]
    routers = requests_for(llm, Role.ROUTER)
    assert NO_MEMORY in system_text(executors[0])
    assert "MEMORY-1" in system_text(executors[1])
    assert "MEMORY-2" in system_text(executors[2])
    assert NO_MEMORY in routers[0].text
    assert "MEMORY-1" in routers[1].text


async def test_dependency_run_retries_moves_forward_and_terminates_on_router_choice() -> None:
    llm = scripted(act_once, router=in_order(["a", "b", END]))
    result = await run_graph(dependency_graph(("a", "b"), retry=True), llm)

    assert result.outcome.end_reason is EndReason.TERMINATED
    assert not result.outcome.won
    assert [next_enum(r) for r in requests_for(llm, Role.ROUTER)] == [
        ["b", "a", GIVE_UP],
        ["b", "a", GIVE_UP],
        ["b", GIVE_UP],
    ]
    assert [(t.source, t.target, t.type, t.valid) for t in transitions(result)] == [
        ("a", "a", TransitionType.RETRY, True),
        ("a", "b", TransitionType.FORWARD, True),
        ("b", END, TransitionType.TERMINAL, True),
    ]
    assert all(segment.router is not None for segment in result.segments)


async def test_no_retry_sink_terminates_without_a_router_call() -> None:
    llm = scripted(act_once, router=lambda _: "b")
    result = await run_graph(dependency_graph(("a", "b"), retry=False), llm)

    routers = requests_for(llm, Role.ROUTER)
    assert [next_enum(r) for r in routers] == [["b", GIVE_UP]]
    assert result.outcome.end_reason is EndReason.TERMINATED
    sink = result.segments[-1]
    assert (sink.node, sink.router) == ("b", None)
    assert sink.transition == TransitionRecord(
        source="b",
        target=END,
        allowed=[END],
        nominal=END,
        perturbed=False,
        type=TransitionType.TERMINAL,
        valid=True,
    )


async def test_fault_injection_redirects_away_from_the_routers_choice() -> None:
    llm = scripted(act_once, router=lambda _: "a")
    result = await run_graph(
        cyclic_graph(("a", "b", "c")),
        llm,
        budgets=Budgets(global_tool_calls=6, local_tool_calls=3),
        perturbation=1.0,
    )

    taken = transitions(result)
    assert len(taken) == 5
    for transition in taken:
        assert (transition.nominal, transition.perturbed, transition.valid) == ("a", True, True)
        assert transition.target in ("b", "c")
        expected = TransitionType.SELF if transition.target == transition.source else TransitionType.CROSS
        assert transition.type is expected
    assert [s.node for s in result.segments[1:]] == [t.target for t in taken]


async def test_fault_injection_is_reproducible_and_off_at_zero() -> None:
    async def run(perturbation: float, seed: int) -> list[tuple[str, str, bool]]:
        llm = scripted(act_once, router=lambda _: "a")
        result = await run_graph(
            cyclic_graph(("a", "b", "c")),
            llm,
            budgets=Budgets(global_tool_calls=12, local_tool_calls=3),
            perturbation=perturbation,
            rng=random.Random(seed),
        )
        return [(t.nominal, t.target, t.perturbed) for t in transitions(result)]

    first = await run(0.5, seed=5)
    assert first == await run(0.5, seed=5)
    assert {perturbed for _, _, perturbed in first} == {True, False}
    assert all(target == nominal and not perturbed for nominal, target, perturbed in await run(0.0, seed=5))


async def test_fault_injection_is_rejected_for_dependency_graphs() -> None:
    with pytest.raises(ValueError):
        await run_graph(dependency_graph(), scripted(act_once), perturbation=0.5)


async def test_unusable_executor_output_ends_the_episode_and_keeps_the_segment_so_far() -> None:
    def executor(request: FakeRequest) -> AssistantTurn:
        if "b-description" in system_text(request) and tool_results(request) == 1:
            raise UNUSABLE
        return act_once(request)

    llm = scripted(executor, router=lambda _: "b")
    result = await run_graph(cyclic_graph(), llm)

    assert result.outcome == Outcome(won=False, end_reason=EndReason.OUTPUT_FAILURE, tool_calls=2, segments=2)
    assert result.failure == f"LLMOutputError: {UNUSABLE}"
    first, last = result.segments
    assert first.router is not None and first.transition is not None
    assert (last.node, last.summary, last.router, last.transition) == ("b", UNUSABLE_OUTPUT, None, None)
    assert [(c.tool, c.tool_calls_after) for c in last.tool_calls] == [("act", 2)]
    assert sum(len(s.tool_calls) for s in result.segments) == result.outcome.tool_calls
    assert len(requests_for(llm, Role.ROUTER)) == 1


async def test_unusable_router_output_ends_the_episode_and_keeps_the_unrouted_segment() -> None:
    def router(_: FakeRequest) -> str:
        raise UNUSABLE

    result = await run_graph(cyclic_graph(), scripted(act_once, router=router))

    assert result.outcome == Outcome(won=False, end_reason=EndReason.OUTPUT_FAILURE, tool_calls=1, segments=1)
    assert result.failure == f"LLMOutputError: {UNUSABLE}"
    [segment] = result.segments
    assert (segment.summary, segment.router, segment.transition) == ("acted once", None, None)
    assert len(segment.tool_calls) == 1


@pytest.mark.parametrize(
    "error", [SpendCapExceededError("spend cap reached"), BackendUnavailableError("API key rejected")]
)
@pytest.mark.parametrize("role", [Role.EXECUTOR, Role.ROUTER])
async def test_infrastructure_errors_from_a_model_call_propagate(error: RuntimeError, role: Role) -> None:
    def executor(request: FakeRequest) -> AssistantTurn:
        if role is Role.EXECUTOR and tool_results(request) == 1:
            raise error
        return act_once(request)

    def router(_: FakeRequest) -> str:
        if role is Role.ROUTER:
            raise error
        return "b"

    with pytest.raises(type(error)):
        await run_graph(cyclic_graph(), scripted(executor, router=router))
