import json
from typing import cast

import pytest
from control_fakes import (
    BENCHMARK,
    CAPPED,
    DESCRIPTION,
    MODELS,
    OBJECTIVE,
    FakeEnv,
    subtask,
    system_text,
    tool_results,
)

from ctg.config import Budgets, Role
from ctg.control.executor import (
    EPISODE_FINISHED,
    NO_ACTIONS,
    NO_WRAP_UP,
    SKIPPED,
    UNUSABLE_OUTPUT,
    SegmentResult,
    run_segment,
    segment_allowance,
)
from ctg.envs.base import GENERALIST, ExecutorProfile
from ctg.llm.base import AssistantTurn, LLMOutputError, Message, ToolCall
from ctg.llm.fake import FakeLLM, FakeRequest, text_turn, tool_turn
from ctg.runs.records import Inconsistency, Memory, Resource

GENERALIST_PROFILE = BENCHMARK.roster[GENERALIST]


async def run(
    env: FakeEnv,
    llm: FakeLLM,
    allowance: int,
    profile: ExecutorProfile = GENERALIST_PROFILE,
    memory: Memory | None = None,
    wrap_up: bool = True,
) -> SegmentResult:
    return await run_segment(
        env=env,
        benchmark=BENCHMARK,
        llm=llm,
        model=MODELS[Role.EXECUTOR],
        subtask=subtask("a"),
        profile=profile,
        memory=memory,
        allowance=allowance,
        wrap_up=wrap_up,
    )


def act(request: FakeRequest) -> AssistantTurn:
    return tool_turn("act", {"value": f"v{tool_results(request) + 1}"})


@pytest.mark.parametrize(
    ("local", "cap", "used", "expected"),
    [
        (5, None, 0, 5),  # C_l binds
        (5, 2, 0, 2),  # the profile's cap binds
        (5, None, 17, 3),  # the remaining global budget binds
        (5, 4, 19, 1),
        (2, 4, 0, 2),
    ],
)
def test_allowance_is_the_tightest_of_local_profile_and_remaining_global(
    local: int, cap: int | None, used: int, expected: int
) -> None:
    budgets = Budgets(global_tool_calls=20, local_tool_calls=local)
    profile = ExecutorProfile(name="p", description="p", tools=("act",), max_tool_calls=cap)

    assert segment_allowance(budgets, profile, used) == expected


async def test_first_call_is_required_and_a_text_reply_ends_the_segment() -> None:
    def handler(request: FakeRequest) -> AssistantTurn:
        return act(request) if tool_results(request) < 2 else text_turn("  Made two moves.  ")

    env, llm = FakeEnv(), FakeLLM(handler)
    result = await run(env, llm, allowance=5)

    assert [r.tool_choice for r in llm.requests] == ["required", "auto", "auto"]
    assert result.summary == "Made two moves."
    assert [(c.tool, c.arguments, c.tool_calls_after) for c in result.tool_calls] == [
        ("act", {"value": "v1"}, 1),
        ("act", {"value": "v2"}, 2),
    ]
    assert result.tool_calls[1].observation == "Observed v2."
    assert env.tool_calls == 2


async def test_every_tool_call_is_answered_with_its_call_id() -> None:
    turns: list[AssistantTurn] = []

    def handler(request: FakeRequest) -> AssistantTurn:
        turns.append(act(request) if tool_results(request) < 2 else text_turn("done"))
        return turns[-1]

    llm = FakeLLM(handler)
    await run(FakeEnv(), llm, allowance=5)

    messages = llm.requests[-1].messages
    answered = [(m["tool_call_id"], m["content"]) for m in messages if m["role"] == "tool"]
    assert answered == [
        (turns[0].tool_calls[0].id, "Observed v1."),
        (turns[1].tool_calls[0].id, "Observed v2."),
    ]


async def test_exhausted_allowance_gets_one_tool_free_wrap_up_call() -> None:
    def handler(request: FakeRequest) -> AssistantTurn:
        return text_turn("Wrapped up.") if request.tool_choice == "none" else act(request)

    env, llm = FakeEnv(), FakeLLM(handler)
    result = await run(env, llm, allowance=3)

    assert env.tool_calls == 3
    assert [r.tool_choice for r in llm.requests] == ["required", "auto", "auto", "none"]
    assert llm.requests[-1].messages[-1]["role"] == "user"
    assert result.summary == "Wrapped up."
    assert len(result.tool_calls) == 3


async def test_exhausted_allowance_without_wrap_up_makes_no_summary_call() -> None:
    def handler(request: FakeRequest) -> AssistantTurn:
        return text_turn("Wrapped up.") if request.tool_choice == "none" else act(request)

    env, llm = FakeEnv(), FakeLLM(handler)
    result = await run(env, llm, allowance=3, wrap_up=False)

    assert env.tool_calls == 3
    assert [r.tool_choice for r in llm.requests] == ["required", "auto", "auto"]
    assert result.summary == NO_WRAP_UP
    assert len(result.tool_calls) == 3


async def test_a_text_reply_is_the_summary_even_without_wrap_up() -> None:
    def handler(request: FakeRequest) -> AssistantTurn:
        return act(request) if tool_results(request) == 0 else text_turn("Stopped early.")

    llm = FakeLLM(handler)
    result = await run(FakeEnv(), llm, allowance=3, wrap_up=False)

    assert result.summary == "Stopped early."


async def test_unusable_wrap_up_output_keeps_the_segments_tool_calls() -> None:
    unusable = LLMOutputError("fake-executor returned unusable executor output 3 times: refusal")

    def handler(request: FakeRequest) -> AssistantTurn:
        if request.tool_choice == "none":
            raise unusable
        return act(request)

    env = FakeEnv()
    result = await run(env, FakeLLM(handler), allowance=2)

    assert [(c.tool, c.tool_calls_after) for c in result.tool_calls] == [("act", 1), ("act", 2)]
    assert (result.summary, result.failure) == (UNUSABLE_OUTPUT, f"LLMOutputError: {unusable}")
    assert env.tool_calls == 2


async def test_profile_cap_limits_the_segment_and_its_tools() -> None:
    def handler(request: FakeRequest) -> AssistantTurn:
        return text_turn("Capped.") if request.tool_choice == "none" else act(request)

    env, llm = FakeEnv(), FakeLLM(handler)
    allowance = segment_allowance(Budgets(global_tool_calls=20, local_tool_calls=5), CAPPED, 0)
    await run(env, llm, allowance=allowance, profile=CAPPED)

    assert env.tool_calls == 2
    assert all([tool.name for tool in r.tools] == ["act"] for r in llm.requests)


async def test_a_tool_outside_the_profile_is_refused_and_still_spends_the_allowance() -> None:
    turns: list[AssistantTurn] = []

    def handler(request: FakeRequest) -> AssistantTurn:
        turns.append(text_turn("Gave up on win.") if request.tool_choice == "none" else tool_turn("win", {}))
        return turns[-1]

    env, llm = FakeEnv(), FakeLLM(handler)
    allowance = segment_allowance(Budgets(global_tool_calls=20, local_tool_calls=5), CAPPED, 0)
    result = await run(env, llm, allowance=allowance, profile=CAPPED)

    refusal = "Error: tool 'win' is not available to this executor. Available tools: act."
    assert not env.won and not env.done
    assert env.tool_calls == allowance == 2
    assert [(c.tool, c.observation, c.tool_calls_after) for c in result.tool_calls] == [
        ("win", refusal, 1),
        ("win", refusal, 2),
    ]
    answered = [(m["tool_call_id"], m["content"]) for m in llm.requests[-1].messages if m["role"] == "tool"]
    assert answered == [(turn.tool_calls[0].id, refusal) for turn in turns[:2]]
    assert [r.tool_choice for r in llm.requests] == ["required", "auto", "none"]
    assert result.summary == "Gave up on win."


async def test_episode_end_stops_the_segment_without_a_wrap_up() -> None:
    def handler(request: FakeRequest) -> AssistantTurn:
        return act(request) if tool_results(request) == 0 else tool_turn("win", {})

    env, llm = FakeEnv(), FakeLLM(handler)
    result = await run(env, llm, allowance=5)

    assert env.won
    assert len(llm.requests) == 2
    assert result.summary == EPISODE_FINISHED


async def test_no_available_tools_ends_the_segment_without_calling_the_model() -> None:
    llm = FakeLLM(act)
    result = await run(FakeEnv(available=()), llm, allowance=3)

    assert llm.requests == []
    assert result == SegmentResult([], NO_ACTIONS)


async def test_tools_are_rebuilt_from_the_current_state_before_every_call() -> None:
    env = FakeEnv(available=("act", "win"))

    def handler(request: FakeRequest) -> AssistantTurn:
        if tool_results(request) == 0:
            env.available = {"act"}
            return act(request)
        return text_turn("done")

    llm = FakeLLM(handler)
    await run(env, llm, allowance=3)

    assert [[tool.name for tool in r.tools] for r in llm.requests] == [["act", "win"], ["act"]]


async def test_calls_beyond_the_allowance_are_answered_but_not_executed() -> None:
    calls = (
        ToolCall(id="call_x", name="act", arguments={"value": "x"}),
        ToolCall(id="call_y", name="act", arguments={"value": "y"}),
    )
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in calls
        ],
    }
    both = AssistantTurn(content=None, tool_calls=calls, message=cast(Message, message))

    def handler(request: FakeRequest) -> AssistantTurn:
        return text_turn("summary") if request.tool_choice == "none" else both

    env, llm = FakeEnv(), FakeLLM(handler)
    result = await run(env, llm, allowance=1)

    assert env.tool_calls == 1
    assert [c.arguments for c in result.tool_calls] == [{"value": "x"}]
    tool_messages = [m for m in llm.requests[-1].messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == [call.id for call in both.tool_calls]
    assert tool_messages[1]["content"] == SKIPPED


async def test_system_prompt_carries_benchmark_objective_subtask_and_memory() -> None:
    memory = Memory(
        segment_analysis=["looked around"],
        inconsistencies=[
            Inconsistency(description="claimed 3 logs", cause="miscount", correction_plan=["recount"])
        ],
        held_resources=[Resource(name="oak log", quantity=2)],
        environment_notes=["the chest is empty"],
        summary="SUMMARY-SO-FAR",
        subtask_progress=False,
        failure_analysis=["looping on the chest"],
    )
    llm = FakeLLM(lambda request: text_turn("done"))
    await run(FakeEnv(), llm, allowance=4, memory=memory)

    system = system_text(llm.requests[0])
    for expected in (
        DESCRIPTION,
        OBJECTIVE,
        "a-name",
        "a-description",
        "a-criteria",
        "SUMMARY-SO-FAR",
        "oak log: 2",
        "the chest is empty",
        "claimed 3 logs",
        "looping on the chest",
        "at most 4 tool calls",
    ):
        assert expected in system


async def test_a_non_positive_allowance_is_rejected() -> None:
    with pytest.raises(ValueError):
        await run(FakeEnv(), FakeLLM(act), allowance=0)
