import pytest
from control_fakes import (
    ALL_TOOLS,
    BENCHMARK,
    DESCRIPTION,
    MODELS,
    OBJECTIVE,
    FakeEnv,
    system_text,
    tool_results,
)

from ctg.config import Budgets, Role
from ctg.control.react import run_react_episode
from ctg.control.workflow import EpisodeResult
from ctg.errors import BackendUnavailableError
from ctg.llm.base import AssistantTurn, LLMOutputError, SpendCapExceededError
from ctg.llm.fake import FakeLLM, FakeRequest, text_turn, tool_turn
from ctg.runs.records import EndReason, Outcome

BUDGETS = Budgets(global_tool_calls=5, local_tool_calls=5)


async def run(llm: FakeLLM, env: FakeEnv | None = None) -> EpisodeResult:
    return await run_react_episode(env or FakeEnv(), BENCHMARK, llm, MODELS, BUDGETS)


async def test_react_acts_until_the_global_budget_with_a_required_call_each_step() -> None:
    llm = FakeLLM(lambda request: tool_turn("act", {"value": f"step-{tool_results(request) + 1}"}))
    result = await run(llm)

    assert result.outcome.end_reason is EndReason.BUDGET
    assert (result.outcome.tool_calls, result.outcome.segments) == (5, 0)
    assert len(llm.requests) == 5
    assert all(r.role is Role.EXECUTOR and r.tool_choice == "required" for r in llm.requests)
    assert all([tool.name for tool in r.tools] == list(ALL_TOOLS) for r in llm.requests)
    assert [(s.arguments["value"], s.tool_calls_after) for s in result.react_steps] == [
        (f"step-{n}", n) for n in range(1, 6)
    ]
    assert result.segments == []


async def test_react_prompt_has_benchmark_objective_and_budget() -> None:
    llm = FakeLLM(lambda request: tool_turn("win", {}))
    await run(llm)

    system = system_text(llm.requests[0])
    assert DESCRIPTION in system
    assert OBJECTIVE in system
    assert "at most 5 tool calls" in system


async def test_react_stops_when_the_goal_is_reached() -> None:
    def handler(request: FakeRequest) -> AssistantTurn:
        return tool_turn("act", {"value": "x"}) if tool_results(request) < 2 else tool_turn("win", {})

    llm = FakeLLM(handler)
    result = await run(llm)

    assert result.outcome.won
    assert result.outcome.end_reason is EndReason.WON
    assert len(result.react_steps) == len(llm.requests) == 3


async def test_react_submission_without_a_win_is_a_final_answer() -> None:
    result = await run(FakeLLM(lambda request: tool_turn("submit", {})))

    assert not result.outcome.won
    assert result.outcome.end_reason is EndReason.FINAL_ANSWER
    assert len(result.react_steps) == 1


async def test_react_replies_without_a_tool_call_still_use_up_steps() -> None:
    llm = FakeLLM(lambda request: text_turn("thinking"))
    result = await run(llm)

    assert len(llm.requests) == 5
    assert result.outcome.end_reason is EndReason.BUDGET
    assert result.outcome.tool_calls == 0


async def test_unusable_output_ends_the_episode_and_keeps_the_steps_taken() -> None:
    unusable = LLMOutputError("fake-executor returned unusable executor output 3 times: empty reply")

    def handler(request: FakeRequest) -> AssistantTurn:
        if tool_results(request) == 2:
            raise unusable
        return tool_turn("act", {"value": f"step-{tool_results(request) + 1}"})

    result = await run(FakeLLM(handler))

    assert result.outcome == Outcome(won=False, end_reason=EndReason.OUTPUT_FAILURE, tool_calls=2, segments=0)
    assert result.failure == f"LLMOutputError: {unusable}"
    assert [(s.arguments["value"], s.tool_calls_after) for s in result.react_steps] == [
        ("step-1", 1),
        ("step-2", 2),
    ]


@pytest.mark.parametrize(
    "error", [SpendCapExceededError("spend cap reached"), BackendUnavailableError("API key rejected")]
)
async def test_infrastructure_errors_from_the_model_propagate(error: RuntimeError) -> None:
    def handler(request: FakeRequest) -> AssistantTurn:
        if tool_results(request) == 1:
            raise error
        return tool_turn("act", {"value": "x"})

    with pytest.raises(type(error)):
        await run(FakeLLM(handler))
