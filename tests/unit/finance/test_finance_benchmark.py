from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from ctg.config import BenchmarkName, Controller, ExperimentConfig, ModelSpec, Role
from ctg.envs.base import GENERALIST
from ctg.envs.finance.benchmark import FinanceAgentBenchmark, required_env_vars
from ctg.envs.finance.dataset import get_question
from ctg.envs.finance.tools import FINAL_ANSWER, TOOL_NAMES
from ctg.envs.registry import get_benchmark
from ctg.llm.base import AssistantTurn
from ctg.llm.fake import FakeLLM, FakeRequest

MODELS = {role: ModelSpec(name=f"{role.value}-model") for role in Role}
ALL_VARS = ("SEC_API_KEY", "SEC_USER_AGENT", "SERPAPI_API_KEY", "EXA_API_KEY")


def config(web_search: Literal["serpapi", "exa"] = "serpapi") -> ExperimentConfig:
    return ExperimentConfig(
        name="finance-test",
        benchmark=BenchmarkName.FINANCE_AGENT,
        controller=Controller.GEN_CYC,
        web_search=web_search,
    )


def judge_everything_met(request: FakeRequest) -> BaseModel | AssistantTurn:
    assert request.schema is not None
    question = get_question("q02")
    fields = {
        f"criterion_{n}": {"reasoning": "r", "verdict": criterion.operator == "correctness"}
        for n, criterion in enumerate(question.rubric, 1)
    }
    return request.schema.model_validate(fields)


def test_roster_has_a_generalist_and_one_specialist_per_tool() -> None:
    benchmark = get_benchmark(BenchmarkName.FINANCE_AGENT)

    generalist = benchmark.roster[GENERALIST]
    assert (generalist.tools, generalist.max_tool_calls) == (TOOL_NAMES, None)
    specialists = benchmark.executor_roster(specialists=True)
    assert {s.name: s.tools for s in specialists} == {
        "edgar_researcher": ("finance_agent_edgar_search",),
        "web_researcher": ("finance_agent_google_web_search",),
        "page_parser": ("finance_agent_parse_html_page",),
        "retriever": ("finance_agent_retrieve_information",),
        "answerer": ("finance_agent_final_answer",),
    }
    assert all(s.max_tool_calls == 5 for s in specialists)


def test_budgets_and_instances() -> None:
    benchmark = FinanceAgentBenchmark()

    budgets = benchmark.budgets(config())
    assert (budgets.global_tool_calls, budgets.local_tool_calls) == (20, 20)
    instances = benchmark.enumerate_instances("finance_agent")
    assert [i.id for i in instances] == [f"q{n:02d}" for n in range(50)]
    assert instances[0].meta == {"question_type": "Market Analysis"}
    with pytest.raises(ValueError):
        benchmark.enumerate_instances("textcraft_depth2")


def test_description_states_the_reference_date_and_every_tool() -> None:
    description = FinanceAgentBenchmark().description

    assert "2025-04-07" in description
    assert all(name in description for name in TOOL_NAMES)


@pytest.mark.parametrize(
    ("web_search", "present", "missing"),
    [
        ("serpapi", (), ("SEC_API_KEY", "SEC_USER_AGENT", "SERPAPI_API_KEY")),
        ("exa", ("SERPAPI_API_KEY",), ("SEC_API_KEY", "SEC_USER_AGENT", "EXA_API_KEY")),
        ("serpapi", ("SEC_API_KEY", "SEC_USER_AGENT", "EXA_API_KEY"), ("SERPAPI_API_KEY",)),
        ("exa", ("SEC_API_KEY", "EXA_API_KEY"), ("SEC_USER_AGENT",)),
    ],
)
async def test_make_env_fails_fast_naming_every_missing_variable(
    monkeypatch: pytest.MonkeyPatch,
    web_search: Literal["serpapi", "exa"],
    present: tuple[str, ...],
    missing: tuple[str, ...],
) -> None:
    for name in ALL_VARS:
        monkeypatch.setenv(name, "value" if name in present else "  ")
    llm = FakeLLM(judge_everything_met)
    benchmark = FinanceAgentBenchmark()

    with pytest.raises(RuntimeError) as caught:
        await benchmark.make_env(
            benchmark.enumerate_instances("finance_agent")[0], config(web_search), llm, MODELS
        )

    assert str(caught.value).endswith(", ".join(missing))
    assert set(required_env_vars(web_search)) >= set(missing)


async def test_make_env_judges_the_final_answer_with_the_judge_model(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ALL_VARS:
        monkeypatch.setenv(name, "value")
    llm = FakeLLM(judge_everything_met)
    benchmark = FinanceAgentBenchmark()
    instance = benchmark.enumerate_instances("finance_agent")[2]

    env = await benchmark.make_env(instance, config("exa"), llm, MODELS)
    try:
        assert env.objective == get_question("q02").question
        await env.call(FINAL_ANSWER, {"answer": "80bps beat from low end", "sources": None})
    finally:
        await env.close()

    assert env.done and env.won
    [request] = llm.requests
    assert (request.role, request.model) == (Role.JUDGE, MODELS[Role.JUDGE])
