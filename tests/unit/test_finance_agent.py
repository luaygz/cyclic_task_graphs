import json

import pytest

from luna.actor.schemas import RubricEvaluation, Usage
from luna.benchmarks.finance_agent.adapter import FinanceAgentAdapter
from luna.benchmarks.finance_agent.data import load_cases
from luna.benchmarks.finance_agent.summarize import summarize_content
from luna.benchmarks.finance_agent.tools import FinanceToolClient


class FakeTransport:
    def __init__(self, statuses=None):
        self.statuses = list(statuses or [200])
        self.posts = 0

    async def get_text(self, url, headers, timeout):
        assert headers["User-Agent"] == "Research Group research-contact@example.org"
        return "<html><style>hidden</style><body><h1>Revenue</h1><p>$42</p></body></html>"

    async def post_json(self, url, payload, headers, timeout):
        self.posts += 1
        status = self.statuses.pop(0)
        return status, {"filings": [{"ticker": "TEST", "formType": "10-K"}]}


def test_all_finance_rubrics_parse():
    cases = load_cases()
    assert len(cases) == 50
    assert all(case.rubric for case in cases)
    assert sum(len(case.rubric) for case in cases) == 241


@pytest.mark.asyncio
async def test_html_and_retrieval_are_mockable(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Research Group research-contact@example.org")
    client = FinanceToolClient(transport=FakeTransport())
    result = await client.parse_html_page("https://www.sec.gov/test", "filing")
    assert "SUCCESS" in result
    retrieved = await client.retrieve_information("Use {{filing}}")
    assert "Revenue" in retrieved and "$42" in retrieved


@pytest.mark.asyncio
async def test_edgar_retry_count_and_reporting(monkeypatch):
    monkeypatch.setenv("SEC_EDGAR_API_KEY", "test-key")
    transport = FakeTransport(statuses=[429, 429, 429])
    client = FinanceToolClient(transport=transport)
    monkeypatch.setattr("luna.benchmarks.finance_agent.tools.asyncio.sleep", _no_sleep)
    result = await client.edgar_search(query="material weakness")
    assert result.endswith("after 3 attempts")
    assert transport.posts == 3


async def _no_sleep(_):
    return None


@pytest.mark.asyncio
async def test_recursive_summarization_chunks_large_input():
    calls = []

    async def summarize_once(content, prompt, label):
        calls.append((len(content), label))
        return "short summary"

    content = "word " * 120_000
    result = await summarize_content(content, "question", summarize_once)
    assert result == "short summary"
    assert len(calls) > 2


@pytest.mark.asyncio
async def test_web_search_is_mockable():
    async def search(query):
        return f"evidence for {query}"

    async def summarize_once(content, prompt, label):
        return f"summary: {content}"

    client = FinanceToolClient(search=search, summarize_once=summarize_once)
    result = await client.web_search("cash flow")
    assert result == "evidence for cash flow"


@pytest.mark.asyncio
async def test_finance_rubric_judge_is_injected_and_counted():
    calls = []

    async def judge(question, expected, rubric, answer):
        calls.append((question, expected, rubric, answer))
        return RubricEvaluation(
            overall_reasoning="mock evaluation",
            criteria_evaluations=[],
            overall_passed=True,
            usage=Usage(input_tokens=3, output_tokens=2, total_tokens=5),
        )

    adapter = FinanceAgentAdapter(judge=judge)
    await adapter.initialize(seed=0, index=0, depth=None)
    assert await adapter.finalize("A semantically correct paraphrase.")
    assert calls and calls[0][2]
    assert adapter.evaluation_usage.total_tokens == 5

