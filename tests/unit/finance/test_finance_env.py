from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from pydantic import BaseModel

from ctg.config import ModelSpec, Role
from ctg.envs.finance import env as env_module
from ctg.envs.finance.dataset import Criterion, FinanceQuestion
from ctg.envs.finance.edgar import EdgarQuery, Filing
from ctg.envs.finance.env import Backends, FinanceEnv
from ctg.envs.finance.judge import JudgeError
from ctg.envs.finance.search import SearchResult
from ctg.envs.finance.tools import (
    EDGAR_SEARCH,
    FINAL_ANSWER,
    PARSE_HTML_PAGE,
    RETRIEVE_INFORMATION,
    TOOL_NAMES,
    WEB_SEARCH,
    ToolError,
)
from ctg.errors import BackendUnavailableError
from ctg.llm.base import AssistantTurn, CallSink, LLMOutputError, Message, SchemaT, SpendCapExceededError
from ctg.llm.fake import FakeLLM, FakeRequest, text_turn

FILING_HTML = (Path(__file__).parent / "fixtures" / "filing.html").read_text(encoding="utf-8")
TOOL_MODEL = ModelSpec(name="tool-model")
JUDGE_MODEL = ModelSpec(name="judge-model")
QUESTION = FinanceQuestion(
    id="q99",
    question="What was Acme's FY2024 revenue?",
    answer="$5.0 billion",
    question_type="Quantitative Retrieval",
    rubric=(
        Criterion("correctness", "Revenue of $5.0 billion"),
        Criterion("contradiction", "$5.0 billion"),
    ),
)
FILINGS = [
    Filing(
        company=f"Acme Corp (ACME) (CIK {n:010d})",
        ticker="ACME",
        cik=str(n),
        form_type="10-K",
        filed_at="2025-02-01",
        description="10-K",
        url=f"https://www.sec.gov/Archives/edgar/data/{n}/acme-10k.htm",
    )
    for n in range(1, 13)
]
STORED_UNDER = re.compile(r'The document stored under "(.+?)"')


def capture(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.S)
    assert match is not None, text
    return match.group(1)


@dataclass
class FakeEdgar:
    filings: list[Filing] = field(default_factory=lambda: list(FILINGS))
    queries: list[EdgarQuery] = field(default_factory=list[EdgarQuery])
    error: str | None = None
    outage: str | None = None

    async def search(self, query: EdgarQuery) -> list[Filing]:
        self.queries.append(query)
        if self.outage:
            raise BackendUnavailableError(self.outage)
        if self.error:
            raise ToolError(self.error)
        return self.filings


@dataclass
class FakeWeb:
    results: list[SearchResult] = field(default_factory=list[SearchResult])
    queries: list[str] = field(default_factory=list[str])

    async def search(self, query: str) -> list[SearchResult]:
        self.queries.append(query)
        return self.results


@dataclass
class FakePages:
    pages: dict[str, str] = field(default_factory=dict[str, str])

    async def fetch(self, url: str) -> str:
        if url not in self.pages:
            raise ToolError(f"HTTP 404 Not Found fetching {url}")
        return self.pages[url]


@dataclass
class Harness:
    env: FinanceEnv
    llm: FakeLLM
    edgar: FakeEdgar
    web: FakeWeb
    pages: FakePages
    verdicts: list[bool]
    summarizer_failures: dict[str | None, Exception]
    """Raised by the summarizer instead of summarizing a stored document (key) or web results (None)."""

    def requests(self, role: Role) -> list[FakeRequest]:
        return [request for request in self.llm.requests if request.role is role]


@pytest.fixture
def harness() -> Harness:
    verdicts = [True, False]
    summarizer_failures: dict[str | None, Exception] = {}

    def handler(request: FakeRequest) -> BaseModel | AssistantTurn:
        if request.kind == "structured":
            assert request.schema is not None
            fields = {f"criterion_{n}": {"reasoning": "r", "verdict": v} for n, v in enumerate(verdicts, 1)}
            return request.schema.model_validate(fields)
        stored = STORED_UNDER.search(request.text)
        failure = summarizer_failures.get(stored.group(1) if stored else None)
        if failure is not None:
            raise failure
        return text_turn(f"<summary of {stored.group(1)}>" if stored else "web summary")

    llm = FakeLLM(handler)
    edgar, web, pages = FakeEdgar(), FakeWeb(), FakePages({"https://www.sec.gov/acme-10k.htm": FILING_HTML})
    env = FinanceEnv(QUESTION, llm, TOOL_MODEL, JUDGE_MODEL, Backends(edgar=edgar, web=web, pages=pages))
    return Harness(env, llm, edgar, web, pages, verdicts, summarizer_failures)


def edgar_args(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "query": "revenue",
        "form_types": None,
        "ciks": None,
        "start_date": None,
        "end_date": None,
        "page": None,
        "top_n_results": None,
    }
    return base | overrides


async def store(harness: Harness, key: str, html: str) -> None:
    url = f"https://example.com/{key}"
    harness.pages.pages[url] = html
    observation = await harness.env.call(PARSE_HTML_PAGE, {"url": url, "key": key})
    assert not observation.startswith("Error"), observation


async def test_edgar_search_defaults_span_edgar_history_up_to_the_reference_date(harness: Harness) -> None:
    observation = await harness.env.call(EDGAR_SEARCH, edgar_args())

    [query] = harness.edgar.queries
    assert query == EdgarQuery("revenue", (), (), date(2001, 1, 1), date(2025, 4, 7), 1)
    header, listing = observation.split("\n", 1)
    assert "2001-01-01 to 2025-04-07" in header
    shown = json.loads(listing)
    assert [item["url"] for item in shown] == [filing.url for filing in FILINGS[:10]]
    assert set(shown[0]) == {"company", "ticker", "cik", "form_type", "filed_at", "description", "url"}


async def test_edgar_search_caps_dates_and_passes_filters(harness: Harness) -> None:
    observation = await harness.env.call(
        EDGAR_SEARCH,
        edgar_args(
            form_types=["10-K", "8-K"],
            ciks=["320193"],
            start_date="2024-01-01",
            end_date="2026-01-01",
            page=3,
            top_n_results=2,
        ),
    )

    [query] = harness.edgar.queries
    assert query == EdgarQuery("revenue", ("10-K", "8-K"), ("320193",), date(2024, 1, 1), date(2025, 4, 7), 3)
    assert len(json.loads(observation.split("\n", 1)[1])) == 2


async def test_edgar_search_rejects_a_range_entirely_after_the_reference_date(harness: Harness) -> None:
    observation = await harness.env.call(EDGAR_SEARCH, edgar_args(start_date="2025-06-01"))

    assert observation.startswith("Error: start_date 2025-06-01 is after end_date 2025-04-07")
    assert harness.edgar.queries == []
    assert harness.env.tool_calls == 1


async def test_edgar_search_reports_an_empty_result(harness: Harness) -> None:
    harness.edgar.filings = []

    observation = await harness.env.call(EDGAR_SEARCH, edgar_args())

    assert observation.endswith("no filings found.")


async def test_web_search_returns_the_tool_model_summary(harness: Harness) -> None:
    harness.web.results = [
        SearchResult("Acme results", "https://acme.example/ir", "Revenue $5.0B", "2025-01-30")
    ]

    observation = await harness.env.call(WEB_SEARCH, {"search_query": "Acme FY2024 revenue"})

    assert observation == "web summary"
    assert harness.web.queries == ["Acme FY2024 revenue"]
    [request] = harness.requests(Role.TOOL)
    assert request.model == TOOL_MODEL
    assert "Acme FY2024 revenue" in request.text
    assert "https://acme.example/ir" in request.text and "Revenue $5.0B" in request.text


async def test_web_search_without_results_makes_no_model_call(harness: Harness) -> None:
    observation = await harness.env.call(WEB_SEARCH, {"search_query": "nothing matches"})

    assert observation == "No web results found for 'nothing matches'."
    assert harness.llm.requests == []


async def test_parse_html_page_stores_visible_text_under_the_key(harness: Harness) -> None:
    observation = await harness.env.call(
        PARSE_HTML_PAGE, {"url": "https://www.sec.gov/acme-10k.htm", "key": "acme_10k"}
    )

    assert observation.startswith("Stored ")
    assert "under key 'acme_10k'" in observation and "Stored keys: acme_10k." in observation
    length = int(capture(r"Stored ([\d,]+) characters", observation).replace(",", ""))
    assert harness.env.steps[-1].state == {"documents": ["acme_10k"], "answered": False}

    await harness.env.call(RETRIEVE_INFORMATION, {"prompt": "Revenue? {{acme_10k}}"})
    document = capture(r"<document>\n(.*)\n</document>", harness.llm.requests[0].text)
    assert len(document) == length
    assert "Revenue $5,000 $4,500" in document
    assert "trackingSecret" not in document and "injected" not in document and "font-family" not in document


async def test_parse_html_page_overwrites_and_lists_every_key(harness: Harness) -> None:
    await store(harness, "b", "<p>first</p>")
    await store(harness, "a", "<p>other</p>")

    observation = await harness.env.call(PARSE_HTML_PAGE, {"url": "https://example.com/a", "key": "b"})

    assert "(replacing its previous content)" in observation
    assert observation.endswith("Stored keys: a, b.")


async def test_parse_html_page_failure_stores_nothing(harness: Harness) -> None:
    observation = await harness.env.call(PARSE_HTML_PAGE, {"url": "https://example.com/missing", "key": "k"})

    assert observation == "Error: HTTP 404 Not Found fetching https://example.com/missing"
    assert harness.env.details()["documents"] == []


async def test_retrieve_information_substitutes_one_summary_per_key(harness: Harness) -> None:
    await store(harness, "q4_release", "<p>Q4 revenue was $1.4B</p>")
    await store(harness, "acme_10k", "<p>FY revenue was $5.0B</p>")

    observation = await harness.env.call(
        RETRIEVE_INFORMATION,
        {"prompt": "Compare {{acme_10k}} with {{ q4_release }}; recheck {{acme_10k}}."},
    )

    assert observation == (
        "Compare <summary of acme_10k> with <summary of q4_release>; recheck <summary of acme_10k>."
    )
    requests = harness.requests(Role.TOOL)
    assert len(requests) == 2
    by_key = {capture(STORED_UNDER.pattern, r.text): r.text for r in requests}
    assert "FY revenue was $5.0B" in by_key["acme_10k"] and "Q4 revenue" not in by_key["acme_10k"]
    assert all("Compare {{acme_10k}} with {{ q4_release }}" in text for text in by_key.values())


async def test_retrieve_information_missing_key_lists_available_keys(harness: Harness) -> None:
    await store(harness, "acme_10k", "<p>text</p>")
    await store(harness, "acme_8k", "<p>text</p>")

    observation = await harness.env.call(RETRIEVE_INFORMATION, {"prompt": "{{acme_10k}} vs {{acme_10q}}"})

    assert observation == "Error: no stored document under 'acme_10q'. Available keys: acme_10k, acme_8k."
    assert harness.llm.requests == []


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Summarize {{report}}", "Error: no stored document under 'report'. No documents are stored yet"),
        ("Summarize the report", "Error: the prompt contains no {{key}} placeholder. No documents"),
    ],
)
async def test_retrieve_information_with_an_empty_store(harness: Harness, prompt: str, expected: str) -> None:
    observation = await harness.env.call(RETRIEVE_INFORMATION, {"prompt": prompt})

    assert observation.startswith(expected)
    assert harness.llm.requests == []


async def test_retrieve_information_truncates_documents_beyond_the_summary_budget(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(env_module, "SUMMARY_INPUT_TOKENS", 5)  # 10 UTF-8 bytes at 2 bytes per token
    await store(harness, "short", "<p>0123456789</p>")
    await store(harness, "long", "<p>0123456789ABCDEF</p>")

    observation = await harness.env.call(RETRIEVE_INFORMATION, {"prompt": "{{short}} | {{long}}"})

    assert observation == (
        "<summary of short> | <summary of long>\n(Only the first 10 of 16 characters were read.)"
    )
    long_request = next(r for r in harness.llm.requests if 'stored under "long"' in r.text)
    assert "0123456789\n</document>" in long_request.text


async def test_the_summary_budget_counts_bytes_and_never_splits_a_character(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(env_module, "SUMMARY_INPUT_TOKENS", 5)  # 10 bytes: three 3-byte characters
    await store(harness, "euros", "<p>€€€€€</p>")

    observation = await harness.env.call(RETRIEVE_INFORMATION, {"prompt": "{{euros}}"})

    assert observation == "<summary of euros>\n(Only the first 3 of 5 characters were read.)"
    assert "<document>\n€€€\n</document>" in harness.requests(Role.TOOL)[0].text


def provider_error(status: int, message: str, code: str | int | None = None) -> openai.APIStatusError:
    """The error the OpenAI SDK raises for an HTTP error from OpenAI or OpenRouter."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body: dict[str, Any] = {"message": message, "type": "invalid_request_error", "code": code}
    sdk_message = f"Error code: {status} - {{'error': {body}}}"
    error_type = openai.BadRequestError if status == 400 else openai.APIStatusError
    return error_type(sdk_message, response=httpx.Response(status, request=request), body=body)


TOO_LONG = "the summarizer rejected document 'acme_10k' as too long for the model"


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (
            LLMOutputError("tool-model returned unusable tool output 3 times: empty reply"),
            "the summarizer gave no usable summary of document 'acme_10k': tool-model returned unusable",
        ),
        (
            provider_error(
                400,
                "Your input exceeds the context window of this model. Please adjust your input.",
                "context_length_exceeded",
            ),
            f"{TOO_LONG} (HTTP 400): Error code: 400",
        ),
        (
            provider_error(
                400,
                "This model's maximum context length is 272000 tokens. However, your messages resulted in "
                "301000 tokens. Please reduce the length of the messages.",
            ),
            f"{TOO_LONG} (HTTP 400): Error code: 400",
        ),
        (
            provider_error(
                400,
                "This endpoint's maximum context length is 163840 tokens. However, you requested about "
                "301000 tokens (301000 of text input). Please reduce the length of either one, or use the "
                "context-compression plugin to compress your prompt automatically.",
                400,
            ),
            f"{TOO_LONG} (HTTP 400): Error code: 400",
        ),
        (provider_error(413, "Request entity too large."), f"{TOO_LONG} (HTTP 413): Error code: 413"),
    ],
    ids=["unusable-output", "openai-context-window", "openai-maximum-context", "openrouter", "http-413"],
)
async def test_a_document_the_summarizer_cannot_summarize_is_an_error_observation(
    harness: Harness, failure: Exception, reason: str
) -> None:
    await store(harness, "acme_10k", "<p>FY revenue was $5.0B</p>")
    harness.summarizer_failures["acme_10k"] = failure

    observation = await harness.env.call(RETRIEVE_INFORMATION, {"prompt": "Revenue in {{acme_10k}}?"})

    assert observation.startswith(f"Error: {reason}")
    assert harness.env.tool_calls == 2
    assert harness.env.steps[-1].observation == observation
    assert not harness.env.done


async def test_web_results_the_summarizer_cannot_summarize_are_an_error_observation(harness: Harness) -> None:
    harness.web.results = [SearchResult("Acme FY2024", "https://acme.example/ir", "Revenue $5.0B", None)]
    harness.summarizer_failures[None] = LLMOutputError("tool-model returned unusable tool output 3 times")

    observation = await harness.env.call(WEB_SEARCH, {"search_query": "acme revenue"})

    assert observation.startswith("Error: the summarizer gave no usable summary of the web results for")


@pytest.mark.parametrize(
    "failure",
    [
        BackendUnavailableError("OpenAI rejected the API key"),
        SpendCapExceededError("run 'r' spend $5.0000 has reached its cap $5.0000"),
        provider_error(500, "The server had an error while processing your request."),
        provider_error(
            400,
            "Unsupported parameter: 'temperature' is not supported with this model.",
            "unsupported_parameter",
        ),
        provider_error(400, "deepseek/deepseek-v9 is not a valid model ID", 400),
    ],
    ids=["backend-unavailable", "spend-cap", "http-500", "unsupported-parameter", "invalid-model"],
)
async def test_summarizer_service_failures_propagate(harness: Harness, failure: Exception) -> None:
    await store(harness, "acme_10k", "<p>FY revenue was $5.0B</p>")
    harness.summarizer_failures["acme_10k"] = failure

    with pytest.raises(type(failure)):
        await harness.env.call(RETRIEVE_INFORMATION, {"prompt": "Revenue in {{acme_10k}}?"})

    assert len(harness.env.steps) == 1


@dataclass
class GatedSummarizer:
    """Holds every document summary until `gate` opens. A key in `fail_now` fails at once, a key in
    `fail_later` once the gate opens."""

    fail_now: dict[str, Exception] = field(default_factory=dict[str, Exception])
    fail_later: dict[str, Exception] = field(default_factory=dict[str, Exception])
    gate: asyncio.Event = field(default_factory=asyncio.Event)
    waiting: list[str] = field(default_factory=list[str])
    cancelled: list[str] = field(default_factory=list[str])

    async def chat(
        self,
        role: Role,
        model: ModelSpec,
        messages: Sequence[Message],
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
    ) -> AssistantTurn:
        key = capture(STORED_UNDER.pattern, str(messages[0].get("content")))
        if key in self.fail_now:
            raise self.fail_now[key]
        self.waiting.append(key)
        try:
            await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled.append(key)
            raise
        if key in self.fail_later:
            raise self.fail_later[key]
        return text_turn(f"<summary of {key}>")

    async def structured(
        self, role: Role, model: ModelSpec, messages: Sequence[Message], schema: type[SchemaT]
    ) -> SchemaT:
        raise AssertionError("no judge call expected")

    def scoped(self, sink: CallSink) -> GatedSummarizer:
        return self


async def gated_env(summarizer: GatedSummarizer, keys: Sequence[str]) -> FinanceEnv:
    pages = FakePages({f"https://example.com/{key}": f"<p>{key} text</p>" for key in keys})
    env = FinanceEnv(QUESTION, summarizer, TOOL_MODEL, JUDGE_MODEL, Backends(FakeEdgar(), FakeWeb(), pages))
    for key in keys:
        await env.call(PARSE_HTML_PAGE, {"url": f"https://example.com/{key}", "key": key})
    return env


async def test_a_failed_summary_cancels_the_other_summaries_of_the_call() -> None:
    summarizer = GatedSummarizer(fail_now={"q4_release": LLMOutputError("unusable output")})
    env = await gated_env(summarizer, ["acme_10k", "q4_release"])

    observation = await asyncio.wait_for(
        env.call(RETRIEVE_INFORMATION, {"prompt": "{{acme_10k}} vs {{q4_release}}"}), timeout=5
    )

    assert observation.startswith("Error: the summarizer gave no usable summary of document 'q4_release'")
    assert summarizer.cancelled == ["acme_10k"]


async def test_a_service_failure_outranks_a_concurrent_unusable_summary() -> None:
    summarizer = GatedSummarizer(
        fail_later={
            "acme_10k": LLMOutputError("unusable output"),
            "q4_release": BackendUnavailableError("OpenAI rejected the API key"),
        }
    )
    env = await gated_env(summarizer, ["acme_10k", "q4_release"])
    retrieval = asyncio.create_task(
        env.call(RETRIEVE_INFORMATION, {"prompt": "{{acme_10k}} vs {{q4_release}}"})
    )
    while len(summarizer.waiting) < 2:
        await asyncio.sleep(0)

    summarizer.gate.set()

    with pytest.raises(BackendUnavailableError):
        await retrieval
    assert summarizer.cancelled == []


async def test_a_tool_not_offered_to_the_executor_is_not_run(harness: Harness) -> None:
    unchanged = {"documents": [], "answered": False}

    observation = await harness.env.call(EDGAR_SEARCH, edgar_args(), offered={WEB_SEARCH, PARSE_HTML_PAGE})
    answer = await harness.env.call(FINAL_ANSWER, {"answer": "$5.0 billion"}, offered={EDGAR_SEARCH})

    assert observation == (
        f"Error: tool {EDGAR_SEARCH!r} is not available to this executor. "
        f"Available tools: {WEB_SEARCH}, {PARSE_HTML_PAGE}."  # sorted: finance_agent_google_web_search first
    )
    assert answer.startswith(f"Error: tool {FINAL_ANSWER!r} is not available")
    assert harness.edgar.queries == []
    assert not harness.env.done and harness.requests(Role.JUDGE) == []
    assert harness.env.tool_calls == 2
    assert [(step.tool, step.state) for step in harness.env.steps] == [
        (EDGAR_SEARCH, unchanged),
        (FINAL_ANSWER, unchanged),
    ]
    assert not (await harness.env.call(EDGAR_SEARCH, edgar_args(), offered={EDGAR_SEARCH})).startswith(
        "Error"
    )
    assert len(harness.edgar.queries) == 1


async def test_final_answer_ends_the_episode_and_is_judged(harness: Harness) -> None:
    harness.verdicts[:] = [True, False]
    sources = [{"url": "https://www.sec.gov/acme-10k.htm", "name": "Acme 10-K"}]

    observation = await harness.env.call(FINAL_ANSWER, {"answer": "$5.0 billion", "sources": sources})

    assert observation == "Final answer submitted with 1 source(s)."
    assert harness.env.done and harness.env.won
    assert harness.env.steps[-1].state == {"documents": [], "answered": True}
    [judge_request] = harness.requests(Role.JUDGE)
    assert judge_request.model == JUDGE_MODEL
    details = harness.env.details()
    assert details["answer"] == "$5.0 billion" and details["sources"] == sources and details["passed"]
    assert [(v["operator"], v["verdict"], v["satisfied"]) for v in details["verdicts"]] == [
        ("correctness", True, True),
        ("contradiction", False, True),
    ]
    with pytest.raises(RuntimeError):
        await harness.env.call(WEB_SEARCH, {"search_query": "more"})
    assert harness.env.tool_calls == 1


async def test_an_unusable_judgement_is_not_an_output_failure_and_records_no_answer() -> None:
    def handler(request: FakeRequest) -> BaseModel:
        raise LLMOutputError("judge-model returned unusable judge output 3 times: empty reply")

    env = FinanceEnv(
        QUESTION, FakeLLM(handler), TOOL_MODEL, JUDGE_MODEL, Backends(FakeEdgar(), FakeWeb(), FakePages())
    )

    with pytest.raises(JudgeError) as caught:
        await env.call(FINAL_ANSWER, {"answer": "$5.0 billion", "sources": None})

    assert not isinstance(caught.value, LLMOutputError)
    assert not env.done
    assert env.details()["answer"] is None


async def test_a_contradicting_final_answer_ends_the_episode_lost(harness: Harness) -> None:
    harness.verdicts[:] = [True, True]

    await harness.env.call(FINAL_ANSWER, {"answer": "$5.0 billion, or maybe $4 billion", "sources": None})

    assert harness.env.done and not harness.env.won
    assert harness.env.details()["passed"] is False


async def test_an_empty_final_answer_ends_the_episode_without_a_judge_call(harness: Harness) -> None:
    await harness.env.call(FINAL_ANSWER, {"answer": "  ", "sources": None})

    assert harness.env.done and not harness.env.won
    assert harness.llm.requests == []
    assert harness.env.details() == {
        "answer": "  ",
        "sources": [],
        "passed": False,
        "verdicts": [],
        "documents": [],
    }


async def test_failed_calls_count_and_are_recorded(harness: Harness) -> None:
    harness.edgar.error = "EDGAR search failed with HTTP 500: upstream"
    calls: list[tuple[str, dict[str, Any]]] = [
        ("finance_agent_unknown", {}),
        (EDGAR_SEARCH, edgar_args(end_date="April 7")),
        (EDGAR_SEARCH, edgar_args()),
        (PARSE_HTML_PAGE, {"url": "https://example.com/missing", "key": "k"}),
        (RETRIEVE_INFORMATION, {"prompt": "{{nothing}}"}),
        (FINAL_ANSWER, {"sources": None}),
    ]

    observations = [await harness.env.call(name, arguments) for name, arguments in calls]

    assert all(observation.startswith("Error: ") for observation in observations)
    assert observations[0].endswith(f"Available tools: {', '.join(TOOL_NAMES)}.")
    assert "invalid arguments for finance_agent_edgar_search: end_date" in observations[1]
    assert "invalid arguments for finance_agent_final_answer: answer" in observations[5]
    assert harness.env.tool_calls == len(calls)
    assert [step.step for step in harness.env.steps] == list(range(1, len(calls) + 1))
    assert [step.tool for step in harness.env.steps] == [name for name, _ in calls]
    assert not harness.env.done


async def test_an_unavailable_backend_raises_instead_of_becoming_an_observation(harness: Harness) -> None:
    harness.edgar.outage = "sec-api.io quota exhausted (HTTP 429): Upgrade your plan."

    with pytest.raises(BackendUnavailableError, match="quota exhausted"):
        await harness.env.call(EDGAR_SEARCH, edgar_args())

    assert list(harness.env.steps) == []
    assert not harness.env.done


def test_tool_specs_follow_the_requested_order_and_reject_unknown_names(harness: Harness) -> None:
    assert [spec.name for spec in harness.env.tool_specs([FINAL_ANSWER, EDGAR_SEARCH])] == [
        FINAL_ANSWER,
        EDGAR_SEARCH,
    ]
    assert harness.env.initial_state() == {"documents": [], "answered": False}
    with pytest.raises(ValueError):
        harness.env.tool_specs(["textcraft_craft"])


async def test_close_closes_the_episode_http_client(harness: Harness) -> None:
    http = httpx.AsyncClient()
    backends = Backends(edgar=harness.edgar, web=harness.web, pages=harness.pages)
    env = FinanceEnv(QUESTION, harness.llm, TOOL_MODEL, JUDGE_MODEL, backends, http=http)

    await env.close()

    assert http.is_closed
