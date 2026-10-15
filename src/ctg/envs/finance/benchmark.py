"""Finance-Agent benchmark family (Appendix B.1.3, B.2.4; budgets Sec. 4.1.5 and Table 8)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

import httpx

from ctg.config import BenchmarkName, Budgets, ExperimentConfig, ModelSpec, Role
from ctg.envs.base import GENERALIST, Benchmark, Environment, ExecutorProfile, InstanceRef
from ctg.envs.finance.dataset import REFERENCE_DATE, get_question, load_questions
from ctg.envs.finance.edgar import SecApiFullTextSearch
from ctg.envs.finance.env import Backends, FinanceEnv
from ctg.envs.finance.pages import HttpPageFetcher
from ctg.envs.finance.search import ExaSearch, SerpApiSearch, WebSearch
from ctg.envs.finance.tools import (
    EDGAR_SEARCH,
    FINAL_ANSWER,
    PARSE_HTML_PAGE,
    RETRIEVE_INFORMATION,
    TOOL_NAMES,
    WEB_SEARCH,
)
from ctg.llm.base import LLM
from ctg.prompts import render

VARIANT = "finance_agent"
SPECIALIST_MAX_TOOL_CALLS = 5
HTTP_TIMEOUT_S = 60.0

_SEARCH_KEY_VARS = {"serpapi": "SERPAPI_API_KEY", "exa": "EXA_API_KEY"}


def _specialist(name: str, description: str, tool: str) -> ExecutorProfile:
    return ExecutorProfile(name, description, (tool,), SPECIALIST_MAX_TOOL_CALLS)


_ROSTER: Mapping[str, ExecutorProfile] = MappingProxyType(
    {
        GENERALIST: ExecutorProfile(
            GENERALIST,
            "Has every Finance-Agent tool: EDGAR search, web search, page parsing, document retrieval and "
            "final-answer submission.",
            TOOL_NAMES,
        ),
        "edgar_researcher": _specialist(
            "edgar_researcher",
            "Finds SEC filings with EDGAR full-text search and reports their form types, dates and URLs.",
            EDGAR_SEARCH,
        ),
        "web_researcher": _specialist(
            "web_researcher",
            "Searches the web for facts, figures, guidance and the URLs of relevant documents.",
            WEB_SEARCH,
        ),
        "page_parser": _specialist(
            "page_parser",
            "Fetches web pages and filing documents by URL and stores their text under named keys.",
            PARSE_HTML_PAGE,
        ),
        "retriever": _specialist(
            "retriever",
            "Extracts information from stored documents with prompts that reference them as {{key}}.",
            RETRIEVE_INFORMATION,
        ),
        "answerer": _specialist(
            "answerer",
            "Submits the final answer with its sources, which ends the episode.",
            FINAL_ANSWER,
        ),
    }
)


def required_env_vars(web_search: Literal["serpapi", "exa"]) -> tuple[str, ...]:
    """Environment variables a Finance-Agent episode needs with the given web-search backend."""
    return ("SEC_API_KEY", "SEC_USER_AGENT", _SEARCH_KEY_VARS[web_search])


class FinanceAgentBenchmark(Benchmark):
    name = BenchmarkName.FINANCE_AGENT

    @property
    def description(self) -> str:
        return render("benchmarks/finance_agent", reference_date=REFERENCE_DATE.isoformat())

    @property
    def roster(self) -> Mapping[str, ExecutorProfile]:
        return _ROSTER

    def budgets(self, config: ExperimentConfig) -> Budgets:
        return Budgets(global_tool_calls=20, local_tool_calls=20)

    def enumerate_instances(self, variant: str) -> list[InstanceRef]:
        if variant != VARIANT:
            raise ValueError(f"unknown Finance-Agent variant {variant!r}; the only variant is {VARIANT!r}")
        return [InstanceRef(q.id, {"question_type": q.question_type}) for q in load_questions()]

    async def make_env(
        self,
        instance: InstanceRef,
        config: ExperimentConfig,
        llm: LLM,
        models: Mapping[Role, ModelSpec],
    ) -> Environment:
        """Reads credentials from os.environ (create_llm loads .env) and fails fast if any is missing."""
        names = required_env_vars(config.web_search)
        missing = [name for name in names if not os.environ.get(name, "").strip()]
        if missing:
            raise RuntimeError(
                f"Finance-Agent needs these environment variables (see .env.example): {', '.join(missing)}"
            )
        sec_api_key, sec_user_agent, search_key = (os.environ[name].strip() for name in names)
        question = get_question(instance.id)
        http = httpx.AsyncClient(timeout=HTTP_TIMEOUT_S)
        web: WebSearch = (
            SerpApiSearch(http, search_key) if config.web_search == "serpapi" else ExaSearch(http, search_key)
        )
        backends = Backends(
            edgar=SecApiFullTextSearch(http, sec_api_key),
            web=web,
            pages=HttpPageFetcher(http, sec_user_agent),
        )
        return FinanceEnv(question, llm, models[Role.TOOL], models[Role.JUDGE], backends, http=http)
