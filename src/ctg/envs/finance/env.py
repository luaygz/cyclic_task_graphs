"""One Finance-Agent episode: the five tools over an episode-local document store, ended by a final answer."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from openai import APIStatusError
from pydantic import BaseModel, ValidationError

from ctg.config import ModelSpec, Role
from ctg.envs.base import Environment
from ctg.envs.finance.dataset import REFERENCE_DATE, FinanceQuestion
from ctg.envs.finance.edgar import EDGAR_START_DATE, EdgarQuery, EdgarSearch
from ctg.envs.finance.judge import Judgement, judge_answer
from ctg.envs.finance.pages import PageFetcher, extract_text
from ctg.envs.finance.search import WebSearch, format_results
from ctg.envs.finance.tools import (
    DEFAULT_PAGE,
    DEFAULT_TOP_N_RESULTS,
    EDGAR_SEARCH,
    FINAL_ANSWER,
    PARSE_HTML_PAGE,
    RETRIEVE_INFORMATION,
    TOOL_NAMES,
    TOOL_SPECS,
    WEB_SEARCH,
    EdgarSearchArgs,
    FinalAnswerArgs,
    ParseHtmlPageArgs,
    RetrieveInformationArgs,
    Source,
    ToolError,
    WebSearchArgs,
)
from ctg.llm.base import LLM, LLMOutputError, ToolSpec
from ctg.prompts import render
from ctg.runs.records import StepRecord

PLACEHOLDER = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")

SUMMARY_INPUT_TOKENS = 250_000
"""Estimated tokens of document text given to the summarizer; gpt-5-mini accepts 272k input tokens."""
DENSE_TEXT_BYTES_PER_TOKEN = 2
"""UTF-8 bytes per token assumed by the estimate. Dense documents set it: financial tables and XBRL JSON
measure 2.2-2.4 characters per o200k token and prose about 3.3, so their prefixes stay within the budget."""
CONTEXT_EXCEEDED = re.compile(
    r"context[ _](?:length|window)|maximum context|too many tokens|(?:prompt|input) is too long"
    r"|maximum number of tokens",
    re.IGNORECASE,
)
"""Provider wording, in an HTTP 400 error, for input longer than the model's context: OpenAI ("maximum context
length", "exceeds the context window"), OpenRouter ("This endpoint's maximum context length is N tokens"),
Anthropic ("prompt is too long") and Gemini ("exceeds the maximum number of tokens allowed")."""

ArgsT = TypeVar("ArgsT", bound=BaseModel)


@dataclass(frozen=True)
class Backends:
    edgar: EdgarSearch
    web: WebSearch
    pages: PageFetcher


class FinanceEnv(Environment):
    def __init__(
        self,
        question: FinanceQuestion,
        llm: LLM,
        tool_model: ModelSpec,
        judge_model: ModelSpec,
        backends: Backends,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        """`tool_model` summarizes search results and documents; `http` is closed with the episode."""
        self._question = question
        self._llm = llm
        self._tool_model = tool_model
        self._judge_model = judge_model
        self._backends = backends
        self._http = http
        self._documents: dict[str, str] = {}
        self._answer: str | None = None
        self._sources: list[Source] = []
        self._judgement: Judgement | None = None
        self._tool_calls = 0
        self._steps: list[StepRecord] = []
        self._handlers: dict[str, Callable[[Mapping[str, Any]], Awaitable[str]]] = {
            EDGAR_SEARCH: self._edgar_search,
            WEB_SEARCH: self._web_search,
            PARSE_HTML_PAGE: self._parse_html_page,
            RETRIEVE_INFORMATION: self._retrieve_information,
            FINAL_ANSWER: self._final_answer,
        }

    @property
    def objective(self) -> str:
        return self._question.question

    @property
    def won(self) -> bool:
        return self._judgement is not None and self._judgement.passed

    @property
    def done(self) -> bool:
        return self._answer is not None

    @property
    def tool_calls(self) -> int:
        return self._tool_calls

    @property
    def steps(self) -> Sequence[StepRecord]:
        return self._steps

    def initial_state(self) -> dict[str, Any]:
        return {"documents": [], "answered": False}

    def _state(self) -> dict[str, Any]:
        return {"documents": sorted(self._documents), "answered": self.done}

    def tool_specs(self, names: Sequence[str]) -> list[ToolSpec]:
        unknown = [name for name in names if name not in TOOL_SPECS]
        if unknown:
            raise ValueError(f"not Finance-Agent tools: {unknown}")
        return [TOOL_SPECS[name] for name in names]

    async def call(
        self, name: str, arguments: Mapping[str, Any], offered: Collection[str] | None = None
    ) -> str:
        if self.done:
            raise RuntimeError("the episode is over: a final answer was already submitted")
        self._tool_calls += 1
        handler = self._handlers.get(name)
        if offered is not None and name not in offered:
            observation = (
                f"Error: tool {name!r} is not available to this executor. "
                f"Available tools: {', '.join(sorted(offered))}."
            )
        elif handler is None:
            observation = f"Error: unknown tool {name!r}. Available tools: {', '.join(TOOL_NAMES)}."
        else:
            try:
                observation = await handler(arguments)
            except ToolError as exc:
                observation = f"Error: {exc}"
        self._steps.append(
            StepRecord(
                step=self._tool_calls,
                tool=name,
                arguments=dict(arguments),
                observation=observation,
                state=self._state(),
            )
        )
        return observation

    def details(self) -> dict[str, Any]:
        verdicts = self._judgement.verdicts if self._judgement else ()
        return {
            "answer": self._answer,
            "sources": [source.model_dump() for source in self._sources],
            "passed": self.won,
            "verdicts": [verdict.as_dict() for verdict in verdicts],
            "documents": sorted(self._documents),
        }

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()

    async def _edgar_search(self, arguments: Mapping[str, Any]) -> str:
        args = _parse(EdgarSearchArgs, EDGAR_SEARCH, arguments)
        start = args.start_date or EDGAR_START_DATE
        end = min(args.end_date, REFERENCE_DATE) if args.end_date else REFERENCE_DATE
        if start > end:
            raise ToolError(
                f"start_date {start} is after end_date {end} (filings are searchable up to {REFERENCE_DATE})"
            )
        page = args.page or DEFAULT_PAGE
        query = EdgarQuery(
            query=args.query,
            form_types=tuple(args.form_types or ()),
            ciks=tuple(args.ciks or ()),
            start_date=start,
            end_date=end,
            page=page,
        )
        filings = (await self._backends.edgar.search(query))[: args.top_n_results or DEFAULT_TOP_N_RESULTS]
        header = f"EDGAR filings matching {args.query!r} filed {start} to {end}, page {page}"
        if not filings:
            return f"{header}: no filings found."
        listing = json.dumps([filing.as_dict() for filing in filings], indent=2)
        return f"{header} ({len(filings)} shown):\n{listing}"

    async def _web_search(self, arguments: Mapping[str, Any]) -> str:
        args = _parse(WebSearchArgs, WEB_SEARCH, arguments)
        results = await self._backends.web.search(args.search_query)
        if not results:
            return f"No web results found for {args.search_query!r}."
        return await self._summarize(
            render("finance_search_summary", query=args.search_query, results=format_results(results)),
            f"the web results for {args.search_query!r}",
        )

    async def _parse_html_page(self, arguments: Mapping[str, Any]) -> str:
        args = _parse(ParseHtmlPageArgs, PARSE_HTML_PAGE, arguments)
        page = await self._backends.pages.fetch(args.url)
        text = await asyncio.to_thread(extract_text, page)
        if not text:
            raise ToolError(f"no text could be extracted from {args.url}")
        replaced = args.key in self._documents
        self._documents[args.key] = text
        note = " (replacing its previous content)" if replaced else ""
        return (
            f"Stored {len(text):,} characters from {args.url} under key {args.key!r}{note}. "
            f"Stored keys: {', '.join(sorted(self._documents))}."
        )

    async def _retrieve_information(self, arguments: Mapping[str, Any]) -> str:
        args = _parse(RetrieveInformationArgs, RETRIEVE_INFORMATION, arguments)
        keys = list(dict.fromkeys(match.group(1) for match in PLACEHOLDER.finditer(args.prompt)))
        if not keys:
            raise ToolError("the prompt contains no {{key}} placeholder. " + self._available_keys())
        missing = [key for key in keys if key not in self._documents]
        if missing:
            raise ToolError(
                f"no stored document under {', '.join(map(repr, missing))}. {self._available_keys()}"
            )
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(self._summarize_document(args.prompt, key)) for key in keys]
        except ExceptionGroup as failures:
            raise _decisive(failures) from None
        by_key = {key: task.result() for key, task in zip(keys, tasks, strict=True)}
        return PLACEHOLDER.sub(lambda match: by_key[match.group(1)], args.prompt)

    async def _final_answer(self, arguments: Mapping[str, Any]) -> str:
        args = _parse(FinalAnswerArgs, FINAL_ANSWER, arguments)
        sources = list(args.sources or ())
        judgement = await judge_answer(self._llm, self._judge_model, self._question, args.answer, sources)
        self._answer, self._sources, self._judgement = args.answer, sources, judgement
        return f"Final answer submitted with {len(sources)} source(s)."

    def _available_keys(self) -> str:
        if not self._documents:
            return f"No documents are stored yet; store pages with {PARSE_HTML_PAGE} first."
        return f"Available keys: {', '.join(sorted(self._documents))}."

    async def _summarize_document(self, prompt: str, key: str) -> str:
        document = self._documents[key]
        text = _summary_input(document)
        summary = await self._summarize(
            render("finance_retrieve", prompt=prompt, key=key, document=text), f"document {key!r}"
        )
        if len(text) == len(document):
            return summary
        return f"{summary}\n(Only the first {len(text):,} of {len(document):,} characters were read.)"

    async def _summarize(self, prompt: str, subject: str) -> str:
        """The tool model's reply to `prompt`. Unusable output and input refused as too long are tool
        errors; every other failure propagates (a provider 400 for another reason is a setup error)."""
        try:
            turn = await self._llm.chat(Role.TOOL, self._tool_model, [{"role": "user", "content": prompt}])
        except LLMOutputError as exc:
            raise ToolError(f"the summarizer gave no usable summary of {subject}: {exc}") from exc
        except APIStatusError as exc:
            if not _input_too_long(exc):
                raise
            raise ToolError(
                f"the summarizer rejected {subject} as too long for the model "
                f"(HTTP {exc.status_code}): {exc.message}"
            ) from exc
        if not turn.content or not turn.content.strip():
            raise ToolError(f"the summarizer returned an empty summary of {subject}")
        return turn.content.strip()


def _input_too_long(error: APIStatusError) -> bool:
    """The provider refused the request because its input is too long (HTTP 413, or a context-length 400)."""
    if error.status_code == 413:
        return True
    return error.status_code == 400 and (
        error.code == "context_length_exceeded" or CONTEXT_EXCEEDED.search(error.message) is not None
    )


def _summary_input(document: str) -> str:
    """The longest prefix of `document` whose estimated size fits SUMMARY_INPUT_TOKENS."""
    budget = SUMMARY_INPUT_TOKENS * DENSE_TEXT_BYTES_PER_TOKEN
    encoded = document.encode(errors="replace")  # one byte per lone surrogate keeps characters aligned
    if len(encoded) <= budget:
        return document
    return document[: len(encoded[:budget].decode(errors="ignore"))]


def _decisive(failures: ExceptionGroup[Exception]) -> Exception:
    """The failure to raise for concurrent summaries: the first that is not a ToolError, else the first."""
    errors = list(failures.exceptions)
    return next((error for error in errors if not isinstance(error, ToolError)), errors[0])


def _parse(model: type[ArgsT], tool: str, arguments: Mapping[str, Any]) -> ArgsT:
    try:
        return model.model_validate(dict(arguments))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, error['loc'])) or 'arguments'}: {error['msg']}" for error in exc.errors()
        )
        raise ToolError(f"invalid arguments for {tool}: {problems}") from exc
