"""Web search backends for finance_agent_google_web_search: Google through SerpAPI (default) or Exa."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import httpx

from ctg.envs.finance.tools import ToolError
from ctg.errors import BackendUnavailableError

SERPAPI_URL = "https://serpapi.com/search.json"
EXA_URL = "https://api.exa.ai/search"

_SERPAPI_ACCOUNT_ERRORS = ("invalid api key", "run out of searches")
"""Fragments of SerpAPI's documented `error` messages for a rejected key and a used-up plan."""

_EXA_QUERY_REJECTED = frozenset({"PROHIBITED_CONTENT", "CONTENT_FILTER_ERROR"})
"""Exa 403 tags that reject the query's content rather than the account."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    content: str
    """The search snippet (SerpAPI) or the beginning of the page text (Exa)."""
    date: str | None = None


class WebSearch(Protocol):
    async def search(self, query: str) -> list[SearchResult]:
        """Results for `query`, best first; empty when nothing matched.

        Raises ToolError on a failure the agent should see and BackendUnavailableError when the service
        cannot serve the run.
        """
        ...


def format_results(results: list[SearchResult]) -> str:
    """Numbered plain-text listing of the results, as shown to the summarizer."""
    blocks: list[str] = []
    for number, result in enumerate(results, 1):
        lines = [f"[{number}] {result.title}", f"URL: {result.url}"]
        if result.date:
            lines.append(f"Date: {result.date}")
        lines.append(result.content)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


class SerpApiSearch:
    """Google organic results through SerpAPI. The key travels in the query string, so it is never echoed.

    SerpAPI cannot serve the run when it answers HTTP 401 (key rejected), 402, or 403 (account deleted),
    or when its `error` says "Invalid API key" or "Your account has run out of searches" (sent with HTTP
    429, which otherwise means the hourly throughput limit and stays a ToolError).
    """

    def __init__(self, client: httpx.AsyncClient, api_key: str) -> None:
        self._client = client
        self._api_key = api_key

    async def search(self, query: str) -> list[SearchResult]:
        params = {"engine": "google", "q": query, "api_key": self._api_key}
        request = self._client.get(SERPAPI_URL, params=params)
        data = await _get_json(request, "SerpAPI", self._api_key, _serpapi_unavailable)
        information = cast(dict[str, Any], data.get("search_information") or {})
        if "error" in data and information.get("organic_results_state") != "Fully empty":
            message = _redact(str(data["error"]), self._api_key)
            if _serpapi_account_error(message):
                raise BackendUnavailableError(f"SerpAPI cannot serve this run: {message}")
            raise ToolError(f"SerpAPI search failed: {message}")
        return [
            SearchResult(
                title=str(item.get("title") or ""),
                url=str(item.get("link") or ""),
                content=str(item.get("snippet") or ""),
                date=str(item["date"]) if item.get("date") else None,
            )
            for item in cast(list[dict[str, Any]], data.get("organic_results") or [])
        ]


class ExaSearch:
    """Exa neural search with page text.

    Exa cannot serve the run when it answers HTTP 401 (key rejected), 402 (credits or spending budget
    used up), or 403 (feature not on the plan) unless the 403 rejects the query's content. Its 429 is rate
    limiting and stays a ToolError.
    """

    def __init__(
        self, client: httpx.AsyncClient, api_key: str, num_results: int = 5, max_characters: int = 5000
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._num_results = num_results
        self._max_characters = max_characters

    async def search(self, query: str) -> list[SearchResult]:
        body = {
            "query": query,
            "numResults": self._num_results,
            "contents": {"text": {"maxCharacters": self._max_characters}},
        }
        request = self._client.post(EXA_URL, json=body, headers={"x-api-key": self._api_key})
        data = await _get_json(request, "Exa", self._api_key, _exa_unavailable)
        return [
            SearchResult(
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                content=str(item.get("text") or ""),
                date=str(item["publishedDate"]) if item.get("publishedDate") else None,
            )
            for item in cast(list[dict[str, Any]], data.get("results") or [])
        ]


async def _get_json(
    request: Awaitable[httpx.Response],
    service: str,
    api_key: str,
    unavailable: Callable[[int, dict[str, Any]], bool],
) -> dict[str, Any]:
    """Await an httpx request and decode its JSON body. Error messages never contain the key.

    An error response for which `unavailable(status, body)` holds raises BackendUnavailableError; every
    other failure raises ToolError.
    """
    try:
        response = await request
    except httpx.HTTPError as exc:
        raise ToolError(
            f"{service} search failed: {type(exc).__name__}: {_redact(str(exc), api_key)}"
        ) from exc
    try:
        data: dict[str, Any] | None = cast(dict[str, Any], response.json())
    except ValueError:
        data = None
    if response.is_error:
        body = data or {}
        status = response.status_code
        detail = _redact(str(body.get("error") or response.reason_phrase), api_key)
        if unavailable(status, body):
            raise BackendUnavailableError(f"{service} cannot serve this run (HTTP {status}): {detail}")
        raise ToolError(f"{service} search failed with HTTP {status}: {detail}")
    if data is None:
        raise ToolError(f"{service} search returned a response that is not JSON")
    return data


def _serpapi_unavailable(status: int, body: dict[str, Any]) -> bool:
    return status in (401, 402, 403) or _serpapi_account_error(str(body.get("error") or ""))


def _serpapi_account_error(message: str) -> bool:
    lowered = message.lower()
    return any(fragment in lowered for fragment in _SERPAPI_ACCOUNT_ERRORS)


def _exa_unavailable(status: int, body: dict[str, Any]) -> bool:
    if status == 403:
        return body.get("tag") not in _EXA_QUERY_REJECTED
    return status in (401, 402)


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "***")
