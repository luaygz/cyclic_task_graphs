"""SEC EDGAR full-text search through sec-api.io."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Protocol, cast

import httpx

from ctg.envs.finance.tools import ToolError
from ctg.errors import BackendUnavailableError

SEC_API_URL = "https://api.sec-api.io/full-text-search"

EDGAR_START_DATE = date(2001, 1, 1)
"""Full-text search covers filings since 2001. sec-api.io would otherwise default to the last 30 days."""


@dataclass(frozen=True)
class EdgarQuery:
    query: str
    form_types: tuple[str, ...]
    ciks: tuple[str, ...]
    start_date: date
    end_date: date
    page: int


@dataclass(frozen=True)
class Filing:
    company: str
    ticker: str
    cik: str
    form_type: str
    filed_at: str
    description: str
    url: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


class EdgarSearch(Protocol):
    async def search(self, query: EdgarQuery) -> list[Filing]:
        """One page of matching filings, best match first.

        Raises ToolError on a failure the agent should see and BackendUnavailableError when the service
        cannot serve the run.
        """
        ...


class SecApiFullTextSearch:
    """POST https://api.sec-api.io/full-text-search, retrying with exponential backoff on HTTP 429.

    The service cannot serve the run (BackendUnavailableError, no retry) when it rejects the key (HTTP
    401 or 403) or answers 429 with a request-limit message that asks to upgrade the plan or mentions
    the trial: the account's quota is used up.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        max_retries: int = 5,
        backoff_s: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._sleep = sleep

    async def search(self, query: EdgarQuery) -> list[Filing]:
        payload = {
            "query": query.query,
            "formTypes": list(query.form_types),
            "ciks": list(query.ciks),
            "startDate": query.start_date.isoformat(),
            "endDate": query.end_date.isoformat(),
            "page": str(query.page),
        }
        headers = {"Authorization": self._api_key}
        attempt = 0
        while True:
            try:
                response = await self._client.post(SEC_API_URL, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                raise ToolError(f"EDGAR search failed: {type(exc).__name__}: {exc}") from exc
            if response.status_code in (401, 403):
                raise BackendUnavailableError(
                    f"sec-api.io rejected the API key (HTTP {response.status_code}): {_api_error(response)}"
                )
            if response.status_code != 429:
                break
            message = _api_error(response)
            if _quota_exhausted(message):
                raise BackendUnavailableError(f"sec-api.io quota exhausted (HTTP 429): {message}")
            if attempt == self._max_retries:
                raise ToolError(
                    f"EDGAR search is rate limited (HTTP 429 after {attempt + 1} attempts): {message}"
                )
            await self._sleep(self._backoff_s * 2**attempt)
            attempt += 1
        if response.is_error:
            raise ToolError(f"EDGAR search failed with HTTP {response.status_code}: {response.text[:300]}")
        try:
            data = cast(dict[str, Any], response.json())
        except ValueError as exc:
            raise ToolError("EDGAR search returned a response that is not JSON") from exc
        return [_filing(cast(dict[str, Any], item)) for item in data.get("filings") or []]


def _api_error(response: httpx.Response) -> str:
    try:
        error = cast(dict[str, Any], response.json()).get("error")
    except ValueError:
        error = None
    return str(error) if error else response.text[:300]


def _quota_exhausted(message: str) -> bool:
    lowered = message.lower()
    return "limit" in lowered and ("upgrade" in lowered or "trial" in lowered)


def _filing(item: dict[str, Any]) -> Filing:
    def text(key: str) -> str:
        value = item.get(key)
        return "" if value is None else str(value)

    return Filing(
        company=text("companyNameLong"),
        ticker=text("ticker"),
        cik=text("cik"),
        form_type=text("formType"),
        filed_at=text("filedAt"),
        description=text("description"),
        url=text("filingUrl"),
    )
