"""Mockable Finance Agent search, EDGAR, HTML, and retrieval tools."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any, Protocol


class HttpTransport(Protocol):
    async def get_text(self, url: str, headers: dict[str, str], timeout: int) -> str: ...
    async def post_json(
        self, url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int
    ) -> tuple[int, dict[str, Any]]: ...


class AioHttpTransport:
    async def get_text(self, url: str, headers: dict[str, str], timeout: int) -> str:
        try:
            import aiohttp
        except ImportError as exc:
            raise RuntimeError("Install Finance tools with: pip install '.[finance]'") from exc
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                response.raise_for_status()
                return await response.text()

    async def post_json(
        self, url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int
    ) -> tuple[int, dict[str, Any]]:
        try:
            import aiohttp
        except ImportError as exc:
            raise RuntimeError("Install Finance tools with: pip install '.[finance]'") from exc
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 429:
                    return response.status, {}
                response.raise_for_status()
                return response.status, await response.json()


class FinanceToolClient:
    def __init__(
        self,
        transport: HttpTransport | None = None,
        search: Callable[[str], Awaitable[str]] | None = None,
        summarize_once: Callable[[str, str, str], Awaitable[str]] | None = None,
    ):
        self.transport = transport or AioHttpTransport()
        self.search = search
        self.summarize_once = summarize_once
        self.data_storage: dict[str, str] = {}

    async def web_search(self, search_query: str) -> str:
        if not search_query:
            return "Error: Search query not provided."
        if self.search is not None:
            return await self.search(search_query)
        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            return "Error: EXA_API_KEY is not set."
        try:
            from exa_py import Exa
        except ImportError as exc:
            raise RuntimeError("Install Finance search support with: pip install '.[finance]'") from exc

        def run_search() -> str:
            result = Exa(api_key=api_key).search(
                query=search_query,
                contents={"context": True, "text": {"maxCharacters": 5000}},
                type="fast",
                num_results=5,
            )
            return result.context

        context = await asyncio.to_thread(run_search)
        return await self._summarize(context, search_query)

    async def edgar_search(self, **arguments: Any) -> str:
        api_key = os.getenv("SEC_EDGAR_API_KEY")
        if not api_key:
            return "Error: SEC_EDGAR_API_KEY is not set."
        payload = {
            "query": arguments.get("query", ""),
            "formTypes": arguments.get("form_types", []),
            "ciks": arguments.get("ciks", []),
            "startDate": arguments.get("start_date", ""),
            "endDate": min(arguments.get("end_date") or "2025-04-07", "2025-04-07"),
            "page": str(arguments.get("page", 1)),
        }
        max_retries = 3
        for attempt in range(max_retries):
            status, result = await self.transport.post_json(
                "https://api.sec-api.io/full-text-search",
                payload,
                {"Content-Type": "application/json", "Authorization": api_key},
                timeout=60,
            )
            if status != 429:
                top_n = int(arguments.get("top_n_results", 10))
                fields = (
                    "companyNameLong", "ticker", "cik", "formType", "filedAt",
                    "description", "filingUrl", "accessionNo", "type",
                )
                filings = [{key: filing.get(key, "") for key in fields} for filing in result.get("filings", [])[:top_n]]
                return json.dumps(filings, indent=2)
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)
        return f"Error: HTTP 429 - Rate limit exceeded after {max_retries} attempts"

    async def parse_html_page(self, url: str, key: str) -> str:
        user_agent = os.getenv("SEC_USER_AGENT")
        if not user_agent:
            return "Error: SEC_USER_AGENT is required for HTML retrieval."
        if not url or not key:
            return "Error: Both url and key are required."
        html = await self.transport.get_text(url, {"User-Agent": user_agent}, timeout=60)
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise RuntimeError("Install Finance tools with: pip install '.[finance]'") from exc
        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style"]):
            element.extract()
        text = "\n".join(
            chunk for line in soup.get_text().splitlines() for chunk in [line.strip()] if chunk
        )
        overwritten = key in self.data_storage
        self.data_storage[key] = text
        warning = "WARNING: Existing key overwritten. " if overwritten else ""
        return f"{warning}SUCCESS: Saved {len(text)} characters under {key}."

    async def retrieve_information(self, prompt: str) -> str:
        keys = re.findall(r"{{([^{}]+)}}", prompt)
        if not keys:
            return "Error: Prompt must include at least one {{key_name}} placeholder."
        missing = [key for key in keys if key not in self.data_storage]
        if missing:
            return f"Error: Unknown data keys: {', '.join(missing)}"
        values = {key: await self._summarize(self.data_storage[key], prompt) for key in keys}
        formatted = re.sub(r"{{([^{}]+)}}", r"{\1}", prompt)
        return formatted.format(**values)

    async def _summarize(self, content: str, prompt: str) -> str:
        if self.summarize_once is None:
            return content
        from luna.benchmarks.finance_agent.summarize import summarize_content

        return await summarize_content(content, prompt, self.summarize_once)

