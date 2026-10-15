from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from ctg.envs.finance.edgar import EdgarQuery, SecApiFullTextSearch
from ctg.envs.finance.pages import (
    GENERIC_USER_AGENT,
    MAX_PAGE_BYTES,
    HttpPageFetcher,
    extract_text,
    is_sec_host,
)
from ctg.envs.finance.search import ExaSearch, SearchResult, SerpApiSearch, WebSearch, format_results
from ctg.envs.finance.tools import ToolError
from ctg.errors import BackendUnavailableError

FIXTURES = Path(__file__).parent / "fixtures"
SEC_USER_AGENT = "Jane Researcher jane@example.org"
QUERY = EdgarQuery("going concern", ("10-K",), ("320193",), date(2024, 1, 1), date(2025, 4, 7), 2)
SEC_API_RATE_LIMITED = {
    "status": 429,
    "error": "Too many requests. Rate limit exceeded, slow down and retry.",
}
SEARCH_KEY = "search-secret-key"


def fixture_json(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


async def test_sec_api_request_and_filing_metadata() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=fixture_json("sec_api_full_text_search.json"))

    async with client(handler) as http:
        filings = await SecApiFullTextSearch(http, "sec-key").search(QUERY)

    [request] = requests
    assert (request.method, str(request.url)) == ("POST", "https://api.sec-api.io/full-text-search")
    assert request.headers["Authorization"] == "sec-key"
    assert json.loads(request.content) == {
        "query": "going concern",
        "formTypes": ["10-K"],
        "ciks": ["320193"],
        "startDate": "2024-01-01",
        "endDate": "2025-04-07",
        "page": "2",
    }
    assert [f.form_type for f in filings] == ["10-K", "8-K", "10-Q"]
    first = filings[0]
    assert (first.company, first.ticker, first.cik, first.filed_at) == (
        "Apple Inc. (AAPL) (CIK 0000320193)",
        "AAPL",
        "320193",
        "2024-11-01",
    )
    assert first.url.startswith("https://www.sec.gov/Archives/edgar/data/320193/")
    assert filings[1].description == "EXHIBIT 99.1"
    assert filings[2].description == ""


async def test_sec_api_retries_rate_limits_with_backoff() -> None:
    statuses = iter([429, 429, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(
            status,
            json=fixture_json("sec_api_full_text_search.json") if status == 200 else SEC_API_RATE_LIMITED,
        )

    sleeps = Sleeps()
    async with client(handler) as http:
        filings = await SecApiFullTextSearch(http, "k", max_retries=3, backoff_s=0.5, sleep=sleeps).search(
            QUERY
        )

    assert len(filings) == 3
    assert sleeps.delays == [0.5, 1.0]


async def test_sec_api_gives_up_after_max_retries() -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(429, json=SEC_API_RATE_LIMITED)

    sleeps = Sleeps()
    async with client(handler) as http:
        search = SecApiFullTextSearch(http, "k", max_retries=2, backoff_s=1.0, sleep=sleeps)
        with pytest.raises(ToolError, match="rate limited"):
            await search.search(QUERY)

    assert len(attempts) == 3
    assert sleeps.delays == [1.0, 2.0]


@pytest.mark.parametrize(
    "message",
    [
        "You exceeded the free query limit of 100 requests. Upgrade your account.",
        "Your free trial request limit has been exceeded.",
    ],
)
async def test_sec_api_quota_exhaustion_makes_the_backend_unavailable_without_retrying(message: str) -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(429, json={"status": 429, "error": message})

    sleeps = Sleeps()
    async with client(handler) as http:
        search = SecApiFullTextSearch(http, "k", max_retries=5, backoff_s=1.0, sleep=sleeps)
        with pytest.raises(
            BackendUnavailableError, match=r"^sec-api.io quota exhausted \(HTTP 429\)"
        ) as caught:
            await search.search(QUERY)

    assert message in str(caught.value)
    assert len(attempts) == 1
    assert sleeps.delays == []


@pytest.mark.parametrize("status", [401, 403])
async def test_sec_api_rejected_key_makes_the_backend_unavailable(status: int) -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(
            status, json={"status": status, "error": "The API key is missing or not valid."}
        )

    async with client(handler) as http:
        with pytest.raises(BackendUnavailableError, match=f"rejected the API key \\(HTTP {status}\\)"):
            await SecApiFullTextSearch(http, "k").search(QUERY)

    assert len(attempts) == 1


@pytest.mark.parametrize("status", [400, 404, 500])
async def test_sec_api_other_http_errors_are_tool_errors(status: int) -> None:
    async with client(lambda request: httpx.Response(status, text="upstream failure")) as http:
        with pytest.raises(ToolError, match=f"HTTP {status}: upstream failure"):
            await SecApiFullTextSearch(http, "k").search(QUERY)


async def test_serpapi_parses_organic_results() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=fixture_json("serpapi_google.json"))

    async with client(handler) as http:
        results = await SerpApiSearch(http, "serp-key").search("Netflix ARM 2024")

    params = requests[0].url.params
    assert (params["engine"], params["q"], params["api_key"]) == ("google", "Netflix ARM 2024", "serp-key")
    assert results[0] == SearchResult(
        title="Netflix Q4 2024 Shareholder Letter",
        url="https://ir.netflix.net/files/doc_financials/2024/q4/FINAL-Q4-24-Shareholder-Letter.pdf",
        content="Average monthly revenue per paying membership (ARM) was $11.70 in 2024.",
        date="Jan 21, 2025",
    )
    assert results[1].date is None


async def test_serpapi_without_results_is_empty() -> None:
    body = {
        "search_information": {"organic_results_state": "Fully empty"},
        "error": "Google hasn't returned any results for this query.",
    }
    async with client(lambda request: httpx.Response(200, json=body)) as http:
        assert await SerpApiSearch(http, "k").search("zzzz") == []


def web_search(service: str, http: httpx.AsyncClient) -> WebSearch:
    return SerpApiSearch(http, SEARCH_KEY) if service == "SerpAPI" else ExaSearch(http, SEARCH_KEY)


@pytest.mark.parametrize(
    ("service", "status", "body"),
    [
        (
            "SerpAPI",
            401,
            {"error": "Invalid API key. Your API key should be here: https://serpapi.com/manage-api-key"},
        ),
        ("SerpAPI", 402, {"error": "Payment Required"}),
        ("SerpAPI", 403, {"error": "Forbidden"}),
        ("SerpAPI", 429, {"error": "Your account has run out of searches."}),
        ("SerpAPI", 200, {"error": "Your account has run out of searches."}),
        ("Exa", 401, {"requestId": "r1", "error": "Invalid API key", "tag": "INVALID_API_KEY"}),
        ("Exa", 402, {"requestId": "r2", "error": "No more credits", "tag": "NO_MORE_CREDITS"}),
        ("Exa", 402, {"requestId": "r3", "error": "Budget exceeded", "tag": "API_KEY_BUDGET_EXCEEDED"}),
        ("Exa", 403, {"requestId": "r4", "error": "Not on your plan", "tag": "FEATURE_DISABLED"}),
    ],
)
async def test_account_failures_make_the_search_backend_unavailable(
    service: str, status: int, body: dict[str, Any]
) -> None:
    async with client(lambda request: httpx.Response(status, json=body)) as http:
        with pytest.raises(BackendUnavailableError, match=f"^{service} cannot serve this run") as caught:
            await web_search(service, http).search("q")

    assert str(body["error"]) in str(caught.value)


@pytest.mark.parametrize(
    ("service", "status", "body"),
    [
        ("SerpAPI", 400, {"error": "Missing query `q` parameter."}),
        ("SerpAPI", 404, {"error": "Not Found"}),
        ("SerpAPI", 429, {"error": "You have exceeded the hourly throughput limit."}),
        ("SerpAPI", 503, {"error": "We couldn't get valid results for this search. Please try again later."}),
        ("SerpAPI", 200, {"error": "We couldn't get valid results for this search. Please try again later."}),
        ("Exa", 400, {"requestId": "r1", "error": "Invalid request body", "tag": "INVALID_REQUEST_BODY"}),
        ("Exa", 403, {"requestId": "r2", "error": "Rejected by moderation", "tag": "PROHIBITED_CONTENT"}),
        ("Exa", 404, {"requestId": "r3", "error": "Not found"}),
        ("Exa", 429, {"requestId": "r4", "error": "Rate limit exceeded", "tag": "RATE_LIMIT_EXCEEDED"}),
        ("Exa", 500, {"requestId": "r5", "error": "Internal server error"}),
    ],
)
async def test_other_search_failures_are_tool_errors(service: str, status: int, body: dict[str, Any]) -> None:
    async with client(lambda request: httpx.Response(status, json=body)) as http:
        with pytest.raises(ToolError, match=f"^{service} search failed") as caught:
            await web_search(service, http).search("q")

    assert str(body["error"]) in str(caught.value)


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (200, {"error": f"Invalid API key: {SEARCH_KEY}"}, BackendUnavailableError),
        (401, {"error": f"Invalid API key: {SEARCH_KEY}"}, BackendUnavailableError),
        (400, {"error": f"Unsupported parameter api_key={SEARCH_KEY}"}, ToolError),
    ],
)
async def test_serpapi_errors_never_reveal_the_key(
    status: int, body: dict[str, Any], error: type[Exception]
) -> None:
    async with client(lambda request: httpx.Response(status, json=body)) as http:
        with pytest.raises(error) as caught:
            await SerpApiSearch(http, SEARCH_KEY).search("q")

    assert SEARCH_KEY not in str(caught.value)
    assert "***" in str(caught.value)


async def test_exa_request_and_results() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=fixture_json("exa_search.json"))

    async with client(handler) as http:
        results = await ExaSearch(http, "exa-key", num_results=2, max_characters=100).search("TJX margin")

    [request] = requests
    assert (request.method, str(request.url), request.headers["x-api-key"]) == (
        "POST",
        "https://api.exa.ai/search",
        "exa-key",
    )
    assert json.loads(request.content) == {
        "query": "TJX margin",
        "numResults": 2,
        "contents": {"text": {"maxCharacters": 100}},
    }
    assert [(r.url, r.date) for r in results] == [
        ("https://investors.tjx.com/news/q4-fy25", "2025-02-26T00:00:00.000Z"),
        ("https://example.com/tjx-analysis", None),
    ]
    assert "11.6%" in results[0].content
    listing = format_results(results)
    assert listing.startswith(
        "[1] TJX Reports Q4 and FY25 Results\nURL: https://investors.tjx.com/news/q4-fy25"
    )
    assert "[2] TJX beats margin guidance" in listing


@pytest.mark.parametrize(
    ("host", "is_sec"),
    [
        ("sec.gov", True),
        ("www.sec.gov", True),
        ("EFTS.SEC.GOV", True),
        ("notsec.gov", False),
        ("sec.gov.example.com", False),
        ("example.com", False),
    ],
)
def test_sec_host_detection(host: str, is_sec: bool) -> None:
    assert is_sec_host(host) is is_sec


async def test_sec_user_agent_is_sent_only_to_sec_gov() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.host] = request.headers["User-Agent"]
        return httpx.Response(200, text="<p>ok</p>", headers={"content-type": "text/html"})

    async with client(handler) as http:
        fetcher = HttpPageFetcher(http, SEC_USER_AGENT)
        for url in ("https://www.sec.gov/a.htm", "https://example.com/b", "https://notsec.gov/c"):
            assert await fetcher.fetch(url) == "<p>ok</p>"

    assert seen == {
        "www.sec.gov": SEC_USER_AGENT,
        "example.com": GENERIC_USER_AGENT,
        "notsec.gov": GENERIC_USER_AGENT,
    }
    assert "@" not in GENERIC_USER_AGENT


async def test_redirects_choose_the_user_agent_per_host() -> None:
    hops: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hops.append((str(request.url), request.headers["User-Agent"]))
        if request.url.host == "www.sec.gov":
            return httpx.Response(302, headers={"location": "https://cdn.example.com/doc.htm"})
        return httpx.Response(200, text="<p>moved</p>", headers={"content-type": "text/html; charset=utf-8"})

    async with client(handler) as http:
        assert (
            await HttpPageFetcher(http, SEC_USER_AGENT).fetch("https://www.sec.gov/doc.htm") == "<p>moved</p>"
        )

    assert hops == [
        ("https://www.sec.gov/doc.htm", SEC_USER_AGENT),
        ("https://cdn.example.com/doc.htm", GENERIC_USER_AGENT),
    ]


@pytest.mark.parametrize(
    ("url", "response", "message"),
    [
        ("https://example.com/x", httpx.Response(404), "HTTP 404 Not Found"),
        (
            "https://example.com/x.pdf",
            httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}),
            "application/pdf, not an HTML or text page",
        ),
        (
            "https://example.com/loop",
            httpx.Response(301, headers={"location": "https://example.com/loop"}),
            "too many redirects",
        ),
        ("ftp://example.com/file", httpx.Response(200), "only http and https"),
    ],
)
async def test_page_fetch_failures_are_tool_errors(url: str, response: httpx.Response, message: str) -> None:
    async with client(lambda request: response) as http:
        with pytest.raises(ToolError, match=message):
            await HttpPageFetcher(http, SEC_USER_AGENT).fetch(url)


@pytest.mark.parametrize(
    "content_type",
    [
        "text/html; charset=utf-8",
        "text/plain",
        "TEXT/HTML",
        "application/xhtml+xml",
        "application/xml",
        "application/json",
        "application/atom+xml",
        "application/vnd.api+json; charset=utf-8",
    ],
)
async def test_text_media_types_are_read(content_type: str) -> None:
    response = httpx.Response(200, content=b"<p>ok</p>", headers={"content-type": content_type})
    async with client(lambda request: response) as http:
        assert await HttpPageFetcher(http, SEC_USER_AGENT).fetch("https://example.com/doc") == "<p>ok</p>"


@pytest.mark.parametrize(
    "content_type",
    [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
        "application/zip",
    ],
)
async def test_binary_media_types_are_tool_errors(content_type: str) -> None:
    response = httpx.Response(200, content=b"PK\x03\x04\x14\x00", headers={"content-type": content_type})
    async with client(lambda request: response) as http:
        with pytest.raises(ToolError, match=f"is {re.escape(content_type)}, not an HTML or text page"):
            await HttpPageFetcher(http, SEC_USER_AGENT).fetch("https://example.com/Financial_Report.xlsx")


@pytest.mark.parametrize(
    ("body", "kind"),
    [(b"PK\x03\x04\x14\x00\x06\x00", "a ZIP archive"), (b"%PDF-1.7\n%\xe2\xe3", "a PDF document")],
)
async def test_untyped_binary_bodies_are_tool_errors(body: bytes, kind: str) -> None:
    async with client(lambda request: httpx.Response(200, content=body)) as http:
        with pytest.raises(ToolError, match=f"is {kind}, not an HTML or text page"):
            await HttpPageFetcher(http, SEC_USER_AGENT).fetch("https://example.com/download")


async def test_untyped_text_bodies_are_read() -> None:
    async with client(lambda request: httpx.Response(200, content=b"<p>ok</p>")) as http:
        assert await HttpPageFetcher(http, SEC_USER_AGENT).fetch("https://example.com/doc") == "<p>ok</p>"


class Chunks:
    """A streamed body of 1 MiB chunks without a Content-Length, recording how many were read."""

    def __init__(self, total_bytes: int) -> None:
        self.total_bytes = total_bytes
        self.read = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        remaining = self.total_bytes
        while remaining > 0:
            size = min(1 << 20, remaining)
            self.read += 1
            yield b"a" * size
            remaining -= size


def streamed(body: Chunks, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/plain", **(headers or {})}, content=body)


async def test_a_page_of_exactly_the_size_limit_is_read() -> None:
    async with client(lambda request: streamed(Chunks(MAX_PAGE_BYTES))) as http:
        text = await HttpPageFetcher(http, SEC_USER_AGENT).fetch("https://example.com/big.txt")
    assert len(text) == MAX_PAGE_BYTES


@pytest.mark.parametrize("total_bytes", [MAX_PAGE_BYTES + 1, 3 * MAX_PAGE_BYTES])
async def test_reading_a_larger_page_stops_at_the_size_limit(total_bytes: int) -> None:
    body = Chunks(total_bytes)
    async with client(lambda request: streamed(body)) as http:
        with pytest.raises(ToolError, match="larger than 20 MB"):
            await HttpPageFetcher(http, SEC_USER_AGENT).fetch("https://example.com/huge.txt")
    assert body.read == MAX_PAGE_BYTES // (1 << 20) + 1


async def test_a_declared_length_over_the_size_limit_is_refused_unread() -> None:
    body = Chunks(MAX_PAGE_BYTES + 1)
    response = streamed(body, {"content-length": str(MAX_PAGE_BYTES + 1)})
    async with client(lambda request: response) as http:
        with pytest.raises(ToolError, match="larger than 20 MB"):
            await HttpPageFetcher(http, SEC_USER_AGENT).fetch("https://example.com/huge.txt")
    assert body.read == 0


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("http://xn--zz.com/10k.htm", "invalid host name"),
        ("http://127.0.0.1:99999/10k.htm", "port 99999 is outside 1-65535"),
        ("https://www.sec.gov:0/10k.htm", "port 0 is outside 1-65535"),
        ("http:///10k.htm", "has no host"),
    ],
)
async def test_unrequestable_urls_are_tool_errors(url: str, message: str) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text="<p>ok</p>")

    async with client(handler) as http:
        with pytest.raises(ToolError, match=message):
            await HttpPageFetcher(http, SEC_USER_AGENT).fetch(url)
    assert requests == []


@pytest.mark.parametrize(
    ("location", "message"),
    [
        ("http://xn--zz.com/doc", "invalid host name"),
        ("https://ir.example.com:70000/doc", "port 70000 is outside 1-65535"),
        ("ftp://ir.example.com/doc", "only http and https"),
    ],
)
async def test_redirects_to_unrequestable_urls_are_tool_errors(location: str, message: str) -> None:
    hops: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hops.append(str(request.url))
        return httpx.Response(302, headers={"location": location})

    async with client(handler) as http:
        with pytest.raises(ToolError, match=message):
            await HttpPageFetcher(http, SEC_USER_AGENT).fetch("https://ir.example.com/q4")
    assert hops == ["https://ir.example.com/q4"]


async def test_an_out_of_range_port_is_a_tool_error_with_the_real_transport() -> None:
    async with httpx.AsyncClient(timeout=5) as http:
        with pytest.raises(ToolError, match="port 99999"):
            await HttpPageFetcher(http, SEC_USER_AGENT).fetch("http://127.0.0.1:99999/10k.htm")


def test_extract_text_drops_scripts_and_styles_and_keeps_layout() -> None:
    text = extract_text((FIXTURES / "filing.html").read_text(encoding="utf-8"))

    assert text.splitlines() == [
        "Acme Corp Form 10-K",
        "Acme Corp reported fiscal 2024 results.",
        "Item 7. Management’s Discussion",
        "Metric 2024 2023",
        "Revenue $5,000 $4,500",
        "Net income $700 $650",
        "Amounts in millions.",
        "Unaudited supplement follows.",
    ]


@pytest.mark.parametrize(
    ("html", "text"),
    [
        ("<table><tr><td><p>Revenue</p></td><td><p>$5,000</p></td></tr></table>", "Revenue $5,000"),
        (
            "<table><tr><th><div>Metric</div></th><th><div>2024</div><div>(audited)</div></th></tr>"
            "<tr><td><div>Net<br>income</div></td><td><p>$700</p></td></tr></table>"
            "<p>Amounts in millions.</p>",
            "Metric 2024 (audited)\nNet income $700\nAmounts in millions.",
        ),
        ("<div>Revenue<br>grew</div><ul><li>a</li><li>b</li></ul>", "Revenue\ngrew\na\nb"),
    ],
)
def test_extract_text_keeps_each_table_row_on_one_line(html: str, text: str) -> None:
    assert extract_text(html) == text
