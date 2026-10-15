"""Fetching web pages and filings, and extracting their text.

SEC requires automated clients to identify themselves with a contact User-Agent. That contact string is
sent to sec.gov hosts only; every other host gets a generic browser User-Agent. Redirects are followed
manually so the URL is checked and the User-Agent chosen again for every hop.

Only text documents are read: the Content-Type media type must be text, HTML, XML or JSON, and a body
without a Content-Type must not start with a ZIP or PDF signature. Bodies are streamed and refused once
they exceed MAX_PAGE_BYTES.
"""

from __future__ import annotations

from typing import Protocol

import httpx
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from ctg.envs.finance.tools import ToolError

GENERIC_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
MAX_REDIRECTS = 5
MAX_PAGE_BYTES = 20 * 1024 * 1024
"""Largest page body read (after content decoding); a larger page is a tool error."""

_TEXT_MEDIA_TYPES = frozenset({"application/xhtml+xml", "application/xml", "application/json"})
_TEXT_MEDIA_SUFFIXES = ("+xml", "+json")
_BINARY_SIGNATURES = {b"PK\x03\x04": "a ZIP archive", b"%PDF": "a PDF document"}
_CELL_TAGS = ("td", "th")
_BLOCK_TAGS = (
    "p", "div", "tr", "li", "ul", "ol", "table", "section", "article", "header", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "dt", "dd", "title",
)  # fmt: skip


def is_sec_host(host: str) -> bool:
    host = host.lower().rstrip(".")
    return host == "sec.gov" or host.endswith(".sec.gov")


def user_agent_for(url: httpx.URL, sec_user_agent: str) -> str:
    return sec_user_agent if is_sec_host(url.host) else GENERIC_USER_AGENT


class PageFetcher(Protocol):
    async def fetch(self, url: str) -> str:
        """The page body as text (HTML or plain text). Raises ToolError on failure."""
        ...


class HttpPageFetcher:
    def __init__(self, client: httpx.AsyncClient, sec_user_agent: str) -> None:
        self._client = client
        self._sec_user_agent = sec_user_agent

    async def fetch(self, url: str) -> str:
        try:
            target = httpx.URL(url)
        except httpx.InvalidURL as exc:
            raise ToolError(f"invalid URL {url!r}: {exc}") from exc
        for _ in range(MAX_REDIRECTS + 1):
            _check_target(target)
            headers = {"User-Agent": user_agent_for(target, self._sec_user_agent)}
            try:
                async with self._client.stream(
                    "GET", target, headers=headers, follow_redirects=False
                ) as response:
                    if response.next_request is None:
                        return await _page_text(response)
                    target = response.next_request.url
            except httpx.HTTPError as exc:
                raise ToolError(f"could not fetch {target}: {type(exc).__name__}: {exc}") from exc
            except UnicodeError as exc:  # httpx decodes the host name of a redirect's target
                raise ToolError(f"{target} redirects to a URL with an invalid host name: {exc}") from exc
        raise ToolError(f"too many redirects fetching {url}")


def _check_target(url: httpx.URL) -> None:
    """Raise ToolError unless `url` is an http(s) URL with a valid host name and port."""
    if url.scheme not in ("http", "https"):
        raise ToolError(f"cannot fetch {url}: only http and https URLs are supported")
    try:
        host = url.host
    except UnicodeError as exc:  # an undecodable internationalized (IDNA) host name
        raise ToolError(f"cannot fetch {url}: invalid host name: {exc}") from exc
    if not host:
        raise ToolError(f"cannot fetch {url}: the URL has no host")
    if url.port is not None and not 1 <= url.port <= 65535:
        raise ToolError(f"cannot fetch {url}: port {url.port} is outside 1-65535")


def _is_text_media_type(media_type: str) -> bool:
    """Text, HTML, XML or JSON, given a lowercase media type without parameters."""
    return (
        media_type.startswith("text/")
        or media_type in _TEXT_MEDIA_TYPES
        or media_type.endswith(_TEXT_MEDIA_SUFFIXES)
    )


async def _page_text(response: httpx.Response) -> str:
    """The body of a streamed response decoded as text, read only if it is a text document."""
    if response.is_error:
        raise ToolError(f"HTTP {response.status_code} {response.reason_phrase} fetching {response.url}")
    media_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if media_type and not _is_text_media_type(media_type):
        raise ToolError(f"{response.url} is {media_type}, not an HTML or text page")
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_PAGE_BYTES:
        raise ToolError(_too_large(response.url))
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body += chunk
        if len(body) > MAX_PAGE_BYTES:
            raise ToolError(_too_large(response.url))
    if not media_type:
        for signature, kind in _BINARY_SIGNATURES.items():
            if body.startswith(signature):
                raise ToolError(f"{response.url} is {kind}, not an HTML or text page")
    return body.decode(response.encoding or "utf-8", errors="replace")


def _too_large(url: httpx.URL) -> str:
    return f"{url} is larger than {MAX_PAGE_BYTES // (1024 * 1024)} MB, the page size limit"


def extract_text(html: str) -> str:
    """Visible text of an HTML document: scripts and styles dropped, one line per block outside tables and
    one line per table row, with its cells separated by spaces."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    tags: list[Tag] = soup.find_all(True)  # document order: every parent precedes its children
    in_cells: set[int] = set()  # id() of each tag inside a table cell; `tags` keeps them all alive
    for tag in tags:
        parent = tag.parent
        if parent is not None and (parent.name in _CELL_TAGS or id(parent) in in_cells):
            in_cells.add(id(tag))
    for tag in tags:
        # Inside a cell, a block or `<br>` ends with a space so the whole row stays on one line.
        separator = " " if id(tag) in in_cells else "\n"
        if tag.name == "br":
            tag.replace_with(NavigableString(separator))
        elif tag.name in _CELL_TAGS:
            tag.append(NavigableString(" "))
        elif tag.name in _BLOCK_TAGS:
            tag.append(NavigableString(separator))
    lines = (" ".join(line.split()) for line in soup.get_text().splitlines())
    return "\n".join(line for line in lines if line)
