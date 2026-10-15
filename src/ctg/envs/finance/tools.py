"""The five Finance-Agent tools (Appendix B.1.3): names, argument models and strict JSON schemas.

The schemas are what the model sees. The argument models validate what it sends back; they are lenient
about omitted nullable arguments because not every provider enforces strict schemas.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints

from ctg.llm.base import ToolSpec

EDGAR_SEARCH = "finance_agent_edgar_search"
WEB_SEARCH = "finance_agent_google_web_search"
PARSE_HTML_PAGE = "finance_agent_parse_html_page"
RETRIEVE_INFORMATION = "finance_agent_retrieve_information"
FINAL_ANSWER = "finance_agent_final_answer"
TOOL_NAMES = (EDGAR_SEARCH, WEB_SEARCH, PARSE_HTML_PAGE, RETRIEVE_INFORMATION, FINAL_ANSWER)

DEFAULT_PAGE = 1
DEFAULT_TOP_N_RESULTS = 10


class ToolError(Exception):
    """A tool call failed in a way the agent should see. The message becomes the observation."""


_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _require_iso_date(value: object) -> object:
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        raise ValueError("expected a date string in yyyy-mm-dd format")
    return value


IsoDate = Annotated[date, BeforeValidator(_require_iso_date)]
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EdgarSearchArgs(_Args):
    query: NonBlank
    form_types: list[str] | None = None
    ciks: list[str] | None = None
    start_date: IsoDate | None = None
    end_date: IsoDate | None = None
    page: Annotated[int, Field(ge=1)] | None = None
    top_n_results: Annotated[int, Field(ge=1)] | None = None


class WebSearchArgs(_Args):
    search_query: NonBlank


class ParseHtmlPageArgs(_Args):
    url: NonBlank
    key: NonBlank


class RetrieveInformationArgs(_Args):
    prompt: NonBlank


class Source(_Args):
    url: str
    name: str


class FinalAnswerArgs(_Args):
    answer: str
    sources: list[Source] | None = None


def _object(properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_SOURCE_SCHEMA = _object(
    {
        "url": {"type": "string", "description": "URL of the source."},
        "name": {
            "type": "string",
            "description": "Descriptive name of the source, e.g. 'Apple FY2024 10-K'.",
        },
    }
)

TOOL_SPECS: dict[str, ToolSpec] = {
    EDGAR_SEARCH: ToolSpec(
        name=EDGAR_SEARCH,
        description=(
            "Search the SEC EDGAR database for regulatory filings (full-text search). Returns filing "
            "metadata: company name, ticker, CIK, form type, filing date, description and document URL. It "
            "does not return filing text; open a document with finance_agent_parse_html_page."
        ),
        parameters=_object(
            {
                "query": {
                    "type": "string",
                    "description": (
                        'Keyword or phrase to search, e.g. "substantial doubt" OR "material weakness".'
                    ),
                },
                "form_types": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": 'Filter by form types, e.g. ["10-K", "10-Q", "8-K"]. null for all forms.',
                },
                "ciks": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Filter by company CIK numbers. null for all companies.",
                },
                "start_date": {
                    "type": ["string", "null"],
                    "description": "Start of the filing-date range, yyyy-mm-dd. null for the earliest.",
                },
                "end_date": {
                    "type": ["string", "null"],
                    "description": "End of the filing-date range, yyyy-mm-dd. null for the latest.",
                },
                "page": {
                    "type": ["integer", "null"],
                    "description": f"Page number for pagination. null for {DEFAULT_PAGE}.",
                },
                "top_n_results": {
                    "type": ["integer", "null"],
                    "description": f"Maximum number of results to return. null for {DEFAULT_TOP_N_RESULTS}.",
                },
            }
        ),
    ),
    WEB_SEARCH: ToolSpec(
        name=WEB_SEARCH,
        description=(
            "Search the web for financial information using Google Search. Returns a summary of the search "
            "results with the relevant information extracted and their URLs."
        ),
        parameters=_object(
            {"search_query": {"type": "string", "description": "The query to search for on the web."}}
        ),
    ),
    PARSE_HTML_PAGE: ToolSpec(
        name=PARSE_HTML_PAGE,
        description=(
            "Fetch and parse an HTML page, storing the extracted text content for later retrieval with "
            "finance_agent_retrieve_information. Returns a confirmation with the content length and the "
            "current storage keys."
        ),
        parameters=_object(
            {
                "url": {"type": "string", "description": "The URL of the HTML page to parse."},
                "key": {
                    "type": "string",
                    "description": "The key under which to store the parsed content for later retrieval.",
                },
            }
        ),
    ),
    RETRIEVE_INFORMATION: ToolSpec(
        name=RETRIEVE_INFORMATION,
        description=(
            "Retrieve and summarize previously stored documents using placeholder substitution. Each "
            "{{key_name}} placeholder in the prompt is replaced with a summary of the stored document "
            "focused on the prompt. Returns the prompt with the placeholders replaced."
        ),
        parameters=_object(
            {
                "prompt": {
                    "type": "string",
                    "description": (
                        "A prompt containing {{key_name}} placeholders that will be replaced with stored "
                        "content, e.g. 'Summarize this filing: {{company_10k}}'."
                    ),
                }
            }
        ),
    ),
    FINAL_ANSWER: ToolSpec(
        name=FINAL_ANSWER,
        description=(
            "Submit the final answer to the financial question. This ends the task; only one answer can "
            "be submitted."
        ),
        parameters=_object(
            {
                "answer": {"type": "string", "description": "The final answer to the financial question."},
                "sources": {
                    "type": ["array", "null"],
                    "items": _SOURCE_SCHEMA,
                    "description": "List of sources with URLs and descriptive names.",
                },
            }
        ),
    ),
}
