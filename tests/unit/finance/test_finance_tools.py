from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from ctg.envs.finance.tools import (
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
    WebSearchArgs,
)

ARGS_MODELS: dict[str, type[BaseModel]] = {
    EDGAR_SEARCH: EdgarSearchArgs,
    WEB_SEARCH: WebSearchArgs,
    PARSE_HTML_PAGE: ParseHtmlPageArgs,
    RETRIEVE_INFORMATION: RetrieveInformationArgs,
    FINAL_ANSWER: FinalAnswerArgs,
}


def test_tool_names_and_parameters_follow_appendix_b13() -> None:
    assert TOOL_NAMES == (
        "finance_agent_edgar_search",
        "finance_agent_google_web_search",
        "finance_agent_parse_html_page",
        "finance_agent_retrieve_information",
        "finance_agent_final_answer",
    )
    parameters = {name: list(spec.parameters["properties"]) for name, spec in TOOL_SPECS.items()}
    assert parameters == {
        EDGAR_SEARCH: ["query", "form_types", "ciks", "start_date", "end_date", "page", "top_n_results"],
        WEB_SEARCH: ["search_query"],
        PARSE_HTML_PAGE: ["url", "key"],
        RETRIEVE_INFORMATION: ["prompt"],
        FINAL_ANSWER: ["answer", "sources"],
    }
    assert TOOL_SPECS[FINAL_ANSWER].parameters["properties"]["sources"]["items"]["required"] == [
        "url",
        "name",
    ]


def _objects(schema: dict[str, Any]) -> list[dict[str, Any]]:
    found = [schema] if "properties" in schema else []
    for child in schema.get("properties", {}).values():
        found += _objects(child)
    if "items" in schema:
        found += _objects(schema["items"])
    return found


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_schemas_are_strict_and_match_the_argument_models(name: str) -> None:
    spec = TOOL_SPECS[name]

    assert spec.name == name
    for obj in _objects(spec.parameters):
        assert obj["type"] == "object"
        assert obj["additionalProperties"] is False
        assert obj["required"] == list(obj["properties"])
    assert set(spec.parameters["properties"]) == set(ARGS_MODELS[name].model_fields)


@pytest.mark.parametrize("value", ["2024/01/31", "31-01-2024", "20240131", "2024-02-30", 20240131])
def test_edgar_dates_must_be_iso_calendar_dates(value: object) -> None:
    with pytest.raises(ValidationError):
        EdgarSearchArgs.model_validate({"query": "revenue", "end_date": value})


@pytest.mark.parametrize("field", ["page", "top_n_results"])
def test_edgar_page_and_result_count_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError):
        EdgarSearchArgs.model_validate({"query": "revenue", field: 0})


def test_omitted_nullable_arguments_default_to_null() -> None:
    args = EdgarSearchArgs.model_validate({"query": "revenue", "start_date": "2024-01-31"})

    assert args.start_date == date(2024, 1, 31)
    assert (args.form_types, args.ciks, args.end_date, args.page, args.top_n_results) == (None,) * 5
    assert FinalAnswerArgs.model_validate({"answer": "42"}).sources is None


@pytest.mark.parametrize(
    ("model", "arguments"),
    [
        (EdgarSearchArgs, {"query": "  "}),
        (WebSearchArgs, {"search_query": ""}),
        (ParseHtmlPageArgs, {"url": "https://example.com", "key": " "}),
        (RetrieveInformationArgs, {"prompt": "x", "extra": 1}),
        (FinalAnswerArgs, {"answer": "x", "sources": [{"url": "https://example.com"}]}),
    ],
)
def test_blank_extra_or_incomplete_arguments_are_rejected(
    model: type[BaseModel], arguments: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(arguments)
