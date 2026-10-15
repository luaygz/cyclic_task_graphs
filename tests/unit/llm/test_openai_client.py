"""OpenAIClient against a mocked HTTP transport: request shaping, output retries, accounting, caps and
provider errors."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from pydantic import BaseModel

from ctg.config import ModelSpec, Provider, Role, load_model_presets
from ctg.errors import BackendUnavailableError
from ctg.llm.base import (
    CallSink,
    LLMCallRecord,
    LLMOutputError,
    Message,
    SpendCapExceededError,
    ToolCall,
    ToolSpec,
)
from ctg.llm.client import SDK_MAX_RETRIES, OpenAIClient, create_llm
from ctg.llm.pricing import SpendLedger

GPT_4O_MINI = ModelSpec(
    name="gpt-4o-mini", input_usd_per_mtok=0.15, cached_input_usd_per_mtok=0.075, output_usd_per_mtok=0.60
)
GPT_5 = ModelSpec(
    name="gpt-5",
    reasoning_effort="low",
    supports_temperature=False,
    input_usd_per_mtok=1.25,
    cached_input_usd_per_mtok=0.125,
    output_usd_per_mtok=10.0,
)
GEMINI = ModelSpec(name="google/gemini-2.5-pro", provider=Provider.OPENROUTER)
PROMPT: list[Message] = [{"role": "user", "content": "go"}]
SEARCH = ToolSpec(
    name="search",
    description="Search the web.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
)
# Cost of a default reply (1000 prompt tokens, 200 completion tokens) on gpt-4o-mini.
GPT_4O_MINI_REPLY_COST = (1000 * 0.15 + 200 * 0.60) / 1e6
VALID_VERDICT = '{"answer": "42", "confidence": 0.9}'


class Verdict(BaseModel):
    answer: str
    confidence: float


def usage(
    prompt: int = 1000, completion: int = 200, cached: int = 0, reasoning: int = 0, **extra: Any
) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": reasoning},
        **extra,
    }


def reply(
    content: str | None = None,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    refusal: str | None = None,
    usage_: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "refusal": refusal,
                    "tool_calls": tool_calls,
                },
            }
        ],
        "usage": usage_ if usage_ is not None else usage(),
    }


def tool_call(name: str, arguments: str, call_id: str = "call_1") -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


@dataclass(frozen=True)
class ApiError:
    """A scripted provider error response (`{"error": error}` with HTTP `status`).

    It asks for a 1 ms retry delay, so the SDK's own retries of transient errors are fast.
    """

    status: int
    error: dict[str, Any]

    def response(self) -> httpx.Response:
        return httpx.Response(self.status, json={"error": self.error}, headers={"retry-after-ms": "1"})


@dataclass
class FakeServer:
    """Serves scripted chat completions and errors; any request beyond the script gets a 400."""

    replies: list[dict[str, Any] | ApiError]
    requests: list[httpx.Request] = field(default_factory=list[httpx.Request])

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.replies:
            return httpx.Response(400, json={"error": {"message": "unexpected request", "type": "test"}})
        scripted = self.replies.pop(0)
        if isinstance(scripted, ApiError):
            return scripted.response()
        return httpx.Response(200, json=scripted)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]


def mock_http(server: FakeServer) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(server.handle))


def make_client(server: FakeServer, ledger_path: Path, **kwargs: Any) -> OpenAIClient:
    return OpenAIClient(
        api_keys={Provider.OPENAI: "sk-test", Provider.OPENROUTER: "or-test"},
        ledger=SpendLedger(ledger_path),
        http_client=mock_http(server),
        **kwargs,
    )


def seed_spend(ledger_path: Path, cost: float, provider: Provider, run: str | None) -> None:
    record = LLMCallRecord(
        role=Role.EXECUTOR, model="seed", provider=provider, cost_usd=cost, started_at=datetime.now(UTC)
    )
    SpendLedger(ledger_path).append(record, run)


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "spend.jsonl"


async def test_structured_returns_validated_instance_from_strict_schema(ledger_path: Path) -> None:
    server = FakeServer([reply(VALID_VERDICT, usage_=usage(prompt=1000, completion=200, cached=400))])
    sink = CallSink()
    client = make_client(server, ledger_path).scoped(sink)

    result = await client.structured(Role.ROUTER, GPT_4O_MINI, PROMPT, Verdict)

    assert result == Verdict(answer="42", confidence=0.9)
    (request,) = server.requests
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    response_format = server.bodies[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Verdict"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False
    assert response_format["json_schema"]["schema"]["required"] == ["answer", "confidence"]
    (record,) = sink.calls
    assert (record.role, record.model, record.provider) == (Role.ROUTER, "gpt-4o-mini", Provider.OPENAI)
    assert (record.input_tokens, record.cached_input_tokens, record.output_tokens) == (1000, 400, 200)
    assert record.cost_usd == pytest.approx((600 * 0.15 + 400 * 0.075 + 200 * 0.60) / 1e6)
    assert record.error is None
    assert client.spent(Provider.OPENAI) == pytest.approx(record.cost_usd)


async def test_structured_retries_unusable_replies_then_succeeds(ledger_path: Path) -> None:
    server = FakeServer(
        [
            reply(None, refusal="I can't help with that."),
            reply('{"answer": "4', finish_reason="length"),
            reply(VALID_VERDICT),
        ]
    )
    sink = CallSink()
    client = make_client(server, ledger_path).scoped(sink)

    result = await client.structured(Role.ROUTER, GPT_4O_MINI, PROMPT, Verdict)

    assert result.answer == "42"
    assert len(server.requests) == 3
    errors = [record.error for record in sink.calls]
    assert errors[0] is not None and "refusal" in errors[0]
    assert errors[1] is not None and "length" in errors[1]
    assert errors[2] is None
    assert client.spent() == pytest.approx(3 * GPT_4O_MINI_REPLY_COST)


async def test_structured_raises_after_three_unusable_replies(ledger_path: Path) -> None:
    server = FakeServer(
        [reply("not json"), reply('{"answer": "42"}'), reply('{"answer": "42"}'), reply(VALID_VERDICT)]
    )
    sink = CallSink()
    client = make_client(server, ledger_path).scoped(sink)

    with pytest.raises(LLMOutputError, match="confidence"):
        await client.structured(Role.ROUTER, GPT_4O_MINI, PROMPT, Verdict)

    assert len(server.requests) == 3
    assert len(sink.calls) == 3
    assert all(record.error is not None for record in sink.calls)
    # Failed attempts still cost money.
    assert client.spent() == pytest.approx(3 * GPT_4O_MINI_REPLY_COST)


async def test_chat_parses_tool_calls_and_returns_appendable_history_message(ledger_path: Path) -> None:
    arguments = '{"query": "revenue"}'
    server = FakeServer(
        [
            reply("Looking it up.", tool_calls=[tool_call("search", arguments)], finish_reason="tool_calls"),
            reply("Revenue was 42."),
        ]
    )
    client = make_client(server, ledger_path)

    turn = await client.chat(Role.EXECUTOR, GPT_4O_MINI, PROMPT, tools=[SEARCH], tool_choice="required")

    assert turn.content == "Looking it up."
    assert turn.tool_calls == (ToolCall(id="call_1", name="search", arguments={"query": "revenue"}),)
    assert turn.message == {
        "role": "assistant",
        "content": "Looking it up.",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": arguments}}
        ],
    }
    first = server.bodies[0]
    assert first["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web.",
                "parameters": SEARCH.parameters,
                "strict": True,
            },
        }
    ]
    assert first["tool_choice"] == "required"
    assert first["parallel_tool_calls"] is False

    history: list[Message] = [
        *PROMPT,
        turn.message,
        {"role": "tool", "tool_call_id": "call_1", "content": "42"},
    ]
    final = await client.chat(Role.EXECUTOR, GPT_4O_MINI, history, tool_choice="none")

    assert final.tool_calls == ()
    assert final.message == {"role": "assistant", "content": "Revenue was 42."}
    second = server.bodies[1]
    assert second["messages"][1] == turn.message
    assert not {"tools", "tool_choice", "parallel_tool_calls"} & second.keys()


async def test_chat_retries_invalid_tool_arguments_and_truncated_replies(ledger_path: Path) -> None:
    server = FakeServer(
        [
            reply(None, tool_calls=[tool_call("search", '{"query": "rev')], finish_reason="tool_calls"),
            reply(None, tool_calls=[tool_call("search", '["revenue"]')], finish_reason="tool_calls"),
            reply(
                None,
                tool_calls=[tool_call("search", '{"query": "revenue"}', "call_3")],
                finish_reason="tool_calls",
            ),
        ]
    )
    sink = CallSink()
    client = make_client(server, ledger_path).scoped(sink)

    turn = await client.chat(Role.EXECUTOR, GPT_4O_MINI, PROMPT, tools=[SEARCH], tool_choice="required")

    assert turn.tool_calls == (ToolCall(id="call_3", name="search", arguments={"query": "revenue"}),)
    assert [record.error is None for record in sink.calls] == [False, False, True]

    truncated = FakeServer([reply("The answer is", finish_reason="length")] * 3)
    with pytest.raises(LLMOutputError, match="length"):
        await make_client(truncated, ledger_path).chat(Role.EXECUTOR, GPT_4O_MINI, PROMPT)
    assert len(truncated.requests) == 3


@pytest.mark.parametrize(
    ("model", "role", "expected", "absent"),
    [
        pytest.param(
            GPT_5,
            Role.EXECUTOR,
            {"reasoning_effort": "low", "max_completion_tokens": 2048},
            {"temperature", "max_tokens"},
            id="gpt-5-executor",
        ),
        pytest.param(
            GPT_4O_MINI,
            Role.EXECUTOR,
            {"temperature": 0.0, "max_completion_tokens": 2048},
            {"reasoning_effort", "max_tokens"},
            id="gpt-4o-mini-executor",
        ),
        pytest.param(
            GPT_4O_MINI,
            Role.PLANNER,
            {"temperature": 1.0, "max_completion_tokens": 16384},
            {"reasoning_effort", "max_tokens"},
            id="gpt-4o-mini-planner",
        ),
        pytest.param(
            GPT_5,
            Role.JUDGE,
            {"reasoning_effort": "low"},
            {"temperature", "max_completion_tokens", "max_tokens"},
            id="gpt-5-judge-provider-defaults",
        ),
    ],
)
async def test_decoding_parameters_follow_role_and_model(
    ledger_path: Path, model: ModelSpec, role: Role, expected: dict[str, Any], absent: set[str]
) -> None:
    server = FakeServer([reply("ok")])

    await make_client(server, ledger_path).chat(role, model, PROMPT)

    body = server.bodies[0]
    assert body["model"] == model.name
    assert {key: body.get(key) for key in expected} == expected
    assert not absent & body.keys()


async def test_openrouter_uses_its_endpoint_usage_accounting_and_reported_cost(ledger_path: Path) -> None:
    deepseek = ModelSpec(
        name="deepseek/deepseek-v3.1-terminus", provider=Provider.OPENROUTER, reasoning_effort="low"
    )
    server = FakeServer(
        [reply("ok", usage_=usage(prompt=500, completion=50, cached=100, reasoning=20, cost=0.0123))]
    )
    sink = CallSink()
    client = make_client(server, ledger_path, run="b5").scoped(sink)

    await client.chat(Role.EXECUTOR, deepseek, PROMPT)

    (request,) = server.requests
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer or-test"
    body = server.bodies[0]
    assert body["usage"] == {"include": True}
    assert body["reasoning"] == {"effort": "low"}
    assert body["max_tokens"] == 2048
    assert body["temperature"] == 0.0
    assert not {"max_completion_tokens", "reasoning_effort"} & body.keys()
    (record,) = sink.calls
    assert (record.provider, record.cached_input_tokens, record.reasoning_tokens) == (
        Provider.OPENROUTER,
        100,
        20,
    )
    assert record.cost_usd == 0.0123
    assert client.spent(Provider.OPENROUTER) == pytest.approx(0.0123)
    assert client.spent(Provider.OPENAI) == 0.0
    assert client.spent(run="b5") == pytest.approx(0.0123)


@pytest.mark.parametrize(
    ("seed", "caps"),
    [
        pytest.param(
            (1.0, Provider.OPENAI, "old-run"), {"provider_caps_usd": {Provider.OPENAI: 1.0}}, id="provider"
        ),
        pytest.param(
            (0.5, Provider.OPENROUTER, "r1"), {"run": "r1", "run_cap_usd": 0.5}, id="run-spans-providers"
        ),
    ],
)
async def test_reached_cap_refuses_before_any_request(
    ledger_path: Path, seed: tuple[float, Provider, str], caps: dict[str, Any]
) -> None:
    seed_spend(ledger_path, *seed)
    server = FakeServer([reply("unused")])
    sink = CallSink()
    client = make_client(server, ledger_path, **caps).scoped(sink)

    with pytest.raises(SpendCapExceededError):
        await client.chat(Role.EXECUTOR, GPT_4O_MINI, PROMPT)

    assert server.requests == []
    assert sink.calls == []


async def test_caps_only_count_their_own_provider_and_run(ledger_path: Path) -> None:
    seed_spend(ledger_path, 1.0, Provider.OPENAI, "other-run")
    server = FakeServer([reply("ok", usage_=usage(cost=0.01))])
    client = make_client(
        server,
        ledger_path,
        provider_caps_usd={Provider.OPENAI: 1.0, Provider.OPENROUTER: 1.0},
        run="mine",
        run_cap_usd=1.0,
    )

    await client.chat(Role.EXECUTOR, GEMINI, PROMPT)

    assert len(server.requests) == 1


async def test_cap_is_checked_again_before_each_retry(ledger_path: Path) -> None:
    server = FakeServer(
        [reply("not json", usage_=usage(cost=0.02)), reply(VALID_VERDICT, usage_=usage(cost=0.02))]
    )
    client = make_client(server, ledger_path, provider_caps_usd={Provider.OPENROUTER: 0.01})

    with pytest.raises(SpendCapExceededError):
        await client.structured(Role.ROUTER, GEMINI, PROMPT, Verdict)

    assert len(server.requests) == 1


async def test_missing_key_fails_only_when_its_provider_is_used(ledger_path: Path) -> None:
    server = FakeServer([reply("ok")])
    client = OpenAIClient(
        api_keys={Provider.OPENAI: "sk-test", Provider.OPENROUTER: None},
        ledger=SpendLedger(ledger_path),
        http_client=mock_http(server),
    )

    await client.chat(Role.EXECUTOR, GPT_4O_MINI, PROMPT)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        await client.chat(Role.EXECUTOR, GEMINI, PROMPT)

    assert len(server.requests) == 1


async def test_from_env_reads_keys_and_treats_blank_cap_as_none(ledger_path: Path) -> None:
    seed_spend(ledger_path, 0.5, Provider.OPENAI, None)
    server = FakeServer([reply("ok", usage_=usage(cost=0.01))])
    environ = {
        "OPENAI_API_KEY": "sk-env",
        "OPENROUTER_API_KEY": "or-env",
        "CTG_SPEND_CAP_OPENAI_USD": "0.5",
        "CTG_SPEND_CAP_OPENROUTER_USD": " ",
    }
    client = OpenAIClient.from_env(environ, ledger=SpendLedger(ledger_path), http_client=mock_http(server))

    with pytest.raises(SpendCapExceededError, match="CTG_SPEND_CAP_OPENAI_USD"):
        await client.chat(Role.EXECUTOR, GPT_4O_MINI, PROMPT)
    await client.chat(Role.EXECUTOR, GEMINI, PROMPT)

    (request,) = server.requests
    assert request.headers["authorization"] == "Bearer or-env"


def test_from_env_rejects_non_numeric_cap(ledger_path: Path) -> None:
    with pytest.raises(ValueError, match="CTG_SPEND_CAP_OPENROUTER_USD"):
        OpenAIClient.from_env({"CTG_SPEND_CAP_OPENROUTER_USD": "ten"}, ledger=SpendLedger(ledger_path))


async def test_ledger_totals_persist_across_client_instances(ledger_path: Path) -> None:
    first = make_client(FakeServer([reply("a", usage_=usage(cost=0.25)), reply("b")]), ledger_path, run="r1")
    await first.chat(Role.EXECUTOR, GEMINI, PROMPT)
    await first.chat(Role.ROUTER, GPT_4O_MINI, PROMPT)

    second = make_client(FakeServer([]), ledger_path, run="r2")

    assert second.spent() == pytest.approx(0.25 + GPT_4O_MINI_REPLY_COST)
    assert second.spent(Provider.OPENROUTER) == pytest.approx(0.25)
    assert second.spent(Provider.OPENAI, run="r1") == pytest.approx(GPT_4O_MINI_REPLY_COST)
    assert second.spent(run="r2") == 0.0
    lines = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert [(line["provider"], line["model"], line["role"], line["run"]) for line in lines] == [
        ("openrouter", "google/gemini-2.5-pro", "executor", "r1"),
        ("openai", "gpt-4o-mini", "router", "r1"),
    ]
    assert (lines[1]["input_tokens"], lines[1]["output_tokens"]) == (1000, 200)


async def test_concurrent_calls_are_all_recorded(ledger_path: Path) -> None:
    server = FakeServer([reply("ok", usage_=usage(cost=0.01)) for _ in range(20)])
    client = make_client(server, ledger_path)

    await asyncio.gather(*(client.chat(Role.EXECUTOR, GEMINI, PROMPT) for _ in range(20)))

    assert len(ledger_path.read_text().splitlines()) == 20
    assert client.spent() == pytest.approx(0.2)
    assert SpendLedger(ledger_path).total() == pytest.approx(0.2)


async def test_scoped_views_record_into_their_own_and_inherited_sinks(ledger_path: Path) -> None:
    server = FakeServer([reply("not json"), reply(VALID_VERDICT), reply("plain")])
    base = make_client(server, ledger_path)
    episode, step, sibling = CallSink(), CallSink(), CallSink()
    episode_llm = base.scoped(episode)
    base.scoped(sibling)

    await episode_llm.scoped(step).structured(Role.PLANNER, GPT_4O_MINI, PROMPT, Verdict)
    await episode_llm.chat(Role.EXECUTOR, GPT_4O_MINI, PROMPT)

    assert [(record.role, record.error is None) for record in step.calls] == [
        (Role.PLANNER, False),
        (Role.PLANNER, True),
    ]
    assert episode.calls[:2] == step.calls
    assert [record.role for record in episode.calls] == [Role.PLANNER, Role.PLANNER, Role.EXECUTOR]
    assert sibling.calls == []
    assert base.spent() == pytest.approx(sum(record.cost_usd for record in episode.calls))


@pytest.mark.parametrize(
    ("model", "script", "raised"),
    [
        pytest.param(GPT_4O_MINI, [], openai.BadRequestError, id="openai-bad-request"),
        pytest.param(
            GEMINI,
            [ApiError(403, {"code": 403, "message": "Input was flagged.", "metadata": {"reasons": ["x"]}})],
            openai.PermissionDeniedError,
            id="openrouter-moderation-flag",
        ),
        pytest.param(
            GEMINI,
            [
                ApiError(
                    402,
                    {
                        "code": 402,
                        "message": "In-flight budget exceeded.",
                        "metadata": {"limit_source": "openrouter_in_flight_budget"},
                    },
                )
            ],
            openai.APIStatusError,
            id="openrouter-in-flight-budget",
        ),
    ],
)
async def test_request_errors_are_recorded_and_propagate(
    ledger_path: Path, model: ModelSpec, script: list[ApiError], raised: type[openai.APIStatusError]
) -> None:
    server = FakeServer([*script])
    sink = CallSink()
    client = make_client(server, ledger_path).scoped(sink)

    with pytest.raises(raised):
        await client.chat(Role.EXECUTOR, model, PROMPT)

    (record,) = sink.calls
    assert record.error is not None and raised.__name__ in record.error
    assert record.cost_usd == 0.0
    assert len(ledger_path.read_text().splitlines()) == 1


QUOTA_EXHAUSTED = ApiError(
    429,
    {
        "message": "You exceeded your current quota, please check your plan and billing details.",
        "type": "insufficient_quota",
        "code": "insufficient_quota",
    },
)


@pytest.mark.parametrize(
    ("model", "script", "status", "detail"),
    [
        pytest.param(
            GPT_4O_MINI,
            [
                ApiError(
                    401,
                    {
                        "message": "Incorrect API key provided: sk-test.",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    },
                )
            ],
            401,
            "Incorrect API key provided",
            id="openai-rejected-key",
        ),
        pytest.param(
            GEMINI,
            [ApiError(401, {"code": 401, "message": "User not found."})],
            401,
            "User not found.",
            id="openrouter-rejected-key",
        ),
        pytest.param(
            GPT_4O_MINI,
            [
                ApiError(
                    403,
                    {
                        "message": "Country, region, or territory not supported",
                        "type": "request_forbidden",
                        "code": "unsupported_country_region_territory",
                    },
                )
            ],
            403,
            "territory not supported",
            id="openai-permission-denied",
        ),
        pytest.param(
            GPT_4O_MINI,
            [
                ApiError(
                    404,
                    {
                        "message": "The model `gpt-4o-mini` does not exist or you do not have access to it.",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    },
                )
            ],
            404,
            "does not exist",
            id="openai-unknown-model",
        ),
        pytest.param(
            GPT_4O_MINI,
            [QUOTA_EXHAUSTED] * (1 + SDK_MAX_RETRIES),
            429,
            "exceeded your current quota",
            id="openai-quota-exhausted",
        ),
        pytest.param(
            GEMINI,
            [
                ApiError(
                    402,
                    {
                        "code": 402,
                        "message": "Insufficient credits. Add more using https://openrouter.ai/settings/credits",
                        "metadata": {"limit_source": "openrouter_credits"},
                    },
                )
            ],
            402,
            "Insufficient credits",
            id="openrouter-no-credits",
        ),
    ],
)
async def test_provider_that_cannot_serve_the_run_raises_backend_unavailable(
    ledger_path: Path, model: ModelSpec, script: list[ApiError], status: int, detail: str
) -> None:
    server = FakeServer([*script])
    sink = CallSink()
    client = make_client(server, ledger_path).scoped(sink)

    with pytest.raises(BackendUnavailableError) as raised:
        await client.chat(Role.EXECUTOR, model, PROMPT)

    message = str(raised.value)
    assert all(part in message for part in (str(model.provider), model.name, f"HTTP {status}", detail))
    assert "sk-test" not in message and "or-test" not in message
    (record,) = sink.calls
    assert record.error is not None and "BackendUnavailableError" in record.error
    assert "sk-test" not in record.error
    assert record.cost_usd == 0.0
    assert len(ledger_path.read_text().splitlines()) == 1


async def test_transient_rate_limit_is_retried_and_the_call_succeeds(ledger_path: Path) -> None:
    rate_limited = ApiError(
        429,
        {"message": "Rate limit reached for gpt-4o-mini.", "type": "requests", "code": "rate_limit_exceeded"},
    )
    server = FakeServer([rate_limited, reply("ok")])
    sink = CallSink()

    turn = await make_client(server, ledger_path).scoped(sink).chat(Role.EXECUTOR, GPT_4O_MINI, PROMPT)

    assert turn.content == "ok"
    assert len(server.requests) == 2
    (record,) = sink.calls
    assert record.error is None


async def test_null_usage_counts_are_recorded_as_zero(ledger_path: Path) -> None:
    null_usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "cost": 0.004}
    server = FakeServer([reply("ok", usage_=null_usage)])
    sink = CallSink()
    client = make_client(server, ledger_path).scoped(sink)

    turn = await client.chat(Role.EXECUTOR, GEMINI, PROMPT)

    assert turn.content == "ok"
    (record,) = sink.calls
    assert (record.input_tokens, record.output_tokens, record.error) == (0, 0, None)
    assert client.spent(Provider.OPENROUTER) == pytest.approx(0.004)


async def test_empty_arguments_for_a_parameterless_tool_parse_as_an_empty_object(ledger_path: Path) -> None:
    look = ToolSpec(
        name="look",
        description="Describe the surroundings.",
        parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    )
    server = FakeServer([reply(None, tool_calls=[tool_call("look", "")], finish_reason="tool_calls")])

    turn = await make_client(server, ledger_path).chat(Role.EXECUTOR, GPT_4O_MINI, PROMPT, tools=[look])

    assert turn.tool_calls == (ToolCall(id="call_1", name="look", arguments={}),)
    assert turn.message == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "look", "arguments": "{}"}}],
    }
    assert len(server.requests) == 1


class Count(BaseModel):
    value: int


@pytest.mark.live
async def test_live_gpt_4o_mini_structured_call() -> None:
    llm = create_llm(run="live-test")
    sink = CallSink()
    messages: list[Message] = [{"role": "user", "content": "Return the number seven as `value`."}]

    result = await llm.scoped(sink).structured(
        Role.ROUTER, load_model_presets()["gpt-4o-mini"], messages, Count
    )

    assert result.value == 7
    (record,) = sink.calls
    assert record.input_tokens > 0
    assert record.cost_usd > 0
