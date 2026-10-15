"""The `LLM` implementation for real runs: OpenAI and OpenRouter through the official SDK (Chat Completions).

Every attempt, failed or not, becomes an `LLMCallRecord` that goes to the spend ledger and to every
attached sink. Spend caps are checked before each request is sent. A provider response that means no
call of the run can succeed (rejected key, unknown model, no credits or quota left) raises
`BackendUnavailableError`; transient errors are retried inside the SDK.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar, cast

import httpx
from dotenv import load_dotenv
from openai import (
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    NotFoundError,
    Omit,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    omit,
)
from openai.lib._pydantic import to_strict_json_schema
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import BaseModel, ValidationError

from ctg.config import REPO_ROOT, ROLE_SETTINGS, ModelSpec, Provider, ReasoningEffort, Role
from ctg.errors import BackendUnavailableError
from ctg.llm.base import (
    AssistantTurn,
    CallSink,
    LLMCallRecord,
    LLMOutputError,
    Message,
    SchemaT,
    SpendCapExceededError,
    ToolCall,
    ToolChoice,
    ToolSpec,
)
from ctg.llm.pricing import SpendLedger, has_prices, token_cost_usd

log = logging.getLogger(__name__)

ENV_FILE = REPO_ROOT / ".env"
BASE_URLS: dict[Provider, str] = {
    Provider.OPENAI: "https://api.openai.com/v1",
    Provider.OPENROUTER: "https://openrouter.ai/api/v1",
}
API_KEY_ENV: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.OPENROUTER: "OPENROUTER_API_KEY",
}
SPEND_CAP_ENV: dict[Provider, str] = {
    Provider.OPENAI: "CTG_SPEND_CAP_OPENAI_USD",
    Provider.OPENROUTER: "CTG_SPEND_CAP_OPENROUTER_USD",
}
OUTPUT_RETRIES = 2  # extra attempts after an unusable reply
SDK_MAX_RETRIES = 5  # transport errors and 429s, handled inside the SDK
TIMEOUT_S = 600.0

ResultT = TypeVar("ResultT")
ValueT = TypeVar("ValueT")


class _UnusableOutput(Exception):
    """A reply that cannot be used (refusal, invalid structure, truncation). The call is retried."""


def create_llm(run: str | None = None, run_cap_usd: float | None = None) -> OpenAIClient:
    """Build the client from `<repo>/.env` and the environment (existing variables win over `.env`)."""
    load_dotenv(ENV_FILE, override=False)
    return OpenAIClient.from_env(os.environ, ledger=SpendLedger(), run=run, run_cap_usd=run_cap_usd)


class OpenAIClient:
    """Views returned by `scoped` share the SDK clients, the ledger and the caps."""

    def __init__(
        self,
        *,
        api_keys: Mapping[Provider, str | None],
        ledger: SpendLedger,
        provider_caps_usd: Mapping[Provider, float] | None = None,
        run: str | None = None,
        run_cap_usd: float | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if run_cap_usd is not None and run is None:
            raise ValueError("run_cap_usd needs a run name to attribute spend to")
        self._api_keys = dict(api_keys)
        self._ledger = ledger
        self._provider_caps_usd = dict(provider_caps_usd or {})
        self._run = run
        self._run_cap_usd = run_cap_usd
        self._http_client = http_client
        self._clients: dict[Provider, AsyncOpenAI] = {}
        self._sinks: tuple[CallSink, ...] = ()

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str],
        *,
        ledger: SpendLedger,
        run: str | None = None,
        run_cap_usd: float | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> OpenAIClient:
        """Keys and cumulative caps from environment variables. A blank cap means no cap."""
        caps: dict[Provider, float] = {}
        for provider, name in SPEND_CAP_ENV.items():
            raw = environ.get(name, "").strip()
            if raw:
                try:
                    caps[provider] = float(raw)
                except ValueError as exc:
                    raise ValueError(f"{name} must be an amount in USD, got {raw!r}") from exc
        return cls(
            api_keys={provider: environ.get(name) for provider, name in API_KEY_ENV.items()},
            ledger=ledger,
            provider_caps_usd=caps,
            run=run,
            run_cap_usd=run_cap_usd,
            http_client=http_client,
        )

    def scoped(self, sink: CallSink) -> OpenAIClient:
        view = copy.copy(self)
        view._sinks = (*self._sinks, sink)
        return view

    def spent(self, provider: Provider | None = None, run: str | None = None) -> float:
        return self._ledger.total(provider, run)

    async def structured(
        self,
        role: Role,
        model: ModelSpec,
        messages: Sequence[Message],
        schema: type[SchemaT],
    ) -> SchemaT:
        def parse(completion: ChatCompletion) -> SchemaT:
            message = _usable_choice(completion).message
            if message.refusal:
                raise _UnusableOutput(f"refusal: {message.refusal}")
            if not message.content:
                raise _UnusableOutput("empty reply")
            try:
                return schema.model_validate_json(message.content)
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(map(str, error['loc'])) or '<root>'}: {error['msg']}"
                    for error in exc.errors(include_url=False, include_input=False)
                )
                raise _UnusableOutput(f"invalid {schema.__name__}: {problems}") from exc

        return await self._complete(role, model, messages, parse, response_format=_response_format(schema))

    async def chat(
        self,
        role: Role,
        model: ModelSpec,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        tool_choice: ToolChoice = "auto",
    ) -> AssistantTurn:
        tool_params = [_function_tool(tool) for tool in tools]
        return await self._complete(
            role, model, messages, _parse_turn, tools=tool_params, tool_choice=tool_choice
        )

    async def _complete(
        self,
        role: Role,
        model: ModelSpec,
        messages: Sequence[Message],
        parse: Callable[[ChatCompletion], ResultT],
        *,
        response_format: ResponseFormatJSONSchema | Omit = omit,
        tools: Sequence[ChatCompletionFunctionToolParam] = (),
        tool_choice: ToolChoice = "auto",
    ) -> ResultT:
        client = self._client(model.provider)
        if model.provider is Provider.OPENAI and not has_prices(model):
            raise ValueError(
                f"OpenAI model {model.name!r} needs prices in configs/models.yaml for the ledger"
            )
        decoding = _Decoding.of(role, model)
        failures: list[str] = []
        for _ in range(1 + OUTPUT_RETRIES):
            self._check_caps(model.provider)
            started_at = datetime.now(UTC)
            start = time.perf_counter()
            try:
                completion = await client.chat.completions.create(
                    model=model.name,
                    messages=messages,
                    response_format=response_format,
                    tools=list(tools) if tools else omit,
                    tool_choice=tool_choice if tools else omit,
                    parallel_tool_calls=False if tools else omit,
                    temperature=decoding.temperature,
                    reasoning_effort=decoding.reasoning_effort,
                    max_completion_tokens=decoding.max_completion_tokens,
                    max_tokens=decoding.max_tokens,
                    extra_body=decoding.extra_body,
                )
            except OpenAIError as exc:
                latency_s = time.perf_counter() - start
                unavailable = _backend_unavailable(exc, model, self._api_keys.get(model.provider))
                if unavailable is None:
                    self._record(role, model, None, started_at, latency_s, f"{exc!r}")
                    raise
                self._record(role, model, None, started_at, latency_s, f"{unavailable!r}")
                raise unavailable from exc
            latency_s = time.perf_counter() - start
            try:
                result = parse(completion)
            except _UnusableOutput as exc:
                failures.append(str(exc))
                self._record(role, model, completion.usage, started_at, latency_s, str(exc))
                log.warning("unusable %s output from %s: %s", role, model.name, exc)
                continue
            self._record(role, model, completion.usage, started_at, latency_s, None)
            return result
        raise LLMOutputError(
            f"{model.name} returned unusable {role} output {len(failures)} times: " + " | ".join(failures)
        )

    def _client(self, provider: Provider) -> AsyncOpenAI:
        client = self._clients.get(provider)
        if client is None:
            api_key = self._api_keys.get(provider)
            if not api_key:
                raise RuntimeError(f"{API_KEY_ENV[provider]} is not set")
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=BASE_URLS[provider],
                max_retries=SDK_MAX_RETRIES,
                timeout=TIMEOUT_S,
                # The SDK accepts `httpx` clients at runtime but annotates its own `httpx2` fork.
                http_client=cast(Any, self._http_client),
            )
            self._clients[provider] = client
        return client

    async def aclose(self) -> None:
        """Close the provider connections opened by this client and its scoped views."""
        clients = list(self._clients.values())
        self._clients.clear()
        for client in clients:
            await client.close()

    def _check_caps(self, provider: Provider) -> None:
        cap = self._provider_caps_usd.get(provider)
        if cap is not None:
            spent = self._ledger.total(provider=provider)
            if spent >= cap:
                raise SpendCapExceededError(
                    f"{provider} spend ${spent:.4f} has reached the {SPEND_CAP_ENV[provider]} cap ${cap:.4f}"
                )
        if self._run_cap_usd is not None:
            spent = self._ledger.total(run=self._run)
            if spent >= self._run_cap_usd:
                raise SpendCapExceededError(
                    f"run {self._run!r} spend ${spent:.4f} has reached its cap ${self._run_cap_usd:.4f}"
                )

    def _record(
        self,
        role: Role,
        model: ModelSpec,
        usage: CompletionUsage | None,
        started_at: datetime,
        latency_s: float,
        error: str | None,
    ) -> None:
        tokens = _Tokens.of(usage)
        record = LLMCallRecord(
            role=role,
            model=model.name,
            provider=model.provider,
            input_tokens=tokens.input,
            cached_input_tokens=tokens.cached_input,
            output_tokens=tokens.output,
            reasoning_tokens=tokens.reasoning,
            cost_usd=_cost_usd(model, usage, tokens) if usage is not None else 0.0,
            latency_s=latency_s,
            started_at=started_at,
            error=error,
        )
        self._ledger.append(record, self._run)
        for sink in self._sinks:
            sink.calls.append(record)


@dataclass(frozen=True)
class _Decoding:
    """Per-role decoding settings (Table 8) shaped for the model's provider."""

    temperature: float | Omit
    reasoning_effort: ReasoningEffort | Omit
    max_completion_tokens: int | Omit
    max_tokens: int | Omit
    extra_body: dict[str, Any] | None

    @classmethod
    def of(cls, role: Role, model: ModelSpec) -> _Decoding:
        settings = ROLE_SETTINGS[role]
        temperature = settings.temperature if model.supports_temperature else None
        if model.provider is Provider.OPENROUTER:
            extra_body: dict[str, Any] = {"usage": {"include": True}}
            if model.reasoning_effort is not None:
                extra_body["reasoning"] = {"effort": model.reasoning_effort}
            return cls(
                temperature=_given(temperature),
                reasoning_effort=omit,
                max_completion_tokens=omit,
                max_tokens=_given(settings.max_tokens),
                extra_body=extra_body,
            )
        return cls(
            temperature=_given(temperature),
            reasoning_effort=_given(model.reasoning_effort),
            max_completion_tokens=_given(settings.max_tokens),
            max_tokens=omit,
            extra_body=None,
        )


@dataclass(frozen=True)
class _Tokens:
    """Token counts of one reply. Counts the provider leaves null are 0."""

    input: int = 0
    cached_input: int = 0
    output: int = 0
    reasoning: int = 0

    @classmethod
    def of(cls, usage: CompletionUsage | None) -> _Tokens:
        if usage is None:
            return cls()
        prompt_details, completion_details = usage.prompt_tokens_details, usage.completion_tokens_details
        return cls(
            input=usage.prompt_tokens or 0,
            cached_input=(prompt_details.cached_tokens or 0) if prompt_details else 0,
            output=usage.completion_tokens or 0,
            reasoning=(completion_details.reasoning_tokens or 0) if completion_details else 0,
        )


def _given(value: ValueT | None) -> ValueT | Omit:
    return omit if value is None else value


def _cost_usd(model: ModelSpec, usage: CompletionUsage, tokens: _Tokens) -> float:
    """OpenRouter reports the exact cost; OpenAI calls are priced from the preset."""
    if model.provider is Provider.OPENROUTER:
        reported = (usage.model_extra or {}).get("cost")
        if isinstance(reported, int | float):
            return float(reported)
    if has_prices(model):
        return token_cost_usd(model, tokens.input, tokens.cached_input, tokens.output)
    log.warning("no cost reported for %s and the preset has no prices; recording $0", model.name)
    return 0.0


def _backend_unavailable(
    exc: OpenAIError, model: ModelSpec, api_key: str | None
) -> BackendUnavailableError | None:
    """The run-level error for a response meaning that no call of this run can succeed, else `None`.

    That is a rejected key (401), an unknown model (404), an OpenAI permission refusal (403), an
    exhausted OpenAI quota (429 `insufficient_quota`) or no credits left (402). OpenRouter answers 403
    when moderation or a guardrail blocks one request, and 402 with `limit_source`
    `openrouter_in_flight_budget` while running requests hold its in-flight budget; both stay request
    errors. The message never contains the API key.
    """
    if not isinstance(exc, APIStatusError):
        return None
    body = _error_body(exc)
    if isinstance(exc, AuthenticationError | NotFoundError):
        unavailable = True
    elif isinstance(exc, PermissionDeniedError):
        unavailable = model.provider is Provider.OPENAI
    elif isinstance(exc, RateLimitError):
        unavailable = exc.code == "insufficient_quota"
    else:
        metadata = body.get("metadata")
        limit_source = (
            cast(dict[str, object], metadata).get("limit_source") if isinstance(metadata, dict) else None
        )
        unavailable = exc.status_code == 402 and limit_source != "openrouter_in_flight_budget"
    if not unavailable:
        return None
    message = body.get("message")
    detail = message if isinstance(message, str) and message else exc.message
    if api_key:
        detail = detail.replace(api_key, "***")
    return BackendUnavailableError(
        f"{model.provider} cannot serve this run with {model.name} (HTTP {exc.status_code}): {detail}"
    )


def _error_body(exc: APIStatusError) -> dict[str, object]:
    """The provider's error object (the SDK unwraps the top-level `error` key); empty if not JSON."""
    return cast(dict[str, object], exc.body) if isinstance(exc.body, dict) else {}


def _response_format(schema: type[BaseModel]) -> ResponseFormatJSONSchema:
    return {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": to_strict_json_schema(schema), "strict": True},
    }


def _function_tool(tool: ToolSpec) -> ChatCompletionFunctionToolParam:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": True,
        },
    }


def _usable_choice(completion: ChatCompletion) -> Choice:
    if not completion.choices:
        raise _UnusableOutput("response has no choices")
    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise _UnusableOutput("reply truncated at the output token cap (finish_reason=length)")
    return choice


def _parse_turn(completion: ChatCompletion) -> AssistantTurn:
    message = _usable_choice(completion).message
    calls: list[ToolCall] = []
    call_params: list[ChatCompletionMessageFunctionToolCallParam] = []
    for tool_call in message.tool_calls or ():
        if tool_call.type != "function":
            raise _UnusableOutput(f"unsupported tool call type {tool_call.type!r}")
        function = tool_call.function
        # A tool without parameters may be called with an empty arguments string.
        raw_arguments = function.arguments or "{}"
        try:
            arguments: object = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise _UnusableOutput(f"invalid JSON arguments for {function.name}: {exc}") from exc
        if not isinstance(arguments, dict):
            raise _UnusableOutput(f"arguments for {function.name} are not a JSON object")
        calls.append(ToolCall(id=tool_call.id, name=function.name, arguments=cast(dict[str, Any], arguments)))
        call_params.append(
            {
                "id": tool_call.id,
                "type": "function",
                "function": {"name": function.name, "arguments": raw_arguments},
            }
        )
    if not calls and not message.content:
        raise _UnusableOutput("empty reply (no content and no tool calls)")
    history: ChatCompletionAssistantMessageParam = {"role": "assistant", "content": message.content}
    if call_params:
        history["tool_calls"] = call_params
    return AssistantTurn(content=message.content, tool_calls=tuple(calls), message=history)
