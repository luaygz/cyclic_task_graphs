"""The interface every LLM caller in the package depends on.

`ctg.llm.client.OpenAIClient` implements it against OpenAI and OpenRouter; `ctg.llm.fake.FakeLLM`
implements it for offline tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol, TypeVar

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict

from ctg.config import ModelSpec, Provider, Role

Message = ChatCompletionMessageParam
SchemaT = TypeVar("SchemaT", bound=BaseModel)
ToolChoice = Literal["auto", "required", "none"]


@dataclass(frozen=True)
class ToolSpec:
    """A function tool offered to a model. `parameters` is a strict-mode JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AssistantTurn:
    """One model reply. `message` is ready to append to the conversation."""

    content: str | None
    tool_calls: tuple[ToolCall, ...]
    message: Message


class LLMCallRecord(BaseModel):
    """Accounting for one model call (successful or not)."""

    model_config = ConfigDict(frozen=True)

    role: Role
    model: str
    provider: Provider
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    started_at: datetime
    error: str | None = None


class SpendCapExceededError(RuntimeError):
    """Raised before a call that would run past a cumulative or per-run spend cap."""


class LLMOutputError(RuntimeError):
    """The model kept returning unusable output (invalid structure, truncation) after retries."""


@dataclass
class CallSink:
    """Collects the calls made on behalf of one episode or one planning step."""

    calls: list[LLMCallRecord] = field(default_factory=list[LLMCallRecord])


class LLM(Protocol):
    async def structured(
        self,
        role: Role,
        model: ModelSpec,
        messages: Sequence[Message],
        schema: type[SchemaT],
    ) -> SchemaT:
        """Return a validated instance of `schema` (strict JSON-schema structured output)."""
        ...

    async def chat(
        self,
        role: Role,
        model: ModelSpec,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        tool_choice: ToolChoice = "auto",
    ) -> AssistantTurn:
        """One chat completion, optionally with tools. Parallel tool calls are disabled."""
        ...

    def scoped(self, sink: CallSink) -> LLM:
        """A view that shares connections and spend accounting but also records calls into `sink`."""
        ...
