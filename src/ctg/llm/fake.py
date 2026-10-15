"""A scripted stand-in for the LLM interface, for offline tests.

The handler sees every request and returns either a schema instance (for `structured`) or an
`AssistantTurn` (for `chat`). Use `tool_turn` and `text_turn` to build chat replies.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast

from pydantic import BaseModel

from ctg.config import ModelSpec, Role
from ctg.llm.base import (
    AssistantTurn,
    CallSink,
    LLMCallRecord,
    Message,
    SchemaT,
    ToolCall,
    ToolChoice,
    ToolSpec,
)


@dataclass(frozen=True)
class FakeRequest:
    kind: Literal["structured", "chat"]
    role: Role
    model: ModelSpec
    messages: tuple[Message, ...]
    schema: type[BaseModel] | None = None
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: ToolChoice = "auto"

    @property
    def text(self) -> str:
        """All message contents joined, for handlers that branch on prompt text."""
        parts: list[str] = []
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
        return "\n".join(parts)


Handler = Callable[[FakeRequest], BaseModel | AssistantTurn]


def tool_turn(name: str, arguments: dict[str, Any], content: str | None = None) -> AssistantTurn:
    call = ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=arguments)
    message = cast(
        Message,
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
    )
    return AssistantTurn(content=content, tool_calls=(call,), message=message)


def text_turn(content: str) -> AssistantTurn:
    return AssistantTurn(
        content=content, tool_calls=(), message=cast(Message, {"role": "assistant", "content": content})
    )


@dataclass
class FakeLLM:
    handler: Handler
    requests: list[FakeRequest] = field(default_factory=list[FakeRequest])
    sinks: tuple[CallSink, ...] = ()

    def _record(self, request: FakeRequest) -> None:
        self.requests.append(request)
        record = LLMCallRecord(
            role=request.role,
            model=request.model.name,
            provider=request.model.provider,
            input_tokens=sum(len(str(m.get("content") or "")) for m in request.messages) // 4,
            output_tokens=10,
            started_at=datetime.now(UTC),
        )
        for sink in self.sinks:
            sink.calls.append(record)

    async def structured(
        self,
        role: Role,
        model: ModelSpec,
        messages: Sequence[Message],
        schema: type[SchemaT],
    ) -> SchemaT:
        request = FakeRequest("structured", role, model, tuple(messages), schema=schema)
        self._record(request)
        result = self.handler(request)
        if not isinstance(result, schema):
            # Round-trip through the schema so handlers may return compatible models or dicts.
            return schema.model_validate(result.model_dump() if isinstance(result, BaseModel) else result)
        return result

    async def chat(
        self,
        role: Role,
        model: ModelSpec,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        tool_choice: ToolChoice = "auto",
    ) -> AssistantTurn:
        request = FakeRequest(
            "chat", role, model, tuple(messages), tools=tuple(tools), tool_choice=tool_choice
        )
        self._record(request)
        result = self.handler(request)
        if not isinstance(result, AssistantTurn):
            raise TypeError(f"chat handler must return AssistantTurn, got {type(result).__name__}")
        return result

    def scoped(self, sink: CallSink) -> FakeLLM:
        view = FakeLLM(self.handler, self.requests, (*self.sinks, sink))
        return view

    async def aclose(self) -> None:
        return None
