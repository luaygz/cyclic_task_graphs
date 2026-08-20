"""Provider-neutral decision model boundary and optional OpenAI adapter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Protocol

from luna.actor.schemas import Decision, DecisionRequest, RubricEvaluation, Usage


class DecisionModel(Protocol):
    async def decide(self, request: DecisionRequest) -> Decision: ...


class ScriptedModel:
    """Side-effect-free model used by tests and programmatic smoke checks."""

    def __init__(self, callback: Callable[[DecisionRequest], Decision | Awaitable[Decision]]):
        self.callback = callback

    async def decide(self, request: DecisionRequest) -> Decision:
        result = self.callback(request)
        return await result if isinstance(result, Awaitable) else result


class OpenAIResponsesModel:
    """Structured decision adapter using the OpenAI Responses API.

    The optional dependency and API key are resolved only when this class is
    instantiated; importing the benchmark package never contacts a provider.
    """

    def __init__(self, model: str, reasoning_effort: str | None = None, judge_model: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install the live adapter with: pip install '.[openai]'") from exc
        self.client = OpenAI()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.judge_model = judge_model or model

    async def decide(self, request: DecisionRequest) -> Decision:
        response = await asyncio.to_thread(self._request, request)
        parsed: Decision = response.output_parsed
        return parsed.model_copy(update={"usage": _response_usage(response)})

    async def judge_finance(
        self,
        question: str,
        expected_answer: str,
        rubric: list[dict],
        answer: str,
    ) -> RubricEvaluation:
        response = await asyncio.to_thread(
            self._judge_request, question, expected_answer, rubric, answer
        )
        parsed: RubricEvaluation = response.output_parsed
        if len(parsed.criteria_evaluations) != len(rubric):
            raise ValueError("Finance judge did not evaluate every rubric criterion")
        return parsed.model_copy(update={"usage": _response_usage(response)})

    def _request(self, request: DecisionRequest):
        system = (
            "You are an experiment executor. Choose one action that follows an admissible "
            "action or action grammar. Preserve exact object names when actions are enumerated. "
            "Set done=true only when a final answer is required or the task is already complete. "
            "Never invent a tool or command outside admissible_actions."
        )
        kwargs = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": request.model_dump_json()},
            ],
            "text_format": Decision,
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        return self.client.responses.parse(**kwargs)

    def _judge_request(
        self,
        question: str,
        expected_answer: str,
        rubric: list[dict],
        answer: str,
    ):
        system = (
            "You are an expert financial analyst grading an answer. Evaluate every rubric item "
            "in its given order. A correctness item passes when the answer contains, implies, or "
            "is consistent with it. A contradiction item passes only when the answer does not "
            "contradict it. overall_passed is true only when every item passes."
        )
        payload = {
            "question": question,
            "expected_answer": expected_answer,
            "agent_answer": answer,
            "rubric": rubric,
        }
        kwargs = {
            "model": self.judge_model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "text_format": RubricEvaluation,
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        return self.client.responses.parse(**kwargs)


def _response_usage(response) -> Usage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage()
    return Usage(
        input_tokens=getattr(usage, "input_tokens", 0),
        output_tokens=getattr(usage, "output_tokens", 0),
        total_tokens=getattr(usage, "total_tokens", 0),
    )
