"""Public Finance Agent environment with dependency-injected tools."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from luna.actor.schemas import RubricEvaluation, Usage
from luna.benchmarks.base import BenchmarkAdapter, StepResult
from luna.benchmarks.finance_agent.data import FinanceCase, load_cases
from luna.benchmarks.finance_agent.tools import FinanceToolClient


class FinanceAgentAdapter(BenchmarkAdapter):
    benchmark_name = "finance_agent"

    def __init__(
        self,
        tools: FinanceToolClient | None = None,
        judge: Callable[[str, str, list[dict], str], Awaitable[RubricEvaluation]] | None = None,
    ):
        self.tools = tools or FinanceToolClient()
        self.judge = judge
        self.case: FinanceCase | None = None
        self.actions: list[dict] = []
        self.final_answer: str | None = None
        self.evaluation: RubricEvaluation | None = None
        self._success = False

    def set_judge(self, judge) -> None:
        self.judge = judge

    async def initialize(
        self, seed: int, index: int, depth: int | None, output_dir: Path | None = None
    ) -> None:
        cases = load_cases()
        if seed < 0 or seed >= len(cases):
            raise ValueError(f"Finance Agent seed must be in 0..{len(cases) - 1}")
        self.case = cases[seed]
        self.actions, self.final_answer, self.evaluation, self._success = [], None, None, False
        self.tools.data_storage.clear()

    def task_prompt(self) -> str:
        return self._require_case().question

    def state(self) -> dict:
        case = self._require_case()
        return {
            "question": case.question,
            "question_type": case.question_type,
            "data_keys": sorted(self.tools.data_storage),
            "steps": len(self.actions),
            "final_answer_submitted": self.final_answer is not None,
            "evaluation": self.evaluation.model_dump(mode="json") if self.evaluation else None,
        }

    def admissible_actions(self) -> list[str]:
        return [
            'web_search({"search_query": "..."})',
            'edgar_search({"query": "...", "form_types": [], "ciks": []})',
            'parse_html_page({"url": "https://...", "key": "document_name"})',
            'retrieve_information({"prompt": "Analyze {{document_name}}"})',
            'final_answer({"answer": "...", "sources": []})',
        ]

    async def step(self, action: str) -> StepResult:
        name, arguments = parse_tool_action(action)
        if name == "web_search":
            result = await self.tools.web_search(arguments.get("search_query", ""))
        elif name == "edgar_search":
            result = await self.tools.edgar_search(**arguments)
        elif name == "parse_html_page":
            result = await self.tools.parse_html_page(arguments.get("url", ""), arguments.get("key", ""))
        elif name == "retrieve_information":
            result = await self.tools.retrieve_information(arguments.get("prompt", ""))
        elif name == "final_answer":
            self.final_answer = arguments.get("answer", "")
            self._success = await self.finalize(self.final_answer)
            result = "Final answer submitted."
        else:
            raise ValueError(f"unknown Finance Agent tool: {name}")
        self.actions.append({"tool": name, "arguments": arguments, "result": result})
        return StepResult(
            observation=result,
            success=self._success,
            terminal=name == "final_answer",
            metadata={"tool": name},
        )

    async def finalize(self, answer: str | None) -> bool:
        if not answer:
            self._success = False
            return False
        case = self._require_case()
        if self.judge is not None:
            try:
                rubric = [criterion.model_dump() for criterion in case.rubric]
                self.evaluation = await self.judge(case.question, case.answer, rubric, answer)
                self._success = self.evaluation.overall_passed
                return self._success
            except Exception as exc:
                self.evaluation = RubricEvaluation(
                    overall_reasoning=f"Judge failed: {type(exc).__name__}: {exc}",
                    criteria_evaluations=[],
                    overall_passed=False,
                )
                self._success = False
                return False
        expected = normalize_text(case.answer)
        supplied = normalize_text(answer)
        # Exact/contained matching is used only by injected-model smoke tests.
        self._success = expected == supplied or expected in supplied
        return self._success

    @property
    def evaluation_usage(self) -> Usage:
        return self.evaluation.usage if self.evaluation else Usage()

    @property
    def success(self) -> bool:
        return self._success

    def _require_case(self) -> FinanceCase:
        if self.case is None:
            raise RuntimeError("Finance Agent adapter is not initialized")
        return self.case


def parse_tool_action(action: str) -> tuple[str, dict]:
    match = re.fullmatch(r"([a-z_]+)\((.*)\)", action.strip(), flags=re.DOTALL)
    if match is None:
        raise ValueError("Finance Agent actions must be tool_name({...})")
    arguments = json.loads(match.group(2))
    if not isinstance(arguments, dict):
        raise ValueError("Finance Agent tool arguments must be a JSON object")
    return match.group(1), arguments


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9.$%]+", " ", value.lower()).split())

