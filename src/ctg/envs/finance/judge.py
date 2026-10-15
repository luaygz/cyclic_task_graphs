"""LLM judge for Finance-Agent answers.

The judge model returns one verdict per rubric criterion. Whether the answer passes is decided here, in
code: every correctness criterion must be stated and no contradiction criterion may be contradicted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, cast

from pydantic import BaseModel, Field, create_model

from ctg.config import ModelSpec, Role
from ctg.envs.finance.dataset import Criterion, FinanceQuestion
from ctg.envs.finance.tools import Source
from ctg.llm.base import LLM, LLMOutputError
from ctg.prompts import render

_QUESTIONS = {
    "correctness": "Does the answer state this, or something equivalent to it?",
    "contradiction": "Does the answer contradict this reference statement?",
}


class JudgeError(RuntimeError):
    """The judge model kept returning unusable verdicts. Judging is evaluation, not the method under test,
    so this is an infrastructure failure: the runner records the episode as an error and re-runs it on
    resume."""


class CriterionVerdict(BaseModel):
    reasoning: str = Field(description="Why the verdict holds, citing the relevant part of the answer.")
    verdict: bool = Field(description="The yes/no answer to the criterion's question.")


@cache
def verdict_schema(criteria: int) -> type[BaseModel]:
    """Structured-output schema with one required field `criterion_<n>` per rubric criterion."""
    fields: dict[str, Any] = {f"criterion_{n}": (CriterionVerdict, ...) for n in range(1, criteria + 1)}
    return create_model(f"RubricVerdicts{criteria}", **fields)


@dataclass(frozen=True)
class Verdict:
    criterion: Criterion
    reasoning: str
    verdict: bool
    """The judge's answer to the criterion's question (see `_QUESTIONS`)."""

    @property
    def satisfied(self) -> bool:
        return self.verdict if self.criterion.operator == "correctness" else not self.verdict

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": self.criterion.operator,
            "criterion": self.criterion.text,
            "reasoning": self.reasoning,
            "verdict": self.verdict,
            "satisfied": self.satisfied,
        }


@dataclass(frozen=True)
class Judgement:
    passed: bool
    verdicts: tuple[Verdict, ...]


def aggregate(verdicts: Sequence[Verdict]) -> bool:
    """Pass iff there is at least one verdict and every criterion is satisfied."""
    return bool(verdicts) and all(verdict.satisfied for verdict in verdicts)


async def judge_answer(
    llm: LLM, model: ModelSpec, question: FinanceQuestion, answer: str, sources: Sequence[Source]
) -> Judgement:
    """Grade `answer` against the question's rubric. An empty answer fails without a model call.

    Raises JudgeError when the judge's output stays unusable after the client's retries.
    """
    if not answer.strip():
        return Judgement(passed=False, verdicts=())
    prompt = render(
        "finance_judge",
        question=question.question,
        expected_answer=question.answer,
        answer=answer,
        sources=_format_sources(sources),
        criteria=_format_criteria(question.rubric),
    )
    schema = verdict_schema(len(question.rubric))
    try:
        result = await llm.structured(Role.JUDGE, model, [{"role": "user", "content": prompt}], schema)
    except LLMOutputError as exc:
        raise JudgeError(f"the Finance-Agent judge {model.name} gave no usable verdicts: {exc}") from exc
    verdicts: list[Verdict] = []
    for number, criterion in enumerate(question.rubric, 1):
        item = cast(CriterionVerdict, getattr(result, f"criterion_{number}"))
        verdicts.append(Verdict(criterion, item.reasoning, item.verdict))
    return Judgement(passed=aggregate(verdicts), verdicts=tuple(verdicts))


def _format_sources(sources: Sequence[Source]) -> str:
    if not sources:
        return "(none)"
    return "\n".join(f"- {source.name}: {source.url}" for source in sources)


def _format_criteria(rubric: Sequence[Criterion]) -> str:
    blocks = [
        f"criterion_{number} ({criterion.operator}): {_QUESTIONS[criterion.operator]}\n"
        f"<criterion>\n{criterion.text}\n</criterion>"
        for number, criterion in enumerate(rubric, 1)
    ]
    return "\n\n".join(blocks)
