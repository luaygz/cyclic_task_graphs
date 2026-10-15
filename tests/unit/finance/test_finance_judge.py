from __future__ import annotations

import pytest
from pydantic import BaseModel

from ctg.config import ModelSpec, Role
from ctg.envs.finance.dataset import Criterion, FinanceQuestion
from ctg.envs.finance.judge import JudgeError, aggregate, judge_answer
from ctg.envs.finance.tools import Source
from ctg.errors import BackendUnavailableError
from ctg.llm.base import LLMOutputError, SpendCapExceededError
from ctg.llm.fake import FakeLLM, FakeRequest

JUDGE_MODEL = ModelSpec(name="judge-model")
QUESTION = FinanceQuestion(
    id="q99",
    question="What were Acme's FY2024 revenue and net income?",
    answer="Revenue $5.0 billion; net income $700 million.",
    question_type="Quantitative Retrieval",
    rubric=(
        Criterion("correctness", "Revenue of $5.0 billion"),
        Criterion("correctness", "Net income of $700 million"),
        Criterion("contradiction", "Revenue $5.0 billion; net income $700 million."),
    ),
)


def judge_llm(*verdicts: bool) -> FakeLLM:
    def handler(request: FakeRequest) -> BaseModel:
        assert request.schema is not None
        fields = {
            f"criterion_{n}": {"reasoning": f"reason {n}", "verdict": verdict}
            for n, verdict in enumerate(verdicts, 1)
        }
        return request.schema.model_validate(fields)

    return FakeLLM(handler)


@pytest.mark.parametrize(
    ("verdicts", "passed"),
    [
        ((True, True, False), True),
        ((True, False, False), False),
        ((False, False, False), False),
        ((True, True, True), False),
        ((False, False, True), False),
    ],
    ids=["all-met", "one-correctness-missing", "nothing-met", "contradiction-only", "everything-wrong"],
)
async def test_pass_needs_every_correctness_and_no_contradiction(
    verdicts: tuple[bool, bool, bool], passed: bool
) -> None:
    llm = judge_llm(*verdicts)

    judgement = await judge_answer(llm, JUDGE_MODEL, QUESTION, "Revenue was $5.0B.", [])

    assert judgement.passed is passed
    assert [v.verdict for v in judgement.verdicts] == list(verdicts)
    assert [v.satisfied for v in judgement.verdicts] == [verdicts[0], verdicts[1], not verdicts[2]]
    assert [v.reasoning for v in judgement.verdicts] == ["reason 1", "reason 2", "reason 3"]
    [request] = llm.requests
    assert (request.kind, request.role, request.model) == ("structured", Role.JUDGE, JUDGE_MODEL)
    assert request.schema is not None
    assert list(request.schema.model_fields) == ["criterion_1", "criterion_2", "criterion_3"]


@pytest.mark.parametrize("answer", ["", "   \n\t"])
async def test_empty_answer_fails_without_a_judge_call(answer: str) -> None:
    llm = judge_llm(True, True, False)

    judgement = await judge_answer(llm, JUDGE_MODEL, QUESTION, answer, [])

    assert not judgement.passed
    assert judgement.verdicts == ()
    assert llm.requests == []


async def test_judge_sees_question_expected_answer_answer_sources_and_criteria() -> None:
    llm = judge_llm(True, True, False)
    sources = [Source(url="https://www.sec.gov/acme-10k.htm", name="Acme FY2024 10-K")]

    await judge_answer(llm, JUDGE_MODEL, QUESTION, "Acme earned $5.0B on $700M net income.", sources)

    text = llm.requests[0].text
    for expected in (
        QUESTION.question,
        QUESTION.answer,
        "Acme earned $5.0B on $700M net income.",
        "Acme FY2024 10-K: https://www.sec.gov/acme-10k.htm",
        "criterion_1 (correctness)",
        "criterion_3 (contradiction)",
        *(criterion.text for criterion in QUESTION.rubric),
    ):
        assert expected in text


def test_no_verdicts_never_pass() -> None:
    assert aggregate([]) is False


def failing_llm(failure: Exception) -> FakeLLM:
    def handler(request: FakeRequest) -> BaseModel:
        raise failure

    return FakeLLM(handler)


async def test_unusable_judge_output_is_an_evaluation_failure_not_an_output_failure() -> None:
    unusable = LLMOutputError("judge-model returned unusable judge output 3 times: refusal: no")

    with pytest.raises(JudgeError, match="judge judge-model gave no usable verdicts") as caught:
        await judge_answer(failing_llm(unusable), JUDGE_MODEL, QUESTION, "Revenue was $5.0B.", [])

    assert not isinstance(caught.value, LLMOutputError)
    assert caught.value.__cause__ is unusable


@pytest.mark.parametrize(
    "failure",
    [SpendCapExceededError("run spend reached its cap"), BackendUnavailableError("OpenAI rejected the key")],
)
async def test_judge_spend_and_backend_failures_propagate_unchanged(failure: Exception) -> None:
    with pytest.raises(type(failure)) as caught:
        await judge_answer(failing_llm(failure), JUDGE_MODEL, QUESTION, "Revenue was $5.0B.", [])

    assert caught.value is failure
