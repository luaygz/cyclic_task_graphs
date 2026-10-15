"""The Vals.ai Finance-Agent public validation set: 50 questions with expert answers and grading rubrics."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from datetime import date
from functools import cache
from pathlib import Path
from typing import Literal, cast, get_args

from ctg.config import DATA_DIR

DATA_FILE = DATA_DIR / "finance_agent" / "public.csv"

REFERENCE_DATE = date(2025, 4, 7)
"""The benchmark's "today": questions are posed as of this date and EDGAR searches stop here."""

Operator = Literal["correctness", "contradiction"]
"""`correctness`: the answer must state the criterion. `contradiction`: the answer must not contradict it."""


@dataclass(frozen=True)
class Criterion:
    operator: Operator
    text: str


@dataclass(frozen=True)
class FinanceQuestion:
    id: str
    question: str
    answer: str
    """The expert answer."""
    question_type: str
    rubric: tuple[Criterion, ...]


def parse_rubric(text: str) -> tuple[Criterion, ...]:
    """Parse the CSV's Rubric column, a Python-literal list of {'operator': ..., 'criteria': ...} dicts."""
    try:
        raw: object = ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"rubric is not a Python literal: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("rubric must be a non-empty list")
    criteria: list[Criterion] = []
    for item in cast(list[object], raw):
        if not isinstance(item, dict):
            raise ValueError(f"rubric item is not a dict: {item!r}")
        fields = cast(dict[object, object], item)
        if set(fields) != {"operator", "criteria"}:
            raise ValueError(f"rubric item must have exactly 'operator' and 'criteria': {fields!r}")
        operator, criterion = fields["operator"], fields["criteria"]
        if operator not in get_args(Operator):
            raise ValueError(f"unknown rubric operator {operator!r}")
        if not isinstance(criterion, str) or not criterion.strip():
            raise ValueError(f"rubric criterion must be a non-empty string: {criterion!r}")
        criteria.append(Criterion(cast(Operator, operator), criterion.strip()))
    return tuple(criteria)


@cache
def load_questions(path: Path = DATA_FILE) -> tuple[FinanceQuestion, ...]:
    """All questions in CSV row order, with ids q00, q01, ..."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    questions: list[FinanceQuestion] = []
    for index, row in enumerate(rows):
        question_id = f"q{index:02d}"
        try:
            rubric = parse_rubric(row["Rubric"])
        except ValueError as exc:
            raise ValueError(f"{path.name} row {question_id}: {exc}") from exc
        questions.append(
            FinanceQuestion(
                id=question_id,
                question=row["Question"].strip(),
                answer=row["Answer"].strip(),
                question_type=row["Question Type"].strip(),
                rubric=rubric,
            )
        )
    return tuple(questions)


def get_question(question_id: str, path: Path = DATA_FILE) -> FinanceQuestion:
    for question in load_questions(path):
        if question.id == question_id:
            return question
    raise KeyError(f"no Finance-Agent question with id {question_id!r}")
