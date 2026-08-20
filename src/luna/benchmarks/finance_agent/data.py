"""Packaged Finance Agent case loading and strict rubric parsing."""

from __future__ import annotations

import ast
import csv
import json
from importlib import resources
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RubricCriterion(BaseModel):
    model_config = ConfigDict(frozen=True)

    operator: str
    criteria: str

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, value: str) -> str:
        if value not in {"correctness", "contradiction"}:
            raise ValueError(f"unsupported rubric operator: {value}")
        return value


class FinanceCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int
    question: str
    answer: str
    question_type: str
    expert_time_mins: float
    rubric: list[RubricCriterion] = Field(default_factory=list)


def load_cases() -> list[FinanceCase]:
    resource = resources.files("luna.benchmarks.finance_agent") / "data" / "public.csv"
    with resources.as_file(resource) as path, path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        FinanceCase(
            seed=index,
            question=row["Question"],
            answer=row["Answer"],
            question_type=row["Question Type"],
            expert_time_mins=float(row["Expert time (mins)"] or 0),
            rubric=parse_rubric(row.get("Rubric", "[]")),
        )
        for index, row in enumerate(rows)
    ]


def parse_rubric(value: str) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
            if not isinstance(parsed, list):
                raise ValueError("rubric must be a list")
            return parsed
        except (ValueError, SyntaxError, json.JSONDecodeError) as exc:
            last_error = exc
    raise ValueError(f"invalid Finance Agent rubric: {last_error}")

