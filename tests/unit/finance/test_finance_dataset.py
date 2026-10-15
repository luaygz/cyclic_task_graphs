from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from ctg.envs.finance.dataset import get_question, load_questions, parse_rubric


def test_all_fifty_rubrics_parse_in_row_order() -> None:
    questions = load_questions()

    assert [q.id for q in questions] == [f"q{i:02d}" for i in range(50)]
    operators = Counter(c.operator for q in questions for c in q.rubric)
    assert operators == {"correctness": 191, "contradiction": 50}
    for question in questions:
        assert question.question and question.answer and question.question_type
        assert any(c.operator == "correctness" for c in question.rubric), question.id


def test_rubrics_with_apostrophes_keep_their_text() -> None:
    texts = {q.id: " ".join(c.text for c in q.rubric) for q in load_questions()}

    assert "AMD's guidance range" in texts["q03"]
    assert "Microsoft's full-time employees" in texts["q25"]


def test_parse_rubric_accepts_double_quoted_strings() -> None:
    rubric = parse_rubric(
        "[{'operator': 'correctness', 'criteria': \"Salesforce's debt is $7.5 billion\"}, "
        "{'operator': 'contradiction', 'criteria': '$7.5 billion'}]"
    )

    assert [(c.operator, c.text) for c in rubric] == [
        ("correctness", "Salesforce's debt is $7.5 billion"),
        ("contradiction", "$7.5 billion"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "[{'operator': 'correctness', 'criteria': 'unterminated}]",
        "{'operator': 'correctness', 'criteria': 'not a list'}",
        "[]",
        "['just a string']",
        "[{'operator': 'similarity', 'criteria': 'x'}]",
        "[{'operator': 'correctness'}]",
        "[{'operator': 'correctness', 'criteria': 'x', 'weight': 2}]",
        "[{'operator': 'correctness', 'criteria': '   '}]",
    ],
)
def test_parse_rubric_rejects_malformed_rubrics(text: str) -> None:
    with pytest.raises(ValueError):
        parse_rubric(text)


def test_load_questions_names_the_row_with_a_bad_rubric(tmp_path: Path) -> None:
    path = tmp_path / "public.csv"
    path.write_text(
        "Question,Answer,Question Type,Expert time (mins),Rubric\n"
        "Q1?,A1,Trends,5,\"[{'operator': 'correctness', 'criteria': 'A1'}]\"\n"
        "Q2?,A2,Trends,5,\"[{'operator': 'bogus', 'criteria': 'A2'}]\"\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="q01"):
        load_questions(path)


def test_get_question_rejects_unknown_ids() -> None:
    assert get_question("q49").id == "q49"
    with pytest.raises(KeyError):
        get_question("q50")
