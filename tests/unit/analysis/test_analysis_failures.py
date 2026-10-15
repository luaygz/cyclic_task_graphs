from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from analysis_fixtures import (
    config,
    episode,
    error_episode,
    interrupted_segment,
    make_run,
    output_failure,
    path_segments,
    planner_failure,
    segment,
    tool_call,
    transition,
)

from ctg.analysis.failures import FailureLabel, finance_diagnostics, label_failure, unproductive_revisits
from ctg.analysis.report import finance_tables, to_markdown
from ctg.config import BenchmarkName, Controller
from ctg.runs.records import EndReason, EpisodeRecord, SegmentRecord, TransitionType

FINANCE = BenchmarkName.FINANCE_AGENT
REACT = config(name="react", controller=Controller.REACT, benchmark=FINANCE)
GEN_CYC = config(name="gen", controller=Controller.GEN_CYC, benchmark=FINANCE)
SPEC_CYC = config(name="spec", controller=Controller.SPEC_CYC, benchmark=FINANCE)
DEPDAG = config(name="dag", controller=Controller.DEPDAG, benchmark=FINANCE)

SEARCH = "finance_agent_google_web_search"
PARSE = "finance_agent_parse_html_page"
RETRIEVE = "finance_agent_retrieve_information"


def details(
    answer: str | None = None,
    sources: int = 0,
    documents: tuple[str, ...] = (),
    satisfied: tuple[bool, ...] = (),
    contradicted: bool | None = None,
) -> dict[str, Any]:
    """`satisfied`: the correctness verdicts. `contradicted`: adds the rubric's contradiction criterion."""
    verdicts = [
        {"operator": "correctness", "criterion": "c", "reasoning": "r", "verdict": s, "satisfied": s}
        for s in satisfied
    ]
    if contradicted is not None:
        verdicts.append(
            {
                "operator": "contradiction",
                "criterion": "c",
                "reasoning": "r",
                "verdict": contradicted,
                "satisfied": not contradicted,
            }
        )
    return {
        "answer": answer,
        "sources": [{"url": f"https://example.com/{i}", "name": f"s{i}"} for i in range(sources)],
        "passed": all(v["satisfied"] for v in verdicts) and bool(verdicts),
        "verdicts": verdicts,
        "documents": list(documents),
    }


def react_episode(
    instance: str,
    end_reason: EndReason,
    env: dict[str, Any],
    calls: Sequence[tuple[str, str]] = (),
    won: bool = False,
    tool_calls: int | None = None,
) -> EpisodeRecord:
    steps = [tool_call(tool, observation, k=i + 1) for i, (tool, observation) in enumerate(calls)]
    return episode(
        REACT,
        instance,
        won=won,
        end_reason=end_reason,
        tool_calls=tool_calls if tool_calls is not None else len(steps),
        react_steps=steps,
        env_details=env,
    )


def loop_segments(progress: tuple[bool, ...]) -> list[SegmentRecord]:
    """Segments alternating A, B, A, B, ... with the given router progress flags, then a final segment."""
    nodes = ["A", "B"] * len(progress)
    segments = [
        segment(i, nodes[i], transition(nodes[i], nodes[i + 1], TransitionType.CROSS), progress=flag)
        for i, flag in enumerate(progress)
    ]
    return [*segments, segment(len(progress), nodes[len(progress)], progress=None)]


def test_budget_exhaustion_without_final_answer() -> None:
    record = react_episode("q", EndReason.BUDGET, details(), [(SEARCH, "results")] * 3)
    assert label_failure(record) is FailureLabel.BUDGET_EXHAUSTION


def test_routing_loop_needs_two_unproductive_revisits() -> None:
    looping = episode(GEN_CYC, "q", segments=loop_segments((True, True, False, False)), env_details=details())
    assert unproductive_revisits(looping) == 2
    assert label_failure(looping) is FailureLabel.ROUTING_LOOP

    # the first visits of A and B never count; one unproductive revisit is not a loop
    once = episode(GEN_CYC, "q", segments=loop_segments((False, False, False)), env_details=details())
    assert unproductive_revisits(once) == 1
    assert label_failure(once) is FailureLabel.BUDGET_EXHAUSTION

    # revisits the router judged productive are not a loop
    productive = episode(GEN_CYC, "q", segments=loop_segments((True,) * 6), env_details=details())
    assert unproductive_revisits(productive) == 0
    assert label_failure(productive) is FailureLabel.BUDGET_EXHAUSTION


def test_executor_drift_from_empty_answer_or_mostly_failed_calls() -> None:
    empty = react_episode("q", EndReason.FINAL_ANSWER, details(answer="  "), [(SEARCH, "results")])
    assert label_failure(empty) is FailureLabel.EXECUTOR_DRIFT

    half_failed = [
        (SEARCH, "Error: HTTP 500"),
        (SEARCH, "results"),
        (PARSE, "Error: bad url"),
        (SEARCH, "ok"),
    ]
    assert (
        label_failure(react_episode("q", EndReason.BUDGET, details(), half_failed))
        is FailureLabel.EXECUTOR_DRIFT
    )

    third_failed = [(SEARCH, "Error: HTTP 500"), (SEARCH, "results"), (SEARCH, "ok")]
    assert (
        label_failure(react_episode("q", EndReason.BUDGET, details(), third_failed))
        is FailureLabel.BUDGET_EXHAUSTION
    )


def test_depdag_termination_without_answer_is_a_graph_mismatch() -> None:
    record = episode(DEPDAG, "q", end_reason=EndReason.TERMINATED, env_details=details())
    assert label_failure(record) is FailureLabel.GRAPH_MISMATCH


def test_answered_failures_follow_the_evidence_trail() -> None:
    wrong = EndReason.FINAL_ANSWER
    read = [(PARSE, "stored"), (RETRIEVE, "summary")]
    no_documents = react_episode("q", wrong, details(answer="42"), [(SEARCH, "results")])
    assert label_failure(no_documents) is FailureLabel.RETRIEVAL_MISS

    never_read = react_episode("q", wrong, details(answer="42", documents=("k",)), [(PARSE, "stored")])
    assert label_failure(never_read) is FailureLabel.GROUNDING_MISMATCH

    failed_read = [(PARSE, "stored"), (RETRIEVE, "Error: unknown key")]
    assert (
        label_failure(react_episode("q", wrong, details(answer="42", documents=("k",)), failed_read))
        is FailureLabel.GROUNDING_MISMATCH
    )

    partial = react_episode("q", wrong, details(answer="42", documents=("k",), satisfied=(True, False)), read)
    assert label_failure(partial) is FailureLabel.SYNTHESIS_ERROR

    all_wrong = react_episode(
        "q", wrong, details(answer="42", documents=("k",), satisfied=(False, False)), read
    )
    assert label_failure(all_wrong) is FailureLabel.OTHER


@pytest.mark.parametrize(
    ("satisfied", "contradicted", "label"),
    [
        # no correctness criterion met: the satisfied contradiction criterion shows no reached content
        ((False, False), False, FailureLabel.OTHER),
        ((True, False), False, FailureLabel.SYNTHESIS_ERROR),
        ((True, True), True, FailureLabel.SYNTHESIS_ERROR),
        ((True,), None, FailureLabel.OTHER),
    ],
)
def test_synthesis_error_needs_a_met_correctness_criterion_and_an_unmet_verdict(
    satisfied: tuple[bool, ...], contradicted: bool | None, label: FailureLabel
) -> None:
    env = details(answer="42", documents=("k",), satisfied=satisfied, contradicted=contradicted)
    record = react_episode("q", EndReason.FINAL_ANSWER, env, [(PARSE, "stored"), (RETRIEVE, "summary")])
    assert label_failure(record) is label


def test_won_episodes_have_no_failure_label() -> None:
    with pytest.raises(ValueError, match="won"):
        label_failure(react_episode("q", EndReason.WON, details(answer="42"), won=True))


def test_finance_diagnostics_rates_over_non_error_episodes() -> None:
    episodes = [
        react_episode(
            "q1",
            EndReason.WON,
            details(answer="42", sources=1, documents=("k",), satisfied=(True,)),
            won=True,
            tool_calls=6,
        ),
        react_episode("q2", EndReason.BUDGET, details(), [(SEARCH, "results")] * 20),
        react_episode("q3", EndReason.FINAL_ANSWER, details(answer="41"), [(SEARCH, "r")] * 4),
        react_episode("q4", EndReason.FINAL_ANSWER, details(answer=""), [(SEARCH, "r")] * 2),
        error_episode(REACT, "q5", env_details=details(answer="x", sources=2), tool_calls=19),
    ]
    d = finance_diagnostics(make_run(REACT, episodes))

    assert (d.episodes, d.errors, d.wins, d.failed) == (4, 1, 1, 3)
    assert d.final_answer_rate == 0.5  # q1 and q3; q4's empty answer is not one
    assert d.source_submission_rate == 0.25  # q1
    assert d.budget_exhaustion_rate == 0.25  # q2
    assert d.tool_calls is not None and d.tool_calls.mean == 8.0  # (6 + 20 + 4 + 2) / 4
    assert d.failures[FailureLabel.BUDGET_EXHAUSTION] == 1
    assert d.failures[FailureLabel.RETRIEVAL_MISS] == 1
    assert d.failures[FailureLabel.EXECUTOR_DRIFT] == 1
    assert sum(d.failures.values()) == d.failed


def test_output_failures_are_labeled_from_the_trajectory_they_reached() -> None:
    won = episode(
        SPEC_CYC,
        "q1",
        won=True,
        tool_calls=6,
        segments=path_segments(["A"]),
        env_details=details(answer="42", sources=1, documents=("k",), satisfied=(True,)),
    )
    # two of three calls failed, both in the segment the unusable output cut short
    drifting = output_failure(
        SPEC_CYC,
        "q2",
        segments=[
            segment(
                0, "A", transition("A", "B", TransitionType.CROSS), calls=[tool_call(SEARCH, "results", k=1)]
            ),
            interrupted_segment(
                1,
                "B",
                calls=[tool_call(PARSE, "Error: HTTP 404", k=2), tool_call(RETRIEVE, "Error: no key", k=3)],
            ),
        ],
        env_details=details(),
    )
    # two unproductive revisits, then a revisit without a router record
    looping = output_failure(
        SPEC_CYC, "q3", segments=loop_segments((True, True, False, False)), env_details=details()
    )
    unplanned = planner_failure(SPEC_CYC, "q4", env_details=details())

    assert label_failure(drifting) is FailureLabel.EXECUTOR_DRIFT
    assert unproductive_revisits(looping) == 2
    assert label_failure(looping) is FailureLabel.ROUTING_LOOP
    assert label_failure(unplanned) is FailureLabel.OTHER
    run = make_run(SPEC_CYC, [won, drifting, looping, unplanned])
    d = finance_diagnostics(run)
    assert (d.episodes, d.errors, d.wins, d.failed) == (4, 0, 1, 3)
    assert d.final_answer_rate == 0.25
    assert d.tool_calls is not None and d.tool_calls.mean == pytest.approx((6 + 3 + 0 + 0) / 4)
    assert (d.failures[FailureLabel.EXECUTOR_DRIFT], d.failures[FailureLabel.ROUTING_LOOP]) == (1, 1)
    assert d.failures[FailureLabel.OTHER] == 1
    tool_use = to_markdown(finance_tables([run])[0])
    assert "| Spec-Cyc / gpt-4o-mini | 1/4 | 0.250 | 0.250 | 0.000 | 2.25 |" in tool_use


def test_diagnostics_are_finance_only() -> None:
    with pytest.raises(ValueError, match="Finance-Agent"):
        finance_diagnostics(make_run(config(controller=Controller.REACT), []))
