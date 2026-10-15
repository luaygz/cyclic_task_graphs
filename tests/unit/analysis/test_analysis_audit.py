from __future__ import annotations

import pytest
from analysis_fixtures import (
    config,
    dependency_graph,
    episode,
    error_episode,
    interrupted_segment,
    make_run,
    output_failure,
    planner_failure,
    segment,
    tool_call,
    transition,
)

from ctg.analysis.audit import audit_transitions
from ctg.config import Controller
from ctg.graphs.model import END
from ctg.runs.records import TransitionType

RETRY, FORWARD, TERMINAL = TransitionType.RETRY, TransitionType.FORWARD, TransitionType.TERMINAL


def test_audit_counts_types_and_flags_invalid_transitions() -> None:
    depdag = config(name="depdag", controller=Controller.DEPDAG)
    clean = episode(
        depdag,
        "a",
        segments=[
            segment(0, "t1", transition("t1", "t1", RETRY, allowed=["t2", "t1", END])),
            segment(1, "t1", transition("t1", "t2", FORWARD, allowed=["t2", "t1", END])),
            segment(2, "t2", transition("t2", END, TERMINAL, allowed=["t3", "t2", END])),
        ],
    )
    broken = episode(
        depdag,
        "b",
        segments=[
            segment(0, "t1", transition("t1", "t2", FORWARD, allowed=["t2", "t1", END])),
            # a backtrack typed as cyclic: outside the allowed set although flagged valid
            segment(1, "t2", transition("t2", "t1", TransitionType.CROSS, allowed=["t3", "t2", END])),
            # inside the allowed set but flagged invalid by the controller
            segment(2, "t1", transition("t1", "t2", FORWARD, allowed=["t2", "t1", END], valid=False)),
            segment(3, "t2", progress=None),
        ],
    )
    skipped = error_episode(
        depdag, "c", segments=[segment(0, "t1", transition("t1", "t3", FORWARD, valid=False))]
    )
    noretry = config(name="noretry", controller=Controller.DEPDAG_NORETRY)
    sink = episode(noretry, "a", segments=[segment(0, "t1", transition("t1", END, TERMINAL, allowed=[END]))])

    audit = audit_transitions([make_run(depdag, [clean, broken, skipped]), make_run(noretry, [sink])])

    assert (audit.runs, audit.test_cases) == (2, 3)
    assert audit.transitions == 7
    assert (audit.retry, audit.forward, audit.terminal, audit.other) == (1, 3, 2, 1)
    assert audit.invalid == 2


def test_audit_inspects_output_failures_up_to_the_unusable_output() -> None:
    depdag = config(name="depdag", controller=Controller.DEPDAG)
    allowed = ["t2", "t1", END]
    # the router's output after the third segment was unusable: that segment has no transition
    cut_short = output_failure(
        depdag,
        "a",
        segments=[
            segment(0, "t1", transition("t1", "t1", RETRY, allowed=allowed), calls=[tool_call(k=1)]),
            segment(1, "t1", transition("t1", "t2", FORWARD, allowed=allowed), calls=[tool_call(k=2)]),
            interrupted_segment(2, "t2", calls=[tool_call(k=3), tool_call(k=4)]),
        ],
        graph=dependency_graph(["t1", "t2"]),
    )
    audit = audit_transitions([make_run(depdag, [cut_short, planner_failure(depdag, "b")])])

    assert (audit.runs, audit.test_cases, audit.transitions) == (1, 2, 2)
    assert (audit.retry, audit.forward, audit.terminal, audit.other, audit.invalid) == (1, 1, 0, 0, 0)


def test_audit_refuses_cyclic_runs() -> None:
    cyclic = config(controller=Controller.GEN_CYC)
    with pytest.raises(ValueError, match="DepDAG runs only"):
        audit_transitions([make_run(cyclic, [])])
