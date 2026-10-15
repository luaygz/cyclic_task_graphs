from __future__ import annotations

import pytest
from analysis_fixtures import (
    config,
    cyclic_graph,
    dependency_graph,
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

from ctg.analysis.report import Column, format_cell, routing_tables, to_markdown
from ctg.analysis.routing import Range, run_routing, summarize_routing
from ctg.config import Controller
from ctg.graphs.model import END
from ctg.runs.records import TransitionType

SELF, CROSS = TransitionType.SELF, TransitionType.CROSS


def test_gen_cyc_run_statistics() -> None:
    cfg = config(name="gen", controller=Controller.GEN_CYC)
    graph = cyclic_graph(["A", "B", "C"]).model_copy(
        update={
            "criteria": {
                **cyclic_graph(["A", "B", "C"]).criteria,
                "A": {"A": "a b c d e", "B": "x y", "C": "x y"},
            }
        }
    )
    run = make_run(
        cfg,
        [
            episode(
                cfg, "i1", graph=graph, segments=path_segments(["A", "A", "B", "C"], [SELF, CROSS, CROSS])
            ),
            episode(cfg, "i2", graph=graph, segments=path_segments(["B", "B", "B"], [SELF, SELF])),
            error_episode(cfg, "i3", graph=graph, segments=path_segments(["A", "A", "A", "A"], [SELF] * 3)),
        ],
    )
    stats = run_routing(run)
    assert stats.test_cases == 2
    assert stats.nodes == 3.0
    # eight 2-word criteria and one 5-word criterion
    assert stats.criterion_words == pytest.approx(21 / 9)
    assert stats.self_loops == pytest.approx(1.5)  # (1 + 2) / 2
    assert stats.inter_node == pytest.approx(1.0)  # (2 + 0) / 2


def test_spec_cyc_averages_each_episode_graph() -> None:
    cfg = config(name="spec", controller=Controller.SPEC_CYC)
    small, large = cyclic_graph(["A", "B"], words=3), cyclic_graph(["A", "B", "C", "D"], words=1)
    run = make_run(
        cfg,
        [
            episode(cfg, "i1", graph=small, segments=path_segments(["A"])),
            episode(cfg, "i2", graph=large, segments=path_segments(["A", "B"])),
        ],
    )
    stats = run_routing(run)
    assert stats.nodes == 3.0
    assert stats.criterion_words == 2.0
    assert (stats.self_loops, stats.inter_node) == (0.0, 0.5)


def test_depdag_retry_counts_as_self_loop_and_forward_as_inter_node() -> None:
    cfg = config(name="dag", controller=Controller.DEPDAG)
    graph = dependency_graph(["t1", "t2"])
    segments = path_segments(
        ["t1", "t1", "t2", END], [TransitionType.RETRY, TransitionType.FORWARD, TransitionType.TERMINAL]
    )
    stats = run_routing(make_run(cfg, [episode(cfg, "i1", graph=graph, segments=segments[:-1])]))
    assert stats.nodes == 2.0
    assert stats.criterion_words == 3.0
    assert (stats.self_loops, stats.inter_node) == (1.0, 1.0)


def test_output_failures_count_as_test_cases_with_the_transitions_taken_before_them() -> None:
    cfg = config(name="spec", controller=Controller.SPEC_CYC)
    small, large = cyclic_graph(["A", "B"], words=3), cyclic_graph(["A", "B", "C", "D"], words=1)
    cut_short = output_failure(
        cfg,
        "i2",
        segments=[
            segment(0, "A", transition("A", "B", CROSS), calls=[tool_call(k=1)]),
            segment(1, "B", transition("B", "B", SELF), calls=[tool_call(k=2)]),
            interrupted_segment(2, "B", calls=[tool_call(k=3)]),
        ],
        graph=large,
    )
    run = make_run(
        cfg,
        [
            episode(cfg, "i1", won=True, graph=small, segments=path_segments(["A", "B"])),
            cut_short,
            planner_failure(cfg, "i3"),
        ],
    )
    stats = run_routing(run)

    assert stats.test_cases == 3
    # graph size over the two episodes that have a graph; the planner failure has none
    assert (stats.nodes, stats.criterion_words) == (3.0, 2.0)
    assert stats.self_loops == pytest.approx(1 / 3)  # (0 + 1 + 0) / 3
    assert stats.inter_node == pytest.approx(2 / 3)  # (1 + 1 + 0) / 3
    assert "| spec | TextCraft-2 | Spec-Cyc | 3 | 3.0 | 2.0 | 0.3 | 0.7 |" in to_markdown(
        routing_tables([run])[0]
    )


def test_react_runs_have_no_routing() -> None:
    with pytest.raises(ValueError, match="ReAct"):
        run_routing(make_run(config(controller=Controller.REACT), []))


def test_summary_ranges_across_runs_of_one_controller_and_benchmark() -> None:
    rows = []
    for name, ids in (("g1", ["A", "B", "C"]), ("g2", ["A", "B", "C", "D", "E"])):
        cfg = config(name=name, controller=Controller.GEN_CYC)
        rows.append(
            run_routing(
                make_run(cfg, [episode(cfg, "i", graph=cyclic_graph(ids), segments=path_segments(ids))])
            )
        )
    spec = config(name="s", controller=Controller.SPEC_CYC)
    rows.append(
        run_routing(
            make_run(spec, [episode(spec, "i", graph=cyclic_graph(["A"]), segments=path_segments(["A"]))])
        )
    )

    gen, only_spec = summarize_routing(rows)
    assert (gen.controller, gen.runs) == (Controller.GEN_CYC, 2)
    assert gen.nodes == Range(3.0, 5.0)
    assert gen.inter_node == Range(2.0, 4.0)
    assert only_spec.nodes == Range(1.0, 1.0)
    column = Column("nodes")
    assert format_cell(gen.nodes, column) == "3.0–5.0"
    assert format_cell(only_spec.nodes, column) == "1.0"
