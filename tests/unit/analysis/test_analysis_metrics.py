from __future__ import annotations

import pytest
from analysis_fixtures import (
    config,
    cyclic_graph,
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

from ctg.analysis.load import Run
from ctg.analysis.metrics import (
    MeanStd,
    robustness,
    run_metrics,
    unique_transitions,
    visitation,
    wilson_interval,
)
from ctg.analysis.report import (
    Cell,
    Column,
    Interval,
    format_cell,
    main_table,
    summary_table,
    to_markdown,
    wilson_table,
)
from ctg.config import BenchmarkName, Controller, ExecutorMode, ModelAssignment
from ctg.runs.records import TransitionType


def test_sr_over_complete_repeats_while_tc_and_wilson_pool_every_non_error_episode() -> None:
    cfg = config(controller=Controller.REACT)
    episodes = [
        # repeat 0: 2 of 4 won
        episode(cfg, "a", 0, won=True, tool_calls=4),
        episode(cfg, "b", 0, won=True, tool_calls=6),
        episode(cfg, "c", 0, tool_calls=30),
        episode(cfg, "d", 0, tool_calls=30),
        # repeat 1: 1 of 3 won; d errored, so the repeat is incomplete
        episode(cfg, "a", 1, won=True, tool_calls=5),
        episode(cfg, "b", 1, tool_calls=30),
        episode(cfg, "c", 1, tool_calls=30),
        error_episode(cfg, "d", 1, tool_calls=2),
        # repeat 2: 3 of 4 won
        episode(cfg, "a", 2, won=True, tool_calls=3),
        episode(cfg, "b", 2, won=True, tool_calls=8),
        episode(cfg, "c", 2, won=True, tool_calls=4),
        episode(cfg, "d", 2, tool_calls=30),
    ]
    m = run_metrics(make_run(cfg, episodes))

    assert (m.episodes, m.errors, m.wins) == (11, 1, 6)
    assert m.sr is not None and m.tc is not None and m.str_ is not None
    # SR over the complete repeats 0 and 2: 50, 75 -> mean 62.5, population std 12.5
    assert (m.sr.mean, m.sr.std, m.sr.n) == (62.5, 12.5, 2)
    # TC over wins 4, 6, 5, 3, 8, 4: mean 5, sample std sqrt(16 / 5)
    assert m.tc.mean == pytest.approx(5.0)
    assert m.tc.std == pytest.approx(1.78885, abs=1e-5)
    assert m.str_ == pytest.approx(62.5 / 5.0)
    # Wilson interval on pooled 6 / 11
    wilson = m.wilson
    assert wilson is not None
    assert wilson == pytest.approx(wilson_interval(6, 11))


def _scheduled(instances: str, repeats: int) -> set[tuple[int, str]]:
    return {(r, i) for r in range(repeats) for i in instances}


def test_partly_finished_repeat_does_not_count_toward_sr_and_the_run_is_named() -> None:
    cfg = config(name="partial", controller=Controller.GEN_CYC)
    episodes = [
        episode(cfg, "a", 0, won=True),
        episode(cfg, "b", 0, won=True),
        episode(cfg, "c", 0, won=True),
        episode(cfg, "d", 0),
        episode(cfg, "a", 1),  # repeat 1 has one of four episodes; repeat 2 none
    ]
    run = make_run(cfg, episodes, expected=_scheduled("abcd", 3))
    m = run_metrics(run)

    # repeat 0 alone, not (75 + 0) / 2 = 37.5 ± 37.5
    assert m.sr is not None and (m.sr.mean, m.sr.std, m.sr.n) == (75.0, 0.0, 1)
    assert (m.episodes, m.wins) == (5, 3)
    table = main_table([run])
    assert any("partial (5/12 episodes; repeats 1, 2 incomplete)" in note for note in table.notes)


def test_run_without_a_complete_repeat_reports_the_pooled_sr_without_spread() -> None:
    cfg = config(name="started", controller=Controller.GEN_CYC)
    episodes = [episode(cfg, "a", 0, won=True), episode(cfg, "b", 0), episode(cfg, "c", 0)]
    run = make_run(cfg, episodes, expected=_scheduled("abcd", 3))
    m = run_metrics(run)

    assert m.sr is not None
    assert m.sr.mean == pytest.approx(100 / 3)
    assert m.sr.std is None
    table = main_table([run])
    assert "| TextCraft-2 | 33.3 ± D/A | 1.0 ± 0.0 | 33.3 |" in to_markdown(table).splitlines()
    assert any("started (3/12 episodes; repeats 0, 1, 2 incomplete)" in note for note in table.notes)


def test_complete_runs_get_no_coverage_note() -> None:
    cfg = config(name="done", controller=Controller.GEN_CYC)
    run = make_run(cfg, [episode(cfg, "a", r, won=r == 0) for r in range(3)], expected=_scheduled("a", 3))
    assert not any(note.startswith("Incomplete") for note in main_table([run]).notes)


def test_output_failures_are_losses_counted_in_rates_tool_calls_and_transitions() -> None:
    cfg = config(name="spec", controller=Controller.SPEC_CYC)
    won = episode(
        cfg, "a", won=True, tool_calls=3, segments=path_segments(["A", "B"]), graph=cyclic_graph("AB")
    )
    # the executor's output in the third segment was unusable after two routed segments and four tool calls
    cut_short = output_failure(
        cfg,
        "b",
        segments=[
            segment(
                0, "A", transition("A", "B", TransitionType.CROSS), calls=[tool_call(k=1), tool_call(k=2)]
            ),
            segment(1, "B", transition("B", "B", TransitionType.SELF), calls=[tool_call(k=3)]),
            interrupted_segment(2, "B", calls=[tool_call(k=4)]),
        ],
        graph=cyclic_graph("ABC"),
    )
    run = make_run(cfg, [won, cut_short, planner_failure(cfg, "c")])
    m = run_metrics(run)

    assert (m.episodes, m.errors, m.wins) == (3, 0, 1)
    assert not run.missing
    assert m.sr is not None and m.sr.mean == pytest.approx(100 / 3)
    assert m.wilson == pytest.approx(wilson_interval(1, 3))
    assert m.tc is not None and m.tc.mean == 3.0
    assert m.tc_all is not None and m.tc_all.mean == pytest.approx((3 + 4 + 0) / 3)
    assert m.transitions.all == {("A", "B"), ("B", "B")}
    assert m.transitions.never_won == {("B", "B")}
    assert m.visitation is not None and m.visitation.ats.n == 1
    assert "| spec | TextCraft-2 | Spec-Cyc | generalist | 3 | 0 | 1 | 33.3 ± 0.0 |" in to_markdown(
        summary_table([run])
    )


def test_run_without_wins_has_sr_zero_but_no_tc_or_str() -> None:
    cfg = config(controller=Controller.REACT)
    m = run_metrics(make_run(cfg, [episode(cfg, "a", 0), episode(cfg, "a", 1)]))
    assert m.sr is not None and (m.sr.mean, m.sr.std) == (0.0, 0.0)
    assert m.tc is None
    assert m.str_ is None
    assert m.visitation is None


def test_single_win_reports_zero_tc_spread() -> None:
    cfg = config(controller=Controller.REACT)
    m = run_metrics(make_run(cfg, [episode(cfg, "a", won=True, tool_calls=5)]))
    assert m.tc is not None and (m.tc.mean, m.tc.std) == (5.0, 0.0)


def test_visitation_counts_segments_and_unique_nodes_of_successful_graph_episodes() -> None:
    cfg = config(controller=Controller.GEN_CYC)
    episodes = [
        episode(cfg, "a", won=True, segments=path_segments(["A", "B", "A", "C"])),  # ATS 4, AUS 3
        episode(cfg, "b", won=True, segments=path_segments(["A"])),  # ATS 1, AUS 1
        episode(cfg, "c", won=True, segments=path_segments(["B", "B", "B"])),  # ATS 3, AUS 1
        episode(cfg, "d", segments=path_segments(["A", "B", "A", "B", "A"])),  # lost: ignored
        error_episode(cfg, "e", segments=path_segments(["A", "A"])),
    ]
    v = visitation(episodes)
    assert v is not None
    assert (v.ats.mean, v.ats.std) == pytest.approx((8 / 3, 1.527525), abs=1e-6)
    assert (v.aus.mean, v.aus.std) == pytest.approx((5 / 3, 1.154701), abs=1e-6)
    # ASR per episode: 1, 0, 2
    assert (v.asr.mean, v.asr.std) == pytest.approx((1.0, 1.0))


def test_react_episodes_have_no_visitation() -> None:
    cfg = config(controller=Controller.REACT)
    assert visitation([episode(cfg, "a", won=True)]) is None


def test_unique_transitions_use_executed_targets_and_split_won_from_never_won() -> None:
    cfg = config(controller=Controller.GEN_CYC)
    won = episode(cfg, "a", won=True, segments=path_segments(["A", "B", "A", "C"]))
    lost = episode(
        cfg,
        "b",
        segments=[
            segment(0, "A", transition("A", "B", TransitionType.CROSS)),
            segment(1, "B", transition("B", "B", TransitionType.SELF)),
            # fault injection: the router chose A but the controller went to D
            segment(2, "B", transition("B", "D", TransitionType.CROSS, nominal="A")),
            segment(3, "D", progress=None),
        ],
    )
    errored = error_episode(cfg, "c", segments=[segment(0, "C", transition("C", "D", TransitionType.CROSS))])
    ut = unique_transitions([won, lost, errored])
    assert ut.all == {("A", "B"), ("B", "A"), ("A", "C"), ("B", "B"), ("B", "D")}
    assert ut.won == {("A", "B"), ("B", "A"), ("A", "C")}
    assert ut.never_won == {("B", "B"), ("B", "D")}


def _robustness_run(name: str, perturbation: float, outcomes: dict[int, list[int | None]]) -> Run:
    """`outcomes[repeat]` lists tool calls of won episodes, None for a lost one."""
    cfg = config(name=name, controller=Controller.GEN_CYC, perturbation=perturbation)
    episodes = [
        episode(cfg, f"i{i}", repeat, won=tc is not None, tool_calls=tc or 30)
        for repeat, results in outcomes.items()
        for i, tc in enumerate(results)
    ]
    return make_run(cfg, episodes)


def test_srr_and_delta_tc() -> None:
    nominal = _robustness_run("nominal", 0.0, {0: [10, 10, 12, 12, None], 1: [10, 12, 11, None, None]})
    perturbed = _robustness_run(
        "perturbed", 0.5, {0: [14, 16, 12, None, None], 1: [18, 15, None, None, None]}
    )
    r = robustness(nominal, perturbed)
    # SR 70 -> 50, TC 77 / 7 = 11 -> 75 / 5 = 15
    assert r.srr == pytest.approx(50 / 70)
    assert r.delta_sr == pytest.approx(-20.0)
    assert r.delta_tc == pytest.approx(4.0)
    assert r.relative_drop == pytest.approx(100 * 20 / 70)


def test_srr_is_undefined_without_nominal_success() -> None:
    nominal = _robustness_run("nominal", 0.0, {0: [None, None]})
    perturbed = _robustness_run("perturbed", 0.5, {0: [12, None]})
    r = robustness(nominal, perturbed)
    assert r.srr is None
    assert r.delta_tc is None
    assert r.delta_sr == pytest.approx(50.0)


def test_robustness_rejects_runs_of_different_configurations() -> None:
    nominal = _robustness_run("nominal", 0.0, {0: [10]})
    other_cfg = config(name="other", controller=Controller.SPEC_CYC, perturbation=0.5)
    other = make_run(other_cfg, [episode(other_cfg, "i0", won=True)])
    with pytest.raises(ValueError, match="differ in controller"):
        robustness(nominal, other)
    finance_cfg = config(
        name="fin", controller=Controller.GEN_CYC, benchmark=BenchmarkName.FINANCE_AGENT, perturbation=0.5
    )
    with pytest.raises(ValueError, match="differ in benchmark"):
        robustness(nominal, make_run(finance_cfg, []))
    stronger = config(
        name="stronger",
        controller=Controller.GEN_CYC,
        perturbation=0.5,
        models=ModelAssignment(executor="gpt-5-mini"),
    )
    with pytest.raises(ValueError, match="differ in models"):
        robustness(nominal, make_run(stronger, []))


def test_robustness_rejects_a_generalist_specialist_pair() -> None:
    nominal = _robustness_run("nominal", 0.0, {0: [10]})
    specialist = config(name="spec", controller=Controller.GEN_CYC, perturbation=0.5).model_copy(
        update={"executors": ExecutorMode.SPECIALIST}
    )
    with pytest.raises(ValueError, match="differ in executors"):
        robustness(nominal, make_run(specialist, []))


def test_robustness_rejects_swapped_or_unperturbed_runs() -> None:
    nominal = _robustness_run("nominal", 0.0, {0: [10, None]})
    perturbed = _robustness_run("perturbed", 0.5, {0: [12, None]})
    with pytest.raises(ValueError, match="nominal run needs perturbation 0"):
        robustness(perturbed, nominal)
    with pytest.raises(ValueError, match="nominal run needs perturbation 0"):
        robustness(nominal, nominal)


def test_robustness_pairs_may_differ_in_seed_and_concurrency() -> None:
    nominal = _robustness_run("nominal", 0.0, {0: [10, None]})
    cfg = config(name="perturbed", controller=Controller.GEN_CYC, perturbation=0.5).model_copy(
        update={"seed": 7, "concurrency": 2}
    )
    perturbed = make_run(cfg, [episode(cfg, "i0", won=True, tool_calls=12), episode(cfg, "i1")])
    assert robustness(nominal, perturbed).srr == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("wins", "n", "low", "high"),
    [
        # Table 9 of the paper, in percent printed to one decimal (it prints 1/70's 0.25 as 0.2)
        (16, 70, 14.6, 34.0),
        (12, 70, 10.1, 27.6),
        (1, 70, 0.2, 7.7),
        (2, 70, 0.8, 9.8),
    ],
)
def test_wilson_matches_paper_table_9(wins: int, n: int, low: float, high: float) -> None:
    lower, upper = wilson_interval(wins, n)
    assert (100 * lower, 100 * upper) == pytest.approx((low, high), abs=0.06)


def test_wilson_follows_eq_9_at_the_boundaries() -> None:
    # x = 0, n = 10: center = (z^2 / 20) / (1 + z^2 / 10) = half-width, so the interval is [0, 2 * center]
    z2 = 1.96**2
    center = (z2 / 20) / (1 + z2 / 10)
    lower, upper = wilson_interval(0, 10)
    assert lower == pytest.approx(0.0, abs=1e-12)
    assert upper == pytest.approx(2 * center)
    lower, upper = wilson_interval(10, 10)
    assert (lower, upper) == pytest.approx((1 - 2 * center, 1.0))


def test_wilson_bounds_stay_within_zero_and_one() -> None:
    for n in range(1, 1000):
        for wins in {0, n // 3, n}:
            lower, upper = wilson_interval(wins, n)
            assert 0.0 <= lower <= upper <= 1.0
    # unclamped, Eq. 9 computes about -4e-19 for 0 of 609 (TextCraft-2, 203 × 3) and above 1 for 5 of 5
    assert wilson_interval(0, 609)[0] == 0.0
    assert wilson_interval(5, 5)[1] == 1.0


def test_zero_win_run_prints_no_negative_zero() -> None:
    cfg = config(name="lost", controller=Controller.REACT)
    run = make_run(cfg, [episode(cfg, f"i{i}", r) for i in range(203) for r in range(3)])
    markdown = to_markdown(wilson_table([run]))
    assert "| 0/609 | 0.0 [0.0, 0.6] |" in markdown
    assert "-0.0" not in markdown


@pytest.mark.parametrize(
    ("cell", "column", "text"),
    [
        (-1e-12, Column("ΔSR (pp)", signed=True), "+0.0"),
        (-0.004, Column("x", digits=2), "0.00"),
        (MeanStd(-0.01, 0.0, 3), Column("x"), "0.0 ± 0.0"),
        (Interval(-0.0, -1e-19, 0.6), Column("x"), "0.0 [0.0, 0.6]"),
    ],
)
def test_values_rounding_to_zero_print_without_a_minus_sign(cell: Cell, column: Column, text: str) -> None:
    assert format_cell(cell, column) == text


def test_wilson_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)
