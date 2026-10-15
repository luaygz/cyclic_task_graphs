from __future__ import annotations

import pytest
from analysis_fixtures import T0, config, episode, error_episode, llm_call, make_run

from ctg.analysis.tokens import compare_tokens, episode_tokens, run_usage
from ctg.config import Controller, Role
from ctg.llm.base import LLMCallRecord
from ctg.runs.records import PreviousAttempt

EXECUTOR, JUDGE = Role.EXECUTOR, Role.JUDGE


def _calls(executor_in: int, executor_out: int, judge: int = 0) -> list[LLMCallRecord]:
    return [llm_call(EXECUTOR, executor_in, executor_out), llm_call(JUDGE, judge, 0)]


def test_headline_total_spans_planner_executor_router_and_tool_but_not_judge() -> None:
    cfg = config(controller=Controller.GEN_CYC)
    record = episode(
        cfg,
        "a",
        llm_calls=[
            llm_call(Role.PLANNER, 300, 100),
            llm_call(Role.EXECUTOR, 1000, 200),
            llm_call(Role.ROUTER, 400, 100),
            llm_call(Role.TOOL, 200, 0),
            llm_call(Role.JUDGE, 700, 50),
        ],
    )
    assert episode_tokens(record) == 2300


def test_shared_win_comparison_over_instance_repeat_pairs() -> None:
    react = config(name="react", controller=Controller.REACT)
    cyclic = config(name="cyclic", controller=Controller.GEN_CYC)
    baseline = make_run(
        react,
        [
            episode(react, "a", 0, won=True, llm_calls=_calls(1000, 200, judge=500)),  # 1200
            episode(react, "b", 0, won=True, llm_calls=_calls(2000, 400)),  # 2400
            episode(react, "c", 0, llm_calls=_calls(5000, 0)),
            episode(react, "a", 1, won=True, llm_calls=_calls(1500, 300)),  # 1800
            episode(react, "b", 1, won=True, llm_calls=_calls(900, 100)),  # 1000, B errored here
        ],
    )
    compared = make_run(
        cyclic,
        [
            episode(cyclic, "a", 0, won=True, llm_calls=_calls(2000, 300, judge=700)),  # 2300
            episode(cyclic, "b", 0, llm_calls=_calls(9000, 0)),
            episode(cyclic, "c", 0, won=True, llm_calls=_calls(3000, 600)),  # 3600
            episode(cyclic, "a", 1, won=True, llm_calls=_calls(2000, 500)),  # 2500
            error_episode(cyclic, "b", 1, llm_calls=_calls(100, 0)),
        ],
    )
    result = compare_tokens(baseline, compared)

    shared = result.shared_wins
    assert (shared.cases_a, shared.cases_b) == (2, 2)  # (a, 0) and (a, 1)
    assert (shared.avg_a, shared.avg_b) == (1500.0, 2400.0)
    assert shared.excess == 900.0
    assert shared.relative_pct == pytest.approx(60.0)

    every = result.all_wins
    assert (every.cases_a, every.cases_b) == (4, 3)
    assert every.avg_a == pytest.approx((1200 + 2400 + 1800 + 1000) / 4)
    assert every.avg_b == pytest.approx((2300 + 3600 + 2500) / 3)
    assert every.excess == pytest.approx(2800 - 1600)
    assert every.relative_pct == pytest.approx(75.0)


def test_comparison_without_shared_wins_is_undefined() -> None:
    react = config(name="react", controller=Controller.REACT)
    cyclic = config(name="cyclic", controller=Controller.GEN_CYC)
    result = compare_tokens(
        make_run(react, [episode(react, "a", won=True, llm_calls=_calls(10, 0))]),
        make_run(cyclic, [episode(cyclic, "b", won=True, llm_calls=_calls(20, 0))]),
    )
    assert result.shared_wins.avg_a is None
    assert result.shared_wins.excess is None
    assert result.shared_wins.relative_pct is None
    assert result.all_wins.relative_pct == pytest.approx(100.0)


def test_comparison_rejects_different_benchmarks() -> None:
    react = config(name="react", controller=Controller.REACT)
    other = config(name="other", controller=Controller.GEN_CYC).model_copy(update={"textcraft_depth": 3})
    with pytest.raises(ValueError, match="different benchmark"):
        compare_tokens(make_run(react, []), make_run(other, []))


def test_spend_includes_error_episodes_and_one_time_planning() -> None:
    cfg = config(controller=Controller.GEN_CYC)
    run = make_run(
        cfg,
        [
            episode(
                cfg, "a", llm_calls=[llm_call(EXECUTOR, 100, 10, cost=0.25), llm_call(JUDGE, 50, 5, cost=0.5)]
            ),
            error_episode(cfg, "b", llm_calls=[llm_call(EXECUTOR, 40, 0, cost=0.125)]),
        ],
        planning_calls=[llm_call(Role.PLANNER, 1000, 1000, cost=1.0)],
    )
    usage = run_usage(run)
    assert usage.spend_usd == pytest.approx(1.875)
    assert usage.by_role[EXECUTOR].calls == 2
    assert usage.by_role[EXECUTOR].cost_usd == pytest.approx(0.375)
    # per-test-case totals cover the one non-error episode only, without judge or one-time planning
    assert usage.episodes == 1
    assert usage.tokens_per_case is not None and usage.tokens_per_case.mean == 110.0
    assert usage.judge_tokens_per_case is not None and usage.judge_tokens_per_case.mean == 55.0


def test_spend_includes_earlier_error_attempts_but_per_case_tokens_cover_the_final_attempt() -> None:
    cfg = config(name="rerun", controller=Controller.REACT)
    failed = PreviousAttempt(
        error="ConnectError",
        llm_calls=[llm_call(EXECUTOR, 300, 0, cost=0.25), llm_call(EXECUTOR, 300, 0, cost=0.25)],
        started_at=T0,
        finished_at=T0,
    )
    record = episode(cfg, "a", won=True, llm_calls=[llm_call(EXECUTOR, 100, 20, cost=0.5)]).model_copy(
        update={"previous_attempts": [failed]}
    )
    run = make_run(cfg, [record])
    usage = run_usage(run)

    assert usage.by_role[EXECUTOR].calls == 3
    assert usage.spend_usd == pytest.approx(1.0)
    assert usage.episodes == 1
    assert usage.tokens_per_case is not None and usage.tokens_per_case.mean == 120.0
    assert compare_tokens(run, run).all_wins.avg_a == 120.0
