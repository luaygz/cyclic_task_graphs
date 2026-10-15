"""Every shipped experiment config is valid, uniquely named and points at the right n-shot set, and every
run behind Tables 12 and 15 has its config."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from ctg.config import (
    CONFIGS_DIR,
    BenchmarkName,
    Controller,
    ExecutorMode,
    ExperimentConfig,
    load_model_presets,
    resolve_models,
)

EXPERIMENTS = CONFIGS_DIR / "experiments"
PATHS = sorted(EXPERIMENTS.rglob("*.yaml"))
BY_NAME = {path.stem: path for path in PATHS}
NSHOT_SETS = {
    "textcraft_depth2": "data/nshot/textcraft_depth2.json",
    "textcraft_depth3": "data/nshot/textcraft_depth3.json",
    "textcraft_depth4": "data/nshot/textcraft_depth3.json",
    "alfworld_valid_unseen": "data/nshot/alfworld_train.json",
    "finance_agent": "data/nshot/finance_agent.json",
}
# Benchmark key in config names -> frozen instance list the paper reports on.
BENCHMARKS = {
    "textcraft2": "textcraft_depth2",
    "textcraft3": "textcraft_depth3",
    "textcraft4": "textcraft_depth4",
    "alfworld": "alfworld_valid_unseen",
    "finance": "finance_agent",
}
TIERS = {"4o": "gpt-4o-mini", "5m": "gpt-5-mini"}
# Table 15 rows as planner/executor/router tiers per benchmark and graph.
_SWEEP = ("5m/4o/4o", "4o/5m/5m", "4o/4o/4o")
_TEXTCRAFT_GEN_CYC = (*_SWEEP, "4o/5m/4o", "4o/4o/5m")
TABLE_15: dict[tuple[str, Controller], tuple[str, ...]] = {
    ("textcraft2", Controller.SPEC_CYC): _SWEEP,
    ("textcraft2", Controller.GEN_CYC): _TEXTCRAFT_GEN_CYC,
    ("textcraft3", Controller.SPEC_CYC): _SWEEP,
    ("textcraft3", Controller.GEN_CYC): _TEXTCRAFT_GEN_CYC,
    ("textcraft4", Controller.SPEC_CYC): _SWEEP,
    ("textcraft4", Controller.GEN_CYC): _TEXTCRAFT_GEN_CYC,
    ("alfworld", Controller.SPEC_CYC): (*_SWEEP, "5m/5m/5m"),
    ("alfworld", Controller.GEN_CYC): _SWEEP,
    ("finance", Controller.SPEC_CYC): _SWEEP,
    ("finance", Controller.GEN_CYC): _SWEEP,
}
TABLE_15_CELLS = [
    (f"a2_{benchmark}_{graph}_p{p}_e{e}_r{r}", benchmark, graph, (TIERS[p], TIERS[e], TIERS[r]))
    for (benchmark, graph), allocations in TABLE_15.items()
    for p, e, r in (allocation.split("/") for allocation in allocations)
]


def load(name: str) -> ExperimentConfig:
    assert name in BY_NAME, f"no config named {name}"
    return ExperimentConfig.load(BY_NAME[name])


@pytest.mark.parametrize("path", PATHS, ids=lambda path: path.stem)
def test_config_loads_and_resolves(path: Path) -> None:
    config = ExperimentConfig.load(path)

    assert config.name == path.stem
    resolve_models(config.models, load_model_presets())
    assert config.split == "test"
    if config.nshot is not None:
        assert config.nshot.as_posix() == NSHOT_SETS[config.variant]


def test_families_have_their_cells() -> None:
    configs = [ExperimentConfig.load(path) for path in PATHS]
    by_family: dict[str, list[ExperimentConfig]] = {}
    for config, path in zip(configs, PATHS, strict=True):
        by_family.setdefault(path.parent.name, []).append(config)

    assert len({config.name for config in configs}) == len(configs)
    assert {family: len(cells) for family, cells in by_family.items() if family != "a6_exploration"} == {
        "a1_main": 20,
        "d1_depdag_noretry": 5,
        "a2_a3_roles_nshot": 51,
        "a4_tools": 24,
        "a5_faults": 14,
        "b5_model_coverage": 4,
        "c_token_cost": 5,
    }
    assert all(config.perturbation == 0.5 and config.nshot is not None for config in by_family["a5_faults"])
    a1 = by_family["a1_main"]
    assert Counter(config.controller for config in a1) == dict.fromkeys(
        (Controller.REACT, Controller.DEPDAG, Controller.SPEC_CYC, Controller.GEN_CYC), 5
    )
    assert all(config.nshot is None and config.repeats == 3 for config in a1)
    assert all(
        config.benchmark is BenchmarkName.FINANCE_AGENT and config.repeats == 2
        for config in by_family["b5_model_coverage"]
    )


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_table_12_baselines_cover_the_compared_runs(benchmark: str) -> None:
    """Table 12 compares ReAct with gpt-4o-mini and with gpt-5-mini against cyclic runs; the gpt-5-mini
    baseline is paired with the P/E/R = 4o/5m/5m cyclic runs, so both sides must share one instance list."""
    for name, executor in (
        (f"a1_{benchmark}_react", "gpt-4o-mini"),
        (f"c_{benchmark}_react_e5m", "gpt-5-mini"),
    ):
        baseline = load(name)
        assert baseline.controller is Controller.REACT
        assert baseline.models.executor == executor
        assert baseline.nshot is None and baseline.budgets is None and baseline.repeats == 3
        assert (baseline.variant, baseline.split, baseline.instances) == (BENCHMARKS[benchmark], "test", None)
    for graph in (Controller.SPEC_CYC, Controller.GEN_CYC):
        compared = load(f"a2_{benchmark}_{graph}_p4o_e5m_r5m")
        assert (compared.variant, compared.split, compared.instances) == (BENCHMARKS[benchmark], "test", None)


@pytest.mark.parametrize(
    ("name", "benchmark", "graph", "tiers"), TABLE_15_CELLS, ids=[c[0] for c in TABLE_15_CELLS]
)
def test_table_15_row_has_its_config(
    name: str, benchmark: str, graph: Controller, tiers: tuple[str, str, str]
) -> None:
    config = load(name)

    assert config.controller is graph
    assert (config.models.planner, config.models.executor, config.models.router) == tiers
    assert config.variant == BENCHMARKS[benchmark]
    assert config.executors is ExecutorMode.GENERALIST and config.nshot is not None and config.repeats == 3
