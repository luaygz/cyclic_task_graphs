from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
from analysis_fixtures import (
    config,
    episode,
    error_episode,
    path_segments,
    segment,
    split_ids,
    transition,
    write_run,
)
from typer.testing import CliRunner

from ctg.analysis.cli import app
from ctg.config import BenchmarkName, Controller
from ctg.graphs.model import END
from ctg.runs.records import TransitionType

runner = CliRunner()


def _table2_runs(root: Path) -> list[Path]:
    """Complete one-repeat TextCraft runs, and an ALFWorld run whose second episode errored."""
    tc2, alf_ids = split_ids("textcraft_depth2", 2), split_ids("alfworld_valid_seen", 2)
    react = config(name="react-tc2", controller=Controller.REACT, instances=tc2, repeats=1)
    gen = config(name="gen-tc2", controller=Controller.GEN_CYC, instances=tc2, repeats=1)
    alf = config(
        name="gen-alf",
        controller=Controller.GEN_CYC,
        benchmark=BenchmarkName.ALFWORLD,
        instances=alf_ids,
        repeats=1,
    )
    return [
        write_run(
            root,
            react,
            [episode(react, tc2[0], won=True, tool_calls=4), episode(react, tc2[1], tool_calls=30)],
        ),
        write_run(
            root,
            gen,
            [
                episode(gen, tc2[0], won=True, tool_calls=6, segments=path_segments(["A", "B"])),
                episode(gen, tc2[1], won=True, tool_calls=10, segments=path_segments(["A"])),
            ],
        ),
        write_run(root, alf, [episode(alf, alf_ids[0], tool_calls=75), error_episode(alf, alf_ids[1])]),
    ]


def test_main_renders_table_2_layout_and_csv(tmp_path: Path) -> None:
    runs = _table2_runs(tmp_path / "runs")
    out = tmp_path / "out" / "table2.csv"
    result = runner.invoke(app, ["main", *map(str, runs), "--csv", str(out)])

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert (
        "| Benchmark | ReAct SR (%) | ReAct TC | ReAct STR | Gen-Cyc SR (%) | Gen-Cyc TC | Gen-Cyc STR |"
        in lines
    )
    assert "| TextCraft-2 | 50.0 ± 0.0 | 4.0 ± 0.0 | 12.5 | 100.0 ± 0.0 | 8.0 ± 2.8 | 12.5 |" in lines
    # no complete repeat: the pooled rate without a spread
    assert "| ALFWorld (valid seen) | D/A | D/A | D/A | 0.0 ± D/A | D/A | D/A |" in lines
    assert lines.index("| ALFWorld (valid seen) | D/A | D/A | D/A | 0.0 ± D/A | D/A | D/A |") > lines.index(
        "| TextCraft-2 | 50.0 ± 0.0 | 4.0 ± 0.0 | 12.5 | 100.0 ± 0.0 | 8.0 ± 2.8 | 12.5 |"
    )
    assert "Error episodes excluded from all rates: gen-alf (1)." in lines
    incomplete = [line for line in lines if line.startswith("Incomplete runs")]
    assert len(incomplete) == 1
    assert "gen-alf (1/2 episodes; repeat 0 incomplete)" in incomplete[0]
    assert "tc2" not in incomplete[0]

    with out.open() as f:
        rows = list(csv.DictReader(f))
    textcraft = rows[0]
    assert textcraft["Benchmark"] == "TextCraft-2"
    assert float(textcraft["Gen-Cyc TC mean"]) == 8.0
    assert float(textcraft["Gen-Cyc TC std"]) == 2.8284271247461903
    assert rows[1]["ReAct SR (%) mean"] == ""
    assert rows[1]["Gen-Cyc SR (%) mean"] == "0.0"
    assert rows[1]["Gen-Cyc SR (%) std"] == ""


def test_main_refuses_two_runs_for_one_cell(tmp_path: Path) -> None:
    first = config(name="gen-a", controller=Controller.GEN_CYC)
    second = config(name="gen-b", controller=Controller.GEN_CYC)
    runs = [write_run(tmp_path, cfg, [episode(cfg, "a", won=True)]) for cfg in (first, second)]
    result = runner.invoke(app, ["main", *map(str, runs)])
    assert result.exit_code == 1
    assert "gen-a and gen-b both fill TextCraft-2 / Gen-Cyc" in result.stderr


def test_audit_renders_table_14_and_skips_cyclic_runs(tmp_path: Path) -> None:
    dag = config(name="dag", controller=Controller.DEPDAG)
    allowed = ["t2", "t1", END]
    depdag_run = write_run(
        tmp_path,
        dag,
        [
            episode(
                dag,
                "a",
                segments=[
                    segment(0, "t1", transition("t1", "t1", TransitionType.RETRY, allowed=allowed)),
                    segment(1, "t1", transition("t1", "t2", TransitionType.FORWARD, allowed=allowed)),
                    segment(2, "t2", transition("t2", END, TransitionType.TERMINAL, allowed=["t2", END])),
                ],
            ),
            episode(
                dag,
                "b",
                segments=[segment(0, "t1", transition("t1", "t3", TransitionType.FORWARD, allowed=allowed))],
            ),
        ],
    )
    gen = config(name="gen", controller=Controller.GEN_CYC)
    cyclic_run = write_run(tmp_path, gen, [episode(gen, "a", won=True)])

    result = runner.invoke(app, ["audit", str(depdag_run), str(cyclic_run)])

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    for row in (
        "| DepDAG runs inspected | 1 |",
        "| Test cases inspected | 2 |",
        "| Typed transitions inspected | 4 |",
        "| Same-subtask retry transitions | 1 |",
        "| DAG-forward transitions | 2 |",
        "| Terminal transitions | 1 |",
        "| Other transition types | 0 |",
        "| Invalid transitions | 1 |",
    ):
        assert row in lines
    assert "Skipping runs that are not DepDAG runs: gen" in result.stderr


def test_sweep_writes_one_csv_per_table(tmp_path: Path) -> None:
    gen = config(name="gen", controller=Controller.GEN_CYC)
    run = write_run(
        tmp_path / "runs",
        gen,
        [
            episode(gen, "a", won=True, tool_calls=6, segments=path_segments(["A", "B", "A"])),
            episode(gen, "b", tool_calls=30, segments=path_segments(["A", "C"])),
        ],
    )
    result = runner.invoke(app, ["sweep", str(run), "--csv", str(tmp_path / "sweep.csv")])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "sweep.csv").exists()
    with (tmp_path / "sweep.sweep.csv").open() as f:
        (sweep_row,) = list(csv.DictReader(f))
    assert (sweep_row["ATS mean"], sweep_row["AUS mean"], sweep_row["ASR mean"]) == ("3.0", "2.0", "1.0")
    with (tmp_path / "sweep.unique_transitions.csv").open() as f:
        (ut_row,) = list(csv.DictReader(f))
    assert (ut_row["UT (all)"], ut_row["UT (won)"], ut_row["UT (never won)"]) == ("3", "2", "1")


def test_robustness_refuses_swapped_nominal_and_perturbed_runs(tmp_path: Path) -> None:
    nominal = config(name="nom", controller=Controller.GEN_CYC)
    perturbed = config(name="pert", controller=Controller.GEN_CYC, perturbation=0.5)
    nom, pert = (write_run(tmp_path, cfg, [episode(cfg, "a", won=True)]) for cfg in (nominal, perturbed))

    result = runner.invoke(app, ["robustness", "--nominal", str(pert), "--perturbed", str(nom)])

    assert result.exit_code == 1
    assert "Error: the nominal run needs perturbation 0" in result.stderr
    assert "SRR" not in result.stdout


def test_figures_without_matplotlib_name_the_plots_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gen = config(name="gen", controller=Controller.GEN_CYC)
    run = write_run(tmp_path / "runs", gen, [episode(gen, "a", won=True, segments=path_segments(["A"]))])
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    result = runner.invoke(app, ["figures", str(run), "--out", str(tmp_path / "figs")])

    assert result.exit_code == 1
    assert result.stderr.startswith("Error:")
    assert "`plots` extra" in result.stderr
    assert result.exception is None or isinstance(result.exception, SystemExit)
