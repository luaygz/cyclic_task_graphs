"""`ctg analyze`: paper tables (Markdown on stdout, optional CSV) and figures from run directories."""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, ParamSpec

import typer

from ctg.analysis import figures as figs
from ctg.analysis.audit import audit_transitions
from ctg.analysis.exploration import compare_exploration
from ctg.analysis.load import Run, load_run, load_runs
from ctg.analysis.metrics import robustness as compute_robustness
from ctg.analysis.report import (
    Table,
    audit_table,
    exploration_table,
    finance_tables,
    main_table,
    render_markdown,
    robustness_table,
    routing_tables,
    summary_table,
    sweep_tables,
    token_tables,
    wilson_table,
    write_csv,
)
from ctg.analysis.tokens import compare_tokens
from ctg.config import BenchmarkName, Controller

app = typer.Typer(help="Tables and figures from run directories (runs/<name>).", no_args_is_help=True)

RunDirs = Annotated[
    list[Path], typer.Argument(help="Run directories.", exists=True, file_okay=False, show_default=False)
]
CsvPath = Annotated[
    Path | None,
    typer.Option(
        "--csv", help="Also write CSV here (several tables: <stem>.<table><suffix>).", dir_okay=False
    ),
]

P = ParamSpec("P")


def _user_errors(command: Callable[P, None]) -> Callable[P, None]:
    """Turn analysis errors (incompatible or malformed runs) into a message and exit code 1."""

    @functools.wraps(command)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            command(*args, **kwargs)
        except (ValueError, FileNotFoundError) as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(1) from error

    return wrapper


def _emit(tables: Sequence[Table], csv_path: Path | None) -> None:
    typer.echo(render_markdown(tables), nl=False)
    if csv_path is not None:
        for path in write_csv(tables, csv_path):
            typer.echo(f"Wrote {path}", err=True)


def _select(runs: Sequence[Run], keep: Callable[[Run], bool], what: str) -> list[Run]:
    selected = [r for r in runs if keep(r)]
    if skipped := [r.name for r in runs if not keep(r)]:
        typer.echo(f"Skipping runs that are not {what}: {', '.join(skipped)}", err=True)
    if not selected:
        raise ValueError(f"none of the runs are {what}")
    return selected


@app.command()
@_user_errors
def summary(runs: RunDirs, csv: CsvPath = None) -> None:
    """Every metric of each run: SR, Wilson CI, TC, STR, ATS/AUS/ASR, UT, tokens and spend."""
    _emit([summary_table(load_runs(runs))], csv)


@app.command()
@_user_errors
def main(runs: RunDirs, csv: CsvPath = None) -> None:
    """Table 2: benchmark rows by controller columns (one run per cell)."""
    _emit([main_table(load_runs(runs))], csv)


@app.command()
@_user_errors
def sweep(runs: RunDirs, csv: CsvPath = None) -> None:
    """Tables 15/16 and 19/20: one row per configuration with SR/TC/STR/ATS/AUS/ASR and UT."""
    _emit(sweep_tables(load_runs(runs)), csv)


@app.command()
@_user_errors
def robustness(
    nominal: Annotated[list[Path], typer.Option(exists=True, file_okay=False, help="Unperturbed run.")],
    perturbed: Annotated[
        list[Path], typer.Option(exists=True, file_okay=False, help="Fault-injected run (paired in order).")
    ],
    csv: CsvPath = None,
) -> None:
    """Tables 3/17/18: SRR and ΔTC for nominal and fault-injected runs (repeat the options for more rows)."""
    if len(nominal) != len(perturbed):
        raise ValueError("pass one --perturbed run for every --nominal run")
    pairs = [(load_run(n), load_run(p)) for n, p in zip(nominal, perturbed, strict=True)]
    _emit([robustness_table([(n, p, compute_robustness(n, p)) for n, p in pairs])], csv)


@app.command()
@_user_errors
def tokens(
    baseline: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Reference run (A), e.g. ReAct.")
    ],
    compare: Annotated[
        list[Path], typer.Option(exists=True, file_okay=False, help="Run compared with the baseline (B).")
    ],
    csv: CsvPath = None,
) -> None:
    """Table 12: shared-win token comparison, plus token usage and spend per run."""
    base = load_run(baseline)
    others = load_runs(compare)
    _emit(token_tables(base, [(other, compare_tokens(base, other)) for other in others]), csv)


@app.command()
@_user_errors
def audit(runs: RunDirs, csv: CsvPath = None) -> None:
    """Table 14: DepDAG transition audit."""
    selected = _select(load_runs(runs), lambda r: r.controller.is_dependency, "DepDAG runs")
    _emit([audit_table(audit_transitions(selected), selected)], csv)


@app.command()
@_user_errors
def routing(runs: RunDirs, csv: CsvPath = None) -> None:
    """Tables 21-24: graph size and routing behavior."""
    selected = _select(
        load_runs(runs), lambda r: r.controller is not Controller.REACT, "graph-controller runs"
    )
    _emit(routing_tables(selected), csv)


@app.command()
@_user_errors
def failures(runs: RunDirs, csv: CsvPath = None) -> None:
    """Tables 10-11: Finance-Agent tool-use diagnostics and trace-derived failure labels."""
    selected = _select(
        load_runs(runs), lambda r: r.config.benchmark is BenchmarkName.FINANCE_AGENT, "Finance-Agent runs"
    )
    _emit(finance_tables(selected), csv)


@app.command()
@_user_errors
def exploration(
    cyclic: Annotated[Path, typer.Option(exists=True, file_okay=False, help="Graph-controller run.")],
    react: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="ReAct run on the same instances.")
    ],
    csv: CsvPath = None,
) -> None:
    """A6: unique environment states per instance and the cyclic − ReAct difference."""
    ours, theirs = load_run(cyclic), load_run(react)
    _emit([exploration_table(compare_exploration(ours, theirs), [ours, theirs])], csv)


@app.command()
@_user_errors
def wilson(runs: RunDirs, csv: CsvPath = None) -> None:
    """Table 9: pooled wins with Wilson 95% intervals, TC on successes and on all episodes."""
    _emit([wilson_table(load_runs(runs))], csv)


@app.command()
@_user_errors
def figures(
    out: Annotated[Path, typer.Option(file_okay=False, help="Directory for the figure files.")],
    runs: Annotated[
        list[Path] | None, typer.Argument(help="Runs for Figs. 2-3.", exists=True, file_okay=False)
    ] = None,
    cyclic: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False, help="Fig. 4: graph-controller run.")
    ] = None,
    react: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False, help="Fig. 4: ReAct run.")
    ] = None,
) -> None:
    """Fig. 2 (SR by allocation), Fig. 3 (AUS + ASR) and, with --cyclic and --react, Fig. 4 (exploration)."""
    if importlib.util.find_spec("matplotlib") is None:
        typer.echo(
            "Error: figures need matplotlib; install the `plots` extra (`uv sync --extra plots`).", err=True
        )
        raise typer.Exit(1)
    if (cyclic is None) != (react is None):
        raise ValueError("Fig. 4 needs both --cyclic and --react")
    loaded = load_runs(runs or [])
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if any(r.controller.is_cyclic for r in loaded):
        written.append(figs.save(figs.sr_by_allocation(loaded), out / "fig2_sr_by_allocation.png"))
    if any(r.controller is not Controller.REACT for r in loaded):
        written.append(figs.save(figs.visitation_bars(loaded), out / "fig3_visitation.png"))
    if cyclic is not None and react is not None:
        ours, theirs = load_run(cyclic), load_run(react)
        figure = figs.exploration_bars(compare_exploration(ours, theirs), [ours, theirs])
        written.append(figs.save(figure, out / "fig4_exploration.png"))
    if not written:
        raise ValueError("nothing to draw: pass graph-controller runs, or --cyclic and --react")
    for path in written:
        typer.echo(f"Wrote {path}")
