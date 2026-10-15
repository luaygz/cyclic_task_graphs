"""The `ctg` command line: `run`, `data fetch`, `splits freeze`, `nshot build` and `analyze`."""

from __future__ import annotations

import asyncio
import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import httpx
import typer
from dotenv import load_dotenv
from pydantic import ValidationError

from ctg.analysis.cli import app as analysis_app
from ctg.config import REPO_ROOT, BenchmarkName, ExperimentConfig, load_model_presets
from ctg.envs.base import Benchmark
from ctg.envs.registry import get_benchmark
from ctg.envs.textcraft import recipes
from ctg.llm.base import SpendCapExceededError
from ctg.llm.client import create_llm
from ctg.runs import nshot, splits
from ctg.runs.runner import DryRunReport, RunReport, RunSetupError, run_experiment
from ctg.runs.store import RunStore

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Complete cyclic subtask graphs for tool-using LLM agents (TMLR 2026).",
)
data_app = typer.Typer(no_args_is_help=True, help="Benchmark data downloaded from pinned upstream sources.")
splits_app = typer.Typer(no_args_is_help=True, help="Frozen train/test instance lists in data/splits/.")
nshot_app = typer.Typer(no_args_is_help=True, help="n-shot summaries of successful trajectories.")
app.add_typer(data_app, name="data")
app.add_typer(splits_app, name="splits")
app.add_typer(nshot_app, name="nshot")
app.add_typer(analysis_app, name="analyze")


class Dataset(StrEnum):
    TEXTCRAFT = "textcraft"


@app.callback()
def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    progress = logging.getLogger("ctg.runs")
    if not progress.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        progress.addHandler(handler)
        progress.setLevel(logging.INFO)
        progress.propagate = False


def _error(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)


def _warn(message: str) -> None:
    typer.secho(message, fg=typer.colors.YELLOW)


@app.command()
def run(
    configs: Annotated[
        list[Path], typer.Argument(help="Experiment YAML files.", exists=True, dir_okay=False)
    ],
    runs_dir: Annotated[Path, typer.Option(help="Where run directories are written.")] = Path("runs"),
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate and estimate cost; no model calls, no files written.")
    ] = False,
    limit: Annotated[int | None, typer.Option(min=1, help="Run only the first N pending episodes.")] = None,
    max_cost_usd: Annotated[
        float | None, typer.Option(min=0.0, help="Per-run spend cap in USD (checked before every call).")
    ] = None,
) -> None:
    """Run experiment cells, resuming each from its run directory.

    A spend cap or an unavailable backend stops the sweep; a run that stops for another reason (its Gen-Cyc
    planning failed) is reported and the remaining configs still run. The exit code is 1 if any config
    could not start or stopped.
    """
    failed = False
    estimated_usd = 0.0
    estimated_episodes = 0
    for index, path in enumerate(configs):
        try:
            config = ExperimentConfig.load(path)
        except ValidationError as exc:
            _error(f"{path}: invalid experiment config\n{exc}")
            failed = True
            continue
        if dry_run:
            report = asyncio.run(run_experiment(config, runs_dir, dry_run=True, limit=limit))
            _print_dry_run(report)
            failed |= bool(report.problems)
            if report.estimate is not None:
                estimated_usd += report.estimate.usd
                estimated_episodes += report.estimate.episodes
            continue
        try:
            report = asyncio.run(run_experiment(config, runs_dir, limit=limit, max_cost_usd=max_cost_usd))
        except RunSetupError as exc:
            _error(f"{config.name}: cannot start\n{exc}")
            failed = True
            continue
        _print_run(report)
        if report.stop_reason is not None:
            failed = True
            if report.halts_sweep:
                remaining = [str(p) for p in configs[index + 1 :]]
                if remaining:
                    _error(f"Not starting the remaining configs: {', '.join(remaining)}")
                raise typer.Exit(1)
    if dry_run and len(configs) > 1:
        typer.echo(f"\nAll configs: {estimated_episodes} episodes to run, roughly ${estimated_usd:,.2f}")
    if failed:
        raise typer.Exit(1)


def _print_dry_run(report: DryRunReport) -> None:
    typer.secho(f"{report.name}  ({report.run_dir})", bold=True)
    for note in report.notes:
        typer.echo(f"  {note}")
    if report.total:
        typer.echo(
            f"  episodes: {report.total} total, {report.completed} complete, {report.scheduled} to run"
        )
    if report.estimate is not None:
        line = (
            f"  rough cost: ~${report.estimate.usd:,.2f} (order of magnitude, from the paper's token averages"
        )
        if report.estimate.unpriced:
            line += f"; excludes {', '.join(report.estimate.unpriced)}: no preset prices"
        typer.echo(line + ")")
    for problem in report.problems:
        _error(f"  problem: {problem}")


def _print_run(report: RunReport) -> None:
    typer.secho(
        f"{report.name}: {report.finished}/{report.scheduled} episodes finished "
        f"({report.won} won, {report.errors} errors), ${report.spent_usd:,.4f} spent",
        bold=True,
    )
    complete = report.completed_before + report.finished - report.errors
    typer.echo(f"  {complete}/{report.total} episodes of the experiment are complete ({report.run_dir})")
    if report.output_failures:
        _warn(
            f"  {report.output_failures} episodes ended with model output that stayed unusable; they are "
            "recorded as losses and not rerun"
        )
    if report.errors:
        _warn(
            f"  {report.errors} errored episodes are recorded with tracebacks and rerun on the next `ctg run`"
        )
    if report.stop_reason is not None:
        _error(f"  STOPPED: {report.stop_reason}")
        _error(
            f"  {report.not_started} episodes were not started. Fix the cause and rerun the same command to "
            "resume (spend caps: --max-cost-usd or CTG_SPEND_CAP_*_USD)."
        )


@splits_app.command("freeze")
def splits_freeze(
    variants: Annotated[
        list[str] | None,
        typer.Argument(help=f"Variants to freeze (default: all of {', '.join(splits.VARIANTS)})."),
    ] = None,
) -> None:
    """Enumerate instances and write data/splits/<variant>.json (seed-42 shuffle, Table 4 sizes)."""
    chosen = variants or list(splits.VARIANTS)
    unknown = [variant for variant in chosen if variant not in splits.VARIANTS]
    if unknown:
        raise typer.BadParameter(f"unknown variants {unknown}; choose from {list(splits.VARIANTS)}")
    benchmarks: dict[BenchmarkName, Benchmark] = {}
    for variant in chosen:
        name = splits.VARIANTS[variant]
        if name not in benchmarks:
            benchmarks[name] = get_benchmark(name)
        frozen = splits.freeze(benchmarks[name], variant)
        total = len(frozen.instances)
        typer.echo(
            f"{variant}: {total} instances ({frozen.count('train')} train / {frozen.count('test')} test)"
            f" -> {splits.split_path(variant)}"
        )
        expected = splits.PAPER_TOTALS[variant]
        if total != expected:
            _warn(f"  the paper (Table 4) lists {expected} instances for {variant}")


@nshot_app.command("build")
def nshot_build(
    run_dir: Annotated[
        Path, typer.Argument(help="Run directory of a summary-construction run.", file_okay=False)
    ],
    out: Annotated[Path, typer.Option("--out", help="Output file, e.g. data/nshot/textcraft_depth2.json.")],
    model: Annotated[
        str | None, typer.Option(help="Model preset for the summaries (default: the run's nshot model).")
    ] = None,
) -> None:
    """Summarize every won episode of a run into an n-shot file; nothing is written unless all succeed."""
    try:
        manifest = RunStore(run_dir).manifest()
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="RUN_DIR") from exc
    if manifest is None:
        raise typer.BadParameter(f"{run_dir} has no manifest.json", param_hint="RUN_DIR")
    preset = model or manifest.config.models.nshot
    presets = load_model_presets()
    if preset not in presets:
        raise typer.BadParameter(f"unknown model preset {preset!r}; choose from {sorted(presets)}")
    llm = create_llm(run=f"{manifest.name}-nshot")

    async def build() -> nshot.NShotFile:
        try:
            return await nshot.build(run_dir, out, presets[preset], llm)
        finally:
            await llm.aclose()

    result: nshot.NShotFile | None = None
    try:
        result = asyncio.run(build())
    except* SpendCapExceededError as group:
        _error(f"Spend cap reached; no n-shot file written: {_messages(group)}")
    except* Exception as group:
        _error(f"n-shot build failed; no n-shot file written: {_messages(group)}")
    if result is None:
        raise typer.Exit(1)
    typer.echo(f"{len(result.summaries)} summaries of {manifest.name} ({presets[preset].name}) -> {out}")


def _messages(group: BaseExceptionGroup[Exception]) -> str:
    return "; ".join(f"{type(exc).__name__}: {exc}" for exc in group.exceptions)


@data_app.command("fetch")
def data_fetch(dataset: Annotated[Dataset, typer.Argument(help="Dataset to download.")]) -> None:
    """Download a dataset's pinned files and verify their SHA-256 digests; files already in place are kept."""

    async def fetch() -> recipes.FetchSummary:
        async with httpx.AsyncClient(timeout=recipes.FETCH_TIMEOUT_S) as http:
            return await recipes.fetch_recipes(http)

    try:
        summary = asyncio.run(fetch())
    except recipes.RecipeFetchError as exc:
        _error(f"{dataset}: download failed: {exc}")
        raise typer.Exit(1) from exc
    typer.echo(
        f"{dataset}: {summary.downloaded} files downloaded, {summary.present} already present"
        f" -> {summary.directory}"
    )
