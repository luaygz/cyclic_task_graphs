"""Figures 2-4. matplotlib is an optional dependency (the `plots` extra), imported only when drawing.

Each figure carries the error and coverage notes of the runs it shows below its panels, so a figure of an
unfinished run says so.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ctg.analysis.exploration import ExplorationComparison
from ctg.analysis.load import Run, run_notes
from ctg.analysis.metrics import MeanStd, Visitation, run_metrics
from ctg.analysis.report import CONTROLLER_LABELS, run_order, variant_label, variant_order
from ctg.config import Controller

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

MODEL_ABBREVIATIONS = {"gpt-4o-mini": "4o", "gpt-5-mini": "5m"}
"""Short preset names for allocation labels, as in Fig. 2 (5m = gpt-5-mini, 4o = gpt-4o-mini)."""

DPI = 200
NOTE_FONT_SIZE = 6
NOTE_CHARS_PER_INCH = 18
"""Characters per line of figure width for the notes, at NOTE_FONT_SIZE."""


def allocation_label(run: Run) -> str:
    """Planner/executor/router model presets, abbreviated."""
    models = run.config.models
    return "/".join(MODEL_ABBREVIATIONS.get(m, m) for m in (models.planner, models.executor, models.router))


def save(figure: Figure, path: Path) -> Path:
    """Write `figure` to `path`, notes below the panels included."""
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    return path


def _shared_legend(figure: Figure, axes: Iterable[Axes]) -> None:
    """One legend above all panels, with every series that appears in any of them."""
    entries: dict[str, Artist] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            entries.setdefault(label, handle)
    figure.legend(
        list(entries.values()), list(entries), loc="outside upper center", ncols=len(entries), fontsize=8
    )


def _run_notes(figure: Figure, runs: Sequence[Run]) -> None:
    """The runs' error and coverage notes, wrapped to the figure width, below the panels."""
    notes = run_notes(runs)
    if not notes:
        return
    width = max(40, round(figure.get_figwidth() * NOTE_CHARS_PER_INCH))
    text = "\n".join(textwrap.fill(note, width) for note in notes)
    figure.text(0.0, 0.0, text, ha="left", va="top", fontsize=NOTE_FONT_SIZE)


def sr_by_allocation(runs: Sequence[Run]) -> Figure:
    """Fig. 2: SR mean ± std per role allocation, Spec-Cyc vs Gen-Cyc, one panel per benchmark.

    A run without a complete repeat shows its pooled SR without an error bar.
    """
    from matplotlib.figure import Figure

    points: dict[str, dict[tuple[Controller, str], MeanStd]] = {}
    owners: dict[tuple[str, Controller, str], str] = {}
    shown: list[Run] = []
    for run in runs:
        if not run.controller.is_cyclic:
            continue
        key = (run.variant, run.controller, allocation_label(run))
        if key in owners:
            raise ValueError(f"{owners[key]} and {run.name} share benchmark, graph and allocation {key[2]}")
        owners[key] = run.name
        sr = run_metrics(run).sr
        if sr is not None:
            points.setdefault(run.variant, {})[(run.controller, key[2])] = sr
            shown.append(run)
    if not points:
        raise ValueError("Fig. 2 needs Spec-Cyc or Gen-Cyc runs with evaluated episodes")
    variants = sorted(points, key=variant_order)
    figure = Figure(figsize=(3.2 * len(variants), 3.4), layout="constrained")
    axes = figure.subplots(1, len(variants), squeeze=False)[0]
    for ax, variant in zip(axes, variants, strict=True):
        allocations = sorted({allocation for _, allocation in points[variant]})
        for offset, controller in ((-0.12, Controller.SPEC_CYC), (0.12, Controller.GEN_CYC)):
            present = [
                (i + offset, sr)
                for i, a in enumerate(allocations)
                if (sr := points[variant].get((controller, a)))
            ]
            if present:
                ax.errorbar(
                    [sr.mean for _, sr in present],
                    [y for y, _ in present],
                    xerr=[math.nan if sr.std is None else sr.std for _, sr in present],
                    fmt="o",
                    capsize=3,
                    label=CONTROLLER_LABELS[controller],
                )
        ax.set_yticks(range(len(allocations)), allocations)
        ax.set_xlim(0, 100)
        ax.set_xlabel("SR (%)")
        ax.set_title(variant_label(variant))
    axes[0].set_ylabel("Planner/executor/router")
    _shared_legend(figure, axes)
    _run_notes(figure, shown)
    return figure


def visitation_bars(runs: Sequence[Run]) -> Figure:
    """Fig. 3: AUS stacked with ASR (together ATS) on successful episodes, one bar per run."""
    from matplotlib.figure import Figure

    bars: dict[str, list[tuple[str, Visitation]]] = {}
    shown: list[Run] = []
    for run in sorted(runs, key=run_order):
        visits = run_metrics(run).visitation
        if visits is not None:
            label = f"{CONTROLLER_LABELS[run.controller]}\n{allocation_label(run)}"
            bars.setdefault(run.variant, []).append((label, visits))
            shown.append(run)
    if not bars:
        raise ValueError("Fig. 3 needs graph runs with successful episodes")
    variants = sorted(bars, key=variant_order)
    widest = max(len(group) for group in bars.values())
    figure = Figure(figsize=(max(3.0, 0.9 * widest) * len(variants), 3.6), layout="constrained")
    axes = figure.subplots(1, len(variants), squeeze=False, sharey=True)[0]
    for ax, variant in zip(axes, variants, strict=True):
        group = bars[variant]
        x = list(range(len(group)))
        aus = [v.aus.mean for _, v in group]
        ax.bar(x, aus, label="AUS (unique subtasks)")
        ax.bar(x, [v.asr.mean for _, v in group], bottom=aus, label="ASR (revisits)")
        ax.set_xticks(x, [label for label, _ in group], fontsize=7)
        ax.set_title(variant_label(variant))
    axes[0].set_ylabel("Visits per successful episode")
    _shared_legend(figure, axes)
    _run_notes(figure, shown)
    return figure


def exploration_bars(comparison: ExplorationComparison, runs: Sequence[Run]) -> Figure:
    """Fig. 4: per-instance difference in unique environment states (cyclic minus ReAct) of the two `runs`."""
    from matplotlib.figure import Figure

    if not comparison.instances:
        raise ValueError("Fig. 4 needs instances evaluated by both runs")
    differences = [i.difference for i in comparison.instances]
    figure = Figure(figsize=(max(4.0, 0.22 * len(differences)), 3.2), layout="constrained")
    ax = figure.subplots()
    ax.bar(
        range(1, len(differences) + 1),
        differences,
        color=["tab:blue" if d >= 0 else "tab:red" for d in differences],
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Task")
    ax.set_ylabel("Δ unique states (cyclic − ReAct)")
    _run_notes(figure, runs)
    return figure
