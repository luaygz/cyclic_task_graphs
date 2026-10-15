from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from analysis_fixtures import config, episode, make_run, path_segments

from ctg.analysis import figures
from ctg.analysis.exploration import compare_exploration
from ctg.analysis.load import Run
from ctg.config import BenchmarkName, Controller, ModelAssignment

if TYPE_CHECKING:
    from matplotlib.figure import Figure

pytest.importorskip("matplotlib")


def _notes(figure: Figure) -> str:
    return " ".join(" ".join(text.get_text().split()) for text in figure.texts)


def _run(name: str, controller: Controller, done: dict[int, str]) -> Run:
    """Won episodes of the instances under each repeat in `done`; instances a, b × 2 repeats scheduled."""
    cfg = config(name=name, controller=controller, models=ModelAssignment(executor="gpt-5-mini"))
    episodes = [
        episode(cfg, i, r, won=True, tool_calls=3, segments=path_segments(["A", "B"]))
        for r, ids in done.items()
        for i in ids
    ]
    return make_run(cfg, episodes, expected={(r, i) for r in range(2) for i in "ab"})


def test_figures_name_incomplete_runs_below_the_panels(tmp_path: Path) -> None:
    gen = _run("gen", Controller.GEN_CYC, {0: "ab", 1: "a"})
    spec = _run("spec", Controller.SPEC_CYC, {0: "a"})  # no complete repeat: pooled SR, no error bar

    fig2 = figures.sr_by_allocation([gen, spec])
    fig3 = figures.visitation_bars([gen, spec])

    for figure in (fig2, fig3):
        notes = _notes(figure)
        assert "Incomplete runs" in notes
        assert "gen (3/4 episodes; repeat 1 incomplete)" in notes
        assert "spec (1/4 episodes; repeats 0, 1 incomplete)" in notes
    assert figures.save(fig2, tmp_path / "fig2.png").stat().st_size > 0


def test_figures_of_complete_runs_carry_no_notes() -> None:
    gen = _run("gen", Controller.GEN_CYC, {0: "ab", 1: "ab"})
    assert figures.sr_by_allocation([gen]).texts == []


def test_exploration_figure_names_an_incomplete_run(tmp_path: Path) -> None:
    alfworld = BenchmarkName.ALFWORLD
    cyclic_cfg = config(name="cyclic", controller=Controller.GEN_CYC, benchmark=alfworld)
    react_cfg = config(name="react", controller=Controller.REACT, benchmark=alfworld)
    cyclic = make_run(cyclic_cfg, [episode(cyclic_cfg, "g1", states=[{"room": 1}])])
    react = make_run(react_cfg, [episode(react_cfg, "g1")], expected={(0, "g1"), (0, "g2")})

    figure = figures.exploration_bars(compare_exploration(cyclic, react), [cyclic, react])

    assert "react (1/2 episodes; repeat 0 incomplete)" in _notes(figure)
    assert "cyclic" not in _notes(figure)
    assert figures.save(figure, tmp_path / "fig4.png").stat().st_size > 0
