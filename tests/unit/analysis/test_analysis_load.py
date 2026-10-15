from __future__ import annotations

from pathlib import Path

from analysis_fixtures import (
    config,
    episode,
    error_episode,
    interrupted_segment,
    output_failure,
    split_ids,
    tool_call,
    write_run,
)

from ctg.analysis.load import load_run, run_notes
from ctg.config import Controller


def test_load_run_schedules_repeats_times_the_configured_instances(tmp_path: Path) -> None:
    ids = split_ids("textcraft_depth2", 3)
    cfg = config(name="gen", controller=Controller.GEN_CYC, instances=ids, repeats=2)
    path = write_run(
        tmp_path,
        cfg,
        [
            episode(cfg, ids[0], 0, won=True),
            output_failure(cfg, ids[1], 0, segments=[interrupted_segment(0, "A", calls=[tool_call()])]),
            episode(cfg, ids[2], 0),
            episode(cfg, ids[0], 1),
            error_episode(cfg, ids[1], 1),
        ],
    )
    run = load_run(path)

    assert run.expected == {(r, i) for r in (0, 1) for i in ids}
    # the ERROR episode will be re-run, the output failure is a finished loss
    assert run.missing == {(1, ids[1]), (1, ids[2])}
    assert (run.finished, run.complete_repeats, run.incomplete_repeats) == (4, (0,), (1,))
    assert run_notes([run]) == (
        "Error episodes excluded from all rates: gen (1).",
        "Incomplete runs (scheduled episodes without a non-error record): "
        "gen (4/6 episodes; repeat 1 incomplete).",
    )


def test_load_run_without_pinned_instances_schedules_the_whole_split(tmp_path: Path) -> None:
    cfg = config(name="react", controller=Controller.REACT)
    run = load_run(write_run(tmp_path, cfg, []))
    assert len(run.expected) == 3 * 203  # Table 4: 203 TextCraft-2 test instances, 3 repeats
    assert run.complete_repeats == ()
