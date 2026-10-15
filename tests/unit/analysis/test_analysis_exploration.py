from __future__ import annotations

import pytest
from analysis_fixtures import config, episode, error_episode, make_run

from ctg.analysis.exploration import compare_exploration, unique_states
from ctg.config import BenchmarkName, Controller
from ctg.runs.records import EpisodeRecord

ALFWORLD = BenchmarkName.ALFWORLD
CYCLIC = config(name="cyclic", controller=Controller.GEN_CYC, benchmark=ALFWORLD)
REACT = config(name="react", controller=Controller.REACT, benchmark=ALFWORLD)


def visiting(cfg_name: str, instance: str, repeat: int, distinct: int) -> EpisodeRecord:
    """An episode whose initial state and steps cover exactly `distinct` states (with repeats)."""
    cfg = CYCLIC if cfg_name == "cyclic" else REACT
    states = [{"room": k} for k in range(1, distinct)] + [{"room": 0}]
    return episode(cfg, instance, repeat, initial_state={"room": 0}, states=states)


def test_unique_states_count_canonical_snapshots_including_the_initial_state() -> None:
    record = episode(
        CYCLIC,
        "a",
        initial_state={"location": "kitchen", "inventory": []},
        states=[
            {"location": "hall", "inventory": []},
            {"inventory": [], "location": "kitchen"},  # the initial state, keys reordered
            {"location": "hall", "inventory": ["apple"]},
            {"inventory": ["apple"], "location": "hall"},
        ],
    )
    assert unique_states(record) == 3


def test_episode_without_steps_has_one_state() -> None:
    assert unique_states(episode(CYCLIC, "a", initial_state={"x": 1})) == 1


def test_per_instance_means_and_differences_over_shared_instances() -> None:
    cyclic = make_run(
        CYCLIC,
        [
            visiting("cyclic", "i1", 0, 3),
            visiting("cyclic", "i1", 1, 5),
            visiting("cyclic", "i2", 0, 2),
            visiting("cyclic", "i3", 0, 9),  # ReAct never ran i3
        ],
    )
    react = make_run(
        REACT,
        [
            visiting("react", "i1", 0, 2),
            visiting("react", "i1", 1, 2),
            visiting("react", "i2", 0, 4),
            error_episode(REACT, "i2", 1, initial_state={"room": 0}, states=[{"room": k} for k in range(50)]),
            visiting("react", "i4", 0, 7),
        ],
    )
    comparison = compare_exploration(cyclic, react)

    assert [(i.instance_id, i.cyclic, i.react, i.difference) for i in comparison.instances] == [
        ("i1", 4.0, 2.0, 2.0),
        ("i2", 2.0, 4.0, -2.0),
    ]
    assert comparison.unmatched == 2
    assert comparison.mean_difference == pytest.approx(0.0)


def test_exploration_compares_one_benchmark() -> None:
    textcraft = config(name="tc", controller=Controller.REACT)
    with pytest.raises(ValueError, match="different benchmark"):
        compare_exploration(make_run(CYCLIC, []), make_run(textcraft, []))


def test_exploration_refuses_swapped_or_misassigned_runs() -> None:
    cyclic, react = (
        make_run(CYCLIC, [visiting("cyclic", "i1", 0, 3)]),
        make_run(REACT, [visiting("react", "i1", 0, 2)]),
    )
    with pytest.raises(ValueError, match="ReAct side needs a ReAct run"):
        compare_exploration(react, cyclic)
    with pytest.raises(ValueError, match="ReAct side needs a ReAct run"):
        compare_exploration(cyclic, cyclic)
    with pytest.raises(ValueError, match="cyclic side needs a graph-controller run"):
        compare_exploration(react, react)
