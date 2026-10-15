import pytest
from control_fakes import cyclic_graph, dependency_graph

from ctg.control.policies import candidates, is_forced_end, transition_type
from ctg.graphs.model import END
from ctg.runs.records import TransitionType


def test_cyclic_candidates_are_every_subtask_including_the_current_one() -> None:
    graph = cyclic_graph(("a", "b", "c"))

    assert candidates(graph, "b") == ("a", "b", "c")
    assert not is_forced_end(candidates(graph, "c"))


def test_cyclic_transitions_are_self_or_cross() -> None:
    graph = cyclic_graph(("a", "b"))

    assert transition_type(graph, "a", "a") is TransitionType.SELF
    assert transition_type(graph, "a", "b") is TransitionType.CROSS
    with pytest.raises(ValueError):
        transition_type(graph, "a", END)


@pytest.mark.parametrize(
    ("retry", "current", "expected"),
    [
        (True, "a", ("b", "a", END)),
        (True, "c", ("c", END)),
        (False, "a", ("b", END)),
        (False, "b", ("c", END)),
        (False, "c", (END,)),
    ],
)
def test_dependency_candidates(retry: bool, current: str, expected: tuple[str, ...]) -> None:
    assert candidates(dependency_graph(("a", "b", "c"), retry=retry), current) == expected


def test_only_a_no_retry_sink_forces_termination() -> None:
    no_retry = dependency_graph(("a", "b"), retry=False)
    with_retry = dependency_graph(("a", "b"), retry=True)

    assert is_forced_end(candidates(no_retry, "b"))
    assert not is_forced_end(candidates(no_retry, "a"))
    assert not is_forced_end(candidates(with_retry, "b"))


def test_dependency_transitions_are_retry_forward_or_terminal() -> None:
    graph = dependency_graph(("a", "b", "c"))

    assert transition_type(graph, "b", "b") is TransitionType.RETRY
    assert transition_type(graph, "b", "c") is TransitionType.FORWARD
    assert transition_type(graph, "b", END) is TransitionType.TERMINAL
