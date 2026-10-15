"""Control-flow fault injection for cyclic graphs (Sec. 4.3)."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RoutedChoice:
    nominal: str
    """The router's choice."""
    actual: str
    """The node the controller moves to."""
    perturbed: bool


def inject_fault(
    choice: str, subtasks: Sequence[str], probability: float, rng: random.Random
) -> RoutedChoice:
    """With `probability`, redirect to a uniformly random subtask other than the router's `choice`.

    The rng is consumed only when injection is on and an alternative exists, so p = 0 leaves it untouched.
    """
    alternatives = [subtask for subtask in subtasks if subtask != choice]
    if probability > 0 and alternatives and rng.random() < probability:
        return RoutedChoice(nominal=choice, actual=rng.choice(alternatives), perturbed=True)
    return RoutedChoice(nominal=choice, actual=choice, perturbed=False)
