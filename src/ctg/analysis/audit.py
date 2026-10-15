"""DepDAG transition audit (Appendix D.2, Table 14): checks that the dependency controller never took a
transition outside its allowed set Succ(i) ∪ {i if retry} ∪ {⊥}."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ctg.analysis.load import Run
from ctg.runs.records import TransitionRecord, TransitionType

DEPENDENCY_TYPES = (TransitionType.RETRY, TransitionType.FORWARD, TransitionType.TERMINAL)


@dataclass(frozen=True)
class TransitionAudit:
    runs: int
    test_cases: int
    """Non-error episodes inspected."""
    transitions: int
    retry: int
    forward: int
    terminal: int
    other: int
    """Transitions typed as anything but retry, forward or terminal."""
    invalid: int
    """Transitions flagged invalid or whose target is not in their allowed set."""


def is_invalid(transition: TransitionRecord) -> bool:
    return not transition.valid or transition.target not in transition.allowed


def audit_transitions(runs: Sequence[Run]) -> TransitionAudit:
    if not runs:
        raise ValueError("the audit needs at least one DepDAG run")
    if offending := [r.name for r in runs if not r.controller.is_dependency]:
        raise ValueError(f"the transition audit covers DepDAG runs only; not DepDAG: {', '.join(offending)}")
    transitions = [
        s.transition for run in runs for e in run.valid for s in e.segments if s.transition is not None
    ]
    count = {t: sum(1 for x in transitions if x.type is t) for t in DEPENDENCY_TYPES}
    return TransitionAudit(
        runs=len(runs),
        test_cases=sum(len(r.valid) for r in runs),
        transitions=len(transitions),
        retry=count[TransitionType.RETRY],
        forward=count[TransitionType.FORWARD],
        terminal=count[TransitionType.TERMINAL],
        other=len(transitions) - sum(count.values()),
        invalid=sum(1 for t in transitions if is_invalid(t)),
    )
