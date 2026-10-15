import random

from ctg.control.faults import inject_fault

IDS = ("a", "b", "c", "d")


def test_a_fault_never_returns_the_routers_choice_and_reaches_every_alternative() -> None:
    rng = random.Random(0)
    results = [inject_fault("b", IDS, 1.0, rng) for _ in range(300)]

    assert all(result.perturbed and result.nominal == "b" for result in results)
    assert {result.actual for result in results} == {"a", "c", "d"}


def test_injection_is_reproducible_with_a_seeded_rng() -> None:
    def draw(seed: int) -> list[tuple[str, bool]]:
        rng = random.Random(seed)
        return [(r.actual, r.perturbed) for r in (inject_fault("a", IDS, 0.5, rng) for _ in range(50))]

    assert draw(7) == draw(7)
    assert draw(7) != draw(8)


def test_zero_probability_never_perturbs_and_leaves_the_rng_untouched() -> None:
    rng = random.Random(3)
    state = rng.getstate()

    results = [inject_fault("c", IDS, 0.0, rng) for _ in range(100)]

    assert all(not result.perturbed and result.actual == "c" for result in results)
    assert rng.getstate() == state


def test_injection_rate_follows_the_probability() -> None:
    rng = random.Random(11)
    perturbed = sum(inject_fault("a", IDS, 0.3, rng).perturbed for _ in range(4000))

    assert 0.27 < perturbed / 4000 < 0.33


def test_a_single_subtask_graph_has_nothing_to_redirect_to() -> None:
    result = inject_fault("a", ("a",), 1.0, random.Random(0))

    assert not result.perturbed
    assert result.actual == "a"
