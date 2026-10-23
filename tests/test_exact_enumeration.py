"""Complete micro fronts establish real conflict without solver assumptions."""

import pytest

from timetable.evaluation import evaluate
from timetable.exact import enumerate_pareto
from timetable.model import Faculty, Group, Offering, Problem, Room


def conflict_problem(which="room"):
    rooms = (
        (Room("small", 20, "theory", (1, 3)), Room("large", 40, "theory", (0, 2)))
        if which == "room" else (Room("R", 20, "theory"),)
    )
    faculty = (
        (Faculty("A", 2, 2), Faculty("B", 2, 2))
        if which == "room" else (Faculty("A", 2, 2, (2, 3)), Faculty("B", 2, 2, (0, 1, 2)))
    )
    return Problem(
        name=f"exact-{which}-conflict", days=("Mon",), periods_per_day=4,
        lunch_periods=(), groups=(Group("G", 20),), rooms=rooms,
        faculty=faculty,
        offerings=tuple(
            Offering(subject, ("G",), ("A", "B"), 1, 1, "theory")
            for subject in ("X", "Y")
        ),
    )


def test_complete_room_vs_gaps_front():
    problem = conflict_problem("room")
    result = enumerate_pareto(problem)
    assert result.status == "COMPLETE"
    assert result.objectives == [(0.0, 0.25, 0.0), (1.0, 0.0, 0.0)]
    assert result.evaluated == 64
    assert result.feasible_candidates > len(result.archive)
    assert all(evaluate(problem, s).feasible for s in result.archive)


def test_complete_workload_vs_gaps_front():
    result = enumerate_pareto(conflict_problem("workload"))
    assert result.objectives == [(0.0, 0.0, 0.5), (1.0, 0.0, 0.0)]
    assert result.evaluated == 9


def test_enumeration_refuses_partial_exactness_claim():
    with pytest.raises(ValueError, match="max_combinations"):
        enumerate_pareto(conflict_problem(), max_combinations=10)
    with pytest.raises(ValueError, match="max_combinations"):
        enumerate_pareto(conflict_problem(), max_combinations=0)
