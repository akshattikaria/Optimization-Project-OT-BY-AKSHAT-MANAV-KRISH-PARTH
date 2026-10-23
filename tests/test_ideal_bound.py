"""Independently compare the small load-partition relaxation against brute force."""
import importlib.util
import itertools
from pathlib import Path

import pytest

from timetable.model import Faculty, Group, Offering, Problem, Room, Placement, Schedule

spec = importlib.util.spec_from_file_location("ideal_reference", Path(__file__).parents[1] / "experiments" / "certify_small_ideal.py")
ideal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ideal)


@pytest.mark.parametrize("weights,people,limit", [([2, 3, 3], 2, 6), ([1, 1, 2, 3], 3, 5), ([2, 2, 3, 3], 3, 6)])
def test_integer_dp_matches_full_assignment_oracle(weights, people, limit):
    problem = Problem("bound", ("M", "T", "W"), 8, (), (Group("G", 20),),
                      (Room("R", 20, "classroom"),), tuple(Faculty(str(i), limit, limit) for i in range(people)),
                      tuple(Offering(str(i), ("G",), tuple(str(i) for i in range(people)), w, 1, "classroom", False)
                            for i, w in enumerate(weights)))
    candidates = []
    for assignment in itertools.product(range(people), repeat=len(weights)):
        loads = [sum(w for w, person in zip(weights, assignment) if person == f) for f in range(people)]
        if max(loads) <= limit:
            candidates.append(sum(load*load for load in loads))
    assert ideal.objective_lower_bounds(problem)["relaxed_minimum_sum_squared_loads"] == min(candidates)


def test_exact_attainment_and_unequal_allowances_rejected():
    problem = Problem("ideal", ("M",), 4, (), (Group("G", 20),), (Room("R", 20, "classroom"),),
                      (Faculty("A", 2, 2), Faculty("B", 2, 2)),
                      tuple(Offering(k, ("G",), ("A", "B"), 1, 1, "classroom") for k in ["X", "Y"]))
    bounds = ideal.objective_lower_bounds(problem)
    balanced = Schedule((Placement(0, "R"), Placement(1, "R")), ("A", "B"))
    assert ideal.exact_attainment(problem, balanced, bounds)["exact_sum_squared_loads"] == 2
    unequal = Schedule(balanced.placements, ("A", "A"))
    assert ideal.exact_attainment(problem, unequal, bounds) is None
    from dataclasses import replace
    with pytest.raises(ValueError, match="identical"):
        ideal.objective_lower_bounds(replace(problem, faculty=(Faculty("A", 2, 2), Faculty("B", 3, 2))))
