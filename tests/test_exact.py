"""Small reference problems where CP-SAT can finish an actual proof."""

from dataclasses import replace

import pytest

pytest.importorskip("ortools")

from timetable.evaluation import evaluate
from timetable.exact import cp_sat
from timetable.model import Faculty, Group, Offering, Problem, Room


def reference_problem() -> Problem:
    return Problem(
        name="exact-reference",
        days=("Mon", "Tue"),
        periods_per_day=6,
        lunch_periods=(2,),
        groups=(Group("G", 20),),
        rooms=(Room("small", 30, "theory"), Room("large", 50, "theory")),
        faculty=(Faculty("A", 5, 3), Faculty("B", 5, 3)),
        offerings=(Offering("M", ("G",), ("A", "B"), 2, 1, "theory"),),
    )


def test_reference_feasibility_and_consistent_teacher():
    problem = reference_problem()
    result = cp_sat(problem, time_limit=5)
    assert result.status == "OPTIMAL"
    assert result.proven_optimal
    assert result.schedule is not None
    assert evaluate(problem, result.schedule).feasible
    assert len(result.schedule.teachers) == 1
    assert result.schedule.teachers[0] in {"A", "B"}
    assert result.runtime >= result.metadata["solver_wall_time"]


def test_exact_room_objective_agrees_with_evaluator_and_known_optimum():
    problem = reference_problem()
    result = cp_sat(problem, objective="room", time_limit=5)
    assert result.status == "OPTIMAL"
    assert result.schedule is not None
    assert all(p.room == "small" for p in result.schedule.placements)
    assert result.objective_value == pytest.approx(1 / 3)
    assert result.best_bound == pytest.approx(result.objective_value)
    assert evaluate(problem, result.schedule).objectives[1] == pytest.approx(1 / 3)


def test_exact_gaps_exclude_lunch_but_count_free_teaching_periods():
    problem = Problem(
        name="forced-gaps",
        days=("Mon",),
        periods_per_day=7,
        lunch_periods=(3,),
        groups=(Group("G", 20),),
        rooms=(Room("R", 30, "theory", (1, 2, 4, 6)),),
        faculty=(Faculty("A", 4, 4),),
        offerings=(Offering("M", ("G",), ("A",), 2, 1, "theory", False),),
    )
    result = cp_sat(problem, objective="gaps", time_limit=5)
    assert result.status == "OPTIMAL"
    assert {p.start for p in result.schedule.placements} == {0, 5}
    # Teaching-period gaps 1, 2, 4; the fixed lunch period 3 is excluded.
    assert result.objective_value == pytest.approx(3 + 0.5 * 3)
    assert result.best_bound == pytest.approx(4.5)
    assert evaluate(problem, result.schedule).objectives[0] == pytest.approx(4.5)


def test_exact_multi_period_shared_lab_and_limits():
    problem = Problem(
        name="shared-lab",
        days=("Mon", "Tue"),
        periods_per_day=6,
        lunch_periods=(2,),
        groups=(Group("G1", 20), Group("G2", 15)),
        rooms=(Room("L", 40, "lab"), Room("T", 80, "theory")),
        faculty=(
            Faculty("A", 4, 2, (0, 1, 6, 7)),
            Faculty("B", 4, 2, (3, 4, 5, 9, 10, 11)),
        ),
        offerings=(Offering("Lab", ("G1", "G2"), ("A", "B"), 2, 2, "lab"),),
    )
    result = cp_sat(problem, objective="gaps", time_limit=5)
    assert result.status == "OPTIMAL"
    assert result.schedule is not None
    assert evaluate(problem, result.schedule).feasible
    assert result.objective_value == 0
    assert all(p.room == "L" for p in result.schedule.placements)


@pytest.mark.parametrize("failure", ["capacity", "weekly", "distinct_days", "eligibility"])
def test_infeasibility_is_solver_proven(failure):
    problem = reference_problem()
    if failure == "capacity":
        problem = replace(problem, rooms=(Room("too-small", 10, "theory"),))
    elif failure == "weekly":
        problem = replace(problem, faculty=(Faculty("A", 1, 1), Faculty("B", 1, 1)))
    elif failure == "distinct_days":
        problem = replace(problem, days=("Mon",))
    else:
        problem = replace(problem, offerings=(replace(problem.offerings[0], eligible_faculty=()),))
    result = cp_sat(problem, time_limit=5)
    assert result.status == "INFEASIBLE"
    assert result.schedule is None
    assert result.objective_value is None
    assert result.best_bound is None
    assert not result.proven_optimal


def test_time_limit_can_return_unknown_without_claiming_infeasibility():
    result = cp_sat(reference_problem(), time_limit=1e-9)
    assert result.status == "UNKNOWN"
    assert result.schedule is None
    assert not result.proven_optimal


@pytest.mark.parametrize("kwargs", [{"objective": "balance"}, {"time_limit": 0}, {"time_limit": float("nan")}, {"seed": -1}])
def test_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        cp_sat(reference_problem(), **kwargs)


def test_refuses_unsafe_exact_room_scaling():
    problem = replace(
        reference_problem(),
        rooms=tuple(
            Room(str(capacity), capacity, "theory")
            for capacity in (1_000_000_007, 1_000_000_009, 1_000_000_033)
        ),
    )
    with pytest.raises(ValueError, match="integer scale"):
        cp_sat(problem, objective="room")
