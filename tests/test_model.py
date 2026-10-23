"""Hand-computed fixtures exercise constraints independently of construction."""

import json
from dataclasses import replace
from math import isclose
from itertools import permutations

import pytest

from timetable.evaluation import evaluate
from timetable.model import (
    Faculty, Group, Offering, Placement, Problem, Room, Schedule,
    compatible_rooms, problem_from_dict, problem_to_dict,
    schedule_from_dict, schedule_to_dict, valid_starts, validate_problem,
)


@pytest.fixture
def hand_problem():
    return Problem(
        "hand-computed", ("Mon", "Tue"), 6, (2,),
        (Group("G1", 20), Group("G2", 10)),
        (Room("R", 30, "lecture"), Room("L", 40, "lab")),
        (Faculty("F1", 10, 4), Faculty("F2", 10, 4)),
        (
            Offering("A", ("G1",), ("F1", "F2"), 2, 1, "lecture"),
            Offering("B", ("G1", "G2"), ("F1", "F2"), 1, 2, "lab"),
            Offering("C", ("G2",), ("F1", "F2"), 1, 1, "lecture"),
        ),
    )


@pytest.fixture
def hand_schedule():
    return Schedule(
        (Placement(0, "R"), Placement(6, "R"), Placement(3, "L"), Placement(5, "R")),
        ("F1", "F2", "F1"),
    )


def test_hand_computed_objectives(hand_problem, hand_schedule):
    result = evaluate(hand_problem, hand_schedule)
    assert result.feasible and result.violation == 0
    assert result.objectives == pytest.approx((2.5, 11 / 30, 0.05))
    assert result.metrics["student_gaps"] == 1
    assert result.metrics["faculty_gaps"] == 3
    assert result.metrics["faculty_loads"] == {"F1": 3, "F2": 2}
    assert result.metrics["required_teaching_periods"] == 5


def test_workload_changes_with_real_offering_teacher_decision(hand_problem, hand_schedule):
    unbalanced = replace(hand_schedule, teachers=("F1", "F1", "F1"))
    result = evaluate(hand_problem, unbalanced)
    assert result.feasible
    assert isclose(result.objectives[2], 0.25)
    assert result.objectives[2] > evaluate(hand_problem, hand_schedule).objectives[2]


def test_workload_objective_is_exactly_invariant_under_load_permutation():
    faculty = tuple(Faculty(f"F{i}", 20, 7) for i in range(6))
    problem = Problem(
        "load-permutation", ("Mon", "Tue", "Wed", "Thu", "Fri"), 8, (4,),
        (Group("G", 17),), (Room("R", 30, "classroom"),), faculty,
        tuple(Offering(f"K{i}", ("G",), tuple(f.id for f in faculty),
                       sessions, 1, "classroom", False)
              for i, sessions in enumerate((1, 3, 4, 8, 11))),
    )
    placements = tuple(Placement(t, "R") for t in problem.teaching_periods[:27])
    # The multiset of relative loads is always {0,.05,.15,.20,.40,.55}.
    results = set()
    for teacher_order in permutations(tuple(f.id for f in faculty)):
        result = evaluate(problem, Schedule(placements, teacher_order[:5]))
        assert result.feasible
        results.add(result.objectives[2])
    assert len(results) == 1  # Exact float equality, with no tolerance/rounding.


def test_duration_weighted_room_waste_is_exactly_invariant_under_event_permutation():
    capacities = (23, 31, 47, 61, 89, 113)
    rooms = tuple(Room(f"R{i}", capacity, "lab") for i, capacity in enumerate(capacities))
    offerings = tuple(Offering(f"K{i}", ("G",), ("F",), 1, duration, "lab")
                      for i, duration in enumerate((1, 2, 1, 2, 1, 2)))
    problem = Problem(
        "room-permutation", ("Mon", "Tue"), 8, (4,),
        (Group("G", 17),), rooms, (Faculty("F", 20, 7),), offerings,
    )
    placements = tuple(Placement(t, f"R{i}") for i, t in enumerate((0, 1, 3, 5, 7, 8)))
    original = evaluate(problem, Schedule(placements, ("F",) * 6))
    assert original.feasible
    for order in permutations(range(6)):
        permuted = replace(problem, offerings=tuple(offerings[i] for i in order))
        candidate = Schedule(tuple(placements[i] for i in order), ("F",) * 6)
        result = evaluate(permuted, candidate)
        assert result.feasible
        assert result.objectives == original.objectives


def test_missing_event_does_not_improve_workload(hand_problem, hand_schedule):
    incomplete = replace(hand_schedule, placements=(None, *hand_schedule.placements[1:]))
    result = evaluate(hand_problem, incomplete)
    assert not result.feasible
    assert result.violations["missing_event"] == 1
    assert result.objectives[2] == evaluate(hand_problem, hand_schedule).objectives[2]


def test_lunch_is_not_an_idle_gap(hand_problem):
    problem = replace(hand_problem, offerings=(replace(hand_problem.offerings[0], distinct_days=False),))
    schedule = Schedule((Placement(1, "R"), Placement(3, "R")), ("F1",))
    result = evaluate(problem, schedule)
    assert result.feasible
    assert result.objectives[0] == 0


@pytest.mark.parametrize(
    "event_index,placement,key",
    [
        (0, Placement(2, "R"), "lunch"),
        (2, Placement(1, "L"), "lunch"),
        (2, Placement(5, "L"), "day_boundary"),
        (0, Placement(-1, "R"), "invalid_start"),
        (0, Placement(12, "R"), "invalid_start"),
        (0, Placement(True, "R"), "invalid_start"),
        (0, Placement(0.5, "R"), "invalid_start"),
        (0, Placement(0, "L"), "room_type"),
        (0, Placement(0, "missing"), "invalid_room"),
    ],
)
def test_illegal_placement_domains(hand_problem, hand_schedule, event_index, placement, key):
    placements = list(hand_schedule.placements)
    placements[event_index] = placement
    result = evaluate(hand_problem, replace(hand_schedule, placements=tuple(placements)))
    assert not result.feasible
    assert result.violations[key] > 0


def test_capacity_shared_event_sums_groups(hand_problem, hand_schedule):
    problem = replace(hand_problem, rooms=(hand_problem.rooms[0], Room("L", 25, "lab")))
    result = evaluate(problem, hand_schedule)
    assert result.violations["room_capacity"] == 1


def test_room_faculty_and_student_clashes_all_detected(hand_problem, hand_schedule):
    # A and C now share room and F1, while C and B overlap group G2.
    problem = replace(hand_problem, offerings=(
        hand_problem.offerings[0], hand_problem.offerings[1],
        replace(hand_problem.offerings[2], groups=("G1", "G2")),
    ))
    placements = (*hand_schedule.placements[:3], Placement(0, "R"))
    result = evaluate(problem, replace(hand_schedule, placements=placements))
    assert result.violations["room_clash"] == 1
    assert result.violations["faculty_clash"] == 1
    assert result.violations["student_clash"] == 1


def test_laboratory_occupies_every_period_for_every_group(hand_problem, hand_schedule):
    placements = (*hand_schedule.placements[:3], Placement(4, "R"))
    result = evaluate(hand_problem, replace(hand_schedule, placements=placements))
    assert result.violations["student_clash"] == 1
    assert "faculty_clash" not in result.violations


def test_room_and_faculty_availability_cover_entire_lab(hand_problem, hand_schedule):
    problem = replace(hand_problem,
        rooms=(hand_problem.rooms[0], replace(hand_problem.rooms[1], unavailable=(4,))),
        faculty=(hand_problem.faculty[0], replace(hand_problem.faculty[1], unavailable=(4,))),
    )
    result = evaluate(problem, hand_schedule)
    assert result.violations["room_unavailable"] == 1
    assert result.violations["faculty_unavailable"] == 1


def test_daily_weekly_limits_and_theory_distinct_days(hand_problem, hand_schedule):
    problem = replace(hand_problem, faculty=(
        replace(hand_problem.faculty[0], max_weekly=2, max_daily=1), hand_problem.faculty[1],
    ))
    result = evaluate(problem, hand_schedule)
    assert result.violations["weekly_overload"] == 1
    assert result.violations["daily_overload"] == 1
    same_day = replace(hand_schedule, placements=(Placement(0, "R"), Placement(1, "R"), *hand_schedule.placements[2:]))
    assert evaluate(hand_problem, same_day).violations["same_day_sessions"] == 1


def test_assignment_count_and_invalid_teacher(hand_problem, hand_schedule):
    result = evaluate(hand_problem, replace(hand_schedule, placements=hand_schedule.placements[:-1], teachers=("not-eligible",)))
    assert result.violations["placement_count"] == 1
    assert result.violations["missing_event"] == 1
    assert result.violations["teacher_count"] == 2
    assert result.violations["invalid_teacher"] == 3
    extra = replace(hand_schedule, placements=hand_schedule.placements + (Placement(0, "R"),))
    assert evaluate(hand_problem, extra).violations["placement_count"] == 1


def test_domain_helpers(hand_problem):
    lab = hand_problem.events[2]
    assert valid_starts(hand_problem, lab) == (0, 3, 4, 6, 9, 10)
    assert compatible_rooms(hand_problem, lab) == ("L",)
    assert lab.size == 30
    assert [event.id for event in hand_problem.events] == ["A:1", "A:2", "B:1", "C:1"]


def test_json_roundtrip(hand_problem, hand_schedule):
    validate_problem(hand_problem)
    assert problem_from_dict(json.loads(json.dumps(problem_to_dict(hand_problem)))) == hand_problem
    assert schedule_from_dict(json.loads(json.dumps(schedule_to_dict(hand_schedule)))) == hand_schedule


def test_schema_rejects_bad_data_but_allows_infeasibility(hand_problem):
    with pytest.raises(ValueError, match="IDs"):
        validate_problem(replace(hand_problem, rooms=(hand_problem.rooms[0], hand_problem.rooms[0])))
    with pytest.raises(ValueError, match="unknown group"):
        validate_problem(replace(hand_problem, offerings=(replace(hand_problem.offerings[0], groups=("unknown",)),)))
    with pytest.raises(ValueError, match="positive integer"):
        validate_problem(replace(hand_problem, periods_per_day=True))
    with pytest.raises(ValueError, match="schema version"):
        problem_from_dict({**problem_to_dict(hand_problem), "schema_version": 99})
    validate_problem(replace(hand_problem, offerings=(replace(hand_problem.offerings[0], eligible_faculty=()),)))


def test_empty_problem_and_negative_gap_weight(hand_problem):
    empty = replace(hand_problem, offerings=())
    result = evaluate(empty, Schedule((), ()))
    assert result.feasible and result.objectives == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        evaluate(empty, Schedule((), ()), faculty_gap_weight=-1)
    for invalid in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite"):
            evaluate(empty, Schedule((), ()), faculty_gap_weight=invalid)
