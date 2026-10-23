"""Independently verify domain enhancements and a useful teacher exchange."""
from dataclasses import replace
import random

import pytest

from timetable.evaluation import evaluate
from timetable.hybrid import HybridOperators
from timetable.model import Faculty, Group, Offering, Placement, Problem, Room, Schedule
from timetable.nsga2 import Individual


def exchange_example():
    problem = Problem(
        "whole-offering-exchange", ("Monday", "Tuesday", "Wednesday"), 5, (2,),
        (Group("G", 20),), (Room("R", 20, "classroom"),),
        (Faculty("F1", 6, 4), Faculty("F2", 6, 4)),
        tuple(Offering(name, ("G",), ("F1", "F2"), sessions, 1, "classroom")
              for name, sessions in (("A", 3), ("B", 2), ("C", 2), ("D", 1))),
    )
    # Teaching loads 5/3. Every individual offering transfer has load change
    # 1,2 or3; no single transfer improves SD. Swapping a 3- and 2-hour offering
    # yields 4/4 and preserves all placements and qualifications.
    schedule = Schedule(tuple(Placement(start, "R") for start in (0, 5, 10, 1, 6, 3, 8, 11)),
                        ("F1", "F2", "F1", "F2"))
    return problem, schedule


def test_whole_offering_exchange_improves_where_single_transfer_cannot():
    problem, schedule = exchange_example()
    before = evaluate(problem, schedule)
    assert before.feasible
    assert before.metrics["faculty_loads"] == {"F1": 5, "F2": 3}
    for k in range(len(problem.offerings)):
        teachers = list(schedule.teachers)
        teachers[k] = "F2" if teachers[k] == "F1" else "F1"
        single = evaluate(problem, replace(schedule, teachers=tuple(teachers)))
        assert single.objectives[2] >= before.objectives[2] - 1e-12
    hybrid = HybridOperators(problem)
    exchanged = hybrid.local_proposal(schedule, random.Random(1), mode=3)
    after = evaluate(problem, exchanged)
    assert exchanged.placements == schedule.placements
    assert after.feasible
    assert after.metrics["faculty_loads"] == {"F1": 4, "F2": 4}
    assert after.objectives[2] == pytest.approx(0.0)
    assert after.objectives[0] < before.objectives[0]


def test_exchange_respects_cross_eligibility():
    problem, schedule = exchange_example()
    problem = replace(problem, offerings=tuple(
        replace(offering, eligible_faculty=(teacher,))
        for offering, teacher in zip(problem.offerings, schedule.teachers)
    ))
    hybrid = HybridOperators(problem)
    assert hybrid.local_proposal(schedule, random.Random(1), 3) == schedule


def test_local_search_budget_and_feasible_acceptance():
    problem, schedule = exchange_example()
    hybrid = HybridOperators(problem)
    calls = []

    def counted(candidate):
        calls.append(candidate)
        return Individual(candidate, evaluate(problem, candidate))

    initial = Individual(schedule, evaluate(problem, schedule))
    result, proposals, accepted = hybrid.local_search(
        initial, counted, random.Random(6), steps=100, remaining=lambda: len(calls) < 3,
    )
    assert proposals == len(calls) == 3
    assert 0 <= accepted <= proposals
    assert result.evaluation.feasible


def test_repair_independent_evaluation_and_failure_explicit():
    problem, schedule = exchange_example()
    damaged = replace(schedule, placements=(None,) * len(schedule.placements))
    hybrid = HybridOperators(problem)
    calls = []

    def counted(candidate):
        calls.append(candidate)
        return Individual(candidate, evaluate(problem, candidate))

    repaired = hybrid.repair(Individual(damaged, evaluate(problem, damaged)), counted, random.Random(10))
    assert len(calls) == 1
    assert repaired.evaluation.feasible
    impossible = replace(problem, rooms=())
    failed = HybridOperators(impossible).repair(
        Individual(damaged, evaluate(impossible, damaged)),
        lambda candidate: Individual(candidate, evaluate(impossible, candidate)), random.Random(10),
    )
    assert not failed.evaluation.feasible
