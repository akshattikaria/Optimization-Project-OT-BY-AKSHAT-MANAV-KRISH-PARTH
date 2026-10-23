"""Constructive search is checked by the independent complete evaluator."""

from dataclasses import replace
import random

import pytest

from timetable.evaluation import evaluate
from timetable.generation import generate_instance
from timetable.greedy import SchedulingFailure, construct, greedy
from timetable.model import Faculty, Group, Offering, Problem, Room, Schedule


@pytest.mark.parametrize("size", ["small", "medium", "large"])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_construction_feasible_across_sizes_and_seeds(size, seed):
    problem, _certificate = generate_instance(size, seed)
    schedule = construct(problem, random.Random(seed))
    assert evaluate(problem, schedule).feasible
    assert len(schedule.placements) == len(problem.events)
    assert len(schedule.teachers) == len(problem.offerings)


def test_greedy_seed_reproducibility_and_preference_repair():
    problem, _certificate = generate_instance("small", 19)
    first = greedy(problem, seed=13, restarts=3)
    assert first == greedy(problem, seed=13, restarts=3)
    damaged = list(first.placements)
    damaged[0] = damaged[1]  # resource/cohort collision or duplicate day
    repaired = construct(problem, random.Random(4), Schedule(tuple(damaged), first.teachers))
    assert evaluate(problem, repaired).feasible


def test_impossible_capacity_is_a_heuristic_failure_not_a_valid_timetable():
    problem = Problem("impossible", ("Monday",), 2, (),
                      (Group("g", 50),), (Room("r", 20, "classroom"),),
                      (Faculty("f", 2, 2),),
                      (Offering("k", ("g",), ("f",), 1, 1, "classroom"),))
    partial = construct(problem, random.Random(0))
    assert not evaluate(problem, partial).feasible
    with pytest.raises(SchedulingFailure, match="infeasibility is not proven"):
        greedy(problem, restarts=2)


def test_faculty_consistency_and_load_limits_can_block_an_offering():
    # Two sessions cannot be split between teachers; each teacher has only one
    # weekly period available even though their combined capacity equals demand.
    problem = Problem("indivisible", ("Monday", "Tuesday"), 2, (),
                      (Group("g", 10),), (Room("r", 20, "classroom"),),
                      (Faculty("f1", 1, 1), Faculty("f2", 1, 1)),
                      (Offering("k", ("g",), ("f1", "f2"), 2, 1, "classroom"),))
    with pytest.raises(SchedulingFailure):
        greedy(problem, restarts=2)


def test_empty_problem_and_invalid_restart_count():
    empty = Problem("empty", ("Monday",), 2, (), (), (), (), ())
    assert evaluate(empty, greedy(empty, restarts=1)).feasible
    with pytest.raises(ValueError, match="restarts"):
        greedy(empty, restarts=0)
