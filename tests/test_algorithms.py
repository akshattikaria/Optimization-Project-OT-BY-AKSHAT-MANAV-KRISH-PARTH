"""Algorithm mechanics, budget accounting, reproducibility and feasible archives."""
import random

import pytest

from timetable.evaluation import Evaluation, evaluate
from timetable.generation import generate_instance
from timetable.model import Faculty, Group, Offering, Problem, Room
from timetable.nsga2 import (
    Individual, constrained_dominates, crowding_distance, environmental_selection,
    non_dominated_sort, nsga2, objective_dominates,
)


def ev(objectives, violation=0):
    return Evaluation(tuple(objectives), violation, {}, {})


def easy_problem():
    return Problem(
        "small-test", ("Monday", "Tuesday"), 5, (2,),
        (Group("G", 20),),
        (Room("R1", 20, "classroom"), Room("R2", 40, "classroom")),
        (Faculty("F1", 8, 4), Faculty("F2", 8, 4)),
        (Offering("A", ("G",), ("F1", "F2"), 2, 1, "classroom"),
         Offering("B", ("G",), ("F1", "F2"), 2, 1, "classroom")),
    )


def test_constraint_domination_ignores_apparent_quality_of_infeasible():
    good = ev((10, 1, 0.5))
    invalid = ev((0, 0, 0), 1)
    assert constrained_dominates(good, invalid)
    assert not constrained_dominates(invalid, good)
    assert constrained_dominates(ev((100, 100, 100), 1), ev((0, 0, 0), 2))
    assert not constrained_dominates(ev((1, 1, 1), 1), ev((2, 2, 2), 1))
    assert not objective_dominates((1, 1, 1), (1, 1, 1))


def test_nondominated_layers_known_example():
    outcomes = [ev((0, 2, 0)), ev((2, 0, 0)), ev((2, 2, 0)), ev((3, 3, 1)), ev((0, 0, 0), 1)]
    assert non_dominated_sort(outcomes) == [[0, 1], [2], [3], [4]]
    assert non_dominated_sort([]) == []


def test_crowding_boundaries_constants_and_interior():
    outcomes = [ev((i, 4 - i, 0)) for i in range(5)]
    distances = crowding_distance(outcomes, range(5))
    assert distances[0] == distances[4] == float("inf")
    assert distances[2] == pytest.approx(1.0)
    assert all(value == 0 for value in crowding_distance([ev((1, 1, 1)) for _ in range(4)], range(4)).values())


def test_elitism_retains_nondominated_boundaries():
    candidates = [Individual(None, ev((i, 4 - i, 0))) for i in range(5)]
    selected = environmental_selection(candidates, 2, random.Random(0))
    assert {ind.evaluation.objectives for ind in selected} == {(0, 4, 0), (4, 0, 0)}


def test_budget_counts_every_evaluator_call_and_all_enhancements(monkeypatch):
    import importlib
    module = importlib.import_module("timetable.nsga2")
    calls = []
    original = module.evaluate

    def counted(problem, schedule):
        calls.append(schedule)
        return original(problem, schedule)

    monkeypatch.setattr(module, "evaluate", counted)
    result = nsga2(easy_problem(), seed=0, population_size=8, generations=100,
                   enhanced=True, max_evaluations=80)
    assert result.evaluations == len(calls) == 80
    assert result.metadata["seeded_candidates"] == 4
    assert result.metadata["repairs"] > 0
    assert result.metadata["local_proposals"] > 0
    assert result.history[-1]["evaluations"] == 80


@pytest.mark.parametrize("enhanced", [False, True])
def test_determinism_and_feasible_nondominated_archive(enhanced):
    problem = easy_problem()
    kwargs = dict(seed=33, population_size=12, generations=12, enhanced=enhanced, max_evaluations=120)
    first = nsga2(problem, **kwargs)
    second = nsga2(problem, **kwargs)
    assert first.archive == second.archive
    assert first.history == second.history
    assert first.metadata == second.metadata
    assert first.evaluations == second.evaluations == 120
    assert first.archive
    objectives = [evaluate(problem, schedule).objectives for schedule in first.archive]
    assert len(set(objectives)) == len(objectives)
    for i, schedule in enumerate(first.archive):
        assert evaluate(problem, schedule).feasible
        assert not any(objective_dominates(other, objectives[i]) for other in objectives)
    # External archive quality is monotonic in set dominance, even as the finite
    # population discards points. Every earlier vector survives or is dominated.
    for before, after in zip(first.history, first.history[1:]):
        assert all(any(tuple(old) == tuple(new) or objective_dominates(new, old)
                       for new in after["front"]) for old in before["front"])


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_enhanced_search_returns_valid_generated_timetables(size):
    problem, _unused_certificate = generate_instance(size, seed=5)
    result = nsga2(problem, seed=4, population_size=6, generations=5,
                   enhanced=True, max_evaluations=30)
    assert result.archive
    assert all(evaluate(problem, schedule).feasible for schedule in result.archive)
    assert len(result.population) == 6


def test_plain_has_no_hidden_constructive_initialization():
    result = nsga2(easy_problem(), seed=0, population_size=4, generations=0)
    assert result.metadata["seeded_candidates"] == 0
    assert not result.metadata["repair_enabled"]
    assert not result.metadata["local_search_enabled"]
    assert result.evaluations == 4


def test_seeding_repair_and_local_search_individually_switchable():
    result = nsga2(easy_problem(), seed=0, population_size=6, generations=4,
                   enhanced=False, intelligent_initialization=True, seed_fraction=1.0,
                   repair=False, local_search=False)
    assert result.metadata["seeded_candidates"] == 6
    assert not result.metadata["repair_enabled"]
    assert not result.metadata["local_search_enabled"]
    result = nsga2(easy_problem(), seed=0, population_size=6, generations=4,
                   enhanced=True, intelligent_initialization=False, repair=True,
                   local_search=False)
    assert result.metadata["seeded_candidates"] == 0
    assert result.metadata["repair_enabled"]


def test_infeasible_problem_reports_no_feasible_archive():
    from dataclasses import replace
    problem = replace(easy_problem(), rooms=())
    result = nsga2(problem, seed=2, population_size=4, generations=3, enhanced=True)
    assert result.archive == []
    assert result.metadata["minimum_violation"] > 0
    assert result.best_schedule is not None
    assert not evaluate(problem, result.best_schedule).feasible


@pytest.mark.parametrize("kwargs", [
    {"population_size": 1}, {"generations": -1}, {"seed_fraction": -0.1},
    {"crossover_probability": 2}, {"local_steps": -1}, {"max_evaluations": 1},
])
def test_invalid_search_settings(kwargs):
    with pytest.raises(ValueError):
        nsga2(easy_problem(), **kwargs)
