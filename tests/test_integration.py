"""End-to-end checks keep certificates, optimization and independent audit apart."""

import csv
import importlib
import json
from dataclasses import replace

import pytest

from timetable.evaluation import evaluate
from timetable.export import export_schedule
from timetable.generation import generate_instance
from timetable.greedy import SchedulingFailure, greedy
from timetable.model import (
    problem_from_dict, problem_to_dict, schedule_from_dict, schedule_to_dict,
)
from timetable.nsga2 import nsga2, objective_dominates


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_generated_json_to_baseline_to_pareto_archive(size, tmp_path):
    problem, certificate = generate_instance(size, 22)
    assert evaluate(problem, certificate).feasible
    # Serialize only data, never certificate placements or chosen teachers.
    serialized = json.loads(json.dumps(problem_to_dict(problem)))
    assert "placements" not in serialized and "teachers" not in serialized
    serialized["metadata"] = {}
    loaded = problem_from_dict(serialized)
    baseline = greedy(loaded, seed=1, restarts=1)
    assert evaluate(loaded, baseline).feasible
    result = nsga2(loaded, seed=2, population_size=4, generations=8,
                   enhanced=True, max_evaluations=16)
    assert result.evaluations == 16
    assert result.archive
    objective_vectors = []
    for schedule in result.archive:
        roundtrip = schedule_from_dict(json.loads(json.dumps(schedule_to_dict(schedule))))
        evaluation = evaluate(loaded, roundtrip)
        assert evaluation.feasible, evaluation.violations
        assert len(roundtrip.teachers) == len(loaded.offerings)
        assert len(roundtrip.placements) == len(loaded.events)
        objective_vectors.append(evaluation.objectives)
    assert len(set(objective_vectors)) == len(objective_vectors)
    assert all(not objective_dominates(a, b)
               for i, a in enumerate(objective_vectors)
               for j, b in enumerate(objective_vectors) if i != j)
    exported = export_schedule(loaded, result.archive[0], tmp_path / size)
    assert exported["violation"] == 0
    with (tmp_path / size / "events.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(loaded.events)
    assert len({row["event"] for row in rows}) == len(loaded.events)
    for kind, entities in [("group", loaded.groups), ("faculty", loaded.faculty), ("room", loaded.rooms)]:
        assert len(list((tmp_path / size / kind).glob("*.csv"))) == len(entities)


def test_optimizer_reproducibility_is_independent_of_generator_metadata():
    problem, _ = generate_instance("small", 23)
    kwargs = dict(seed=9, population_size=4, generations=8,
                  enhanced=True, max_evaluations=24)
    first = nsga2(problem, **kwargs)
    second = nsga2(replace(problem, metadata={}), **kwargs)
    assert first.archive == second.archive
    assert first.population == second.population
    assert first.history == second.history
    assert first.metadata == second.metadata
    assert first.evaluations == second.evaluations


def test_all_nsga_evaluator_calls_are_charged_including_repair_and_local(monkeypatch):
    module = importlib.import_module("timetable.nsga2")
    problem, _ = generate_instance("small", 24)
    seen = []
    original = module.evaluate

    def counted(problem, schedule):
        seen.append(schedule)
        return original(problem, schedule)

    monkeypatch.setattr(module, "evaluate", counted)
    result = module.nsga2(problem, seed=6, population_size=4, generations=20,
                          enhanced=True, max_evaluations=32)
    assert len(seen) == result.evaluations == 32
    assert result.metadata["repairs"] > 0
    assert result.metadata["local_proposals"] > 0
    assert result.history[-1]["evaluations"] == result.evaluations


@pytest.mark.parametrize("enhanced", [False, True])
def test_infeasible_instance_cannot_publish_phantom_timetables(enhanced, tmp_path):
    problem, _ = generate_instance("small", 25)
    impossible = replace(problem, rooms=tuple(replace(room, capacity=1) for room in problem.rooms))
    with pytest.raises(SchedulingFailure):
        greedy(impossible, restarts=1)
    result = nsga2(impossible, seed=4, population_size=4, generations=8,
                   enhanced=enhanced, max_evaluations=16)
    assert result.archive == []
    assert all(record["front"] == [] for record in result.history)
    assert result.best_schedule is not None
    assert not evaluate(impossible, result.best_schedule).feasible
    with pytest.raises(ValueError, match="invalid timetable"):
        export_schedule(impossible, result.best_schedule, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
