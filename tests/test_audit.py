"""Independent reporting-boundary checks for exported scientific artifacts."""

import itertools
import json
import math
from dataclasses import replace

import pytest

from timetable.cli import main
from timetable.evaluation import evaluate
from timetable.exact import ExactResult, enumerate_pareto
from timetable.examples import conflict_instance
from timetable.experiments import serialize_run
from timetable.export import export_schedule
from timetable.generation import generate_instance
from timetable.metrics import hypervolume, normalize, objective_scales
from timetable.model import Schedule, problem_to_dict
from timetable.nsga2 import SearchResult


def test_float_hypervolume_with_nonunit_reference_matches_box_inclusion_exclusion():
    # An independent analytic union calculation catches accidental normalized-
    # reference assumptions, same-coordinate slicing errors, and boundary boxes.
    reference = (1.1, 1.3, 1.7)
    points = [(0.13, 0.77, 0.22), (0.13, 0.24, 0.85), (0.67, 0.24, 0.31), (1.1, 0.0, 0.0)]
    expected = 0.0
    for count in range(1, len(points) + 1):
        for subset in itertools.combinations(points, count):
            expected += (-1) ** (count + 1) * math.prod(
                reference[j] - max(vector[j] for vector in subset)
                for j in range(3)
            )
    assert hypervolume(points, reference) == pytest.approx(expected)
    assert hypervolume(points + points, reference) == pytest.approx(expected)


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_feasible_objective_normalization_uses_fixed_instance_bounds(size):
    problem, witness = generate_instance(size, seed=94)
    assessment = evaluate(problem, witness)
    assert assessment.feasible
    scales = objective_scales(problem)
    assert all(0 <= value <= 1 for value in normalize(assessment.objectives, scales))
    # A source label or dataset seed annotation must not fit indicator scales.
    assert objective_scales(replace(problem, metadata={"seed": -999})) == scales


def test_reporting_rejects_invalid_archive_instead_of_plotting_its_objectives():
    problem, witness = generate_instance("small", seed=95)
    malformed = Schedule((None,) + witness.placements[1:], witness.teachers)
    result = SearchResult([malformed], [], 1, 0.01, {}, malformed, [])
    with pytest.raises(AssertionError, match="invalid archive"):
        serialize_run(problem, result, "injected-invalid", seed=0)


def test_resource_ids_do_not_control_export_paths(tmp_path):
    problem = conflict_instance("room")
    result = enumerate_pareto(problem)
    # Rooms/faculty/group IDs are labels, never directory or file components.
    strange_rooms = tuple(replace(room, id=f"../outside/{room.id}") for room in problem.rooms)
    lookup = {old.id: new.id for old, new in zip(problem.rooms, strange_rooms)}
    schedule = result.archive[0]
    schedule = replace(schedule, placements=tuple(replace(p, room=lookup[p.room]) for p in schedule.placements))
    problem = replace(problem, rooms=strange_rooms)
    output = tmp_path / "artifacts"
    export_schedule(problem, schedule, output)
    assert not (tmp_path / "outside").exists()
    assert {path.name for path in (output / "room").glob("*.csv")} == {"000.csv", "001.csv"}


@pytest.mark.parametrize("status", ["UNKNOWN", "INFEASIBLE"])
def test_cli_keeps_exact_unknown_distinct_from_infeasible(status, tmp_path, monkeypatch, capsys):
    import timetable.exact as exact

    problem = conflict_instance("room")
    instance = tmp_path / "instance.json"
    instance.write_text(json.dumps(problem_to_dict(problem)))
    monkeypatch.setattr(exact, "cp_sat", lambda *a, **k: ExactResult(None, status, None, None, 0.01, "gaps"))
    output = tmp_path / status
    code = main(["solve", str(instance), "--method", "cpsat", "--output", str(output)])
    assert code == 2
    payload = json.loads((output / "exact.json").read_text())
    assert payload["status"] == status
    assert payload["schedule"] is None
    assert payload["objective_value"] is None
    assert payload["best_bound"] is None
    assert status in capsys.readouterr().out


def _study_config(tmp_path, **overrides):
    config = {
        "name": "audit", "methods": ["greedy"], "instance_seeds": [7],
        "solver_seeds": [0], "scenarios": [{"size": "small", "budget": 1}],
    }
    config.update(overrides)
    path = tmp_path / "study.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("corruption", ["identity", "budget", "archive", "objectives"])
def test_resume_rejects_corrupt_cached_identity_or_schedule(corruption, tmp_path, monkeypatch):
    import timetable.experiments as module

    monkeypatch.setattr(module, "analyze_suite", lambda *a: {})
    config = _study_config(tmp_path)
    result = module.run_suite(config, root=tmp_path)
    run_path = next((result / "runs").glob("*.json"))
    payload = json.loads(run_path.read_text())
    assert payload["archive"]
    if corruption == "identity":
        payload["instance_sha256"] = "not-the-instance"
    elif corruption == "budget":
        payload["evaluations"] = 999
    elif corruption == "archive":
        payload["archive"][0]["placements"][0] = None
    else:
        payload["objectives"][0][0] += 0.001
    run_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Cached"):
        module.run_suite(config, root=tmp_path, resume=True)


def test_resume_preserves_manifest_and_refuses_another_runtime(tmp_path, monkeypatch):
    import timetable.experiments as module

    monkeypatch.setattr(module, "analyze_suite", lambda *a: {})
    config = _study_config(tmp_path)
    result = module.run_suite(config, root=tmp_path)
    manifest_path = result / "manifest.json"
    original = manifest_path.read_bytes()
    module.run_suite(config, root=tmp_path, resume=True)
    assert manifest_path.read_bytes() == original
    manifest = json.loads(original)
    manifest["versions"]["numpy"] = "different-version"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="recorded Python/dependencies/platform"):
        module.run_suite(config, root=tmp_path, resume=True)


@pytest.mark.parametrize("duplicate", ["scenario", "instance_seed", "solver_seed", "method"])
def test_study_rejects_filename_collisions_before_running(duplicate, tmp_path):
    import timetable.experiments as module

    overrides = {
        "scenario": {"scenarios": [{"size": "small", "budget": 1}, {"size": "small", "budget": 2}]},
        "instance_seed": {"instance_seeds": [7, 7]},
        "solver_seed": {"solver_seeds": [0, 0]},
        "method": {"methods": ["greedy", "greedy"]},
    }[duplicate]
    config = _study_config(tmp_path, **overrides)
    with pytest.raises(ValueError):
        module.run_suite(config, root=tmp_path)
    assert not (tmp_path / "results").exists()
