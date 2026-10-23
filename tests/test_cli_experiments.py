"""Public entry points, evaluation accounting, and recoverable study checkpoints."""
import json
from dataclasses import replace

import pytest

from timetable.cli import main
from timetable.evaluation import evaluate
from timetable.experiments import greedy_search, run_method, run_suite, serialize_run
from timetable.export import export_schedule
from timetable.generation import generate_instance
from timetable.model import Schedule, problem_to_dict, schedule_to_dict


def test_cli_generate_validate_and_invalid_exit(tmp_path, capsys):
    instance, certificate = tmp_path / "instance.json", tmp_path / "certificate.json"
    assert main(["generate", "--output", str(instance), "--certificate", str(certificate), "--seed", "15"]) == 0
    assert main(["evaluate", str(instance), str(certificate)]) == 0
    payload = json.loads(certificate.read_text())
    payload["placements"][0] = None
    certificate.write_text(json.dumps(payload))
    assert main(["validate", str(instance), str(certificate)]) == 2
    assert 'missing_event' in capsys.readouterr().out
    assert main(["generate", "--output", str(instance), "--config", str(tmp_path / "absent.json")]) == 1


def test_greedy_budget_counts_all_evaluations(monkeypatch):
    import timetable.experiments as module
    problem, _ = generate_instance("small", 31)
    calls = 0
    original = module.evaluate
    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "evaluate", counted)
    result = greedy_search(problem, 8, 9)
    assert calls == result.evaluations == 9
    assert result.history[-1]["evaluations"] == 9
    assert all(evaluate(problem, s).feasible for s in result.archive)
    repeat = greedy_search(problem, 8, 9)
    assert result.archive == repeat.archive
    assert result.history == repeat.history


def test_no_infeasible_objectives_enter_hypervolume():
    problem, _ = generate_instance("small", 2)
    result = run_method(problem, "nsga2", 0, 12, 12)
    payload = serialize_run(problem, result, "nsga2", 0)
    assert payload["archive"] == []
    assert payload["hypervolume"] == 0
    assert payload["best_candidate_diagnostic"]["violation"] > 0
    assert all(row["hypervolume"] == 0 for row in payload["history"])


def test_empty_valid_timetable_export(tmp_path):
    problem, _ = generate_instance("small", 2)
    empty = replace(problem, offerings=())
    exported = export_schedule(empty, Schedule((), ()), tmp_path)
    assert exported["violation"] == 0
    assert len((tmp_path / "events.csv").read_text().splitlines()) == 1


def test_study_checkpoint_resume_and_config_identity(tmp_path, monkeypatch):
    import timetable.experiments as module
    monkeypatch.setattr(module, "analyze_suite", lambda *a: {})
    config = {"name": "test", "population_size": 4, "instance_seeds": [7], "solver_seeds": [1],
              "methods": ["greedy", "nsga2", "seeded", "enhanced"], "scenarios": [{"size": "small", "budget": 12}]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = run_suite(path, root=tmp_path)
    runs = sorted((output / "runs").glob("*.json"))
    assert len(runs) == 4
    before = [p.read_bytes() for p in runs]
    assert all(json.loads(p.read_text())["evaluations"] == 12 for p in runs)
    with pytest.raises(ValueError, match="already exists"):
        run_suite(path, root=tmp_path)
    run_suite(path, root=tmp_path, resume=True)
    assert before == [p.read_bytes() for p in runs]
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["source_sha256"]) == 64
    config["scenarios"][0]["budget"] = 16
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="different config"):
        run_suite(path, root=tmp_path, resume=True)
