"""Recheck every certificate and benchmark archive from raw saved JSON.

This is independent of the optimization run and fails on any hard violation,
objective mismatch, dominated archive vector, or incorrect hypervolume.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from timetable.evaluation import evaluate
from timetable.metrics import dominates, hypervolume, normalize, objective_scales
from timetable.model import problem_from_dict, schedule_from_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="benchmark")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_folder, result_folder = root / "data" / args.study, root / "results" / args.study
    manifest = json.loads((result_folder / "manifest.json").read_text())
    problems = {}
    certificates = 0
    for path in data_folder.glob("*.json"):
        problem = problem_from_dict(json.loads(path.read_text()))
        certificate = schedule_from_dict(json.loads((data_folder / "certificates" / path.name).read_text()))
        assert evaluate(problem, certificate).feasible, f"Invalid certificate: {path}"
        problems[path.stem] = problem
        certificates += 1
    runs = schedules = empty_archives = 0
    for path in sorted((result_folder / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        problem = problems[run["instance_id"]]
        assert run["evaluations"] == run["budget"], f"Budget mismatch {path}"
        vals = []
        for encoded in run["archive"]:
            schedule = schedule_from_dict(encoded)
            ev = evaluate(problem, schedule)
            assert ev.feasible, f"Invalid archive: {path}: {ev.violations}"
            vals.append(list(ev.objectives))
            schedules += 1
        assert vals == run["objectives"], f"Objective mismatch: {path}"
        assert len(set(tuple(v) for v in vals)) == len(vals), f"Duplicate vector: {path}"
        assert not any(dominates(a, b) for i, a in enumerate(vals) for j, b in enumerate(vals) if i != j), path
        hv = hypervolume([normalize(v, objective_scales(problem)) for v in vals], run["reference"])
        assert abs(hv - run["hypervolume"]) < 1e-12, f"Hypervolume mismatch: {path}"
        assert all(a["hypervolume"] <= b["hypervolume"] + 1e-12
                   for a, b in zip(run["history"], run["history"][1:])), f"Nonmonotone archive HV: {path}"
        empty_archives += not bool(vals)
        runs += 1
    expected = sum(len(s.get("instance_seeds", manifest["config"].get("instance_seeds", [10]))) *
                   len(s.get("solver_seeds", manifest["config"].get("solver_seeds", [0, 1, 2]))) *
                   len(manifest["config"]["methods"]) for s in manifest["config"]["scenarios"])
    assert runs == expected, f"Incomplete study: {runs}/{expected} runs"
    print(json.dumps({"study": args.study, "certificates_validated": certificates,
                      "runs_validated": runs, "feasible_archived_schedules_validated": schedules,
                      "empty_archives_reported": empty_archives, "expected_runs": expected,
                      "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
