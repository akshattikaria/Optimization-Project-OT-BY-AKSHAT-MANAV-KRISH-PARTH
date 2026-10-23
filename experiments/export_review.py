"""Export audited baseline/optimized grids and comparisons from saved archives.

Usage: python experiments/export_review.py --study benchmark
This never reruns or changes a solver result.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from timetable.evaluation import evaluate
from timetable.experiments import export_representatives
from timetable.model import problem_from_dict, schedule_from_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="benchmark")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / "results" / args.study
    runs = [json.loads(p.read_text()) for p in sorted((folder / "runs").glob("*.json"))]
    rows = []
    for size in sorted(set(r["size"] for r in runs)):
        # Fixed rule chosen without comparing quality: first instance ID of size.
        instance_id = min(r["instance_id"] for r in runs if r["size"] == size)
        problem = problem_from_dict(json.loads((root / "data" / args.study / f"{instance_id}.json").read_text()))
        for method in ["greedy", "enhanced"]:
            schedules = [schedule_from_dict(s) for r in runs if r["instance_id"] == instance_id and r["method"] == method
                         for s in r["archive"]]
            output = folder / "timetables" / instance_id / method
            export_representatives(problem, schedules, output, root / "figures" / args.study / "timetables" / instance_id / method)
            if not schedules:
                continue
            representatives = json.loads((output / "representatives.json").read_text())
            for label in representatives:
                data = json.loads((output / label / "schedule.json").read_text())
                schedule = schedule_from_dict(data["schedule"])
                ev = evaluate(problem, schedule)
                assert ev.feasible
                rows.append({"instance_id": instance_id, "method": method, "representative": label,
                             "gaps": ev.objectives[0], "unused_capacity": ev.objectives[1], "workload_sd": ev.objectives[2],
                             "student_gaps": ev.metrics["student_gaps"], "faculty_gaps": ev.metrics["faculty_gaps"],
                             "max_faculty_daily_load": ev.metrics["max_faculty_daily_load"],
                             "longest_idle_stretch": ev.metrics["longest_idle_stretch"]})
    with (folder / "timetable_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} baseline/optimized representative records; duplicate selections are explicitly retained.")


if __name__ == "__main__":
    main()
