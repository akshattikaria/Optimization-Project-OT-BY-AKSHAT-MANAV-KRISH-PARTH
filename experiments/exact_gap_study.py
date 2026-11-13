"""How close do the heuristics get to proven Pareto-optimal points?

Two tiers, both on deliberately small synthetic instances:

1. **Micro (three objectives).**  The two-event conflict instances are
   exhaustively enumerated, so their complete three-objective Pareto fronts
   are known.  We check whether each heuristic recovers them.

2. **Small (two objectives).**  For the 10-event room-conflict family CP-SAT
   certifies the complete Pareto front of the pair (f1 weighted gaps, f2 room
   wastage) with the lexicographic epsilon-constraint method, and separately
   runs a weighted-sum sweep.  The heuristic archives are projected onto
   (f1, f2) and compared with the certified front.

Scope of the exact claim: CP-SAT does not model the workload standard
deviation (f3), so tier 2 certifies the two-objective (f1, f2) front only.  It
is *not* a certified three-objective front.  Instances whose exact run did not
finish every proof are listed but excluded from the gap statistics.

    python experiments/exact_gap_study.py
    python experiments/exact_gap_study.py --quick
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from fractions import Fraction
from math import dist
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timetable.evaluation import evaluate  # noqa: E402
from timetable.exact import (enumerate_pareto, exact_gaps_room_front, supported_points,  # noqa: E402
                             weighted_sum_cp_sat)
from timetable.examples import conflict_instance, room_conflict_instance  # noqa: E402
from timetable.experiments import greedy_search  # noqa: E402
from timetable.memetic import memetic_nsga2  # noqa: E402
from timetable.model import problem_to_dict  # noqa: E402
from timetable.moead import moead  # noqa: E402

METHODS = {
    "greedy": "Repeated greedy",
    "memetic": "Memetic NSGA-II (final method)",
    "moead_tchebycheff": "MOEA/D, Tchebycheff",
    "moead_weighted_sum": "MOEA/D, weighted sum",
}
PROTOCOL = {"instance_seeds": [1, 2, 3, 4, 5, 6, 7, 8], "solver_seeds": [0, 1, 2, 3, 4], "budget": 300,
            "population": 24, "time_limit_per_solve": 60.0, "weighted_sum_steps": 9}
QUICK = {"instance_seeds": [2, 6], "solver_seeds": [0], "budget": 80, "population": 12,
         "time_limit_per_solve": 30.0, "weighted_sum_steps": 4}
TOLERANCE = 1e-9


def run_method(problem, method: str, seed: int, budget: int, population: int):
    if method == "greedy":
        return greedy_search(problem, seed, budget)
    if method == "memetic":
        return memetic_nsga2(problem, seed=seed, population_size=population, max_evaluations=budget)
    return moead(problem, seed=seed, population_size=population, max_evaluations=budget,
                 scalarization="tchebycheff" if method == "moead_tchebycheff" else "weighted_sum")


def front_2d(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    return [p for p in unique if not any(q != p and q[0] <= p[0] + TOLERANCE and q[1] <= p[1] + TOLERANCE for q in unique)]


def area_2d(points: list[tuple[float, float]], reference: tuple[float, float]) -> float:
    """Dominated area of a two-objective minimization front below ``reference``."""
    total, bound = 0.0, reference[1]
    for x, y in sorted(front_2d(points)):
        if x < reference[0] and y < bound:
            total += (reference[0] - x) * (bound - y)
            bound = y
    return total


def compare(exact: list[tuple[float, float]], found: list[tuple[float, float]],
            unsupported: list[tuple[float, float]]) -> dict:
    """Distance of a heuristic (f1, f2) front from a certified one, in range-normalized units."""
    lo = [min(p[k] for p in exact) for k in (0, 1)]
    span = [max(max(p[k] for p in exact) - lo[k], TOLERANCE) for k in (0, 1)]
    one_point = len(exact) == 1

    def scale(p):
        # A one-point exact front has no range; use the raw value so that the distance stays meaningful.
        return tuple((p[k] - lo[k]) / (1.0 if one_point else span[k]) for k in (0, 1))

    exact_n, found_n = [scale(p) for p in exact], [scale(p) for p in found]
    attained = [any(abs(a[0] - b[0]) <= TOLERANCE and abs(a[1] - b[1]) <= TOLERANCE for b in found) for a in exact]
    reference = (1.1, 1.1) if not one_point else (scale(exact[0])[0] + 1.0, scale(exact[0])[1] + 1.0)
    exact_area = area_2d(exact_n, reference)
    return {
        "exact_points": len(exact),
        "exact_points_attained": sum(attained),
        "fraction_attained": sum(attained) / len(exact),
        "unsupported_exact_points": len(unsupported),
        "unsupported_attained": sum(any(abs(a[0] - b[0]) <= TOLERANCE and abs(a[1] - b[1]) <= TOLERANCE for b in found)
                                    for a in unsupported),
        "igd": mean(min(dist(a, b) for b in found_n) for a in exact_n),
        "additive_epsilon": max(min(max(b[0] - a[0], b[1] - a[1]) for b in found_n) for a in exact_n),
        "hypervolume_ratio": area_2d(found_n, reference) / exact_area if exact_area > 0 else None,
        "f1_gap": min(p[0] for p in found) - lo[0],
        "f2_gap_relative": (min(p[1] for p in found) - lo[1]) / lo[1] if lo[1] > 0 else min(p[1] for p in found) - lo[1],
    }


def micro_tier(protocol: dict) -> list[dict]:
    out = []
    for case in ("room", "workload"):
        problem = conflict_instance(case)
        exact = enumerate_pareto(problem)
        truth = sorted(exact.objectives)
        for method in METHODS:
            recovered = []
            for seed in protocol["solver_seeds"]:
                result = run_method(problem, method, seed, min(protocol["budget"], 120), 8)
                found = {evaluate(problem, s).objectives for s in result.archive}
                recovered.append(sum(any(all(abs(a - b) <= TOLERANCE for a, b in zip(t, f)) for f in found) for t in truth))
            out.append({"instance": problem.name, "method": method, "exact_front_size": len(truth),
                        "exact_front": [list(v) for v in truth], "candidates_enumerated": exact.evaluated,
                        "runs": len(recovered), "mean_points_recovered": mean(recovered),
                        "runs_recovering_whole_front": sum(r == len(truth) for r in recovered)})
    return out


def study(protocol: dict, output: Path, log=print) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    instances, runs = [], []
    for instance_seed in protocol["instance_seeds"]:
        problem = room_conflict_instance(instance_seed)
        exact = exact_gaps_room_front(problem, protocol["time_limit_per_solve"])
        weighted = weighted_sum_cp_sat(problem, protocol["weighted_sum_steps"], protocol["time_limit_per_solve"])
        units = [(p.gap_units, p.room_units) for p in exact.points]
        flags = supported_points(units)
        exact_xy = [(p.objectives[0], p.objectives[1]) for p in exact.points]
        weighted_units = {(p.gap_units, p.room_units) for p in weighted.points}
        record = {
            "instance": problem.name, "instance_seed": instance_seed, "events": len(problem.events),
            "complete": exact.complete, "exact_solves": exact.solves, "exact_runtime_seconds": exact.runtime,
            "exact_statuses": exact.metadata["statuses"],
            "gap_scale": exact.gap_scale, "room_scale": exact.room_scale,
            "exact_front": [{"f1": p.objectives[0], "f2": p.objectives[1], "f3_of_schedule": p.objectives[2],
                             "f1_exact": str(Fraction(p.gap_units, exact.gap_scale)),
                             "f2_exact": str(Fraction(p.room_units, exact.room_scale)),
                             "supported": ok, "found_by_weighted_sum": (p.gap_units, p.room_units) in weighted_units}
                            for p, ok in zip(exact.points, flags)],
            "supported": sum(flags), "unsupported": flags.count(False),
            "weighted_sum_solves": weighted.solves, "weighted_sum_runtime_seconds": weighted.runtime,
            "weighted_sum_distinct_points": len(weighted.points),
            "weighted_sum_all_proven": weighted.metadata.get("all_proven", False),
            "weighted_sum_found_unsupported": sum((u in weighted_units) and not ok for u, ok in zip(units, flags)),
        }
        instances.append(record)
        (output / "instances").mkdir(exist_ok=True)
        (output / "instances" / f"{problem.name}.json").write_text(
            json.dumps(problem_to_dict(problem), indent=2) + "\n", encoding="utf-8")
        log(f"{problem.name}: complete={exact.complete} points={len(units)} unsupported={flags.count(False)} "
            f"weighted-sum points={len(weighted.points)} ({exact.runtime + weighted.runtime:.1f}s)")
        if not exact.complete or not exact_xy:
            continue
        unsupported_xy = [xy for xy, ok in zip(exact_xy, flags) if not ok]
        for method in METHODS:
            for seed in protocol["solver_seeds"]:
                result = run_method(problem, method, seed, protocol["budget"], protocol["population"])
                found = [evaluate(problem, s).objectives[:2] for s in result.archive]
                if not found:
                    runs.append({"instance": problem.name, "method": method, "solver_seed": seed, "feasible": False})
                    continue
                # A heuristic point must never beat a certified front; this guards the whole comparison.
                for f in found:
                    if not any(e[0] <= f[0] + 1e-7 and e[1] <= f[1] + 1e-7 for e in exact_xy):
                        raise AssertionError(f"{method} found {f}, which the certified front does not cover")
                runs.append({"instance": problem.name, "method": method, "solver_seed": seed, "feasible": True,
                             "evaluations": result.evaluations, "runtime_seconds": result.runtime,
                             **compare(exact_xy, front_2d(found), unsupported_xy)})

    certified = [r for r in runs if r.get("feasible")]
    methods = {}
    for method in METHODS:
        rows = [r for r in certified if r["method"] == method]
        if not rows:
            continue
        ratios = [r["hypervolume_ratio"] for r in rows if r["hypervolume_ratio"] is not None]
        methods[method] = {
            "runs": len(rows),
            "runs_attaining_whole_front": sum(r["exact_points_attained"] == r["exact_points"] for r in rows),
            "exact_points_attained": sum(r["exact_points_attained"] for r in rows),
            "exact_points_possible": sum(r["exact_points"] for r in rows),
            "unsupported_attained": sum(r["unsupported_attained"] for r in rows),
            "unsupported_possible": sum(r["unsupported_exact_points"] for r in rows),
            "mean_igd": mean(r["igd"] for r in rows),
            "mean_additive_epsilon": mean(r["additive_epsilon"] for r in rows),
            "mean_hypervolume_ratio": mean(ratios) if ratios else None,
            "runs_with_optimal_f1": sum(abs(r["f1_gap"]) <= TOLERANCE for r in rows),
            "runs_with_optimal_f2": sum(abs(r["f2_gap_relative"]) <= TOLERANCE for r in rows),
            "mean_f2_gap_relative": mean(r["f2_gap_relative"] for r in rows),
            "mean_runtime_seconds": mean(r["runtime_seconds"] for r in rows),
        }
    complete = [i for i in instances if i["complete"]]
    summary = {
        "protocol": protocol, "methods_labels": METHODS,
        "scope": "Certified front of (f1 weighted gaps, f2 room wastage) only; f3 is not modeled by CP-SAT.",
        "instances": instances, "certified_instances": len(complete),
        "exact_points_total": sum(len(i["exact_front"]) for i in complete),
        "unsupported_points_total": sum(i["unsupported"] for i in complete),
        "weighted_sum_points_total": sum(i["weighted_sum_distinct_points"] for i in complete),
        "weighted_sum_found_unsupported_total": sum(i["weighted_sum_found_unsupported"] for i in complete),
        "mean_exact_runtime_seconds": mean(i["exact_runtime_seconds"] for i in complete) if complete else None,
        "methods": methods, "micro": micro_tier(protocol),
        "total_runtime_seconds": perf_counter() - started,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    (output / "runs.json").write_text(json.dumps(runs, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(markdown(summary), encoding="utf-8")
    return summary


def markdown(s: dict) -> str:
    p = s["protocol"]
    lines = ["# Exact reference versus heuristics", "",
             "Generated by `experiments/exact_gap_study.py`. All instances are synthetic and deliberately small.",
             "", f"**Scope.** {s['scope']}", "",
             "## Tier 1: complete three-objective fronts by exhaustive enumeration", "",
             "| Instance | Exact front size | Method | Runs | Mean points recovered | Runs recovering the whole front |",
             "|---|---:|---|---:|---:|---:|"]
    for m in s["micro"]:
        lines.append(f"| {m['instance']} | {m['exact_front_size']} | {s['methods_labels'][m['method']]} | {m['runs']} | "
                     f"{m['mean_points_recovered']:.2f} | {m['runs_recovering_whole_front']} |")
    lines += ["", "## Tier 2: certified (f1, f2) fronts from CP-SAT", "",
              f"Lexicographic epsilon-constraint, {p['time_limit_per_solve']:.0f} s limit per solve; weighted-sum sweep with "
              f"{p['weighted_sum_steps']} weights. A front is *certified* only when every solve ended with a proof.", "",
              "| Instance | Events | Certified | Exact points | Supported | Unsupported | Distinct weighted-sum optima | Unsupported points found by weighted sum | Exact runtime (s) |",
              "|---|---:|:---:|---:|---:|---:|---:|---:|---:|"]
    for i in s["instances"]:
        lines.append(f"| {i['instance']} | {i['events']} | {'yes' if i['complete'] else 'no'} | {len(i['exact_front'])} | "
                     f"{i['supported']} | {i['unsupported']} | {i['weighted_sum_distinct_points']} | "
                     f"{i['weighted_sum_found_unsupported']} | {i['exact_runtime_seconds']:.1f} |")
    lines += ["", f"Across the {s['certified_instances']} certified instances the exact fronts hold "
              f"{s['exact_points_total']} points, {s['unsupported_points_total']} of them unsupported. The weighted-sum "
              f"sweep returned {s['weighted_sum_points_total']} distinct points and "
              f"{s['weighted_sum_found_unsupported_total']} of the unsupported ones.", "",
              f"### Heuristics against the certified fronts (budget {p['budget']} evaluations, "
              f"{len(p['solver_seeds'])} solver seeds per instance)", "",
              "IGD and the additive epsilon indicator are measured in the (f1, f2) plane after scaling each objective by the",
              "range of the certified front; 0 means the certified front was reproduced exactly. The hypervolume ratio is",
              "heuristic area / exact area (1 is the maximum).", "",
              "| Method | Runs | Whole front attained | Exact points attained | Unsupported points attained | Mean IGD | Mean additive ε | Mean HV ratio | Runs with optimal f1 | Runs with optimal f2 | Mean relative f2 gap |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, m in s["methods"].items():
        ratio = f"{m['mean_hypervolume_ratio']:.4f}" if m["mean_hypervolume_ratio"] is not None else "n/a"
        lines.append(f"| {s['methods_labels'][method]} | {m['runs']} | {m['runs_attaining_whole_front']} | "
                     f"{m['exact_points_attained']}/{m['exact_points_possible']} | "
                     f"{m['unsupported_attained']}/{m['unsupported_possible']} | {m['mean_igd']:.4f} | "
                     f"{m['mean_additive_epsilon']:.4f} | {ratio} | {m['runs_with_optimal_f1']} | "
                     f"{m['runs_with_optimal_f2']} | {100 * m['mean_f2_gap_relative']:.2f}% |")
    lines += ["", "The heuristics optimize three objectives, so part of their budget goes to workload balance, which the",
              "two-objective reference ignores. A heuristic point that is not on the certified (f1, f2) front can still be",
              "non-dominated in three objectives.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default="results/exact_gap")
    parser.add_argument("--quick", action="store_true", help="tiny smoke protocol (not a study)")
    args = parser.parse_args(argv)
    summary = study(QUICK if args.quick else PROTOCOL, ROOT / args.output)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
