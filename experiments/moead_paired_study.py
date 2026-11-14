"""Paired comparison of MOEA/D against memetic NSGA-II and repeated greedy.

Protocol (fixed before any run; nothing is tuned against the outcome):

* fresh synthetic instances whose generator seeds were not used by the
  development or held-out validation studies;
* every method sees the same instance, the same solver seed and the same cap on
  complete evaluator calls, so each (instance, seed) is one matched pair;
* quality is the hypervolume of the returned feasible archive under the
  problem's a-priori normalization, exactly as in the validation study;
* pairs are compared with the project's exact two-sided sign test.

MOEA/D is run twice, with Tchebycheff and with weighted-sum scalarization, so
the decomposition question (does the scalarization matter here?) is measured
rather than asserted.  Either outcome is a valid finding.

The script also counts *unsupported* points.  A point of a front is supported
when it minimizes some positive weighted sum over that front, i.e. it lies on
the convex hull facing the ideal point.  Unsupported points are exactly the
ones a weighted-sum method cannot single out, so the share of them on the
pooled reference front says how much of the front weighted sums could miss.

    python experiments/moead_paired_study.py
    python experiments/moead_paired_study.py --quick     # tiny smoke run
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from statistics import mean, median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timetable.evaluation import evaluate  # noqa: E402
from timetable.experiments import greedy_search  # noqa: E402
from timetable.generation import generate_instance  # noqa: E402
from timetable.memetic import memetic_nsga2  # noqa: E402
from timetable.metrics import (hypervolume, nondominated_indices, normalize,  # noqa: E402
                               objective_scales, paired_comparison)
from timetable.moead import moead  # noqa: E402

METHODS = ("greedy", "memetic", "moead_tchebycheff", "moead_weighted_sum")
LABELS = {
    "greedy": "Repeated greedy",
    "memetic": "Memetic NSGA-II (final method)",
    "moead_tchebycheff": "MOEA/D, Tchebycheff",
    "moead_weighted_sum": "MOEA/D, weighted sum",
}
PROTOCOL = {
    "population": 24,
    "scenarios": [
        {"size": "small", "instance_seeds": [3010], "solver_seeds": [0, 1, 2], "budget": 720,
         "note": "Sanity scenario: the 40-event instances usually have a one-point front."},
        {"size": "medium", "instance_seeds": [3010, 3020, 3030], "solver_seeds": [0, 1, 2, 3, 4], "budget": 360},
        {"size": "large", "instance_seeds": [3010, 3020], "solver_seeds": [0, 1, 2], "budget": 240},
    ],
}
QUICK = {
    "population": 12,
    "scenarios": [{"size": "small", "instance_seeds": [3010], "solver_seeds": [0, 1], "budget": 60}],
}


def run_method(problem, method: str, seed: int, budget: int, population: int):
    if method == "greedy":
        return greedy_search(problem, seed, budget)
    if method == "memetic":
        return memetic_nsga2(problem, seed=seed, population_size=population, max_evaluations=budget)
    scalarization = "tchebycheff" if method == "moead_tchebycheff" else "weighted_sum"
    return moead(problem, seed=seed, population_size=population, max_evaluations=budget,
                 scalarization=scalarization)


def simplex_grid(divisions: int) -> list[tuple[float, float, float]]:
    """Strictly positive weight vectors on a regular grid of the 3-simplex."""
    out = []
    for i in range(1, divisions):
        for j in range(1, divisions - i):
            k = divisions - i - j
            out.append((i / divisions, j / divisions, k / divisions))
    return out


def supported_flags(points: list[tuple[float, ...]], divisions: int = 80) -> list[bool]:
    """Which points minimize some positive weighted sum over ``points``.

    A grid of positive weights is searched, so a point reported as unsupported
    is unsupported *for every weight on that grid*; a finer grid can only turn
    an unsupported point into a supported one, never the reverse.
    """
    flags = [False] * len(points)
    if len(points) <= 1:
        return [True] * len(points)
    for w in simplex_grid(divisions):
        values = [sum(a * b for a, b in zip(w, p)) for p in points]
        best = min(values)
        for i, value in enumerate(values):
            if value <= best + 1e-12:
                flags[i] = True
    return flags


def study(protocol: dict, output: Path, log=print) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    population = protocol["population"]
    rows: list[dict] = []
    fronts: dict[tuple[str, int], dict[str, list]] = {}
    started = perf_counter()
    for scenario in protocol["scenarios"]:
        size, budget = scenario["size"], scenario["budget"]
        for instance_seed in scenario["instance_seeds"]:
            problem, _certificate = generate_instance(size, seed=instance_seed)
            scales = objective_scales(problem)
            pooled = fronts.setdefault((size, instance_seed), {m: [] for m in METHODS})
            for solver_seed in scenario["solver_seeds"]:
                for method in METHODS:
                    result = run_method(problem, method, solver_seed, budget, population)
                    evaluations = [evaluate(problem, s) for s in result.archive]
                    if not all(e.feasible for e in evaluations):
                        raise AssertionError(f"{method} returned an infeasible archive timetable")
                    objectives = [e.objectives for e in evaluations]
                    normalized = [normalize(v, scales) for v in objectives]
                    pooled[method].extend(normalized)
                    rows.append({
                        "size": size, "instance_seed": instance_seed, "events": len(problem.events),
                        "solver_seed": solver_seed, "method": method, "budget": budget,
                        "evaluations": result.evaluations, "feasible": bool(objectives),
                        "front_size": len(objectives),
                        "hypervolume": hypervolume(normalized) if normalized else 0.0,
                        "runtime_seconds": result.runtime,
                        "best_f1": min((v[0] for v in objectives), default=None),
                        "best_f2": min((v[1] for v in objectives), default=None),
                        "best_f3": min((v[2] for v in objectives), default=None),
                    })
                    log(f"{size:6s} inst {instance_seed} seed {solver_seed} {method:20s} "
                        f"HV {rows[-1]['hypervolume']:.5f} front {len(objectives):3d} {result.runtime:6.1f}s")

    def values(size: str, method: str, key: str = "hypervolume") -> list[float]:
        return [r[key] for r in rows if r["size"] == size and r["method"] == method]

    summary: dict = {"protocol": protocol, "methods": LABELS, "sizes": {}, "support": {}}
    for scenario in protocol["scenarios"]:
        size = scenario["size"]
        per_method = {}
        for method in METHODS:
            hv = values(size, method)
            per_method[method] = {
                "runs": len(hv), "feasible_runs": sum(values(size, method, "feasible")),
                "mean_hypervolume": mean(hv), "median_hypervolume": median(hv),
                "min_hypervolume": min(hv), "max_hypervolume": max(hv),
                "mean_front_size": mean(values(size, method, "front_size")),
                "mean_runtime_seconds": mean(values(size, method, "runtime_seconds")),
            }
        pairs = {}
        for a, b in (("moead_tchebycheff", "memetic"), ("moead_tchebycheff", "greedy"),
                     ("memetic", "greedy"), ("moead_tchebycheff", "moead_weighted_sum"),
                     ("moead_weighted_sum", "memetic")):
            pairs[f"{a}_vs_{b}"] = paired_comparison(values(size, a), values(size, b))
        summary["sizes"][size] = {"budget": scenario["budget"], "note": scenario.get("note"),
                                  "methods": per_method, "paired": pairs}

    # Supported / unsupported structure of the pooled reference front of each instance.
    for (size, instance_seed), by_method in fronts.items():
        union = [p for m in METHODS for p in by_method[m]]
        reference = [union[i] for i in nondominated_indices(union)]
        flags = supported_flags(reference)
        unsupported = {p for p, ok in zip(reference, flags) if not ok}
        reference_set = set(reference)
        contributions = {}
        for method in METHODS:
            own = set(by_method[method]) & reference_set
            contributions[method] = {"reference_points": len(own),
                                     "unsupported_reference_points": len(own & unsupported)}
        summary["support"][f"{size}-{instance_seed}"] = {
            "reference_front_size": len(reference), "supported": sum(flags),
            "unsupported": len(unsupported), "weight_grid_divisions": 80,
            "by_method": contributions}

    summary["total_runtime_seconds"] = perf_counter() - started
    summary["environment"] = {"python": platform.python_version(), "platform": platform.platform()}

    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "runs.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "summary.md").write_text(markdown(summary), encoding="utf-8")
    return summary


def markdown(summary: dict) -> str:
    lines = ["# MOEA/D paired comparison", "",
             "Generated by `experiments/moead_paired_study.py`. Every row is measured; nothing is estimated.",
             "All instances are synthetic. Each (instance, solver seed) is one matched pair: same instance,",
             "same seed, same cap on complete evaluator calls. Hypervolume uses the fixed a-priori",
             "normalization and reference point (1.1, 1.1, 1.1); higher is better.", ""]
    for size, block in summary["sizes"].items():
        lines += [f"## {size.capitalize()} instances (budget {block['budget']} evaluations)", ""]
        if block.get("note"):
            lines += [f"*{block['note']}*", ""]
        lines += ["| Method | Runs | Feasible | Mean HV | Median HV | Min HV | Max HV | Mean front size | Mean runtime (s) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for method, m in block["methods"].items():
            lines.append(f"| {summary['methods'][method]} | {m['runs']} | {m['feasible_runs']} | "
                         f"{m['mean_hypervolume']:.5f} | {m['median_hypervolume']:.5f} | {m['min_hypervolume']:.5f} | "
                         f"{m['max_hypervolume']:.5f} | {m['mean_front_size']:.1f} | {m['mean_runtime_seconds']:.2f} |")
        lines += ["", "| Paired comparison (A vs B) | Pairs | A wins | B wins | Ties | Median HV difference (A − B) | Sign test p (two-sided) |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, p in block["paired"].items():
            a, b = name.split("_vs_")
            lines.append(f"| {summary['methods'][a]} vs {summary['methods'][b]} | {p['pairs']} | {p['wins']} | "
                         f"{p['losses']} | {p['ties']} | {p['median_difference']:+.5f} | {p['sign_test_p_two_sided']:.4f} |")
        lines.append("")
    lines += ["## Supported and unsupported points", "",
              "For each instance the archives of all methods and seeds are pooled and reduced to their",
              "non-dominated reference front. A point is *supported* if it minimizes some positive weighted",
              "sum over that front (weights searched on a regular simplex grid). Unsupported points lie in",
              "non-convex parts of the front; no weighted sum can single them out.", "",
              "| Instance | Reference front | Supported | Unsupported | Unsupported reference points contributed by: greedy / memetic / MOEA/D-Tch / MOEA/D-WS |",
              "|---|---:|---:|---:|---|"]
    for name, s in summary["support"].items():
        by = s["by_method"]
        contributed = " / ".join(str(by[m]["unsupported_reference_points"]) for m in METHODS)
        lines.append(f"| {name} | {s['reference_front_size']} | {s['supported']} | {s['unsupported']} | {contributed} |")
    lines += ["", "The reference front is the best front *observed* in this study, not a proven Pareto front.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default="results/moead_paired")
    parser.add_argument("--quick", action="store_true", help="tiny smoke protocol (not a study)")
    args = parser.parse_args(argv)
    summary = study(QUICK if args.quick else PROTOCOL, ROOT / args.output)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
