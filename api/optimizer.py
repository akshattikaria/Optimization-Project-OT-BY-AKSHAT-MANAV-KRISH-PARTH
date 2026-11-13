"""Website optimisation run: multi-algorithm benchmark comparing Memetic NSGA-II, MOEA/D, AMOSA, Standard NSGA-II and Greedy.

All algorithms are run under the exact same evaluation budget and seed (the equal-evaluation protocol).
Every timetable is independently verified outside search with the common checker.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from timetable.amosa import amosa
from timetable.evaluation import evaluate
from timetable.experiments import greedy_search, serialize_run
from timetable.memetic import memetic_nsga2
from timetable.metrics import hypervolume, normalize, objective_scales
from timetable.model import Problem, problem_to_dict, schedule_to_dict
from timetable.moead import moead
from timetable.nsga2 import nsga2

from .bridge import problem_to_web, schedule_to_solution

ROOT = Path(__file__).resolve().parents[1]
METHOD = "Feasible memetic NSGA-II (final method)"
BASELINE = "Repeated greedy / DSatur (equal evaluations)"


@dataclass
class RunConfig:
    seed: int = 0
    budget: int = 360
    population: int = 24
    search_seconds: float | None = 6.0
    compare_baseline: bool = True
    label: str = "web"


class NoFeasibleTimetable(RuntimeError):
    pass


def _compromise(values: list[tuple[float, float, float]]) -> int:
    """Index of the normalized compromise (closest to the normalized ideal of the given front)."""
    if not values:
        return 0
    a = np.asarray(values, dtype=float)
    span = np.ptp(a, axis=0)
    scaled = (a - a.min(axis=0)) / np.where(span > 0, span, 1)
    return int(np.argmin(np.linalg.norm(scaled, axis=1)))


def run_optimizer(problem: Problem, cfg: RunConfig, progress: Callable[[dict], None] | None = None,
                  save: bool = True, run_id: str | None = None) -> dict:
    report = progress or (lambda _p: None)
    started = perf_counter()
    scales = objective_scales(problem)

    calls = {"n": 0, "feasible": 0, "last": 0.0}

    def counted_evaluate(prob, schedule):
        ev = evaluate(prob, schedule)
        calls["n"] += 1
        calls["feasible"] += ev.feasible
        now = perf_counter()
        if now - calls["last"] > 0.2:
            calls["last"] = now
            report({"phase": "memetic", "evaluations": calls["n"], "budget": cfg.budget,
                    "feasibleEvaluations": calls["feasible"], "seconds": now - started,
                    "secondsLimit": cfg.search_seconds})
        return ev

    # 1. Primary Algorithm: Feasible Memetic NSGA-II
    report({"phase": "memetic", "evaluations": 0, "budget": cfg.budget, "seconds": 0.0, "secondsLimit": cfg.search_seconds})
    memetic_res = memetic_nsga2(problem, seed=cfg.seed, population_size=cfg.population, max_evaluations=cfg.budget,
                               time_limit=cfg.search_seconds, evaluator=counted_evaluate)
    if not memetic_res.archive:
        best = evaluate(problem, memetic_res.best_schedule) if memetic_res.best_schedule is not None else None
        detail = f" Smallest violation found: {best.violation} ({dict(best.violations)})." if best else ""
        raise NoFeasibleTimetable(
            "No feasible timetable was found within this budget. This is a search failure, not a proof that the "
            "constraints are impossible: try a larger budget/time, or relax teacher availability." + detail)

    run = serialize_run(problem, memetic_res, "memetic", cfg.seed)
    eval_budget = memetic_res.evaluations

    memetic_front = [{"f1": v[0], "f2": v[1], "f3": v[2]} for v in run["objectives"]]

    # Comparison algorithms. Only algorithms that were actually run appear in the result: a run that was not
    # requested, or that failed, is reported as such and never replaced by an estimate.
    comparisons: dict[str, dict] = {}
    baseline_run = baseline_result = None
    baseline = None
    baseline_front: list[dict] = []

    def compare(key: str, search: Callable[[], object]) -> None:
        report({"phase": key, "evaluations": eval_budget, "budget": eval_budget, "seconds": perf_counter() - started})
        t0 = perf_counter()
        try:
            result = search()
        except Exception as error:  # the comparison is optional; say that it failed instead of inventing a result
            comparisons[key] = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                                "runtime": perf_counter() - t0, "front": [], "hv": None, "history": []}
            return
        objectives = []
        for schedule in result.archive:
            ev = evaluate(problem, schedule)
            if ev.feasible:  # independent check of every reported point
                objectives.append(ev.objectives)
        normalized = [normalize(v, scales) for v in objectives]
        comparisons[key] = {
            "status": "ok" if objectives else "no_feasible",
            "runtime": perf_counter() - t0,
            "front": [{"f1": v[0], "f2": v[1], "f3": v[2]} for v in objectives],
            "hv": hypervolume(normalized) if normalized else 0.0,
            "history": result.history,
        }

    if cfg.compare_baseline:
        cmp_time = cfg.search_seconds
        compare("moead", lambda: moead(problem, seed=cfg.seed, population_size=cfg.population,
                                       max_evaluations=eval_budget, time_limit=cmp_time))
        compare("amosa", lambda: amosa(problem, seed=cfg.seed, population_size=cfg.population,
                                       max_evaluations=eval_budget, time_limit=cmp_time))
        compare("nsga2", lambda: nsga2(problem, seed=cfg.seed, population_size=cfg.population,
                                       generations=eval_budget, max_evaluations=eval_budget))

        # Greedy / DSatur baseline under the same evaluation budget
        report({"phase": "baseline", "evaluations": eval_budget, "budget": eval_budget, "seconds": perf_counter() - started})
        baseline_result = greedy_search(problem, cfg.seed, eval_budget)
        baseline_run = serialize_run(problem, baseline_result, "greedy", cfg.seed)
        if baseline_result is not None and baseline_result.archive:
            values = [tuple(v) for v in baseline_run["objectives"]]
            pick = baseline_result.archive[_compromise(values)]
            baseline = schedule_to_solution(problem, pick, "GREEDY", False)
            baseline_front = [{"f1": v[0], "f2": v[1], "f3": v[2]} for v in values]

    report({"phase": "export", "evaluations": memetic_res.evaluations, "budget": cfg.budget, "seconds": perf_counter() - started})

    # Prepare solution objects for the primary Pareto front
    solutions = []
    archive_keys = set(memetic_res.archive)
    for i, schedule in enumerate(memetic_res.archive):
        solutions.append(schedule_to_solution(problem, schedule, f"S{i + 1:03d}", True))
    extra = 0
    for schedule in dict.fromkeys(memetic_res.population):
        if schedule in archive_keys:
            continue
        ev = evaluate(problem, schedule)
        if ev.feasible:
            extra += 1
            solutions.append(schedule_to_solution(problem, schedule, f"S{len(memetic_res.archive) + extra:03d}", False, ev))

    meta = memetic_res.metadata

    # Summary table: the final method plus every comparison algorithm that was run
    look = {"moead": ("MOEA/D (Tchebycheff)", "#059669", "triangle"),
            "amosa": ("AMOSA (Simulated Annealing)", "#7c3aed", "diamond"),
            "nsga2": ("Standard NSGA-II", "#ea580c", "square")}
    algorithm_comparison = [
        {"id": "memetic", "name": "Memetic NSGA-II", "status": "ok", "hv": round(run["hypervolume"], 4),
         "runtime": round(run["runtime"], 2), "frontSize": len(memetic_front), "feasible": True,
         "color": "#2563eb", "marker": "circle"},
    ]
    for key, c in comparisons.items():
        name, color, marker = look[key]
        row = {"id": key, "name": name, "status": c["status"], "hv": None if c["hv"] is None else round(c["hv"], 4),
               "runtime": round(c["runtime"], 2), "frontSize": len(c["front"]), "feasible": bool(c["front"]),
               "color": color, "marker": marker}
        if "error" in c:
            row["error"] = c["error"]
        algorithm_comparison.append(row)
    if baseline_run:
        algorithm_comparison.append(
            {"id": "greedy", "name": "Repeated Greedy", "status": "ok" if baseline_front else "no_feasible",
             "hv": round(baseline_run["hypervolume"], 4), "runtime": round(baseline_run["runtime"], 2),
             "frontSize": len(baseline_front), "feasible": bool(baseline_run["feasible_run"]),
             "color": "#d97706", "marker": "rhombus"})

    # Measured hypervolume trajectories only: no points are added, extended or smoothed.
    def _measured_curve(history_list):
        if not history_list:
            return []
        step = max(1, len(history_list) // 30)
        sampled = [history_list[i] for i in range(0, len(history_list), step)]
        if history_list[-1] is not sampled[-1]:
            sampled.append(history_list[-1])
        pts = []
        for h in sampled:
            if "hypervolume" in h:
                hv = h["hypervolume"]
            else:
                norm = [normalize(v, scales) for v in h.get("front", [])]
                hv = hypervolume(norm) if norm else 0.0
            pts.append({"evaluations": h.get("evaluations", 0), "hypervolume": round(float(hv), 5)})
        return pts

    multi_convergence = {
        "memetic": [{"evaluations": h["evaluations"], "hypervolume": round(float(h["hypervolume"]), 5)} for h in run["history"]],
    }
    for key, c in comparisons.items():
        if c["status"] != "failed":
            multi_convergence[key] = _measured_curve(c["history"])
    if baseline_run:
        multi_convergence["greedy"] = _measured_curve(baseline_run.get("history", []))

    stats = {
        "method": METHOD,
        "baselineMethod": BASELINE if cfg.compare_baseline else None,
        "runtime": run["runtime"],
        "evaluations": run["evaluations"],
        "budget": cfg.budget,
        "population": cfg.population,
        "searchSeconds": cfg.search_seconds,
        "stopReason": meta.get("stop_reason"),
        "generations": meta.get("generations_completed", 0),
        "frontSize": run["archive_size"],
        "hypervolume": run["hypervolume"],
        "spacing": run["spacing"] or 0.0,
        "memeticFront": memetic_front,
        **{f"{key}Front": c["front"] for key, c in comparisons.items() if c["status"] != "failed"},
        "baselineFront": baseline_front,
        "algorithmComparison": algorithm_comparison,
        "multiConvergence": multi_convergence,
        "baselineHypervolume": baseline_run["hypervolume"] if baseline_run else 0.0,
        "baselineFeasible": bool(baseline_run and baseline_run["feasible_run"]),
        "baselineRuntime": baseline_run["runtime"] if baseline_run else None,
        "baselineEvaluations": baseline_run["evaluations"] if baseline_run else None,
        "baselineFrontSize": baseline_run["archive_size"] if baseline_run else 0,
        "normalization": run["normalization"],
        "reference": run["reference"],
        "convergence": [
            {"generation": h["generation"], "evaluations": h["evaluations"], "hypervolume": h["hypervolume"],
             "archiveSize": len(memetic_res.history[i].get("front", [])), "feasibleFraction": h.get("feasible_fraction", 0.0)}
            for i, h in enumerate(run["history"])
        ],
        "verified": True,
    }

    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    saved_to = None
    if save:
        out = Path(os.environ.get("CLOCKWORK_RUNS_DIR", ROOT / "results" / "web")) / run_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "instance.json").write_text(json.dumps(problem_to_dict(problem), indent=2) + "\n")
        (out / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
        if baseline_run:
            (out / "baseline.json").write_text(json.dumps(baseline_run, indent=2, allow_nan=False) + "\n")
        if solutions:
            (out / "compromise_schedule.json").write_text(json.dumps(
                schedule_to_dict(memetic_res.archive[_compromise([tuple(v) for v in run["objectives"]])]), indent=2) + "\n")
        saved_to = str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out).replace("\\", "/")
    stats["savedTo"] = saved_to

    return {
        "runId": run_id,
        "params": {"seed": cfg.seed, "budget": cfg.budget, "population": cfg.population,
                   "searchSeconds": cfg.search_seconds, "compareBaseline": cfg.compare_baseline, "label": cfg.label},
        "instance": problem_to_web(problem),
        "solutions": solutions,
        "baseline": baseline,
        "stats": stats,
    }
