"""Exact lower bounds certify whether a small schedule attains all three ideals.

The workload relaxation drops eligibility/time/resource interactions, retaining
indivisible offering loads and equal weekly allowances. Dynamic programming
exhausts allocations of each load multiplicity to identical faculty bins.
A relaxed minimum is a lower bound on the constrained problem. If a feasible
schedule attains it and the independent gap/room lower bounds, its objective
vector globally dominates every other feasible vector. No heuristic success
or a CP-SAT time limit is mistaken for a proof.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from math import inf, sqrt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from timetable.evaluation import evaluate
from timetable.model import compatible_rooms, problem_from_dict, schedule_from_dict


def objective_lower_bounds(problem) -> dict:
    if not problem.faculty or len(problem.faculty) > 8 or len(problem.offerings) > 24:
        raise ValueError("Exact load-count relaxation is intentionally limited to small instances")
    allowances = {f.max_weekly for f in problem.faculty}
    if len(allowances) != 1:
        raise ValueError("This exact variance identity requires identical weekly allowances")
    allowance = allowances.pop()
    multiplicities = Counter(o.sessions * o.duration for o in problem.offerings)
    if len(multiplicities) > 3:
        raise ValueError("Too many distinct offering loads for this small exact reference")
    weights = tuple(sorted(multiplicities))
    original_counts = tuple(multiplicities[w] for w in weights)
    faculty_count = len(problem.faculty)

    @lru_cache(None)
    def dp(bins: int, counts: tuple[int, ...]):
        if bins == 0:
            return 0 if not any(counts) else inf
        demand = sum(w*c for w, c in zip(weights, counts))
        if demand > bins * allowance:
            return inf
        if bins == 1:
            return demand*demand
        best = inf
        for allocated in product(*(range(c+1) for c in counts)):
            load = sum(w*c for w, c in zip(weights, allocated))
            if load <= allowance:
                rest = tuple(c-a for c, a in zip(counts, allocated))
                best = min(best, load*load + dp(bins-1, rest))
        return best

    squares = dp(faculty_count, original_counts)
    if squares == inf:
        raise ValueError("Even the relaxed weekly-load partition is infeasible")
    total = sum(w*c for w, c in zip(weights, original_counts))
    variance = Fraction(squares, faculty_count * allowance**2) - Fraction(total, faculty_count * allowance)**2
    rooms = {r.id: r for r in problem.rooms}
    demand = sum(e.duration for e in problem.events)
    room_lower = sum((e.duration * min(Fraction(rooms[r].capacity-e.size, rooms[r].capacity)
                                      for r in compatible_rooms(problem, e)) for e in problem.events), Fraction()) / demand
    return {"gaps_lower_bound": 0.0, "room_lower_bound_exact": str(room_lower),
            "room_lower_bound": float(room_lower), "workload_sd_lower_bound": sqrt(float(variance)),
            "relaxed_minimum_sum_squared_loads": squares, "relative_load_variance_exact": str(variance),
            "offering_load_multiplicities": dict(multiplicities), "faculty_count": faculty_count,
            "common_weekly_allowance": allowance, "dp_states": dp.cache_info().currsize,
            "method": "exact integer load-multiplicity DP relaxation; independent static room lower bound; gaps nonnegative"}


def exact_attainment(problem, schedule, bounds) -> dict | None:
    ev = evaluate(problem, schedule)
    if not ev.feasible:
        return None
    rooms = {r.id: r for r in problem.rooms}
    demand = sum(e.duration for e in problem.events)
    room_value = sum((e.duration * Fraction(rooms[p.room].capacity-e.size, rooms[p.room].capacity)
                      for e, p in zip(problem.events, schedule.placements)), Fraction()) / demand
    squared_loads = sum(load*load for load in ev.metrics["faculty_loads"].values())
    if (ev.metrics["student_gaps"] == ev.metrics["faculty_gaps"] == 0
            and room_value == Fraction(bounds["room_lower_bound_exact"])
            and squared_loads == bounds["relaxed_minimum_sum_squared_loads"]):
        return {"exact_room_fraction": str(room_value), "exact_sum_squared_loads": squared_loads,
                "exact_student_gaps": 0, "exact_faculty_gaps": 0,
                "sorted_faculty_loads": sorted(ev.metrics["faculty_loads"].values()),
                "total_required_teaching_periods": demand}
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", default="benchmark")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / "results" / args.study
    cases = {}
    for path in sorted((root / "data" / args.study).glob("small_*.json")):
        problem = problem_from_dict(json.loads(path.read_text()))
        bounds = objective_lower_bounds(problem)
        certificate = None
        for run_path in sorted((folder / "runs").glob(f"{path.stem}_*.json")):
            run = json.loads(run_path.read_text())
            for index, encoded in enumerate(run["archive"]):
                schedule = schedule_from_dict(encoded)
                ev = evaluate(problem, schedule)
                exact = exact_attainment(problem, schedule, bounds)
                if exact is not None:
                    certificate = {"run": str(run_path.relative_to(root)), "archive_index": index,
                                   "objectives": ev.objectives, "violations": ev.violations,
                                   "exact_attainment": exact}
                    break
            if certificate:
                break
        cases[path.stem] = {"problem_sha256": sha256(path.read_bytes()).hexdigest(),
                            "bounds": bounds, "attaining_feasible_certificate": certificate,
                            "status": "CERTIFIED_GLOBAL_IDEAL_VECTOR" if certificate else "LOWER_BOUNDS_ONLY",
                            "interpretation": "If attained, the global three-objective front is the singleton ideal vector; schedule identities may differ."}
    output = folder / "small_ideal_certificates.json"
    output.write_text(json.dumps(cases, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
