"""Multi-Criteria Decision-Making (MCDM) using TOPSIS.

Technique for Order Preference by Similarity to Ideal Solution (TOPSIS).

In multi-objective optimization, metaheuristics discover a front of non-dominated
trade-off solutions. However, a university coordinator or Dean must ultimately
deploy a SINGLE timetable. TOPSIS provides a principled, objective-weighted
decision-making mechanism to select and rank timetables based on stakeholder priorities.

Key Optimization Techniques (OT) Concepts:
------------------------------------------
1. Positive Ideal Solution (PIS, A+) and Negative Ideal Solution (NIS, A-):
   - For minimization objectives (gaps, seat wastage, faculty SD):
     A+_j = min_i { v_ij } (theoretical best across the front)
     A-_j = max_i { v_ij } (worst compromise across the front)

2. Relative Closeness Coefficient (C_i*):
   C_i* = S-_i / (S+_i + S-_i)
   where S+_i is the Euclidean distance to PIS and S-_i is the distance to NIS.
   C_i* in [0, 1]. A higher score indicates the solution is simultaneously close
   to the ideal benchmark and far from the worst alternative.

3. Stakeholder Weight Sensitivity:
   Different campus departments have conflicting interests:
   - Students prioritize minimizing idle teaching gaps.
   - Administration prioritizes room utilization / minimizing seat wastage.
   - Faculty Deans prioritize balanced weekly teaching distribution.
   TOPSIS reveals which timetable best aligns with each stakeholder group.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


STAKEHOLDER_PROFILES: dict[str, tuple[float, float, float]] = {
    "balanced": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    "student_first": (0.6, 0.2, 0.2),
    "admin_room_fit": (0.2, 0.6, 0.2),
    "faculty_fairness": (0.2, 0.2, 0.6),
}


@dataclass(frozen=True)
class TopsisRanking:
    rank: int
    solution_index: int
    objectives: tuple[float, float, float]
    score: float
    distance_ideal: float
    distance_nadir: float


def topsis(
    objectives: Sequence[tuple[float, float, float]],
    weights: Sequence[float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
) -> list[TopsisRanking]:
    """Rank Pareto-optimal objective vectors using the TOPSIS method.
    
    All objectives are assumed to be minimized:
    - Obj 0: Internal gaps (student + 0.5 * faculty)
    - Obj 1: Duration-weighted unused room seat fraction
    - Obj 2: Faculty workload standard deviation
    
    Parameters:
        objectives: List of non-dominated 3D objective tuples.
        weights: 3-element weight vector representing stakeholder priorities.
        
    Returns:
        List of TopsisRanking dataclasses sorted from best (rank 1) to worst.
    """
    if not objectives:
        return []
    if len(weights) != 3:
        raise ValueError("weights must be a 3-element sequence for 3 objectives")
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("Sum of weights must be positive")
    w_norm = [w / weight_sum for w in weights]

    m = len(objectives)
    n = 3

    # Step 1: Vector Normalization
    col_norms = []
    for j in range(n):
        sum_sq = sum(objectives[i][j] ** 2 for i in range(m))
        col_norms.append(math.sqrt(sum_sq) if sum_sq > 0 else 1.0)

    # Step 2: Weighted Normalized Decision Matrix
    # V[i][j] = w_norm[j] * (X[i][j] / col_norms[j])
    v = []
    for i in range(m):
        row = [w_norm[j] * (objectives[i][j] / col_norms[j]) for j in range(n)]
        v.append(row)

    # Step 3: Determine Ideal (PIS / A+) and Anti-Ideal (NIS / A-) Solutions
    # Because all objectives are minimized:
    # A+_j = min_i(V[i][j]), A-_j = max_i(V[i][j])
    pis = [min(v[i][j] for i in range(m)) for j in range(n)]
    nis = [max(v[i][j] for i in range(m)) for j in range(n)]

    # Step 4: Calculate Separation Distances S+ and S-
    scores = []
    for i in range(m):
        d_plus = math.sqrt(sum((v[i][j] - pis[j]) ** 2 for j in range(n)))
        d_minus = math.sqrt(sum((v[i][j] - nis[j]) ** 2 for j in range(n)))
        
        # Step 5: Calculate Relative Closeness to Ideal Solution C_i*
        denom = d_plus + d_minus
        closeness = d_minus / denom if denom > 0 else 1.0
        scores.append((i, closeness, d_plus, d_minus, objectives[i]))

    # Step 6: Rank alternatives by closeness in descending order
    scores.sort(key=lambda item: item[1], reverse=True)

    rankings = []
    for rank, (idx, c_score, d_pos, d_neg, objs) in enumerate(scores, start=1):
        rankings.append(TopsisRanking(
            rank=rank,
            solution_index=idx,
            objectives=objs,
            score=round(c_score, 6),
            distance_ideal=round(d_pos, 6),
            distance_nadir=round(d_neg, 6),
        ))
    return rankings


def analyze_stakeholder_profiles(
    objectives: Sequence[tuple[float, float, float]],
) -> dict[str, dict]:
    """Run TOPSIS across standard institutional stakeholder profiles.
    
    Profiles:
    - balanced: Equal emphasis on students, rooms, and faculty.
    - student_first: 60% weight on minimizing student gap hours.
    - admin_room_fit: 60% weight on room seat utilization.
    - faculty_fairness: 60% weight on equitable teaching workload distribution.
    """
    analysis = {}
    for profile_name, weights in STAKEHOLDER_PROFILES.items():
        ranking = topsis(objectives, weights=weights)
        top = ranking[0] if ranking else None
        analysis[profile_name] = {
            "weights": list(weights),
            "top_choice_index": top.solution_index if top else None,
            "top_choice_objectives": list(top.objectives) if top else None,
            "top_choice_score": top.score if top else None,
            "all_rankings": [
                {
                    "rank": r.rank,
                    "index": r.solution_index,
                    "score": r.score,
                    "objectives": list(r.objectives),
                }
                for r in ranking
            ],
        }
    return analysis


# ---------------------------------------------------------------------------
# AHP-derived weights
# ---------------------------------------------------------------------------

# Saaty's random consistency index for matrices of order 1..10.
_RANDOM_INDEX = (0.0, 0.0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)


@dataclass(frozen=True)
class AhpResult:
    weights: tuple[float, ...]
    lambda_max: float
    consistency_index: float
    consistency_ratio: float

    @property
    def consistent(self) -> bool:
        """Saaty's usual acceptance rule: CR <= 0.10."""
        return self.consistency_ratio <= 0.10


def ahp_weights(matrix: Sequence[Sequence[float]]) -> AhpResult:
    """Priority weights of a positive reciprocal pairwise-comparison matrix.

    ``matrix[i][j]`` says how many times more important criterion ``i`` is
    than criterion ``j`` on Saaty's 1-9 scale, with ``matrix[j][i] = 1 /
    matrix[i][j]``.  The weights are the normalized principal eigenvector
    (power iteration).  The consistency ratio ``CR = CI / RI`` with ``CI =
    (lambda_max - n) / (n - 1)`` measures how far the judgements are from a
    perfectly transitive set; CR <= 0.10 is conventionally acceptable.
    """
    n = len(matrix)
    if not 2 <= n <= len(_RANDOM_INDEX) or any(len(row) != n for row in matrix):
        raise ValueError("AHP needs a square comparison matrix of order 2 to 10")
    for i in range(n):
        for j in range(n):
            value = matrix[i][j]
            if not math.isfinite(value) or value <= 0:
                raise ValueError("comparison judgements must be positive and finite")
            if abs(value * matrix[j][i] - 1.0) > 1e-6:
                raise ValueError("comparison matrix must be reciprocal: a[j][i] = 1 / a[i][j]")
    vector = [1.0 / n] * n
    for _ in range(200):
        product = [sum(matrix[i][j] * vector[j] for j in range(n)) for i in range(n)]
        total = sum(product)
        updated = [value / total for value in product]
        change = max(abs(a - b) for a, b in zip(updated, vector))
        vector = updated
        if change < 1e-13:
            break
    product = [sum(matrix[i][j] * vector[j] for j in range(n)) for i in range(n)]
    lambda_max = sum(product[i] / vector[i] for i in range(n)) / n
    ci = max(0.0, (lambda_max - n) / (n - 1))
    ri = _RANDOM_INDEX[n - 1]
    return AhpResult(tuple(vector), lambda_max, ci, ci / ri if ri > 0 else 0.0)


def ahp_from_judgements(gaps_vs_room: float, gaps_vs_workload: float, room_vs_workload: float) -> AhpResult:
    """AHP weights for (gaps, room fit, workload balance) from three pairwise judgements.

    A judgement above 1 favours the first-named objective (3 = moderately more
    important, 9 = extremely); below 1 favours the second (1/3, ..., 1/9).
    """
    a, b, c = float(gaps_vs_room), float(gaps_vs_workload), float(room_vs_workload)
    if min(a, b, c) <= 0:
        raise ValueError("comparison judgements must be positive")
    return ahp_weights(((1.0, a, b), (1.0 / a, 1.0, c), (1.0 / b, 1.0 / c, 1.0)))


# ---------------------------------------------------------------------------
# Sensitivity of the TOPSIS recommendation to the weights
# ---------------------------------------------------------------------------

def _top(objectives: Sequence[tuple[float, float, float]], weights: Sequence[float]) -> int:
    return topsis(objectives, weights)[0].solution_index


def weight_sensitivity(
    objectives: Sequence[tuple[float, float, float]],
    base_weights: Sequence[float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    divisions: int = 20,
    sweep_steps: int = 50,
) -> dict:
    """How the TOPSIS choice changes when the stakeholder weights change.

    Three complementary views, all computed from real TOPSIS rankings:

    * ``weight_space``: every weight vector on a regular grid of the simplex
      (step ``1 / divisions``) is ranked; for each timetable we report the
      share of that grid on which it is the top choice.
    * ``sweeps``: one weight at a time is moved from 0 to 1 while the other
      two keep their base ratio; each change of the top choice is listed with
      the interval of the swept weight in which it happens.
    * ``stability``: the smallest single-weight change (in either direction)
      that makes the base recommendation lose first place; ``None`` means the
      recommendation never changes along any of the three sweeps.
    """
    if not objectives:
        return {"base_weights": list(base_weights), "base_choice": None, "weight_space": [], "sweeps": [],
                "stability": None, "grid_points": 0}
    if len(base_weights) != 3 or min(base_weights) < 0 or sum(base_weights) <= 0:
        raise ValueError("base_weights must be three nonnegative numbers with a positive sum")
    if divisions < 2 or sweep_steps < 2:
        raise ValueError("divisions and sweep_steps must be at least 2")
    total = sum(base_weights)
    base = tuple(w / total for w in base_weights)
    base_choice = _top(objectives, base)

    counts: dict[int, int] = {}
    grid_points = 0
    for i in range(divisions + 1):
        for j in range(divisions + 1 - i):
            k = divisions - i - j
            # Avoid the all-zero-weight corner cases of a criterion by keeping every weight strictly positive.
            weights = tuple((x + 1e-9) / divisions for x in (i, j, k))
            counts[_top(objectives, weights)] = counts.get(_top(objectives, weights), 0) + 1
            grid_points += 1
    weight_space = [{"index": index, "objectives": list(objectives[index]), "share": count / grid_points}
                    for index, count in sorted(counts.items(), key=lambda item: -item[1])]

    names = ("gaps", "room", "workload")
    sweeps = []
    stability = None
    for axis in range(3):
        others = [base[k] for k in range(3) if k != axis]
        rest = sum(others)
        path = []
        for step in range(sweep_steps + 1):
            value = step / sweep_steps
            weights = [0.0, 0.0, 0.0]
            weights[axis] = value
            for k, share in zip([k for k in range(3) if k != axis], others):
                weights[k] = (1.0 - value) * (share / rest if rest > 0 else 0.5)
            weights = [w + 1e-9 for w in weights]
            path.append((value, _top(objectives, weights)))
        switches = [{"between": [path[s - 1][0], path[s][0]], "from": path[s - 1][1], "to": path[s][1]}
                    for s in range(1, len(path)) if path[s][1] != path[s - 1][1]]
        segments = []
        for value, choice in path:
            if segments and segments[-1]["index"] == choice:
                segments[-1]["to"] = value
            else:
                segments.append({"index": choice, "from": value, "to": value})
        for value, choice in path:
            if choice != base_choice:
                distance = abs(value - base[axis])
                stability = distance if stability is None else min(stability, distance)
        sweeps.append({"objective": names[axis], "base_weight": base[axis], "segments": segments, "switches": switches})

    return {"base_weights": list(base), "base_choice": base_choice, "weight_space": weight_space,
            "distinct_choices": len(counts), "grid_points": grid_points, "divisions": divisions,
            "sweeps": sweeps, "stability": stability}


def recommend(
    objectives: Sequence[tuple[float, float, float]],
    weights: Sequence[float] | None = None,
    ahp: Sequence[float] | None = None,
    divisions: int = 20,
) -> dict:
    """TOPSIS recommendation with the stakeholder profiles and the sensitivity analysis.

    ``ahp`` (three pairwise judgements: gaps vs room, gaps vs workload, room vs
    workload) takes precedence over ``weights`` when both are given.
    """
    objectives = [tuple(float(x) for x in v) for v in objectives]
    if any(len(v) != 3 for v in objectives):
        raise ValueError("every alternative needs three objective values")
    ahp_result = ahp_from_judgements(*ahp) if ahp is not None else None
    chosen = ahp_result.weights if ahp_result else tuple(weights) if weights is not None else STAKEHOLDER_PROFILES["balanced"]
    ranking = topsis(objectives, chosen)
    total = sum(chosen)
    profiles = {}
    for name, profile in STAKEHOLDER_PROFILES.items():
        top = topsis(objectives, profile)
        profiles[name] = {"weights": list(profile), "index": top[0].solution_index if top else None,
                          "score": top[0].score if top else None}
    return {
        "weights": [w / total for w in chosen],
        "weightSource": "ahp" if ahp_result else "weights" if weights is not None else "balanced",
        "ahp": None if ahp_result is None else {
            "weights": list(ahp_result.weights), "lambdaMax": ahp_result.lambda_max,
            "consistencyIndex": ahp_result.consistency_index, "consistencyRatio": ahp_result.consistency_ratio,
            "consistent": ahp_result.consistent},
        "recommended": ranking[0].solution_index if ranking else None,
        "ranking": [{"rank": r.rank, "index": r.solution_index, "score": r.score, "objectives": list(r.objectives),
                     "distanceIdeal": r.distance_ideal, "distanceNadir": r.distance_nadir} for r in ranking],
        "profiles": profiles,
        "sensitivity": weight_sensitivity(objectives, chosen, divisions) if objectives else None,
    }
