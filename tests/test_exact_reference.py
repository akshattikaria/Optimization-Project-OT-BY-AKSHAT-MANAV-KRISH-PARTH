"""Exact (gaps, room) fronts, weighted-sum CP-SAT, AHP weights and TOPSIS sensitivity."""
import importlib.util
import math
from pathlib import Path

import pytest

from timetable.evaluation import evaluate
from timetable.examples import conflict_instance, room_conflict_instance
from timetable.mcdm import ahp_from_judgements, ahp_weights, recommend, topsis, weight_sensitivity
from timetable.model import validate_problem

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- supportedness (pure Python) -------------------------------------------------

def test_supported_points_flags_points_above_the_hull():
    from timetable.exact import supported_points

    assert supported_points([(0, 10), (1, 9), (2, 2), (5, 1), (9, 0)]) == [True, False, True, True, True]
    assert supported_points([(0, 4), (2, 2), (4, 0)]) == [True, True, True]  # collinear points are supported
    assert supported_points([(3, 3)]) == [True]
    assert supported_points([(0, 580), (2, 540), (4, 480), (6, 240)]) == [True, False, False, True]


def test_room_conflict_family_is_valid_and_deterministic():
    a, b = room_conflict_instance(6), room_conflict_instance(6)
    validate_problem(a)
    assert a == b and len(a.events) == 10
    assert room_conflict_instance(7) != a
    with pytest.raises(ValueError):
        room_conflict_instance(1, groups=0)


# --- CP-SAT -----------------------------------------------------------------------

def test_exact_front_matches_exhaustive_enumeration_on_micro_instance():
    pytest.importorskip("ortools")
    from timetable.exact import enumerate_pareto, exact_gaps_room_front

    problem = conflict_instance("room")
    front = exact_gaps_room_front(problem, time_limit_per_solve=10)
    assert front.complete
    assert all(p.proven for p in front.points)
    truth = sorted({v[:2] for v in enumerate_pareto(problem).objectives})
    assert sorted(p.objectives[:2] for p in front.points) == truth == [(0.0, 0.25), (1.0, 0.0)]
    assert all(evaluate(problem, p.schedule).feasible for p in front.points)


def test_weighted_sum_misses_unsupported_points_that_epsilon_constraint_finds():
    pytest.importorskip("ortools")
    from timetable.exact import exact_gaps_room_front, supported_points, weighted_sum_cp_sat

    problem = room_conflict_instance(6)
    exact = exact_gaps_room_front(problem, time_limit_per_solve=30)
    assert exact.complete
    units = [(p.gap_units, p.room_units) for p in exact.points]
    assert units == [(0, 580), (2, 540), (4, 480), (6, 240)]
    flags = supported_points(units)
    assert flags == [True, False, False, True]
    weighted = weighted_sum_cp_sat(problem, steps=9, time_limit_per_solve=30)
    found = {(p.gap_units, p.room_units) for p in weighted.points}
    assert found <= set(units)  # every weighted-sum optimum is on the certified front
    assert not any(u in found for u, ok in zip(units, flags) if not ok)  # and never an unsupported one
    assert found == {u for u, ok in zip(units, flags) if ok}


def test_weighted_objective_validation():
    pytest.importorskip("ortools")
    from timetable.exact import cp_sat

    problem = conflict_instance("room")
    for bad in (None, (1,), (0, 0), (-1, 2), (1.5, 1)):
        with pytest.raises(ValueError):
            cp_sat(problem, "weighted", weights=bad)
    with pytest.raises(ValueError):
        cp_sat(problem, "gaps", weights=(1, 1))
    with pytest.raises(ValueError):
        cp_sat(problem, "gaps", epsilon_objective="room", epsilon_bound=0.1, epsilon_bound_units=3)
    result = cp_sat(problem, "weighted", weights=(1, 0), time_limit=10)
    assert result.proven_optimal and result.metadata["gap_units"] == 0
    assert cp_sat(problem, "gaps", epsilon_objective="room", epsilon_bound_units=-1, time_limit=10).status == "INFEASIBLE"


def test_exact_gap_study_smoke(tmp_path):
    pytest.importorskip("ortools")
    tool = _load("exact_gap_study")
    protocol = {"instance_seeds": [6], "solver_seeds": [0], "budget": 40, "population": 8,
                "time_limit_per_solve": 30.0, "weighted_sum_steps": 3}
    summary = tool.study(protocol, tmp_path, log=lambda _m: None)
    assert summary["certified_instances"] == 1 and summary["unsupported_points_total"] == 2
    assert summary["weighted_sum_found_unsupported_total"] == 0
    for row in summary["methods"].values():
        assert 0 <= row["exact_points_attained"] <= row["exact_points_possible"]
        assert row["mean_igd"] >= 0 and row["mean_additive_epsilon"] >= 0
        assert row["mean_hypervolume_ratio"] <= 1 + 1e-9
    assert (tmp_path / "summary.md").read_text(encoding="utf-8").startswith("# Exact reference")


def test_gap_metrics_are_zero_for_the_exact_front_itself():
    tool = _load("exact_gap_study")
    exact = [(0.0, 0.5), (1.0, 0.3), (3.0, 0.1)]
    same = tool.compare(exact, exact, [(1.0, 0.3)])
    assert same["fraction_attained"] == 1 and same["igd"] == 0 and same["additive_epsilon"] == 0
    assert same["hypervolume_ratio"] == pytest.approx(1.0) and same["unsupported_attained"] == 1
    worse = tool.compare(exact, [(0.0, 0.5)], [(1.0, 0.3)])
    assert worse["exact_points_attained"] == 1 and worse["igd"] > 0 and worse["hypervolume_ratio"] < 1
    assert worse["f1_gap"] == 0 and worse["f2_gap_relative"] == pytest.approx(4.0)


def test_moead_paired_study_smoke(tmp_path):
    tool = _load("moead_paired_study")
    protocol = {"population": 8, "scenarios": [{"size": "small", "instance_seeds": [3010], "solver_seeds": [0, 1], "budget": 30}]}
    summary = tool.study(protocol, tmp_path, log=lambda _m: None)
    block = summary["sizes"]["small"]
    assert set(block["methods"]) == set(tool.METHODS)
    assert all(m["runs"] == 2 for m in block["methods"].values())
    pair = block["paired"]["moead_tchebycheff_vs_memetic"]
    assert pair["pairs"] == 2 and pair["wins"] + pair["losses"] + pair["ties"] == 2
    assert (tmp_path / "runs.csv").exists()


def test_simplex_support_detection():
    tool = _load("moead_paired_study")
    # The middle point lies above the plane through the three extremes: no weighted sum selects it.
    points = [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.8, 0.8, 0.8)]
    assert tool.supported_flags(points, divisions=30) == [True, True, True, False]
    assert tool.supported_flags([(0.2, 0.2, 0.2)]) == [True]


# --- AHP and TOPSIS sensitivity ---------------------------------------------------

def test_ahp_consistent_matrix_recovers_its_weights():
    w = (0.6, 0.3, 0.1)
    matrix = [[w[i] / w[j] for j in range(3)] for i in range(3)]
    result = ahp_weights(matrix)
    assert result.weights == pytest.approx(w)
    assert result.lambda_max == pytest.approx(3.0)
    assert result.consistency_ratio == pytest.approx(0.0, abs=1e-9) and result.consistent


def test_ahp_reports_inconsistent_judgements():
    # gaps >> room, room >> workload, yet workload >> gaps: an intransitive cycle
    result = ahp_from_judgements(9, 1 / 9, 9)
    assert not result.consistent and result.consistency_ratio > 0.10
    assert sum(result.weights) == pytest.approx(1.0)
    assert ahp_from_judgements(1, 1, 1).weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    with pytest.raises(ValueError):
        ahp_weights([[1, 2], [3, 1]])
    with pytest.raises(ValueError):
        ahp_from_judgements(0, 1, 1)


FRONT = [(0.0, 0.30, 0.20), (2.0, 0.20, 0.10), (5.0, 0.10, 0.15), (9.0, 0.05, 0.05)]


def test_weight_sensitivity_is_consistent_with_topsis():
    s = weight_sensitivity(FRONT, (0.6, 0.2, 0.2), divisions=10, sweep_steps=20)
    assert s["base_choice"] == topsis(FRONT, (0.6, 0.2, 0.2))[0].solution_index
    assert sum(w["share"] for w in s["weight_space"]) == pytest.approx(1.0)
    assert s["grid_points"] == 66 and s["distinct_choices"] == len(s["weight_space"]) > 1
    gaps = next(x for x in s["sweeps"] if x["objective"] == "gaps")
    assert gaps["segments"][0]["from"] == 0 and gaps["segments"][-1]["to"] == 1
    assert gaps["segments"][-1]["index"] == 0  # all weight on gaps picks the zero-gap timetable
    assert len(gaps["switches"]) == len(gaps["segments"]) - 1
    assert s["stability"] is not None and 0 < s["stability"] <= 1


def test_single_alternative_is_always_recommended():
    s = weight_sensitivity([(1.0, 0.2, 0.1)])
    assert s["distinct_choices"] == 1 and s["stability"] is None
    assert weight_sensitivity([])["base_choice"] is None


def test_recommend_prefers_ahp_over_weights():
    out = recommend(FRONT, weights=[0, 0, 1], ahp=[9, 9, 1])
    assert out["weightSource"] == "ahp" and out["recommended"] == 0
    assert out["ahp"]["consistent"] and math.isclose(sum(out["weights"]), 1.0)
    assert set(out["profiles"]) == {"balanced", "student_first", "admin_room_fit", "faculty_fairness"}
    assert recommend(FRONT, weights=[0, 0, 1])["recommended"] == 3
    with pytest.raises(ValueError):
        recommend([(1.0, 2.0)])
