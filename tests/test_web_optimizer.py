"""The website run reports measured comparison data only: nothing is estimated for algorithms that did not run."""
import api.optimizer as optimizer
from api.bridge import demo_problem
from api.optimizer import RunConfig, run_optimizer

COMPARED = ("moead", "amosa", "nsga2")


def test_run_without_comparison_reports_only_the_final_method():
    stats = run_optimizer(demo_problem(), RunConfig(budget=40, search_seconds=None, compare_baseline=False), save=False)["stats"]
    assert [a["id"] for a in stats["algorithmComparison"]] == ["memetic"]
    assert list(stats["multiConvergence"]) == ["memetic"]
    assert not any(f"{key}Front" in stats for key in COMPARED)
    assert stats["baselineMethod"] is None and stats["baselineFront"] == []


def test_comparison_curves_are_measured_history_points():
    result = run_optimizer(demo_problem(), RunConfig(budget=60, search_seconds=None), save=False)
    stats = result["stats"]
    rows = {a["id"]: a for a in stats["algorithmComparison"]}
    assert set(rows) == {"memetic", "greedy", *COMPARED}
    for key, row in rows.items():
        assert row["status"] in {"ok", "no_feasible"}
        curve = stats["multiConvergence"][key]
        evaluations = [p["evaluations"] for p in curve]
        assert evaluations == sorted(evaluations) and 0 < evaluations[0] and evaluations[-1] <= stats["evaluations"]
        if key != "greedy":  # greedy records every evaluation and is thinned to about 30 samples
            assert abs(curve[-1]["hypervolume"] - row["hv"]) < 1e-3  # the curve ends at the reported hypervolume
        front = stats["baselineFront" if key == "greedy" else f"{key}Front"]
        assert len(front) == row["frontSize"] and (row["status"] == "ok") == bool(front)
        if row["status"] == "no_feasible":
            assert row["hv"] == 0 and all(p["hypervolume"] == 0 for p in curve)


def test_failed_comparison_is_reported_not_replaced(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("deliberate test failure")

    monkeypatch.setattr(optimizer, "moead", broken)
    stats = run_optimizer(demo_problem(), RunConfig(budget=40, search_seconds=None), save=False)["stats"]
    row = next(a for a in stats["algorithmComparison"] if a["id"] == "moead")
    assert row["status"] == "failed" and row["hv"] is None and row["frontSize"] == 0
    assert "deliberate test failure" in row["error"]
    assert "moead" not in stats["multiConvergence"] and "moeadFront" not in stats
