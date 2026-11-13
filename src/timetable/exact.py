"""Optional CP-SAT reference: feasibility, single objectives and the gaps/room trade-off.

CP-SAT proves a *single-objective* optimum only when its status is OPTIMAL.
The model contains two of the project's three objectives: weighted gaps (f1)
and room seat wastage (f2).  The workload standard deviation (f3) is not
modeled.  The epsilon-constraint and weighted-sum routines therefore describe
the **two-objective (f1, f2) trade-off only**; they do not produce or certify
the three-objective Pareto front.  ``exact_gaps_room_front`` returns the
complete (f1, f2) front when every solve ends with a proof.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import isfinite, lcm
from itertools import product
from time import perf_counter
from typing import Any

from .model import (
    Placement,
    Problem,
    Schedule,
    compatible_rooms,
    valid_starts,
    validate_problem,
)


@dataclass
class ExactResult:
    schedule: Schedule | None
    status: str
    objective_value: float | None
    best_bound: float | None
    runtime: float
    objective: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def proven_optimal(self) -> bool:
        """Whether CP-SAT completed an optimality proof for the chosen objective."""
        return self.status == "OPTIMAL"


@dataclass
class EnumerationResult:
    """A complete, globally non-dominated front for a deliberately tiny model."""

    archive: list[Schedule]
    objectives: list[tuple[float, float, float]]
    evaluated: int
    feasible_candidates: int
    runtime: float
    status: str = "COMPLETE"


@dataclass
class EpsilonParetoResult:
    """Gaps-versus-room trade-off points from an epsilon grid with CP-SAT.

    Each grid step that ends OPTIMAL (``optimal_steps``) proves the minimum of
    the primary objective under that bound on the other one.  The result is a
    set of trade-off points for (f1, f2), not a certified complete front, and
    says nothing about workload balance (f3).
    """

    archive: list[Schedule]
    objectives: list[tuple[float, float, float]]
    evaluated_steps: int
    optimal_steps: int
    feasible_steps: int
    runtime: float
    epsilons: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


def enumerate_pareto(problem: Problem, max_combinations: int = 100_000) -> EnumerationResult:
    """Exhaust all structurally valid genotypes and certify a tiny Pareto front.

    This reference needs no solver dependency.  Room fit, availability and
    block geometry are filtered without excluding any feasible assignment;
    every remaining complete genotype is independently evaluated.  One
    representative is retained per objective vector.  An a priori combination
    bound prevents accidentally starting an institute-scale exhaustive run.
    Exceeding the bound raises ValueError and does not produce a partial front
    that could be mistaken for an exact result.
    """
    from .evaluation import evaluate

    validate_problem(problem)
    if type(max_combinations) is not int or max_combinations <= 0:
        raise ValueError("max_combinations must be a positive integer")
    started = perf_counter()
    faculty = {f.id: f for f in problem.faculty}
    rooms = {r.id: r for r in problem.rooms}
    offering_index = {o.id: i for i, o in enumerate(problem.offerings)}
    teacher_choices = 1
    for offering in problem.offerings:
        teacher_choices *= len(offering.eligible_faculty)
        if teacher_choices > max_combinations:
            raise ValueError("Teacher assignment space exceeds max_combinations")

    domains = {}
    for event in problem.events:
        offering = problem.offerings[offering_index[event.offering_id]]
        for teacher_id in offering.eligible_faculty:
            domains[event.id, teacher_id] = tuple(
                Placement(start, room_id)
                for start in valid_starts(problem, event)
                for room_id in compatible_rooms(problem, event)
                if not any(
                    slot in rooms[room_id].unavailable
                    or slot in faculty[teacher_id].unavailable
                    for slot in range(start, start + event.duration)
                )
            )
    # Count once before evaluating; teacher products are safely bounded above.
    combinations = 0
    for selected_teachers in product(*(o.eligible_faculty for o in problem.offerings)):
        count = 1
        for event in problem.events:
            teacher_id = selected_teachers[offering_index[event.offering_id]]
            count *= len(domains[event.id, teacher_id])
        combinations += count
        if combinations > max_combinations:
            raise ValueError("Scheduling space exceeds max_combinations")

    archive = []
    objective_vectors = []
    evaluated = feasible = 0

    def dominates(first, second):
        return all(a <= b for a, b in zip(first, second)) and any(
            a < b for a, b in zip(first, second)
        )

    for selected_teachers in product(*(o.eligible_faculty for o in problem.offerings)):
        event_domains = [
            domains[e.id, selected_teachers[offering_index[e.offering_id]]]
            for e in problem.events
        ]
        for selected_placements in product(*event_domains):
            schedule = Schedule(tuple(selected_placements), tuple(selected_teachers))
            assessment = evaluate(problem, schedule)
            evaluated += 1
            if not assessment.feasible:
                continue
            feasible += 1
            vector = assessment.objectives
            if vector in objective_vectors or any(
                dominates(other, vector) for other in objective_vectors
            ):
                continue
            kept = [
                i for i, other in enumerate(objective_vectors)
                if not dominates(vector, other)
            ]
            archive = [archive[i] for i in kept] + [schedule]
            objective_vectors = [objective_vectors[i] for i in kept] + [vector]
    ordered = sorted(zip(objective_vectors, archive), key=lambda item: item[0])
    return EnumerationResult(
        archive=[schedule for _, schedule in ordered],
        objectives=[vector for vector, _ in ordered],
        evaluated=evaluated,
        feasible_candidates=feasible,
        runtime=perf_counter() - started,
    )


def cp_sat(
    problem: Problem,
    objective: str = "feasibility",
    time_limit: float = 10.0,
    seed: int = 0,
    epsilon_objective: str | None = None,
    epsilon_bound: float | None = None,
    epsilon_bound_units: int | None = None,
    weights: tuple[int, int] | None = None,
) -> ExactResult:
    """Solve a small instance without consulting its generation witness.

    ``objective`` is ``feasibility``, ``gaps`` or ``room``.  The gaps objective
    uses the project's fixed faculty weight 0.5 and excludes lunch.  The room
    objective uses exact integer coefficients obtained from the least common
    multiple of room capacities, rather than rounded decimal penalties.

    When ``epsilon_objective`` and ``epsilon_bound`` are supplied, the solver
    minimizes ``objective`` subject to ``epsilon_objective <= epsilon_bound``.
    ``epsilon_bound_units`` states the same bound in the model's exact integer
    units (half-periods for gaps, ``room_scale`` units for room wastage) and
    avoids any floating-point rounding of the bound.

    ``objective="weighted"`` minimizes ``weights[0] * gap_units + weights[1] *
    room_units`` for nonnegative integer weights: a weighted-sum scalarization
    of the gaps and room objectives in exact integer arithmetic.

    Whenever the gap or room terms are part of the model, their exact integer
    values for the returned schedule are reported in the metadata as
    ``gap_units`` / ``room_units`` with ``gap_scale`` / ``room_scale``.

    One Boolean alternative jointly fixes an event's start, room and teacher;
    offering-level teacher variables enforce teaching consistency.  Resource
    constraints cover every period of multi-period sessions.

    OR-Tools is optional and imported only when this function is called.
    """
    if objective not in {"feasibility", "gaps", "room", "weighted"}:
        raise ValueError("objective must be 'feasibility', 'gaps', 'room' or 'weighted'")
    if objective == "weighted":
        if (weights is None or len(weights) != 2
                or any(isinstance(w, bool) or not isinstance(w, int) or w < 0 for w in weights)
                or sum(weights) == 0):
            raise ValueError("weighted objective needs two nonnegative integer weights, not both zero")
    elif weights is not None:
        raise ValueError("weights are only used with objective='weighted'")
    if epsilon_bound is not None and epsilon_bound_units is not None:
        raise ValueError("give epsilon_bound or epsilon_bound_units, not both")
    if epsilon_bound_units is not None and (isinstance(epsilon_bound_units, bool)
                                            or not isinstance(epsilon_bound_units, int)):
        raise ValueError("epsilon_bound_units must be an integer")
    if epsilon_objective is not None and epsilon_objective not in {"gaps", "room"}:
        raise ValueError("epsilon_objective must be 'gaps' or 'room'")
    if not isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be a finite positive number")
    if not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be an integer in [0, 2**31 - 1]")
    try:
        try:
            import pandas
        except Exception:
            import sys, types
            mock_pd = types.ModuleType('pandas')
            mock_pd.Index = type('Index', (), {})
            mock_pd.Series = type('Series', (), {})
            sys.modules['pandas'] = mock_pd
        from ortools.sat.python import cp_model
        import ortools
    except ImportError as exc:
        raise ImportError(
            "The CP-SAT reference requires OR-Tools. Install with "
            "`pip install -e '.[exact]'`."
        ) from exc

    validate_problem(problem)
    started = perf_counter()
    model = cp_model.CpModel()
    events = problem.events
    faculty = {f.id: f for f in problem.faculty}
    rooms = {r.id: r for r in problem.rooms}
    offering_index = {o.id: i for i, o in enumerate(problem.offerings)}
    offerings = {o.id: o for o in problem.offerings}
    width = problem.periods_per_day

    teachers = {}
    for offering in problem.offerings:
        for teacher in offering.eligible_faculty:
            teachers[offering.id, teacher] = model.NewBoolVar(
                f"teacher_{offering.id}_{teacher}"
            )
        model.AddExactlyOne(
            teachers[offering.id, f] for f in offering.eligible_faculty
        )

    room_occupancy: dict[tuple[str, int], list] = defaultdict(list)
    faculty_occupancy: dict[tuple[str, int], list] = defaultdict(list)
    group_occupancy: dict[tuple[str, int], list] = defaultdict(list)
    faculty_weekly: dict[str, list] = defaultdict(list)
    faculty_daily: dict[tuple[str, int], list] = defaultdict(list)
    offering_days: dict[tuple[str, int], list] = defaultdict(list)
    choices: list[list[tuple[Any, int, str, str]]] = []
    room_terms = []
    room_coefficient_sum = 0
    total_duration = sum(e.duration for e in events)
    capacity_lcm = lcm(*(r.capacity for r in problem.rooms))
    room_scale = capacity_lcm * max(total_duration, 1)

    need_room_terms = objective in {"room", "weighted"} or epsilon_objective == "room"
    need_gap_terms = objective in {"gaps", "weighted"} or epsilon_objective == "gaps"

    if need_room_terms and room_scale > 2**60:
        raise ValueError(
            "Room capacities produce an unsafe exact integer scale; "
            "use fewer distinct capacities for this small-instance reference."
        )

    for event in events:
        offering = offerings[event.offering_id]
        options = []
        event_teacher: dict[str, list] = defaultdict(list)
        for start in valid_starts(problem, event):
            block = tuple(range(start, start + event.duration))
            day = start // width
            for room_id in compatible_rooms(problem, event):
                room = rooms[room_id]
                if any(t in room.unavailable for t in block):
                    continue
                for teacher_id in offering.eligible_faculty:
                    teacher = faculty[teacher_id]
                    if any(t in teacher.unavailable for t in block):
                        continue
                    variable = model.NewBoolVar(
                        f"x_{event.id}_{start}_{room_id}_{teacher_id}"
                    )
                    options.append((variable, start, room_id, teacher_id))
                    event_teacher[teacher_id].append(variable)
                    faculty_weekly[teacher_id].append(event.duration * variable)
                    faculty_daily[teacher_id, day].append(event.duration * variable)
                    if offering.distinct_days:
                        offering_days[offering.id, day].append(variable)
                    for slot in block:
                        room_occupancy[room_id, slot].append(variable)
                        faculty_occupancy[teacher_id, slot].append(variable)
                        for group in event.groups:
                            group_occupancy[group, slot].append(variable)
                    if need_room_terms:
                        coefficient = (
                            event.duration
                            * (room.capacity - event.size)
                            * (capacity_lcm // room.capacity)
                        )
                        room_coefficient_sum += coefficient
                        if room_coefficient_sum > 2**60:
                            raise ValueError(
                                "Room objective coefficients exceed the safe "
                                "exact integer range for this reference."
                            )
                        room_terms.append(coefficient * variable)
        choices.append(options)
        model.AddExactlyOne(x[0] for x in options)
        for teacher_id in offering.eligible_faculty:
            model.Add(
                sum(event_teacher[teacher_id])
                == teachers[offering.id, teacher_id]
            )

    for occupancy in (room_occupancy, faculty_occupancy, group_occupancy):
        for variables in occupancy.values():
            if len(variables) > 1:
                model.AddAtMostOne(variables)
    for teacher in problem.faculty:
        model.Add(sum(faculty_weekly[teacher.id]) <= teacher.max_weekly)
        for day in range(len(problem.days)):
            model.Add(
                sum(faculty_daily[teacher.id, day]) <= teacher.max_daily
            )
    for variables in offering_days.values():
        model.Add(sum(variables) <= 1)

    # Build gap terms if needed for objective or epsilon bound
    gap_terms = []
    gap_scale = 2
    if need_gap_terms:
        teachable_periods = [
            p for p in range(width) if p not in problem.lunch_periods
        ]
        for resources, occupancy, weight, label in (
            (problem.groups, group_occupancy, 2, "group"),
            (problem.faculty, faculty_occupancy, 1, "faculty"),
        ):
            for resource in resources:
                for day in range(len(problem.days)):
                    busy = []
                    for period in teachable_periods:
                        slot = day * width + period
                        indicator = model.NewBoolVar(
                            f"busy_{label}_{resource.id}_{slot}"
                        )
                        model.Add(indicator == sum(occupancy[resource.id, slot]))
                        busy.append(indicator)
                    for index in range(1, len(busy) - 1):
                        prefix = model.NewBoolVar(
                            f"before_{label}_{resource.id}_{day}_{index}"
                        )
                        suffix = model.NewBoolVar(
                            f"after_{label}_{resource.id}_{day}_{index}"
                        )
                        model.AddMaxEquality(prefix, busy[:index])
                        model.AddMaxEquality(suffix, busy[index + 1 :])
                        gap = model.NewBoolVar(
                            f"gap_{label}_{resource.id}_{day}_{index}"
                        )
                        model.Add(gap <= prefix)
                        model.Add(gap <= suffix)
                        model.Add(gap <= 1 - busy[index])
                        model.Add(gap >= prefix + suffix - busy[index] - 1)
                        gap_terms.append(weight * gap)

    # Apply Epsilon Constraint if requested
    if epsilon_objective is not None and (epsilon_bound is not None or epsilon_bound_units is not None):
        scale = room_scale if epsilon_objective == "room" else gap_scale
        scaled_bound = (epsilon_bound_units if epsilon_bound_units is not None
                        else int(round(epsilon_bound * scale)))
        model.Add(sum(room_terms if epsilon_objective == "room" else gap_terms) <= scaled_bound)

    objective_scale = 1
    if objective == "weighted":
        if weights[0] * gap_scale * len(gap_terms) + weights[1] * room_coefficient_sum > 2**60:
            raise ValueError("Weighted objective coefficients exceed the safe exact integer range.")
        model.Minimize(weights[0] * sum(gap_terms) + weights[1] * sum(room_terms))
    elif objective == "room":
        objective_scale = room_scale
        model.Minimize(sum(room_terms))
    elif objective == "gaps":
        objective_scale = gap_scale
        model.Minimize(sum(gap_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed
    status_code = solver.Solve(model)
    status = solver.StatusName(status_code)
    metadata = {
        "ortools_version": ortools.__version__,
        "option_variables": sum(len(options) for options in choices),
        "branches": solver.NumBranches(),
        "conflicts": solver.NumConflicts(),
        "solver_wall_time": solver.WallTime(),
        "time_limit": time_limit,
        "seed": seed,
        "workers": 1,
        "objective_scale": objective_scale,
        "epsilon_objective": epsilon_objective,
        "epsilon_bound": epsilon_bound,
    }
    if status_code == cp_model.MODEL_INVALID:
        metadata["validation_error"] = model.Validate()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ExactResult(
            None, status, None, None, perf_counter() - started, objective, metadata
        )

    placements = []
    assigned_teachers = [""] * len(problem.offerings)
    for event, options in zip(events, choices):
        selected = next(option for option in options if solver.Value(option[0]))
        _, start, room_id, teacher_id = selected
        placements.append(Placement(start, room_id))
        assigned_teachers[offering_index[event.offering_id]] = teacher_id
    schedule = Schedule(tuple(placements), tuple(assigned_teachers))
    
    from .evaluation import evaluate
    assessment = evaluate(problem, schedule)
    if not assessment.feasible:
        raise RuntimeError(
            f"CP-SAT extracted an invalid schedule: {assessment.violations}"
        )
    if need_gap_terms:
        metadata["gap_units"] = sum(int(solver.Value(term)) for term in gap_terms)
        metadata["gap_scale"] = gap_scale
        if abs(metadata["gap_units"] / gap_scale - assessment.objectives[0]) > 1e-6:
            raise RuntimeError("CP-SAT gap units and evaluation disagree")
    if need_room_terms:
        metadata["room_units"] = sum(int(solver.Value(term)) for term in room_terms)
        metadata["room_scale"] = room_scale
        if abs(metadata["room_units"] / room_scale - assessment.objectives[1]) > 1e-6:
            raise RuntimeError("CP-SAT room units and evaluation disagree")
    value = solver.ObjectiveValue() / objective_scale
    bound = solver.BestObjectiveBound() / objective_scale
    if objective in {"gaps", "room"}:
        expected = assessment.objectives[0 if objective == "gaps" else 1]
        if abs(value - expected) > 1e-6:
            raise RuntimeError(f"CP-SAT objective ({value}) and evaluation ({expected}) disagree")
    return ExactResult(
        schedule, status, value, bound, perf_counter() - started, objective, metadata
    )


def epsilon_constraint_cp_sat(
    problem: Problem,
    primary_objective: str = "gaps",
    epsilon_objective: str = "room",
    steps: int = 5,
    time_limit_per_step: float = 10.0,
    seed: int = 0,
) -> EpsilonParetoResult:
    """Trade-off points between gaps (f1) and room wastage (f2) by epsilon-constraint.

    The method minimizes one objective while the other is bounded,
    ``f_other <= eps``, for ``steps`` values of ``eps`` spread evenly between
    the two single-objective anchor solutions.  A weighted sum can only return
    supported points; a bounded objective can also return unsupported ones.

    What is and is not proven:

    * a step with status OPTIMAL proves the minimum of the primary objective
      for that bound; steps that hit the time limit prove nothing;
    * the bounded objective is not minimized afterwards, so a returned point
      can be only weakly efficient until the final dominance filter;
    * a fixed grid can step over front points, so the result is generally a
      subset of the (f1, f2) front;
    * workload balance (f3) is not in the CP-SAT model.  Its value is reported
      for each schedule and used by the final dominance filter, but it is not
      optimized: this is not a three-objective Pareto front.

    Use ``exact_gaps_room_front`` for the complete, certified (f1, f2) front.

    Parameters:
        problem: Target instance.
        primary_objective: Objective to minimize ('gaps' or 'room').
        epsilon_objective: Objective to bound ('room' or 'gaps').
        steps: Number of grid points between the minimum and maximum bound.
        time_limit_per_step: Solver time limit (seconds) for each step.
        seed: Solver seed.
    """
    from .evaluation import evaluate

    if primary_objective not in {"gaps", "room"}:
        raise ValueError("primary_objective must be 'gaps' or 'room'")
    if epsilon_objective not in {"gaps", "room"}:
        raise ValueError("epsilon_objective must be 'gaps' or 'room'")
    if primary_objective == epsilon_objective:
        raise ValueError("primary_objective and epsilon_objective must differ")
    if steps < 2:
        raise ValueError("steps must be an integer >= 2")

    started = perf_counter()

    # Step 1: Find lower bound of epsilon_objective (anchor point 1)
    min_res = cp_sat(problem, objective=epsilon_objective, time_limit=time_limit_per_step, seed=seed)
    if min_res.schedule is None:
        return EpsilonParetoResult([], [], 0, 0, 0, perf_counter() - started, [], {"error": f"No solution for the {epsilon_objective} anchor within the time limit (status {min_res.status}); this is not an infeasibility proof unless the status is INFEASIBLE"})
    min_eps = min_res.objective_value

    # Step 2: Find anchor point 2 (minimizing primary_objective)
    prim_res = cp_sat(problem, objective=primary_objective, time_limit=time_limit_per_step, seed=seed)
    if prim_res.schedule is None:
        return EpsilonParetoResult([], [], 0, 0, 0, perf_counter() - started, [], {"error": f"No solution for the {primary_objective} anchor within the time limit (status {prim_res.status}); this is not an infeasibility proof unless the status is INFEASIBLE"})
    
    ev_prim = evaluate(problem, prim_res.schedule)
    max_eps = ev_prim.objectives[1 if epsilon_objective == "room" else 0]

    # Step 3: Construct epsilon grid
    if max_eps <= min_eps + 1e-6:
        eps_grid = [min_eps]
    else:
        eps_grid = [min_eps + i * (max_eps - min_eps) / (steps - 1) for i in range(steps)]

    discovered_schedules: list[Schedule] = []
    discovered_objectives: list[tuple[float, float, float]] = []
    optimal_count = 0
    feasible_count = 0

    # Include initial anchor schedules
    for sched in (min_res.schedule, prim_res.schedule):
        ev = evaluate(problem, sched)
        if ev.feasible and ev.objectives not in discovered_objectives:
            discovered_schedules.append(sched)
            discovered_objectives.append(ev.objectives)

    # Step 4: Step through the epsilon grid
    for eps in eps_grid:
        step_res = cp_sat(
            problem,
            objective=primary_objective,
            time_limit=time_limit_per_step,
            seed=seed,
            epsilon_objective=epsilon_objective,
            epsilon_bound=eps,
        )
        if step_res.status == "OPTIMAL":
            optimal_count += 1
        if step_res.schedule is not None:
            feasible_count += 1
            ev = evaluate(problem, step_res.schedule)
            if ev.feasible and ev.objectives not in discovered_objectives:
                discovered_schedules.append(step_res.schedule)
                discovered_objectives.append(ev.objectives)

    # Step 5: keep the points not dominated by another point found here
    def dominates(a, b):
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    pareto_schedules = []
    pareto_objectives = []
    for sched, objs in zip(discovered_schedules, discovered_objectives):
        if not any(dominates(other, objs) for other in discovered_objectives):
            pareto_schedules.append(sched)
            pareto_objectives.append(objs)

    ordered = sorted(zip(pareto_objectives, pareto_schedules), key=lambda it: it[0])

    return EpsilonParetoResult(
        archive=[s for _, s in ordered],
        objectives=[v for v, _ in ordered],
        evaluated_steps=len(eps_grid),
        optimal_steps=optimal_count,
        feasible_steps=feasible_count,
        runtime=perf_counter() - started,
        epsilons=eps_grid,
        metadata={
            "primary_objective": primary_objective,
            "epsilon_objective": epsilon_objective,
            "min_eps": min_eps,
            "max_eps": max_eps,
            "grid_points": len(eps_grid),
        },
    )


@dataclass
class BiObjectivePoint:
    """One point of the (gaps, room wastage) trade-off with its exact integer coordinates."""

    schedule: Schedule
    objectives: tuple[float, float, float]
    gap_units: int
    room_units: int
    proven: bool


@dataclass
class BiObjectiveFront:
    """Trade-off between weighted gaps (f1) and room wastage (f2) from CP-SAT.

    ``complete`` is true only when every solve finished with an optimality (or
    infeasibility) proof.  Then ``points`` is the entire Pareto front of the
    *two-objective* problem (f1, f2).  Workload balance (f3) is reported for
    each schedule but is not optimized, so this is not a three-objective front.
    """

    points: list[BiObjectivePoint]
    complete: bool
    solves: int
    runtime: float
    gap_scale: int
    room_scale: int
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def objectives(self) -> list[tuple[float, float, float]]:
        return [p.objectives for p in self.points]


def exact_gaps_room_front(
    problem: Problem,
    time_limit_per_solve: float = 20.0,
    seed: int = 0,
    max_points: int = 200,
) -> BiObjectiveFront:
    """Complete (gaps, room wastage) Pareto front by lexicographic epsilon-constraint.

    Each front point takes two CP-SAT solves in exact integer units:

    1. minimize gaps subject to ``room <= bound`` (initially unbounded);
    2. minimize room subject to ``gaps <=`` the optimum of step 1.

    Step 2 removes weakly efficient solutions, so the pair is Pareto-optimal
    for (f1, f2).  The bound is then tightened to one integer unit below the
    room value just found and the two steps repeat until the model is proven
    infeasible.  Unlike a fixed epsilon grid this cannot skip a front point,
    and unlike a weighted sum it also returns unsupported points.

    The front is certified only if every solve returns OPTIMAL and the final
    one INFEASIBLE; otherwise ``complete`` is false and the points found so
    far are returned with their individual ``proven`` flags.
    """
    from .evaluation import evaluate

    started = perf_counter()
    points: list[BiObjectivePoint] = []
    solves = 0
    complete = False
    bound: int | None = None
    gap_scale = room_scale = 0
    statuses: list[str] = []
    while len(points) < max_points:
        first = cp_sat(problem, "gaps", time_limit_per_solve, seed,
                       epsilon_objective="room", epsilon_bound_units=bound)
        solves += 1
        statuses.append(first.status)
        if first.status == "INFEASIBLE":
            complete = all(p.proven for p in points)
            break
        if first.schedule is None:
            break
        gap_units = first.metadata["gap_units"]
        second = cp_sat(problem, "room", time_limit_per_solve, seed,
                        epsilon_objective="gaps", epsilon_bound_units=gap_units)
        solves += 1
        statuses.append(second.status)
        if second.schedule is None:
            break
        gap_scale, room_scale = second.metadata["gap_scale"], second.metadata["room_scale"]
        proven = first.proven_optimal and second.proven_optimal
        points.append(BiObjectivePoint(
            second.schedule, evaluate(problem, second.schedule).objectives,
            second.metadata["gap_units"], second.metadata["room_units"], proven))
        if not proven:
            break
        bound = second.metadata["room_units"] - 1
        if bound < 0:
            complete = True
            break
    points.sort(key=lambda p: (p.gap_units, p.room_units))
    return BiObjectiveFront(points, complete, solves, perf_counter() - started, gap_scale, room_scale,
                            "lexicographic_epsilon_constraint",
                            {"statuses": statuses, "time_limit_per_solve": time_limit_per_solve, "seed": seed})


def weighted_sum_cp_sat(
    problem: Problem,
    steps: int = 9,
    time_limit_per_solve: float = 20.0,
    seed: int = 0,
) -> BiObjectiveFront:
    """Weighted-sum CP-SAT runs over the (gaps, room wastage) objectives.

    The two single-objective optima give the ranges of both objectives; each
    run then minimizes ``lambda * gaps/range + (1 - lambda) * room/range`` for
    ``lambda = k / (steps + 1)``, written with integer coefficients.  An optimum
    of a positive weighted sum is Pareto-optimal for (f1, f2), but only
    *supported* points (those on the convex hull of the front) can be optima,
    which is exactly the limitation the epsilon-constraint method avoids.
    """
    from math import gcd

    from .evaluation import evaluate

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    started = perf_counter()
    anchors = [cp_sat(problem, "gaps", time_limit_per_solve, seed, epsilon_objective="room"),
               cp_sat(problem, "room", time_limit_per_solve, seed, epsilon_objective="gaps")]
    solves = 2
    statuses = [a.status for a in anchors]
    if any(a.schedule is None for a in anchors):
        return BiObjectiveFront([], False, solves, perf_counter() - started, 0, 0, "weighted_sum",
                                {"statuses": statuses, "error": "no anchor solution"})
    gap_scale, room_scale = anchors[0].metadata["gap_scale"], anchors[0].metadata["room_scale"]
    gap_range = max(1, abs(anchors[1].metadata["gap_units"] - anchors[0].metadata["gap_units"]))
    room_range = max(1, abs(anchors[0].metadata["room_units"] - anchors[1].metadata["room_units"]))
    found: dict[tuple[int, int], BiObjectivePoint] = {}
    weights_used = []
    for k in range(1, steps + 1):
        # lambda * G / gap_range + (1 - lambda) * R / room_range, multiplied by the common denominator
        a, b = k * room_range, (steps + 1 - k) * gap_range
        common = gcd(a, b)
        a, b = a // common, b // common
        result = cp_sat(problem, "weighted", time_limit_per_solve, seed, weights=(a, b))
        solves += 1
        statuses.append(result.status)
        weights_used.append({"lambda": k / (steps + 1), "gap_weight": a, "room_weight": b, "status": result.status})
        if result.schedule is None:
            continue
        key = (result.metadata["gap_units"], result.metadata["room_units"])
        if key not in found or (result.proven_optimal and not found[key].proven):
            found[key] = BiObjectivePoint(result.schedule, evaluate(problem, result.schedule).objectives,
                                          key[0], key[1], result.proven_optimal)
    keys = list(found)
    points = [found[k] for k in keys
              if not any(o != k and o[0] <= k[0] and o[1] <= k[1] for o in keys)]
    points.sort(key=lambda p: (p.gap_units, p.room_units))
    return BiObjectiveFront(points, False, solves, perf_counter() - started, gap_scale, room_scale, "weighted_sum",
                            {"statuses": statuses, "weights": weights_used,
                             "all_proven": all(p.proven for p in points),
                             "time_limit_per_solve": time_limit_per_solve, "seed": seed})


def supported_points(units: list[tuple[int, int]]) -> list[bool]:
    """Exact supportedness of each point of a two-objective minimization front.

    A point is supported when it minimizes some positive weighted sum over the
    given points, i.e. it is a vertex or lies on an edge of the lower-left
    convex hull.  Integer cross products make the test exact.
    """
    front = sorted(p for p in set(units)
                   if not any(q != p and q[0] <= p[0] and q[1] <= p[1] for q in set(units)))
    hull: list[tuple[int, int]] = []
    for point in front:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1) < 0:
                hull.pop()  # hull[-1] lies strictly above the chord: unsupported
            else:
                break
        hull.append(point)
    return [u in hull for u in units]
