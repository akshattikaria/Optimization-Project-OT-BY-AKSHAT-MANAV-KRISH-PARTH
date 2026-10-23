# Exact references and what they prove

`timetable.exact.enumerate_pareto` needs only Python. It filters impossible
single-event choices (block geometry, room fit, and room/teacher availability),
then evaluates every complete choice of offering teachers and event placements.
It retains one schedule per non-dominated objective vector. As the filtering
cannot discard a feasible timetable, exhaustion certifies the **global Pareto
front for that finite modeled instance**. A combination limit raises an error
before evaluation when the problem is too large; a partial front is never
labeled complete.

The two exhaustive examples in `tests/test_exact_enumeration.py` establish that
the objectives can genuinely conflict:

| Micro instance | Globally non-dominated objective vectors `(gaps, room waste, workload SD)` | Complete candidate evaluations |
|---|---|---:|
| A 20-seat room is available at separated periods 0/2; a 40-seat room at 1/3 | `(0, 0.25, 0)` and `(1, 0, 0)` | 64 |
| Teacher A is available at 0/1, teacher B only at 3; both have a two-period weekly limit | `(0, 0, 0.5)` and `(1, 0, 0)` | 9 |

Both examples have one group, two single-session offerings, one day with four
teaching periods, and no lunch. In the first example, a compact day must use the
larger room for one event; using the best-fit room for both events leaves a gap.
In the second, assigning both events to A gives a compact day, while splitting
the teaching balances workloads and forces a gap. These are controlled
mathematical examples, not evidence that every synthetic college instance has
a large or strongly conflicting Pareto front.

`timetable.exact.cp_sat` optionally uses OR-Tools. Its joint alternative
`x[e,start,room,faculty]` is Boolean. Exactly one alternative is selected for
each event. Offering teacher variables satisfy
`sum(x[e,*,*,f]) = y[offering(e),f]`, so every session of an offering has the
same teacher. Resource occupancy, availability, daily/weekly limits, room fit,
and distinct-day restrictions match the independent evaluator.

Three modes are available:

- `feasibility` finds a valid schedule. `OPTIMAL` here only establishes the
  optimum of the constant-zero feasibility objective; it says nothing about
  timetable quality.
- `gaps` minimizes twice the student gaps plus the faculty gaps. Prefix/suffix
  busy indicators identify internal free teaching periods. Lunch is excluded.
- `room` minimizes exact integer-scaled rational room waste using the least
  common multiple of room capacities. Unsupported integer scales are rejected
  explicitly rather than silently rounded.

An `OPTIMAL` status and matching bound certify only the selected scalar
objective. `FEASIBLE` means a valid incumbent exists without a completed
optimality proof. `UNKNOWN` means the search did not finish a proof or find a
solution within its limit; it is not an infeasibility result. `INFEASIBLE` is
used only when CP-SAT proves infeasibility. The three-objective workload SD is
not approximated by another function and advertised as exact.

The solver runs with one worker and a fixed random seed. Time limits apply to
CP-SAT search; result runtime also includes model construction and extraction.
Runs that reach a wall-clock limit can differ between machines. The model is
intended for small reference instances, not the primary large-instance method.
Every extracted solution is independently checked and its scalar objective is
compared with the evaluator before it is returned.

## Exact ideal-vector certificates for the 40-event instances

`python experiments/certify_small_ideal.py` computes a separate workload lower bound by exact integer dynamic programming over offering-load multiplicities. For six faculty with identical allowance 20, twelve offerings of load 3 and four of load 2 total 44 periods. Dropping eligibility, daily limits and all temporal interactions gives a relaxation. Exhausting assignments of these indivisible loads to faculty bins yields minimum sum of squared loads 328. The relative-load variance lower bound is exactly `328/(6*20^2) - (44/120)^2 = 1/450`. Zero gaps and the per-event static rational room minimum `14/55` provide the other lower bounds.

For all three executed small instance seeds (10,20,30), archived feasible schedules attain the bounds simultaneously. The script checks integer gaps, exact rational seat wastage and integer squared loads. Consequently each global three-objective front is the singleton ideal vector `(0,14/55,sqrt(1/450))`; distinct schedules may have that same vector. This is stronger than a discovered front and explains why inventing a broad small-instance Pareto front would be wrong. It does not say suboptimal schedules never exhibit opposing objective changes. Bounds and exact attaining archive references are saved in `results/benchmark/small_ideal_certificates.json`.

This analysis is deliberately limited to small counts and equal allowances. General unequal-allowance workload optimization remains outside the CP-SAT API. The dynamic program is validated against exhaustive labeled teacher assignments in `tests/test_ideal_bound.py`.
