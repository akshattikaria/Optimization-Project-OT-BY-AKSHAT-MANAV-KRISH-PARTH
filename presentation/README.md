# Presentation and reproducible demonstration guide

This is a ready-to-adapt Markdown presentation outline with speaking points. It does not assert student authorship, faculty approval, or live measurements. Copy measured values and figures only from the saved experiment report; retain unfavorable comparisons and failed-run information.

## Slide 1 — Decision and stakeholders

**Multi-Objective College Timetable Optimization**

A coordinator assigns required weekly sessions to times, compatible rooms, and eligible teachers. Students need clash-free attendance and compact days; teachers need feasible and balanced assignments; the institute needs sensible room fit. The project supports a choice among feasible trade-offs.

State the four real member names and course details only after confirmation.

## Slide 2 — What is controlled, and what is fixed?

Fixed inputs: groups and sizes, session demand/duration, room types/capacities, faculty eligibility/limits, unavailable periods, and lunch. Decisions: one room/time block per event and one teacher per offering. All sessions of an offering share that teacher. Explain why selecting the teacher makes weekly workload imbalance optimizable.

## Slide 3 — Hard constraints before quality

Show a two-period lab occupying both cells of a weekly grid. It cannot cross lunch, leave the day, use an incompatible lab, overlap a group/room/teacher, or exceed a teaching limit. Every required session appears once. A timetable with one clash remains infeasible regardless of its apparent objective quality.

## Slide 4 — Three measurable objectives

Show the equations from `formulation/README.md`:

1. Internal student gaps plus half the faculty gap total, excluding lunch.
2. Duration-weighted unused fraction of seats in occupied rooms.
3. Population standard deviation of weekly load divided by each teacher's allowance.

Explain what each measure omits. Room wastage is not campus room-hour utilization; workload does not include research/administration. These are explicit modeling assumptions.

## Slide 5 — Why multiple objectives?

Use the formulation's micro-example: a 20-seat room only at periods 0 and 2, versus a 40-seat room available at 1 and 3. Two 20-student sessions can have zero room wastage with one internal gap, or zero gap with average wastage 0.25. Explain that generated-instance conflict must still be measured. The three executed 40-event cases are stronger counterexamples to assumed conflict: feasible schedules simultaneously attain exact gap, room-fit, and relaxed workload lower bounds, certifying a singleton ideal objective front `(0, 14/55, sqrt(1/450))`. This says nothing about every random seed or larger case. The micro-examples prove conflict is possible; the ideal certificates prove compromise is not necessary on these three cases.

## Slide 6 — Methods and rationale

The constructive greedy/DSatur-style baseline prioritizes difficult resource conflicts and provides an interpretable reference. Standard NSGA-II uses constrained dominance, ranking, crowding, variation, and parent–offspring survival. The enhancement addresses the observed difficulty of finding or retaining feasible schedules. The separate `hybrid.py` implements inheritance-biased reconstruction repair and four local neighborhoods: compaction, room fit, load-aware teacher transfer, and reciprocal-qualified whole-offering teacher exchange. Describe the enabled switches from the saved configuration; this reconstruction is not a Kempe chain or a minimum-disruption guarantee. Optional CP-SAT checks small instances with explicit statuses and bounds.

## Slide 7 — Data and verification

All main data are synthetic. State the sizes and instance seeds actually executed. A separately stored known-feasible witness verifies that selected generated instances can be scheduled; solvers do not receive that witness. Hand-calculated tests check objective definitions, and constraint tests cover labs, lunch, capacities, clashes, teacher consistency, teaching limits, malformed inputs, and infeasible cases. Cite the actual test result in the final report, not an expected count.

## Slide 8 — Fair experiments

Show the saved configuration and explain instance seed versus algorithm seed. Present declared candidate-evaluation budgets and total runtime; include seeding, repair, and local-search costs according to the experiment's accounting. Show run feasibility, front size, fixed-reference hypervolume, convergence, and objective ranges. Distinguish a small executed study from any larger runnable configuration.

## Slide 9 — Results and practical choice

Insert the generated Pareto projections, convergence chart, and comparison table from `figures/` and `results/`. A size-level plot pooling different instance seeds is a distribution of per-run fronts, not one common feasible front. Explain trade-offs using the fixed-instance diagnostics and schedules. Interpret the measured trade-offs in timetable terms. Show a baseline grid and at least one feasible optimized grid. If NSGA-II fails, or greedy is stronger on some metric, explain the result directly. A heuristic archive is a discovered non-dominated set. The exact micro-instance exhaustions and the three small-instance attaining lower-bound certificates have stronger global guarantees; medium/large archives do not.

## Slide 10 — Limits and next work

Synthetic data, a repeating deterministic week, atomic cohorts, equal periods, one teacher per offering, no travel or uncertain demand, and bounded search restrict the conclusions. Next steps: validate real anonymized inputs, add standard benchmark import with faithful constraint mapping, expand independent instance seeds, evaluate more extensive sensitivity/disruption scenarios, improve incremental feasibility checks, and investigate exact epsilon-constraint fronts on tiny instances.

State explicitly which advanced proposals were not implemented or not experimentally completed. Administrative approval/access and eight weeks of student contribution evidence remain the group's responsibility.

## Live demonstration sequence

Use the exact CLI commands in the root README after installing the documented dependencies:

1. Display `python main.py --help` and the saved experiment configuration.
2. Generate a small synthetic instance with a declared seed; show the problem JSON and separate witness file.
3. Evaluate the witness to establish known feasibility, without passing it into any solver.
4. Run the greedy baseline, evaluate its output, and show a per-group weekly grid.
5. Run standard and enhanced NSGA-II with the declared budget; show explicit feasible archive sizes and violation histories.
6. Re-evaluate a saved representative schedule independently, then display its objectives and grid.
7. Regenerate plots from saved raw results rather than relying on a screenshot or hand-edited table.
8. Show an intentionally infeasible or malformed candidate and its positive violation; do not export it as a valid timetable.

For a time-limited viva, precompute the larger experiment and demonstrate the small case live. Keep the raw records, configuration, schedule exports, and verification results available to inspect.

## Viva questions every member should answer

- Why is the faculty decision per offering rather than per event?
- Which term would become constant if teachers were fixed?
- Why does a two-period lab need two occupied-slot checks?
- Why is lunch excluded from the gap objective?
- What does relative workload balance mean when weekly allowances differ?
- What exactly does constraint domination do when both candidates are infeasible?
- How does crowding preserve diversity, and how are equal objective vectors handled?
- Why can generic crossover destroy a valid timetable?
- What work is counted in the experiment budget and what does runtime add to that comparison?
- What does zero hypervolume mean for a run with no feasible schedule?
- Why does a time-limited feasible CP-SAT solution not certify optimality?
- Why is a discovered non-dominated schedule not necessarily globally Pareto-optimal?
- What evidence would justify replacing the baseline or claiming the enhancement helps?
- Which institutional rules are absent, and how would adding them change the model?
