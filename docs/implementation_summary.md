# Implementation and review summary

The working optimization system, experiments, tests and academic documentation are complete in the cloud repository. Changes remain uncommitted and unpushed; no PR, merge or access change was made. The original attached proposal and assignment are preserved under `problem/references/`, with a requirements/corrections map in `docs/reference_review.md`.

## Implemented system

- Typed problem/schedule schema, versioned JSON serialization, validation and a configurable synthetic engineering-college generator at exactly 40, 150 and 400 events.
- Separate known-feasible certificates. Qualifications and availability preserve the certificate, but optimizer inputs and metadata contain no certificate schedule.
- An independent-from-search hard-constraint checker covering complete demand, eligible offering-consistent teachers, all occupied periods of multi-period labs, lunch/day boundaries, capacity/type, availability, resource clashes, daily/weekly teaching limits and distinct-day rules.
- Three objectives: student gaps plus half faculty gaps; duration-weighted unused seat fraction; population SD of relative faculty teaching loads. Offering-level faculty decisions make the last objective genuinely optimizable.
- A meaningful DSatur-style constructive baseline, strong equal-budget repeated greedy, transparent standard NSGA-II, initialization-only ablation, and enhanced NSGA-II with actual reconstruction repair and feasible local search including whole-offering faculty exchanges.
- Optional exact CP-SAT gaps/room/feasibility models, exhaustive micro-instance Pareto enumeration, and an exact load-partition relaxation with rational/integer ideal-vector certificates.
- Exact 3-D hypervolume, non-dominated sorting, spacing, convergence and feasibility histories, paired descriptive statistics and instance-level sign tests. All algorithms use common objectives and within-scenario evaluation caps; runtime includes constructive work.
- A CLI, event CSV/JSON, Markdown grids, group/faculty/room CSV grids, 24 baseline/enhanced representative selections, Pareto projections/3-D plots, convergence and comparison charts.

## Executed verification

**136 automated tests passed**, and the same **136 passed after a fresh snapshot installation in a new virtual environment using pinned dependencies**. The installed CLI generated, optimized and validated a separate seed-42 instance. Dependency checks passed. A snapshot was used because uncommitted work cannot be retrieved by a remote clone.

Tests include hand-computed objective values; all important hard constraints; invalid and infeasible inputs; laboratory continuation and lunch; shared atomic groups; faculty consistency; variable workload; all three problem sizes; deterministic seeds and JSON round trips; constrained sorting/crowding; hypervolume checked against independent inclusion-exclusion; exact references; load-partition DP checked against brute-force assignments; fair evaluator charging; and resume/cache integrity.

The saved benchmark contains **192 completed runs**, **109,440 charged search evaluations**, seven instance certificates and **769 archived feasible schedules**. The shared checker revalidated every archived schedule and certificate outside search, including in the fresh installation. CP-SAT and hand-derived examples supply additional independent checks; this is not presented as a second independently written constraint checker.

## Measured findings

| Size | Runs per method | Greedy mean HV | Enhanced mean HV | Greedy mean seconds | Enhanced mean seconds |
|---|---:|---:|---:|---:|---:|
| Small | 30 | 0.935319 | 0.935173 | 7.918 | 2.432 |
| Medium | 12 | 1.017029 | 0.981213 | 18.550 | 6.087 |
| Large | 6 | 0.937241 | 0.909322 | 48.520 | 16.236 |

Greedy, seeded and enhanced each found feasible schedules in **48/48 runs**. Plain NSGA-II found none in **48 runs at these caps**, and its empty fronts correctly receive HV=0. This is bounded-search failure on known-feasible inputs, not an infeasibility proof or a claim that NSGA-II can never work.

Enhanced NSGA-II was about three times faster than repeated greedy but had lower mean hypervolume at every size. Small-instance HV was close; medium/large losses were larger. The evidence supports feasibility recovery and runtime savings, not universal quality superiority. Only two or three independent instances per size limit statistical generalization.

A stronger result explains the small cases: all three checked 40-event instances attain the exact lower bounds `(gaps, room waste, load SD) = (0,14/55,sqrt(1/450))`. Integer workload partitioning gives minimum squared-load sum 328 and loads `[6,6,8,8,8,8]`; exact rational room calculations and zero integer gaps certify simultaneous attainment. Their **global objective front is a singleton**. No artificial broad front is needed. Separate exhaustive micro-instances prove that genuine unavoidable objective conflicts are possible. Medium/large fronts remain discovered non-dominated sets, without global optimality proofs.

CP-SAT separately proved gaps=0 and room waste=14/55 on the reference small case, in approximately 2.75 and 2.32 seconds respectively. Those scalar optima do not constrain the other objectives. Bounds, statuses and exact outputs are saved.

## Reproduction

From the extracted project root or cloud repository, in Bash:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
python main.py generate --size small --seed 42 --output data/demo.json
python main.py solve data/demo.json --method enhanced --seed 0 --budget 240 --population 24 --output results/demo --figures figures/demo
python main.py evaluate data/demo.json results/demo/timetables/normalized_compromise/schedule.json
python experiments/verify_saved_results.py --study benchmark
python experiments/certify_small_ideal.py --study benchmark
python main.py plot --results results/benchmark --figures figures/benchmark
python experiments/export_review.py --study benchmark
python experiments/write_report.py
```

For new solver measurements, copy `experiments/benchmark.json`, change its `name`, and run `python main.py experiment --config YOUR_CONFIG.json`. The README provides exact copy commands. `--resume` preserves existing measurements and requires matching source/configuration/runtime versions. Recorded source hashes match the delivered algorithm files. Runtime naturally varies with hardware/load.

## Files changed or created

- Replaced `README.md`; added `.gitignore`, `main.py`, `pyproject.toml`, `requirements.txt` and `requirements-lock.txt`.
- Added 15 modules in `src/timetable/`: model, generation, evaluation, greedy, NSGA-II, hybrid, exact, metrics, experiments, export, visualization, CLI, analytical examples and package entry points.
- Added 12 test modules under `tests/`.
- Added three experiment JSON configurations plus verification, representative export, ideal certification and report-generation scripts under `experiments/`.
- Added original references, problem statement, literature review, full mathematical formulation, requirement corrections, prospective four-member/eight-week plan, exact guarantees, report and presentation/demo/viva material.
- Added synthetic instances/certificates under `data/`; raw run records, tables, grids, exact certificates and verification logs under `results/`; generated charts and timetable images under `figures/`.
- `docs/file_inventory.txt` lists every delivered project file. The full report has the professor's 19 requested sections and derives measured tables from verified records.

## Assumptions, incomplete work and recommendations

The week repeats with equal periods, fixed session demand, disjoint atomic cohorts, one teacher per offering and exact room-kind matching. Default generator labs use entire cohorts. Model input can represent shared atomic groups and specialized room kinds, but individual enrollment overlaps, travel, alternating weeks, fixed starts, team teaching, uncertainty and non-teaching workloads are omitted.

All data are synthetic and planted feasible. The distribution is biased by its construction and generous staffing/resource limits. Larger heuristic archives have no global Pareto certificate; arbitrary inputs can be infeasible. The hypervolume gap normalization is deliberately loose and its policy sensitivity was not tested. Optional aggregate coverage maps empty targets to zero by documented convention; those undefined cases are excluded from substantive conclusions.

The larger five-instance/30-seed-per-instance study is configured but **not executed**. Real-data/ITC validation, systematic sensitivity/disruption studies, equal-runtime comparisons, full repair-versus-local-search ablations, general exact workload optimization, NSGA-III/MOEA/D/SA/weighted GA, Excel export and an interactive UI are not claimed. The full enhancement does not beat strong greedy on HV; further quality improvements need development-instance tuning and new controlled experiments.

Recommended next work: validate anonymized real inputs and resource pressure first; expand independent instances and reference/scale sensitivity; compare quality under equal runtime; profile incremental resource checks and more targeted faculty exchanges; then add true minimum-disruption rescheduling or extra algorithms only where evidence justifies them.

Administrative approval/access, confirmed group/professor details, authentic individual contributions, eight weeks of personal commits and actual presentation/viva delivery remain the group's responsibility. No history or identity was fabricated.
