# Executed verification evidence

- `pytest.txt`: 136 tests passed in the working environment.
- `fresh_pytest.txt`: all 136 passed after copying a clean repository snapshot and installing the pinned dependencies into a new virtual environment. A remote clone could not include uncommitted work, so this is honestly labeled a snapshot test.
- `fresh_install.txt`: actual dependency and editable-package installation log.
- `fresh_dependency_check.txt`: no broken dependency requirements.
- `fresh_cli.txt`: generation, enhanced optimization and independent evaluation of a separate seed-42 demo. The result was feasible.
- `entrypoint_help.txt`: installed `timetable` entry point works.
- `fresh_saved_verification.json`: the fresh installation revalidated all 192 saved runs, 769 archived schedules and seven certificates.
- `resume.txt`: cached study resume completed after checking source/runtime identity and saved schedules.
- `ideal_certification.txt`: exact rational/integer small-instance bound certificates.
- `final_verification.json`: consolidated outcomes and benchmark-source hash match.

All schedule revalidation uses the shared checker outside optimization. Hand-calculated examples, independent hypervolume inclusion-exclusion, brute-force load-partition checks, exhaustive Pareto enumeration and the separately encoded CP-SAT model provide additional cross-checks. Timings in tests and benchmarks are observed wall times, not portability guarantees.
