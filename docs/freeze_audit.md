# Pre-freeze audit

Date: 2026-11-14 · Machine: Windows 11, Python 3.13.7, Node 20.19, OR-Tools 9.14 · Audited revision: the commit
"Sync report, summary and slide sources with the supplementary methods" (`f594524`) on top of merge `82e28e8`.

The previous integration check (61/61, [final_integration_check.md](final_integration_check.md)) was run before the
MOEA/D, AMOSA, MCDM and CP-SAT epsilon-constraint work was merged. This audit repeats the checks on the current tree
after the cleanup fixes.

## Results

| Check | Result |
|---|---|
| Full test suite in the working tree (`python -m pytest -q`) | **267 passed**, 0 failed, 0 skipped |
| Full test suite in a fresh `git clone` of the audited commit (tracked files only) | **267 passed**: nothing depends on untracked or ignored files |
| Frontend type-check (`tsc --noEmit`) and production build (`npm run build`) | Pass |
| End-to-end website check on an empty database (`experiments/web_integration_check.py --web ...`) | **61 passed, 0 failed, 0 skipped** |
| Browser check: normal run | Runs memetic NSGA-II and greedy only; the other three algorithms show "Not run" with no curve, points or numbers |
| Browser check: advanced benchmark | All five algorithms run; chart, table and milestone strip show measured values; standard NSGA-II is reported as "No feasible timetable" |
| Browser check: offline demo data | Labelled as demo data; comparison rows show "Not run / No data" |
| Browser check: TOPSIS panel | Recommendation, ranking, AHP weights and sensitivity are returned by `POST /api/mcdm` and displayed |

## What changed since the previous check

1. The website shows measured results only. Generated convergence curves, estimated hypervolumes, offset copies of
   the memetic front and hard-coded effort figures were removed.
2. MOEA/D, AMOSA and standard NSGA-II run only when *Advanced benchmark* is ticked.
3. The ITC-style dataset is labelled as a synthetic, ITC-inspired instance (`data/itc_inspired`).
4. The CP-SAT epsilon-constraint documentation states that it covers the two-objective (gaps, room wastage)
   trade-off, not the three-objective front.
5. Report, summary and slide sources describe the later additions as supplementary; the validated final method is
   still the memetic NSGA-II.
6. Two supplementary studies were added and measured: `results/moead_paired` and `results/exact_gap`
   ([discussion](ot_advanced_methods.md)).

## Not covered by this audit

- **Installation from scratch was not repeated.** The fresh clone was tested with the Python environment already on
  this machine, not a new virtual environment built from `requirements-lock.txt`, and `npm ci` was not re-run.
- **`report/final_report.pdf` and the PPTX/PDF slides were not rebuilt.** Pandoc, XeLaTeX and python-pptx are not
  installed here. Their Markdown and builder sources are updated; the binaries still show the earlier content.
- **Group details, professor approval and contribution records** remain for the group to complete.
- The supplementary studies are small. In particular, MOEA/D had the higher hypervolume than the memetic method on
  both large instances of `results/moead_paired` (6 of 6 pairs); that result is reported, not resolved.
