# Measured synthetic benchmark results

Each cell summarizes independent solver seeds and planted-feasible instance seeds. Objective minima may belong to different schedules. Empty archives have HV=0. Observed fronts are not certificates of global Pareto optimality.

| Size | Method | Runs | Feasible runs | Mean HV ± SD | Mean runtime (s) | Mean front size | Feasible evaluations |
|---|---|---:|---:|---:|---:|---:|---:|
| large | enhanced | 6 | 6 | 0.909322 ± 0.009554 | 16.236 | 8.67 | 39.7% |
| large | greedy | 6 | 6 | 0.937241 ± 0.003734 | 48.520 | 19.83 | 100.0% |
| large | nsga2 | 6 | 0 | 0.000000 ± 0.000000 | 0.452 | 0.00 | 0.0% |
| large | seeded | 6 | 6 | 0.907620 ± 0.009019 | 2.901 | 6.17 | 8.5% |
| medium | enhanced | 12 | 12 | 0.981213 ± 0.021310 | 6.087 | 7.33 | 43.2% |
| medium | greedy | 12 | 12 | 1.017029 ± 0.011886 | 18.550 | 13.75 | 100.0% |
| medium | nsga2 | 12 | 0 | 0.000000 ± 0.000000 | 0.261 | 0.00 | 0.0% |
| medium | seeded | 12 | 12 | 0.976126 ± 0.020200 | 0.943 | 5.08 | 8.9% |
| small | enhanced | 30 | 30 | 0.935173 ± 0.000169 | 2.432 | 4.27 | 45.9% |
| small | greedy | 30 | 30 | 0.935319 ± 0.000000 | 7.918 | 1.00 | 100.0% |
| small | nsga2 | 30 | 0 | 0.000000 ± 0.000000 | 0.170 | 0.00 | 0.0% |
| small | seeded | 30 | 30 | 0.931300 ± 0.010546 | 0.318 | 2.97 | 17.2% |
