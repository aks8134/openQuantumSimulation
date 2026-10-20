# Experiment6: automatic budget-driven schedule

Controller status: **step_budget_reached**.
Preparation: half_mixed; steps: 500/500; output time: 6.9213266.

Total tolerance 0.02; mean 0.016; sampling 0.004.
Worst forward mean error: 0.0498522812. Classical continuation: 0 steps.

The controller uses continuous bounded timestep optimization and current mean-state feedback. No prior schedules or timestep grid were loaded. Forecast scores are heuristics; actual mean validation checks every node. Sampling is checked at output only.

| Steps | dt |
|---:|---:|
| 4 | 0.498150094 |
| 4 | 0.339612902 |
| 4 | 0.235674914 |
| 4 | 0.395190156 |
| 8 | 0.0813640701 |
| 4 | 0.0287865854 |
| 4 | 0.0195574141 |
| 4 | 0.0133220304 |
| 4 | 0.00896007489 |
| 4 | 0.00599995428 |
| 4 | 0.00403209575 |
| 4 | 0.00269039449 |
| 4 | 0.00181702267 |
| 4 | 0.0012008452 |
| 4 | 0.000805299765 |
| 4 | 0.00053978722 |
| 4 | 0.000359175142 |
| 4 | 0.000239657057 |
| 4 | 0.000159156214 |
| 4 | 0.000105950486 |
| 416 | 0.0001 |

| Replay | Status | Sampled output error | Sampling distance | Two SE |
|---|---|---:|---:|---:|

No sampled replay was performed. A mean-only result is not a sampling-validated success.

Reference-assisted covariance model; no hardware feedback, shots, hardware noise, or compiled-depth calculation. The controller does not guarantee a globally shortest schedule or beat an offline optimized schedule. Validation covers a finite future mean window, not permanent settling.

The numerical optimizer is local; see [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html).
