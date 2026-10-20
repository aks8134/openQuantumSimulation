# Experiment 11 dimensionless randomized-scaling study

Status: complete_with_search_limits. Initial excitation: N//2; J=1, h=0 during the sweep.

Vacuum tolerance: 0.01. Normalized transient tolerance: 0.01.
Recommended timestep bias budget: 0.005; sampling budget: 0.005.
Timing error limit: 5%; averaged-map timing budget: 2.5%.

Transient metric is max over algorithmic nodes and sites/bonds of population error and |current error|/J. Uniform h and zero XY correlations are not optimization axes.

All future-node errors are bounded once remaining excitation sums fall below the error budget. No plotting/save interval inserts steps.

Largest steps are largest *tested* passing values, not a proof of continuous/global optimality. Ratios are refined locally around the best exact grid point; only promising ratios receive timestep sweeps.

Finite-R acceptance requires every listed seed to pass, but this is empirical validation, not a probabilistic guarantee. Random streams are independent per trajectory and differ from existing circuit seeded prefixes.

| N | γ/J | exact Jt_ss | largest tested s (infinite R) | recommended s | tested R | J for target T | γ | Δt | approximate substeps |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 8 | 0.575646 | 0.0175 | 0.00875 | not found | 0.0316605 | 0.253284 | 0.276369 | 67 |
| 4 | 3.81149 | 6.46526 | 0.0175 | 0.00875 | 256 | 0.355589 | 1.35532 | 0.024607 | 741 |
| 7 | 2.87674 | 33.9834 | 0.0225 | 0.01125 | 64 | 1.86908 | 5.37687 | 0.00601899 | 3021 |
| 10 | 2.61968 | 91.6194 | 0.0225 | 0.01125 | 64 | 5.03907 | 13.2008 | 0.00223256 | 8146 |
| 15 | 2.39016 | 279.543 | 0.0225 | 0.01125 | 64 | 15.3748 | 36.7483 | 0.000731715 | 24848 |
| 20 | 2.28443 | 604.29 | 0.0225 | 0.01125 | 32 | 33.236 | 75.9251 | 0.000338489 | 53714 |

Local timestep refinement rounds: 2.

Supplemental trajectory-count extensions: none.

Target T=20. Rates include a 10% speed margin: J=margin·(Jt_ss)/T, γ=rJ, Δt=s/J.

Rate scaling compresses physical time but does not reduce the dimensionless substep count. Missing R means no tested ensemble size passed; do not treat that recommendation as fully validated.

## Search configuration

```json
{
  "steady_tolerance": 0.01,
  "dynamic_tolerance": 0.01,
  "timing_tolerance": 0.05,
  "target_time": 20,
  "rate_margin": 1.1,
  "time_cap": 100000,
  "step_cap": 3000000,
  "horizon_factor": 1.5,
  "promising_ratios": 2,
  "schema_version": 1,
  "sizes": [
    2,
    4,
    7,
    10,
    15,
    20,
    30,
    50
  ],
  "ratios": [
    0.25,
    0.5,
    1,
    1.5,
    2,
    2.5,
    3,
    4,
    6,
    8
  ],
  "steps": [
    0.1,
    0.08,
    0.06,
    0.04,
    0.03,
    0.02,
    0.015,
    0.01,
    0.0075,
    0.005,
    0.003,
    0.002
  ],
  "R_values": [
    4,
    8,
    16,
    32,
    64,
    128,
    256
  ],
  "seeds": [
    29,
    101,
    307,
    509,
    701
  ],
  "bias_tolerance": 0.005,
  "sampling_tolerance": 0.005,
  "bias_timing_tolerance": 0.025
}
```

## Workload through the fixed physical-time target

These counts are algorithmic, not measured transpilation metrics. Each substep has N exchange blocks (N-1 hopping bonds and one selected boundary), plus a reset.

| N | substeps per trajectory to T | R × substeps | exchange blocks per trajectory |
|---|---|---|---|
| 2 | 73 | not found | 146 |
| 4 | 813 | 208128 | 3,252 |
| 7 | 3,323 | 212672 | 23,261 |
| 10 | 8,959 | 573376 | 89,590 |
| 15 | 27,334 | 1.74938e+06 | 410,010 |
| 20 | 59,087 | 1.89078e+06 | 1,181,740 |

A descriptive log-log fit over the studied N>=10 gives τ_ss proportional to N^2.723. This finite-size fit is not an asymptotic scaling proof.

R is selected for local absolute population/current errors: spreading one excitation over more sites can reduce this metric. A lower tested R at higher N is not evidence that generic large-N states are easier to simulate.

N=2 is exceptional: both sites are dissipative boundaries. The fastest exact grid point is at the ratio search boundary, so no finite globally optimal loss ratio is claimed.

Results describe ideal classical expectations, not measurement-shot noise, gate errors or hardware feasibility. The recommended physical rates accelerate the model but leave a potentially very large circuit workload.

Figures: [size scaling](scaling.png), [ratio landscape](ratio_landscape.png), [target rates and workload](target_workload.png). Full case errors, peak times, crossing brackets, seed results and timings are in study.json.
