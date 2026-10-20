# Experiment 11 dimensionless randomized-scaling study

Status: complete. Initial excitation: N//2; J=1, h=0 during the sweep.

Vacuum tolerance: 0.001. Normalized transient tolerance: 0.01.
Recommended timestep bias budget: 0.005; sampling budget: 0.005.
Timing error limit: 5%; averaged-map timing budget: 2.5%.

Transient metric is max over algorithmic nodes and sites/bonds of population error and |current error|/J. Uniform h and zero XY correlations are not optimization axes.

All future-node errors are bounded once remaining excitation sums fall below the error budget. No plotting/save interval inserts steps.

Largest steps are largest *tested* passing values, not a proof of continuous/global optimality. Ratios are refined locally around the best exact grid point; only promising ratios receive timestep sweeps.

Finite-R acceptance requires every listed seed to pass, but this is empirical validation, not a probabilistic guarantee. Random streams are independent per trajectory and differ from existing circuit seeded prefixes.

| N | γ/J | exact Jt_ss | largest tested s (infinite R) | recommended s | tested R | J for target T | γ | Δt | approximate substeps |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 6 | 1.15129 | 0.0225 | 0.0125 | 2048 | 0.0633211 | 0.379927 | 0.197407 | 94 |
| 4 | 3.88959 | 9.54045 | 0.0175 | 0.00875 | 256 | 0.524725 | 2.04097 | 0.0166754 | 1093 |
| 7 | 2.89383 | 53.1079 | 0.0225 | 0.01125 | 64 | 2.92093 | 8.45268 | 0.00385151 | 4721 |
| 10 | 2.64604 | 148.483 | 0.0225 | 0.01125 | 64 | 8.16658 | 21.6091 | 0.00137757 | 13201 |
| 15 | 2.40621 | 474.117 | 0.0225 | 0.01125 | 64 | 26.0764 | 62.7455 | 0.000431424 | 42143 |
| 20 | 2.30089 | 1068 | 0.0225 | 0.01125 | 32 | 58.7401 | 135.155 | 0.000191522 | 94929 |
| 30 | 2.19507 | 3338.59 | 0.0225 | 0.01125 | 16 | 183.622 | 403.064 | 6.1267e-05 | 296740 |
| 50 | 2.11395 | 13863.5 | 0.0225 | 0.01125 | 8 | 762.492 | 1611.87 | 1.47543e-05 | 1.23219e+06 |

Local timestep refinement rounds: 2.

Supplemental trajectory-count extensions: {'2': [512, 1024, 2048, 4096, 8192]}.

Target T=20. Rates include a 10% speed margin: J=margin·(Jt_ss)/T, γ=rJ, Δt=s/J.

Rate scaling compresses physical time but does not reduce the dimensionless substep count. Missing R means no tested ensemble size passed; do not treat that recommendation as fully validated.

## Search configuration

```json
{
  "steady_tolerance": 0.001,
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
| 2 | 102 | 208896 | 204 |
| 4 | 1,200 | 307200 | 4,800 |
| 7 | 5,193 | 332352 | 36,351 |
| 10 | 14,519 | 929216 | 145,190 |
| 15 | 46,359 | 2.96698e+06 | 695,385 |
| 20 | 104,427 | 3.34166e+06 | 2,088,540 |
| 30 | 326,440 | 5.22304e+06 | 9,793,200 |
| 50 | 1,355,542 | 1.08443e+07 | 67,777,100 |

A descriptive log-log fit over the studied N>=10 gives τ_ss proportional to N^2.817. This finite-size fit is not an asymptotic scaling proof.

R is selected for local absolute population/current errors: spreading one excitation over more sites can reduce this metric. A lower tested R at higher N is not evidence that generic large-N states are easier to simulate.

N=2 is exceptional: both sites are dissipative boundaries. The fastest exact grid point is at the ratio search boundary, so no finite globally optimal loss ratio is claimed.

Results describe ideal classical expectations, not measurement-shot noise, gate errors or hardware feasibility. The recommended physical rates accelerate the model but leave a potentially very large circuit workload.

Figures: [size scaling](scaling.png), [ratio landscape](ratio_landscape.png), [target rates and workload](target_workload.png). Full case errors, peak times, crossing brackets, seed results and timings are in study.json.
