# IBM Phoenix: 8192 trajectories with one shot per setting

Run: `ibm_phoenix_20261021T102532_376654Z`. All 48 jobs and 24,576 circuits collected; 8,192 complete trajectories, 3 settings/path, one shot/setting, 24,576 total shots.
Model3 is the unequal-reservoir XX model of writeup Section 4.9. This run uses the ten-site extension: J=gamma=1, reservoirs (0.9,0.1), half-mixed preparation, 72 cosine steps, output time 22.8095429. It does not use the PDF's three-site, gamma=2 target.

**The hardware output fails the 0.02 tolerance:** maximum local error 0.279228516.

## Raw-count verification and ideal references

Every outcome histogram was checked for width, binary labels, and exactly one shot. The three measurement settings were paired by trajectory, current signs were reconstructed for both bond parities, all publication ranges and job IDs were checked, and saved observables/uncertainties/references were independently reproduced. Final-register counts alone were used.

| Comparison | Max population error | Max current error / J | Combined metric |
|---|---:|---:|---:|
| Ideal mean vs physical | 0.0159253878 | 0.0158682092 | 0.0159253878 |
| Ideal same 8192 paths vs mean | 0.00188568368 | 0.00272229504 | 0.00272229504 |
| Ideal same paths vs physical | 0.0171097482 | 0.0144381289 | 0.0171097482 |
| Hardware vs same ideal paths | 0.0898351727 | 0.264790387 | 0.264790387 |
| Hardware vs physical | 0.0825195312 | 0.279228516 | 0.279228516 |

The ideal same-path error fits 0.02, and its distance from the ideal mean is below 0.004. The hardware discrepancy persists after matching all 8192 ideal branch sequences. Increasing trajectory diversity resolved the old four-path reference limitation; it did not remove the hardware mismatch. Maxima in different rows cannot be added as scalar error contributions.

## Actual observable values

[Populations, magnetizations and currents](observables.png) ([PDF](observables.pdf)). All sites/bonds are included, using one-based indices. CSV files contain every hardware/reference value and its nominal standard error.

| Bond | Hardware current / J | Twice nominal SE / J | Ideal same paths / J | Physical target / J |
|---|---:|---:|---:|---:|
| 1–2 | 0.1621094 | 0.0155317 | 0.3110289 | 0.3200000 |
| 2–3 | 0.1359863 | 0.0154890 | 0.3072461 | 0.3200000 |
| 3–4 | 0.1257324 | 0.0155887 | 0.3131765 | 0.3200000 |
| 4–5 | 0.0985107 | 0.0155792 | 0.3066029 | 0.3200000 |
| 5–6 | 0.0771484 | 0.0155212 | 0.3104929 | 0.3200000 |
| 6–7 | 0.0407715 | 0.0157293 | 0.3055619 | 0.3200000 |
| 7–8 | 0.0877686 | 0.0153835 | 0.3119612 | 0.3200000 |
| 8–9 | 0.1026611 | 0.0155047 | 0.3080471 | 0.3200000 |
| 9–10 | 0.1057129 | 0.0154812 | 0.3115695 | 0.3200000 |

| Spatial mean current | Value |
|---|---:|
| hardware | 0.104044596 |
| physical_stationary | 0.32 |
| ideal_mean_channel | 0.308635916 |
| ideal_same_saved_paths | 0.309520769 |

Hardware retains 33.61% of the matched ideal mean current. The nominal SE of its spatial mean is 0.00257455, retaining within-trajectory correlations between bonds.
The worst current discrepancy is on bond [6, 7]; the worst population discrepancy is at site 2. All nine measured currents are below their physical target.

## What uncertainty is estimable with one shot?

A single trajectory has no empirically estimable within-circuit shot variance. Instead, each trajectory contributes a vector of population/current samples, and the covariance of the overall mean is the sample covariance of those vectors divided by R. This includes branch variation, mixture-preparation randomness and measurement fluctuations; these components are not separately identified from one shot. Do not add a separate shot-error estimate to the pooled SE.
The maximum nominal two-SE values are 0.01104833 for populations and 0.01572934 for current/J. These exceed the original 0.004 sampling allowance, but remain far smaller than the observed current mismatch. The 0.004 ideal-trajectory validation excluded hardware measurement noise.
Nominal SE assumes independent trajectories. Twice-SE bars are descriptive per-observable uncertainty estimates, not simultaneous confidence bands, and do not include systematic hardware bias.

## Sensitivity to jobs and submission order

For job diagnostics, 8160 trajectories have all three settings inside a single job, 170 per job. The 32 paths crossing a job boundary are excluded only from this diagnostic; all are retained in the primary estimate.
The per-job spatial mean current ranges from 0.06928105 to 0.1568627, below the target 0.32 in every job. [Job means and current heatmap](batch_currents.png); [per-job CSV](per_job.csv).
Clustering the complete-path subset by job gives mean-current SE 0.003249519, compared with nominal SE 0.002578662 on that same subset and 0.00257455 on all paths. The job-level SE of the paired hardware-minus-ideal current is 0.003270196. The observed SD of per-job paired discrepancies is 0.02265658, versus typical within-job SE 0.01817567 (ratio 1.247).
These diagnostics make the uncertainty somewhat larger but do not account for the large suppression. They assume independent jobs and use a slightly smaller subset; they do not establish time drift or identify a device mechanism. Submission order is not a calibrated time axis, and job timing/calibration data were not fetched.

## Steady-state balance diagnostics

The measured output fails the target steady boundary balances as well. Left residual = gamma*(p_left-n_1)-I_1; right residual = gamma*(n_N-p_right)-I_(N-1). These diagnose agreement with the intended stationary equations; they are not measurements of a hardware time derivative:

| Boundary | Steady-balance residual | Nominal SE |
|---|---:|---:|
| left | 0.11716309 | 0.0094361191 |
| right | 0.14853516 | 0.0093705976 |

## Comparison with the earlier four-trajectory run

| Saved run | Trajectories | Shots/basis/path | Jobs | Total shots | Mean hardware current | Matched ideal mean current | Hardware physical error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Earlier | 4 | 8192 | 1 | 98304 | 0.097446018 | 0.2787983 | 0.25103027 |
| One-shot | 8192 | 1 | 48 | 24576 | 0.1040446 | 0.30952077 | 0.27922852 |
Both show strong current suppression. The different path sets, shot allocation, and execution occasions prevent treating this as a controlled test of whether changing shots itself caused a change in hardware fidelity. The observed mean current is slightly larger in the new run, while the maximum physical error is also larger.

## Scope and resource audit

Native depth range [1095, 1133]; CZ counts/circuit [1440]; resets/circuit [72]; physical chain `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 29]`.
Source integrity: `matches_original_CRLF_line_endings`. The only accepted source-byte conversion is an LF/CRLF transformation that exactly reproduces the original saved hash. ISA QPY available here: False; its hash could not be checked when absent. Original file hashes and all job IDs are in analysis.json.
This is one final-time measurement, so hardware stationarity is not established. Populations sloping through the chain do not demonstrate diffusion in the physical homogeneous XX model, which has a flat interior target. Gate/reset/preparation/readout effects and calibration changes cannot be separated with these data. No hardware jobs, new evolution, mitigation, or backend queries were performed; original inputs were preserved.
