# Phoenix population diagnostic: 128 trajectories × 64 shots

Run `ibm_phoenix_20261023T135506_304864Z`; jobs dbdmeh7i0auc73crdct0. All 131 circuits and 9728 shots collected. Control shots are excluded from evolved populations.

## Main result

Maximum hardware population error against physical stationary values: **0.091064** (target 0.02). MAE: **0.053165**.
Maximum discrepancy against the ideal for the same 128 paths: **0.101882**; MAE **0.055183**.

[Actual population plot](populations.png) ([PDF](populations.pdf)); [controls/calibration](controls_and_calibration.png) ([PDF](controls_and_calibration.pdf)); [per-path residuals](per_path_residuals.png).

| Site | Physical qubit | Hardware | 2 shot SE | Ideal same paths | Physical target |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.630249 | 0.010583 | 0.589565 | 0.580000 |
| 2 | 2 | 0.578857 | 0.010867 | 0.487263 | 0.500000 |
| 3 | 3 | 0.591064 | 0.010829 | 0.512310 | 0.500000 |
| 4 | 4 | 0.539429 | 0.010997 | 0.492084 | 0.500000 |
| 5 | 5 | 0.540894 | 0.011016 | 0.509678 | 0.500000 |
| 6 | 6 | 0.504395 | 0.011038 | 0.489070 | 0.500000 |
| 7 | 7 | 0.468262 | 0.011028 | 0.510657 | 0.500000 |
| 8 | 8 | 0.448242 | 0.010970 | 0.495548 | 0.500000 |
| 9 | 9 | 0.417114 | 0.010866 | 0.518996 | 0.500000 |
| 10 | 19 | 0.359619 | 0.010476 | 0.414952 | 0.420000 |

The ±2-SE plot bars describe shot fluctuations conditional on these fixed paths. The separate across-path SEs include variation between the selected paths and must not be added to those bars. Both assume independent sampling and exclude systematic errors or device drift. There is only one job, so job-cluster uncertainty is not estimable.

## Controls and local-response consistency

Every control used 512 shots. Initial-mixture circuits contain the original preparation measurements and final readout, with no Trotter evolution.

| Site | All-zero population | All-one population | Initial mixture | Predicted mixture from zero/one | Residual / nominal SE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.003906 | 0.980469 | 0.498047 | 0.492188 | 0.262 |
| 2 | 0.001953 | 0.992188 | 0.511719 | 0.497070 | 0.659 |
| 3 | 0.000000 | 0.978516 | 0.462891 | 0.489258 | -1.183 |
| 4 | 0.001953 | 0.988281 | 0.525391 | 0.495117 | 1.361 |
| 5 | 0.003906 | 0.994141 | 0.517578 | 0.499023 | 0.835 |
| 6 | 0.001953 | 0.990234 | 0.496094 | 0.496094 | 0.000 |
| 7 | 0.005859 | 0.947266 | 0.380859 | 0.476562 | -4.329 |
| 8 | 0.007812 | 0.984375 | 0.507812 | 0.496094 | 0.524 |
| 9 | 0.007812 | 0.998047 | 0.509766 | 0.502930 | 0.308 |
| 10 | 0.001953 | 0.986328 | 0.472656 | 0.494141 | -0.965 |

The largest initial-mixture consistency discrepancy is **site 7, physical qubit 7**: measured population **0.380859**, intended **0.500000**, while the zero/one controls predict **0.476562**. The latter discrepancy is **-4.33 nominal SE**, including shot uncertainty in all three controls.


For a stationary independent single-qubit response and a correctly prepared population p, the zero/one controls predict a+(b−a)p. Their shot uncertainty is included in the residual SE. The controls also contain preparation errors and use different circuits at different submission positions; disagreement can reflect preparation, measurement timing, crosstalk, or changing device behavior. These nominal standardized residuals are diagnostics, not certified simultaneous significance tests.

An exploratory inverse affine adjustment (raw−a)/(b−a), without clipping, gives population MAE against physical of **0.054475**, versus raw **0.053165**. Its maximum error is **0.104042**. Because the initial-mixture control tests the response assumptions and contains a substantial discrepancy, these adjusted values are not presented as validated mitigated results.

## Calibration evidence

Snapshot captured at 2026-10-23T13:55:07.835629+00:00. Values describe compilation-time calibration, not a direct measurement of errors during this job.

| Site | Physical qubit | T1 (µs) | T2 (µs) |
|---:|---:|---:|---:|
| 1 | 1 | 90.683 | 101.957 |
| 2 | 2 | 168.707 | 106.336 |
| 3 | 3 | 121.367 | 135.865 |
| 4 | 4 | 213.281 | 110.653 |
| 5 | 5 | 150.341 | 143.295 |
| 6 | 6 | 113.927 | 92.887 |
| 7 | 7 | 16.669 | 19.612 |
| 8 | 8 | 178.886 | 129.005 |
| 9 | 9 | 161.773 | 196.912 |
| 10 | 19 | 216.290 | 84.726 |

The same flagged physical qubit 7 has reported **T1 = 16.67 µs** and **T2 = 19.61 µs**. This makes a preparation/timing/coherence problem a concrete concern, although these data do not isolate it from other errors.

| Physical CZ pair | Reported error (%) |
|---|---:|
| 7–8 | 1.2366 |
| 6–7 | 0.8829 |
| 5–6 | 0.7394 |
| 4–5 | 0.5171 |
| 2–3 | 0.3102 |
| 3–4 | 0.2873 |
| 19–29 | 0.2035 |
| 0–1 | 0.1800 |
| 1–2 | 0.1792 |
| 9–19 | 0.1530 |
| 8–9 | 0.1395 |

Short relaxation times and large adjacent CZ errors make a qubit a concrete layout concern. They do not establish how much of the full-chain population discrepancy is caused by that qubit. The worst evolved-population discrepancies need not occur at that same site because interactions couple the chain.

IBM documents calibration properties in [QPU information](https://quantum.cloud.ibm.com/docs/en/guides/qpu-information), and the timing cost of mid-circuit operations in [dynamic circuits](https://quantum.cloud.ibm.com/docs/en/guides/execute-dynamic-circuits). These mechanisms are plausible interpretations, not measurements of a specific causal noise channel in this dataset.

## Previous population runs

[Population profile comparison](previous_runs.png).

| Run | MAE vs physical | Max vs physical | MAE vs matched ideal | Max vs matched ideal |
|---|---:|---:|---:|---:|
| 4 paths × 8192 shots | 0.041867 | 0.062225 | 0.038661 | 0.082292 |
| 8192 paths × 1 shots | 0.049002 | 0.082520 | 0.048372 | 0.089835 |
| 128 paths × 64 shots (new) | 0.053165 | 0.091064 | 0.055183 | 0.101882 |

Shot allocation, path selection, and execution occasions differ, so this is descriptive rather than a controlled test of shot allocation. Repeated shots sharpen estimates but do not guarantee improved mean hardware fidelity.

## Suggested next step

Select a new connected chain using current calibration quality, avoiding physical qubit 7 and the highest-error adjacent CZ edges. Validate the three short controls on that chain before spending another full-depth run. Keep the cosine schedule fixed during the layout comparison, and retain the newly compiled circuits. This is a recommendation based on the combined control/calibration evidence, not proof that changing one qubit will resolve all population discrepancies.

## Verification and limits

Native depth range: [1125, 1137]; CZ counts [1440]; resets [72]. Saved source integrity: `matches_original_CRLF_line_endings`. Original ISA QPY available: **False**.

All raw count widths, shot counts, trajectory selection, mappings, control separation, job coverage, saved means, references, and nominal across-path SEs were independently reconstructed and checked. Physical stationary populations were independently reconstructed analytically. Hashes verify original files were preserved.

Only final Z populations were measured; currents and hardware stationarity are not established. Submission index is not calibrated execution time. No noisy simulation, trajectory evolution, hardware query, new job, or automatic correction was performed. Job timing metadata are included in analysis.json; job wall time is not an individual circuit duration.
