# IBM Phoenix: saved cosine-run analysis

Job: `dbbrmpc24mrc73ciaq4g`. 4 trajectories, 12 circuits, 8192 shots/circuit, 98,304 total shots.
N=10, J=1, gamma=1, reservoirs (0.9,0.1), half_mixed preparation.
Cosine preparation: 72 steps, output time 22.8095429. All measurements are of this final output, not a time series.

**The hardware output fails the 0.02 accuracy target:** maximum local error 0.251030273.

| Comparison | Max population error | Max current error / J | Combined metric |
|---|---:|---:|---:|
| Ideal mean output vs physical steady state | 0.0159253878 | 0.0158682092 | 0.0159253878 |
| Ideal same 4 paths vs ideal mean output | 0.0370112104 | 0.10305054 | 0.10305054 |
| Ideal same 4 paths vs physical steady state | 0.0507380759 | 0.115248209 | 0.115248209 |
| Hardware vs ideal same 4 paths | 0.0822921768 | 0.257606071 | 0.257606071 |
| Hardware vs physical steady state | 0.0622253418 | 0.251030273 | 0.251030273 |

The signed observable differences decompose exactly as hardware minus same-path ideal, plus same-path ideal minus mean-channel output, plus mean-channel output minus the physical target. These maximum norms do not add as scalar error contributions.

| Spatial mean bond current | Value |
|---|---:|
| hardware | 0.0974460178 |
| physical_stationary | 0.32 |
| ideal_mean_channel | 0.308635916 |
| ideal_same_saved_paths | 0.278798304 |

The measured mean bond current retains 34.95% of the ideal mean current for the same four paths. Its nominal fixed-path shot SE is 0.00130844, computed with within-setting bond correlations retained.

The four-path ideal average already differs appreciably from the target; increasing shots on these four paths cannot remove this trajectory-selection effect. Hardware currents are further suppressed compared with these exact same ideal paths. Extra shots alone would not remove that systematic discrepancy. This run cannot identify how much comes from two-qubit errors, reset errors, mixture preparation, readout, drift, or other device effects.

The physical XX model has a flat interior steady-state population. A sloping measured population profile here is not evidence of diffusive transport. One output measurement also cannot establish that hardware evolution has become stationary.

## Plots and uncertainty

[Final populations, magnetizations and currents](observables.png) ([PDF](observables.pdf)). [Currents for each trajectory](currents_per_trajectory.png). CSV files contain every site/bond value.

Error bars are ±2 nominal shot standard errors conditional on the four selected paths. They include outcome fluctuations from preparation and final measurement, assuming independent shots, and are not simultaneous confidence intervals. Across-path standard errors, which already include shot fluctuations, are stored separately and must not be combined in quadrature with shot SE. The physical tolerance band is ±0.02 in population and current/J, and ±0.04 in magnetization.

## Verification

Source integrity: `matches_original_CRLF_line_endings`. A Windows CRLF to Linux LF conversion is accepted only when reconstructing the original line endings reproduces the exact saved SHA-256.
All 12 raw count histograms have the expected width and shot count. Observables, current parity/sign, saved references, and saved assessments were reconstructed and checked. The physical stationary reference was independently evaluated from the XX formula. Only the final `outcome` register is used; preparation records are ignored.
Recorded native depths: [1127, 1131]; CZ gates/circuit: [1440]; bath resets/circuit: [72]. Physical path: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 29]`.
ISA QPY file present in this copy: False. No circuits were recompiled or submitted, and no calibration data was fetched. Original run files were left unchanged.
