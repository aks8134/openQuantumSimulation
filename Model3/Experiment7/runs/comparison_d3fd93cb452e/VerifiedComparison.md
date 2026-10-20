# Experiment7: verified comparison

N=10, J=gamma=1, half_mixed, total tolerance 0.02, mean allowance 0.016, sampling allowance 0.004.
The new planners did not read baseline schedules. This analysis reads them only after schedule selection.

| Method | Preparation steps | Reduction vs 273 | Reduction vs 117 | Forward mean maximum |
|---|---:|---:|---:|---:|
| two_stage | 73 | 73.26% | 37.61% | 0.0158263033 |
| cosine | 72 | 73.63% | 38.46% | 0.0159253878 |

| Method | Replay | Sampled error | Sampling distance | Two SE |
|---|---|---:|---:|---:|
| two_stage | seed29_R8192 | 0.0160325878 | 0.00323109949 | 0.00237792795 |
| two_stage | seed30_R8192 | 0.0159577773 | 0.0014968217 | 0.00237330854 |
| two_stage | seed31_R8192 | 0.0169250787 | 0.00166113818 | 0.00237807252 |
| cosine | seed29_R8192 | 0.0171097482 | 0.00272229504 | 0.00268756283 |
| cosine | seed30_R8192 | 0.0174284472 | 0.00235330039 | 0.00266951234 |
| cosine | seed31_R8192 | 0.0168577158 | 0.00340830102 | 0.00264118323 |

Same FakeKingston chain, two boundary baths, optimization level 3, transpiler seed 11; four saved trajectories, three final readout settings.

| Method | Full native depth | CZ gates | Two-qubit depth | Unitary gates incl. RZ |
|---|---|---|---|---|
| earlier 117-step schedule | 1882–1887 | 2340–2340 | 468–468 | 14715–14761 |
| two_stage | 1176–1181 | 1460–1460 | 292–292 | 9002–9068 |
| cosine | 1125–1131 | 1440–1440 | 288–288 | 8815–8850 |

Preparation and every future mean node were independently recomputed by direct stepping. Final sampled covariance means and standard errors were recomputed from saved per-trajectory covariances. All replay files passed circuit-source validation. Source hashes are recorded in VerifiedResults.json.

Only preparation steps enter circuits. In particular, a small final timestep can cause thousands of classical validation steps without adding preparation gates. The comparison keeps the earlier finite-window mean/output-sampling protocol; it does not establish permanent settling or noisy hardware accuracy. Three seeds and descriptive two-SE checks do not constitute simultaneous confidence bounds. Search is heuristic; optimizer convergence and global optimality are not established.
