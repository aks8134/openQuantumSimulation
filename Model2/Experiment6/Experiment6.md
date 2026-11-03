# Model2 Experiment6 — FakeKingston ZNE pilot

Apply the final Model3/Experiment5 population mitigation protocol to Model2's
frozen 44-step warped cosine. Run locally in Aer using the installed FakeKingston
calibration, with no account construction or hardware submission.

The comparison includes raw, raw + marginal SPAM correction, twirled,
twirled + SPAM, linear ZNE, and linear ZNE + SPAM. It imports the same manual
CZ twirling, folding, decoding and statistics helpers used in Model3.

The completed comparison and uncertainty are in [results/Results.md](results/Results.md).
The portable CSV tables and PNG/PDF figure live alongside that report. Local raw
counts, native circuits and the exact noise model stay in `runs/` and are ignored
by Git.

## Parameters

The physics and annealing parameters come from
[the frozen Model2 specification](../Experiment5/inputs/cosine44.json):

- N=10, J=1, Γ/J=2, pL=pR=0.25, exchange prefactor 1/4, Δ=0.
- Half-mixed preparation; reservoir jump first, then even/odd exchange layers.
- 44 steps; dt starts at 4.207281841881579 and ends at 3.760760459292396;
  warped-cosine power 16.424973388132297.
- Fixed path seed 29, transpiler seed 11 and Aer seed base 2026.
- Physical chain `[95,94,93,79,73,74,75,59,55,54,53,52]` (baths at the ends).

The main pilot uses 32 trajectories and 128 shots for each path at each arm or
noise factor. Four randomizations split a twirled factor into 32 shots each.
The three factors are 1, 1.25 and 1.5: 880, 1,100 and 1,320 native CZ gates.
All 44 intentional reservoir resets remain. Zero, one and held-out half-mixture
controls each receive 2,048 raw and 2,048 twirled shots.

Total: 431 circuits, 16,384 evolution shots and 12,288 control shots; **28,672
actual Aer shots**. ZNE uses three independently sampled scales, so it consumes
three times the shots of one raw or twirled arm. The fixed linear-intercept
weights are `[17/6, 1/3, -13/6]`, with about 3.58× shot-SE amplification relative
to one equally sampled scale. There is no fit to the exact results.

## Run locally

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/model2_fake_mpl \
  .venv/bin/python Model2/Experiment6/fake_kingston_zne.py \
  --paths 32 --shots 128 --control-shots 2048 --randomizations 4 \
  --seed 29 --sim-seed 2026 --resume
```

The content-addressed directory is printed when analysis completes. Use its
`analysis.json` to export the report:

```bash
.venv/bin/python Model2/Experiment6/summarize_fake.py \
  Model2/Experiment6/runs/fake_zne_<run_hash>/analysis.json

.venv/bin/python Model2/Experiment6/verify_run.py \
  Model2/Experiment6/runs/fake_zne_<run_hash>
```

The independent standard-library audit recounts actual outcome shots and
reproduces all six population estimates and error metrics directly from the
saved counts. It also checks source/noise/QPY hashes and CZ/reset resources.

Each complete path group is checkpointed. Resumption checks the noise model,
source, implementation and QPY hashes. The noise model, trajectories, raw final
outcome counts, circuit metrics, physical mapping, shot seeds, package versions
and Aer metadata are saved. Preparation measurement records are traced out of
the final counts.

## Interpretation

Errors against the exact **same-path** population mean measure circuit noise
and final-shot fluctuations. Errors against the stationary population 0.25 also
include the finite trajectory ensemble's fluctuation and the schedule's
finite-step bias. This pilot does not certify 0.02 total error and does not
measure currents. It does not replay the proposed 16,384 × 32 configuration.

Bootstrap intervals reuse paired paths and independently resample shared
control counts; shot fluctuations already present in the path means are not
resampled twice. They are descriptive and conditional on the static snapshot
and fixed twirl variants. Fixed-path shot SEs are reported separately. All
corrections are unclipped; the held-out half-mixture is not fitted.

The compact 12-qubit target copies the original active-qubit calibration.
Tests compare its noise channels directly to the corresponding channels in
the full 156-qubit noise model, and verify exact Model2 density-matrix
equivalence of folding/twirling at the schedule's large angles.

Standard unscheduled FakeKingston noise includes gate noise, thermal relaxation
over gate durations and readout error. Additional system idling during bath
resets and dynamic-circuit latency, drift and crosstalk are absent. CZ-only ZNE
does not extrapolate all preparation/reset errors. These results are a local
snapshot benchmark, not a prediction for current hardware.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_model2_fake_zne.py tests/test_model3_mitigation.py \
  tests/test_model3_repeated_preparation_controls.py
```
