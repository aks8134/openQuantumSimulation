# Experiment4: shorter preparation using forward validation and a wider search

**Result: a 117-step preparation passes the configured checks at R=8192 for
seeds 29, 30, and 31.** It starts from half_mixed, with N=10, J=gamma=1,
reservoirs (0.9,0.1), exchange prefactor 0.5, and the original rate-weighted,
Hamiltonian-first, even-odd XX channel.

The total tolerance remains 0.02, split into mean allowance 0.016 and sampling
allowance 0.004. All 10 populations and 9 currents are checked. Original
results remain unchanged. Machine-readable metrics and source hashes are in
[ForwardWindowStudy.json](ForwardWindowStudy.json).

## What the new stopping rule means

The circuit stops after preparing the output state. The deterministic mean
then continues classically for at least 10 additional physical-time units at
the same final timestep. Every continuation node and the output must satisfy
the mean allowance. No future refinement is allowed in that check. Sampling
distance, two standard errors, and sampled physical accuracy are checked
**only at the output**. This is a forward mean check plus an output sampling
check; future sampled trajectories were not replayed or validated.

The old 490-step result used a sampled trailing window. Therefore, the step
reduction combines a changed stopping rule with better preparation schedules.
It does not claim the same trailing-window criterion at 117 steps.

The new preparation is:

| Stage | Steps | dt | Physical duration |
|---|---:|---:|---:|
| Coarse burn-in | 8 | 0.75 | 6 |
| Intermediate evolution | 56 | 0.25 | 14 |
| Cleanup | 53 | 0.05 | 2.65 |
| Total preparation | **117** | | **22.65** |

The subsequent classical check uses 200 additional dt=0.05 mean steps,
ending at time 32.65. These mean-only steps are reported separately and do
not appear in the executable replay schedule or per-trajectory step count.

## Deterministic search comparisons

All searches had a 500-preparation-step cap, at most three stages, and the
same 10-time-unit window and accuracy allowances. Requested earlier winners
were reevaluated as mandatory baselines. No search hit its expansion cap.

| Configuration | Best preparation steps | Mean window error | Saved plan |
|---|---:|---:|---|
| Original grid, block 10, beam width 2, forward mode | 328 | 0.0159934 | [6fa3680ca8cb](../runs/budget_6fa3680ca8cb/Study.md) |
| Added cleanup dt=0.075,0.08,0.09,0.10; block 10, width 8 | 210 | 0.0158240 | [755fc46666df](../runs/budget_755fc46666df/Study.md) |
| Added burn-in candidates up to 3; block 10, width 8 | 180 | 0.0142361 | [cc272b1c7739](../runs/budget_cc272b1c7739/Study.md) |
| Same coarse grid, block 5, width 8 | 200 | 0.0150809 | [e382900d19fc](../runs/budget_e382900d19fc/Study.md) |
| Added dt=0.04,0.05; block 2, width 16 | 118 | 0.0159223 | [ce6e6a632f23](../runs/budget_ce6e6a632f23/Study.md) |
| Reevaluated all prefixes of the retained winners | **117** | **0.0159223** | [419029a6f9bc](../runs/budget_419029a6f9bc/Study.md) |

These are heuristic-search comparisons, not an isolated causal attribution
to each grid value: changing grid, switch spacing, or beam width can change
which paths survive. For example, the 180-step winner uses only dt=0.5,
0.25, and 0.03125 despite the grid permitting dt=3. The block-5 trial was
started before the block-10 coarse trial completed, so it retained the
earlier 210-step winner rather than the subsequently found 180-step one.
The final verification pooled the stronger retained schedules.

The winning schedule actually uses a coarser first step, 0.75. Its final
cleanup dt=0.05 is smaller than the old 0.0625, but it requires only 53 cleanup
steps instead of 450. Intermediate-stage state preparation and a short
cleanup matter more here than maximizing the initial timestep.

Compared with 490 steps, the new preparation has 373 fewer steps (76.12%).
Compared with the 328-step original-grid forward baseline, it has 211 fewer
steps (64.33%). These are step-count comparisons, not measured native-depth
or hardware-accuracy improvements.

## Sampled verification

The deterministic schedule was fixed before the three seeds were replayed.
All use the identical preparation and timestep list at R=8192.

| Seed | Status | Sampled output error | Sampled-to-mean distance | Maximum two SE |
|---:|---|---:|---:|---:|
| 29 | passed | 0.01551024 | 0.00130003 | 0.00232404 |
| 30 | passed | 0.01711602 | 0.00253129 | 0.00232508 |
| 31 | passed | 0.01720871 | 0.00262398 | 0.00234336 |

The mean output error is 0.01458473. The maximum over the forward mean window
is 0.01592229. Both fit the 0.016 mean allowance. All sampled output errors
are below 0.02; both sampling quantities are below 0.004 for all three seeds.
The mean replay covariance agrees exactly with the selected planned covariance
in these saved results.

This is evidence across three seeds, not a simultaneous confidence guarantee,
a proof of permanent settling, or a global minimum-step result. The simulation
still analytically averages initial product-mixture bits and excludes
measurement shots and hardware noise. No new circuit transpilation was done.

## Reproduce final verification

The final command reevaluates the already discovered winners and constant
baselines, then replays the selected prefix. `--search-node-cap 0` disables
additional beam exploration; the wider discovery search is saved separately.

```bash
MPLCONFIGDIR=/tmp/model3_exp4_forward_matplotlib_seed29 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python Model3/Experiment4/budgeted_steady_state.py \
  --n-qubits 10 --J 1 --gamma 1 --step-budget 500 \
  --preparations half_mixed --block-steps 2 --beam-width 16 \
  --search-node-cap 0 \
  --step-grid 3 2 1.5 1.25 1 0.75 0.5 0.25 0.125 0.1 0.09 0.08 0.075 0.0625 0.05 0.04 0.03125 0.015625 \
  --steady-tolerance 0.02 --sampling-tolerance 0.004 \
  --trajectories 8192 --seed 29 --validation-mode forward \
  --baseline-study \
    Model3/Experiment4/runs/budget_e8ec77f803de/study.json \
    Model3/Experiment4/runs/budget_cc272b1c7739/study.json \
    Model3/Experiment4/runs/budget_ce6e6a632f23/study.json
```

The other runs use `--seed 30` and `--seed 31`. Their output directories are
`budget_52a1ea7d1155` and `budget_39e2dd5c49ba`, respectively.

To reproduce the wider 118-step discovery search, use the same grid with
`--block-steps 2 --beam-width 16 --search-node-cap 400000 --plan-only --no-plots`
and baseline studies `budget_e8ec77f803de`, `budget_755fc46666df`, and
`budget_e382900d19fc` (their `study.json` files). That plan took about
17 seconds in the executed environment. Retained-winner prefix reevaluation
then found the 117-step result.

## Saved plots and validation

- [Forward mean check and sampled output, seed 29](../runs/budget_419029a6f9bc/figures/forward_validation_R8192.png).
- [Three representative observables, seed 29](../runs/budget_419029a6f9bc/figures/observables_R8192.png).
- [Selected schedule](../runs/budget_419029a6f9bc/figures/schedule.png).
- [Seed 30 report](../runs/budget_52a1ea7d1155/Study.md).
- [Seed 31 report](../runs/budget_39e2dd5c49ba/Study.md).

Fourteen existing Experiment4 tests and eight new forward-window tests passed.
The new tests cover the output/continuation budget split, all-node mean checks,
the prohibition on free future refinement, a small exhaustive-search oracle,
incumbent preservation, plan-only mode, rounding, and saved figures. Final
assessment metrics were independently recomputed from the saved arrays; all
three replays also pass Experiment5's source-format validation. Original
saved-run hashes were checked unchanged.
