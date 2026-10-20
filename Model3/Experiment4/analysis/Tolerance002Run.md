# Experiment4: executed tolerance-0.02 validation

The R=8192 run passes all configured finite-node steady-window checks.
The total tolerance is 0.02, mean allowance 0.016, sampling allowance 0.004,
and per-trajectory step cap 500. No simulator code or original runs were changed.

## Reproduce the successful command

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/model3_exp4_tolerance002_matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python Model3/Experiment4/budgeted_steady_state.py \
  --n-qubits 10 --J 1 --gamma 1 \
  --step-budget 500 --preparations half_mixed --block-steps 10 \
  --steady-tolerance 0.02 --sampling-tolerance 0.004 --trajectories 8192
```

The preceding comparison used the same command with `--R-values 64 4096`
in place of `--trajectories 8192`. Seed 29 and the default grid were retained.

## Selected schedule

- Preparation: half_mixed (independent occupation probability 0.5 at every site;
  zero initial coherence/current).
- 20 steps at dt=0.5, then 20 at dt=0.25, then 450 at dt=0.0625.
- Total: 490 steps per trajectory; final physical time 43.125.
- Assessed window: [33.125,43.125], including every integration node.
- Mean tail error: 0.015682747, below its 0.016 allowance.
- Search-node cap was not reached. This is the shortest passing candidate
  found by this configured search, not a global minimum-depth result.

## Sampling comparison on the same schedule

All errors maximize over every site/bond and every node in the assessed window.

| R | Status | Sampled-to-steady error | Sampled-to-mean distance | Maximum two SE |
|---:|---|---:|---:|---:|
| 64 | sampling_budget_not_met | 0.038007750 | 0.027903359 | 0.026749749 |
| 4096 | sampling_budget_not_met | 0.016521947 | 0.005650905 | 0.002749358 |
| 8192 | passed | 0.016963471 | 0.002717680 | 0.001934807 |

R=4096 meets the physical total tolerance but fails the separate 0.004
sampling-distance check. R=8192 passes both sampling checks and the total
physical tolerance. Sampled error need not decrease monotonically with R;
it combines deterministic error with a finite random sampling contribution.

Increasing R changes repetition count and total workload, not per-circuit
step count. The R=8192 run uses 4,014,080 trajectory channel applications.
Compiled hardware depth is unmeasured. Product-mixture preparation-bit noise,
measurement-shot noise, and hardware noise are not simulated. The two-SE check
is descriptive, not a simultaneous confidence bound or proof of permanent
settling. Only seed 29 was checked.

## Saved outputs

- Successful run: [Study.md](../runs/budget_e8ec77f803de/Study.md).
- Successful replay: [replay_R8192.json](../runs/budget_e8ec77f803de/replay_R8192.json).
- Observables: [observables_R8192.png](../runs/budget_e8ec77f803de/figures/observables_R8192.png).
- Schedule: [schedule.png](../runs/budget_e8ec77f803de/figures/schedule.png).
- Lower-R comparison: [Study.md](../runs/budget_421539f93cf3/Study.md).

Validation independently recomputed the R=8192 assessment from saved arrays,
confirmed identical schedules and preserved trajectory prefixes across R,
and checked that the 12 original analysis source JSON hashes were unchanged.
