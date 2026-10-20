# Experiment7: two-stage feedback and cosine annealing

Both methods automatically choose timesteps for steady-state preparation.
They use the same physical mean channel, preparation, error metric, and
acceptance protocol as the successful forward-window studies in Experiments
4 and 6. No saved schedule, hand-picked timestep grid, or sampled trajectory
enters schedule selection.

## Run

From the repository root:

```bash
MPLCONFIGDIR=/tmp/model3_exp7_matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python Model3/Experiment7/compare_schedules.py
```

Defaults are N=10, J=gamma=1, reservoir populations (0.9,0.1), exchange
prefactor 0.5, half_mixed preparation, rate-weighted single-jump sampling,
Hamiltonian-first even-odd splitting, 500 preparation steps, total tolerance
0.02, mean allowance 0.016, and sampling allowance 0.004. The forward mean
window is at least 10 physical-time units. The selected schedule is replayed
at R=8192 for seeds 29, 30, and 31.

Choose one method with `--methods two_stage` or `--methods cosine`.
`--plan-only` skips sampled replay; `--R-values` and `--seeds` change replay
only. J, gamma, N, preparation, reservoir populations, tolerances, and
step/time caps are configurable. Timestep bounds default to
`[0.0001,1]/max(J,gamma)`; these are bounds, not a timestep grid.

## Two-stage feedback

From the current deterministic mean covariance C, predict a coarse phase
followed by fine evolution:

```text
C_output = Phi_hfine^(K-q) (Phi_hcoarse^q(C))
0 <= q < K, 0 < hfine <= hcoarse.
```

Jointly optimize K, both log timesteps, and a continuous switch fraction
rounded to an integer q. Optimize the actual predicted output and its
constant-final-step forward window; an unacceptable infinite-time fixed
point at the *coarse* timestep does not veto a useful initial coarse phase.
Preparation length is an integer search variable, with no manually chosen
schedule lengths or switch counts.

Commit at most four steps and repeat from the resulting actual mean state.
The remaining suffix of the controller's own passing forecast is retained
and independently revalidated. A new forecast replaces it only if it is
shorter, or has equal length and smaller predicted error. This prevents a
local replan from discarding a feasible continuation. There is no incumbent
from any earlier experiment. Although each forecast has at most two stages,
replanning can produce more stages and can increase or decrease the next
timestep.

## Cosine annealing

Jointly optimize K and positive endpoint steps h_initial >= h_final:

```text
h_k = h_final + (h_initial-h_final)/2 * [1+cos(pi*k/(K-1))]
k = 0,...,K-1.
```

For K=1 use h_final. This is a single monotone cosine decay without restarts.
It reaches a positive floor, and the forward check holds that *same* floor.
The schedule is selected once and then followed. It can stop before its
planned end only if the output and forward check at its actual last timestep
already pass. Such a result is explicitly a truncated cosine schedule.
Unlike two-stage feedback, cosine is not replanned after each short prefix.

## Automatic search and its limits

Both use seeded differential evolution on their small parameter sets. The
initial search uses 80 iterations and population multiplier 10; feedback
updates use 16 iterations. Optimizer seed 1729 is independent of trajectory
seeds. These policy defaults are shared across both methods rather than
selected from earlier winning schedules. This is automatic numerical
optimization, not a guarantee of a globally optimal schedule.

The search first seeks feasibility, then fewer preparation steps. A 1%
margin on the mean allowance helps avoid numerical boundary effects.
Searches that find no passing forecast report `no_feasible_forecast` and do
not sample an unvalidated schedule or fall back to an earlier experiment.

For efficient forecasting, constant mean evolution is represented exactly
as an affine map on symmetric covariances. Integer matrix powers advance
long constant blocks. The search observes up to roughly 64 future nodes,
including the final whole step covering the requested physical-time window.
Sparse observations are a search heuristic and can miss intermediate peaks.
**Every forecast retained for execution and every actual accepted output is
therefore checked separately by direct mean stepping at every future node.**
The replay engine independently recomputes that complete final window.
No sparse-only forecast is reported as a passing result.

The mean metric checks all N populations and N-1 currents:

```text
d(A,B) = max(max_site |n(A)-n(B)|, max_bond |I(A)-I(B)|/J).
```

Sampling at the output requires R>1, sampled physical error <=0.02,
sampled-to-mean distance <=0.004, and maximum descriptive two standard errors
<=0.004. These conditions preserve the earlier protocol. The future window
checks the mean only. Two-SE values are not simultaneous confidence bounds;
a finite window does not establish permanent settling. Product-mixture
initial bits are analytically averaged; measurement shots and hardware
noise are excluded.

## Saved results

Each configuration writes `runs/comparison_<hash>/`:

- `Study.md`, `study.json`, and `comparison.png` compare the methods.
- `two_stage/plan.json` and `cosine/plan.json` retain parameters, decisions,
  passing improvements, the actual preparation prefix, and complete forward
  validation. Classical search work and direct validation work are separate.
- Each method's `replay_seed29_R8192.json` etc. save its sampled observables,
  uncertainty, exact reference, jump choices, and independently repeated
  final-window checks.
- Each method's `figures/` contains the timestep/error curves and three
  representative observable plots, plus sampled-output validation figures.

Only committed preparation steps appear in the executable schedule. Search
forecasts, rejected probes, and future mean steps are classical work. Both
methods optimize a step-count proxy for depth; native depth is not part of
their objective. Experiment5 accepts the saved Experiment7 replay format for
separate local FakeKingston compilation.

`tests/test_model3_experiment7.py` covers accelerated-versus-direct physics,
cosine endpoints, exact sparse block advancement, rejection of intermediate
window failures, absence of stationary-bias vetoes/prior schedules, feasible
continuation retention, independence from sampling seeds/R, speculative
state isolation, failed/zero-step plans, persistence, independent replay,
trajectory prefix preservation, circuit compatibility, and plotting.

## Executed default N=10 comparison

The default command above produced
[comparison_d3fd93cb452e](runs/comparison_d3fd93cb452e/Study.md). Both methods
pass the full forward mean check and all three R=8192 sampling seeds.

| Method | Preparation steps | Physical output time | Maximum forward mean error |
|---|---:|---:|---:|
| Earlier Experiment6 controller | 273 | 15.3977123 | 0.01408550 |
| Earlier searched Experiment4 schedule | 117 | 22.65 | 0.01592229 |
| New two-stage feedback | **73** | 10.7760808 | 0.01582630 |
| New cosine annealing | **72** | 22.8095429 | 0.01592539 |

Cosine uses the complete automatically selected schedule, without early
truncation:

```text
h_k = 0.05733068985582982
    + (0.576267723009639 - 0.05733068985582982)/2
      * [1+cos(pi*k/71)],  k=0,...,71.
```

Thus cosine reduces preparation steps by 73.63% relative to 273, and 38.46%
relative to 117. The two-stage feedback reductions are 73.26% and 37.61%.
These are observed results for this model and optimizer seed, not a general
ranking or a minimum-step proof. Cosine's one-step advantage and smoother
schedule make it an attractive choice here, while the feedback method reaches
its accepted output in less physical time.

The two-stage method initially found a passing 147-step two-stage forecast.
Replanning shortened its own continuation until the actual preparation
required 73 steps. Its executable schedule contains ten constant-step blocks;
it is not restricted to two stages overall. Its final step is approximately
0.00168002, followed by **5,953 classical mean-only validation steps**. Cosine
ends at approximately 0.0573307 and needs **175 classical validation steps**.
Neither continuation adds circuit gates. Search took about 91 and 82 seconds,
respectively, in the executed environment. Prediction-work counters include
equivalent steps covered by exact matrix powers, rather than only individually
executed matrix updates. The affine acceleration is intended for these small
covariance models; its dense matrix dimension is N(N+1)/2+1.

All six sampled output errors are below 0.02. The largest sampled-to-mean
distance is 0.00340830, and the largest descriptive two-SE value is 0.00268756,
both below 0.004. Thirty-four tests passed (13 new schedule-comparison tests,
14 Experiment6 tests, and 7 Experiment5 tests), including 36 subtests.

Separate FakeKingston compilation used the same chain layout, two boundary
baths (12 active qubits), optimization level 3, transpiler seed 11, four saved
trajectories, and three final readout settings as the earlier resource study:

| Schedule | Full native depth range | CZ gates | Two-qubit depth |
|---|---:|---:|---:|
| Earlier 117-step schedule | 1882–1887 | 2340 | 468 |
| New two-stage feedback | 1176–1181 | 1460 | 292 |
| New cosine annealing | 1125–1131 | 1440 | 288 |

These are unweighted dependency-layer depths for the compiled subset, not
scheduled circuit durations. All 32 saved native circuits were reloaded from
QPY, checked against the backend target, and found to reproduce saved depth
and operation counts; see
[QpyVerification.json](runs/comparison_d3fd93cb452e/QpyVerification.json).
No hardware execution or noisy simulation was performed.

See [the verified comparison](runs/comparison_d3fd93cb452e/VerifiedComparison.md)
for independently reconstructed metrics and separate FakeKingston resources,
and [the comparison figure](runs/comparison_d3fd93cb452e/comparison.png) for
the timestep schedules and actual-versus-future mean errors.

Recompute the saved-data verification without selecting schedules or running
sampled trajectories:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python Model3/Experiment7/analyze_results.py \
  Model3/Experiment7/runs/comparison_d3fd93cb452e
```
