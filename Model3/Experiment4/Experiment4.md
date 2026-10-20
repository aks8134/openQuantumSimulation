# Experiment 4: steady-state accuracy within a step budget

Experiment3 selects steps to satisfy separate accuracy budgets and may need
thousands of steps. Experiment4 instead searches for the best **tested**
steady-state preparation and schedule within a fixed per-trajectory step
budget. The default budget is 500. It compares inexpensive initial product
states, includes constant-step baselines, and can replay one selected
schedule at several trajectory counts.

The randomized channel is unchanged: an XX Hamiltonian factor followed by
one rate-weighted boundary pump/loss dilation and ancilla reset per step.
Defaults are `N=10`, `J=gamma=1`, reservoirs `(0.9,0.1)`, exchange prefactor
`0.5`, Hamiltonian-first order, and even-then-odd bond splitting. The
implementation uses the shared `library/reservoir_scaling.py` channels.
`--order`, `--system-formula`, and `--sampling-policy` expose the existing
alternatives. No interactions, circuit construction, transpilation,
hardware access, or higher-order channel modification is introduced.

## Run

From the repository root:

```bash
.venv/bin/python Model3/Experiment4/budgeted_steady_state.py \
  --n-qubits 10 --J 1 --gamma 1 \
  --step-budget 500 --trajectories 64
```

To trade repetitions against per-circuit depth, replay the **same selected
schedule and preparation** at multiple R values:

```bash
.venv/bin/python Model3/Experiment4/budgeted_steady_state.py \
  --step-budget 500 --R-values 64 256 1024
```

The mean search is independent of R and seed. Increasing R changes total
experimental workload, not the number of steps in one trajectory. Existing
paths are preserved as R increases through independent path streams with
fixed draw blocks. Initial product mixtures are averaged analytically;
these simulations have no measurement shots or hardware noise.

## Forward validation and retained schedules

The default remains `--validation-mode trailing`, which counts the entire
observed settling window in the preparation length. The optional
`--validation-mode forward` searches for a shorter output prefix, then checks
the deterministic mean for at least `--steady-window` additional physical time
at the **same final timestep**. The output must include that timestep already;
future refinement cannot be treated as free validation. Every continuation node
and the output node must pass the mean allowance.

Only preparation steps count against `--step-budget` and the optional depth
model. Additional mean-only continuation is classical planning work. Sampling
distance, two standard errors, and sampled physical accuracy are checked at
the output node only. This is a different stopping criterion from a sampled
trailing window. It is not a claim that future sampled trajectories were
validated, or that stability holds forever or continuously between nodes.
The mean window uses whole final-timestep steps, conservatively rounding its
length upward when needed (for example dt=0.075 uses 134 steps for time 10).

Forward replay JSON contains only the executable preparation prefix in
`schedule`, `times`, and trajectory arrays. `mean_forward_validation` stores
the separate future mean times, steps, and errors. `assessment.sampling_scope`
is `output_only`; `workload.classical_forward_validation_steps` reports the
uncounted mean continuation. The existing `tail_mean_error` field names are
retained as compatibility aliases for the forward mean-window maximum.
The forward figure marks the circuit stopping point and subsequent classical
check explicitly.

Use `--baseline-study PATH_TO_STUDY_JSON ...` to include compatible earlier
winning schedules for the requested preparations as mandatory candidates.
Their physics, factor order, and sampling policy must match; their timesteps
must appear in the new grid. Their entire saved preparation and its prefixes
are reevaluated, with an additional constant-step continuation in forward mode.
Their input files are hashed in the new configuration, and they cannot be
discarded by beam pruning. The original files are unchanged.

`--plan-only` saves the deterministic study without sampled replay. It is useful
for comparing grids and beam settings before spending work on large R.
Its report explicitly says sampling was not validated. Forward/baseline/plan-only
configurations use schema/algorithm version 2 and separate output hashes;
ordinary trailing configurations retain version 1 configuration hashes.

For the tolerance-0.02 experiment, a baseline command is:

```bash
MPLCONFIGDIR=/tmp/model3_exp4_forward_matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python Model3/Experiment4/budgeted_steady_state.py \
  --n-qubits 10 --J 1 --gamma 1 --step-budget 500 \
  --preparations half_mixed --block-steps 10 \
  --steady-tolerance 0.02 --sampling-tolerance 0.004 --trajectories 8192 \
  --validation-mode forward \
  --baseline-study Model3/Experiment4/runs/budget_e8ec77f803de/study.json
```

Intermediate cleanup candidates such as 0.075, 0.08, 0.09 and 0.10 can be
supplied in `--step-grid` alongside the old grid, and `--beam-width` can retain
more burn-in paths. Wider searches still produce best-tested results, not
proofs of global minimal depth.

Executed tests of these ideas found a 117-step schedule (8 at dt=0.75,
56 at 0.25, and 53 at 0.05), passing the forward-mean/output-sampling checks
at R=8192 for seeds 29,30,31. The detailed comparison, changed stopping-rule
interpretation, and reproducible commands are in
[analysis/ForwardWindowStudy.md](analysis/ForwardWindowStudy.md).

For a smaller validation or exploration, restrict the grid and preparations:

```bash
.venv/bin/python Model3/Experiment4/budgeted_steady_state.py \
  --step-budget 100 --step-grid 1 0.5 0.25 0.125 0.0625 \
  --preparations neel half_mixed profile_mixed --no-plots
```

All supplied step sizes and `--steady-window` are physical times. The
default step grid is `(1,0.5,0.25,0.125,0.0625,0.03125,0.015625)` divided
by `max(J,gamma)`. The default validation window is `10/J`. The grid is
sorted from largest to smallest; values must be distinct and positive.

## Objective and search

The metric includes **all sites and bonds**:

```text
d(C,C*) = max(max_j |population_j(C)-population_j(C*)|,
              max_b |current_b(C)-current_b(C*)| / J)
```

`C*` is the physical stationary covariance from the continuous Lyapunov
equation. This is a reference-assisted preparation study. No exact transient
trajectory or random sampled path is used to choose a schedule.

For each candidate, evaluate the largest metric value over the final
`--steady-window` of physical time. Every integration node is included;
if the start of the window lies between nodes, the preceding node is
included conservatively. Candidates whose total time is shorter than the
window are marked incomplete. A low endpoint error alone cannot qualify.

The default total tolerance is 0.01, of which 0.004 is reserved for sampling
and 0.006 for the deterministic mean. Set `--steady-tolerance` and
`--sampling-tolerance` to rebalance the tradeoff. Sampling allowance may be
zero, but must be less than the total tolerance. Experiment4 has no separate
stationary-bias/relaxation split: its objective is the actual finite-window
mean error against the physical stationary observables.

Selection follows this order:

1. If any tested candidate satisfies the mean allowance over the covered
   window, select the one using the fewest Trotter steps. With a supplied
   depth model, minimize modeled total depth instead.
2. Otherwise, select the lowest covered-window error among candidates within
   budget. Report `target_not_met_within_budget`, without extending the budget.
3. If no candidate covers the validation window, report
   `tail_window_not_covered`. Available endpoint errors remain diagnostic
   values, not successful steady-state results.

For every preparation and grid value, the search evaluates constant-step
prefixes up to the applicable budget. These baselines are retained even if
the beam search is cut short. The additional beam search explores
nonincreasing step sizes with at most three stages by default. It considers
switches every `--block-steps` steps (default 10). Within each preparation
and final step size, it retains `--beam-width` paths (default 2), using
endpoint error as a search heuristic. All final candidate selection uses
the complete-window objective above. Keeping separate beams for each final
step size preserves coarse burn-in paths for later refinement.

`--search-node-cap` defaults to 50,000 beam expansions. If another complete
search layer would exceed the cap, search stops and records
`search_limit_reached=true`. Preparations receive the same number of full
search layers; constant baselines are always evaluated separately.
`--search-node-cap 0` requests constant baselines only. `--max-stages` and
`--block-steps` control the schedule family/resolution, not circuit depth.

This is a bounded heuristic search, not a global optimum or an accuracy
guarantee. Classical planning work can exceed 500 mean-channel steps;
the 500-step cap applies to each candidate circuit trajectory.

## Initial preparations and their cost

The defaults compare:

| Name | Local occupations | Preparation |
|---|---|---|
| `vacuum` | all zero | supplied all-zero system |
| `filled` | all one | parallel X gates |
| `left_excitation` | left site one, others zero | one X gate |
| `neel` | alternating one/zero, left site one | parallel X gates |
| `half_mixed` | each site occupied with probability 0.5 | independent classical bit randomization and X gates |
| `profile_mixed` | occupations equal the physical stationary site populations | reference-assisted independent classical bit randomization and X gates |

Every preparation has a diagonal initial covariance and zero initial
coherences and current. In particular, `profile_mixed` supplies the product
of local diagonal states, **not** the correlated nonequilibrium steady
state. At the default reservoirs the target current is 0.32 and the initial
current is zero. `half_mixed` and the pure preparations do not need the
stationary population profile. Remove `profile_mixed` from the CLI list
when comparing preparations without that prior information.

Preparation bookkeeping assumes a supplied all-zero register, excludes the
common register-reset cost, and reports the required extra local gate layer
and expected X-gate count per shot. Product mixtures are analytically
averaged in each simulated path. Implementing them on hardware requires
fresh classical bit choices per shot; the associated shot variance is not
included in the reported trajectory standard error.

## Step budget versus circuit depth

The unconditional hard cap is `--step-budget` per trajectory. Actual native
depth depends on compilation, connectivity, resets, and gate scheduling;
the script does not measure it. Preparation costs are reported separately.

If a per-step depth estimate is available, optionally supply both:

```bash
.venv/bin/python Model3/Experiment4/budgeted_steady_state.py \
  --step-budget 500 --estimated-step-depth 6 --depth-budget 3000
```

The additional additive model is

```text
estimated total depth = local preparation layer + estimated_step_depth * steps
```

Each preparation's allowed step count is reduced to fit this modeled cap.
For example, a preparation using one local layer and a six-layer step fits
499 steps into depth 3000, while a zero-layer preparation fits 500. This
accounts for warm-start overhead in the supplied model; it is not a
transpiler-derived circuit-depth guarantee. Compiled depth remains `null`
in saved output. `--system-formula exact` in particular must not be assigned
the same hardware cost as even/odd factors without an appropriate estimate.

## Replay, outputs, and interpretation

Only the selected preparation/schedule is replayed. All per-preparation
mean winners and constant baselines are saved for comparison. The replay
stores exact continuous Lindblad observables at the same times, sampled
means, deterministic means, stationary targets, trajectory standard errors,
jump choices, and final covariance matrices. Observable arrays use
`(initial_state=1, time, site_or_bond)` axes. Covariances use the shared real
gauge: physical `C_ij = (-i)^i i^j M_ij`, with
`C_ij = <c_j^dagger c_i>`. This represents XX bilinear observables, not the
full generally non-Gaussian many-body density matrix or its trace distance.

Each R value gets a separate outcome. `passed` requires a covered window,
mean error within its allowance, realized sampled-versus-mean discrepancy
and descriptive two-SE bands within the sampling allowance, and sampled
stationary error within the total tolerance. `R=1` cannot pass uncertainty
assessment. Other outcomes are `mean_target_not_met`,
`sampling_budget_not_met`, and `tail_window_not_covered`. These are checks
at finite integration nodes; they are not simultaneous confidence bounds,
continuous-time guarantees, or permanent-settling proofs. Increasing R
addresses sampling, not a failed mean target.

Outputs live in `Model3/Experiment4/runs/budget_<configuration hash>/`:

- `Study.md` and `study.json`: selection, per-preparation comparison, outcomes.
- `search.json`: baselines, best tested schedules, targets, search limits.
- `replay_R64.json` (and other requested R values): complete replay data.
- `figures/preparation_comparison.png`, `figures/schedule.png`, and
  `figures/observables_R64.png`: up to three site populations and three
  bond currents, plus stationary distance over all sites/bonds.

Numerical files are saved before plotting. `--no-plots` disables figures;
`--output-directory` changes the parent output folder. Identical settings
regenerate their configuration folder; changed settings use another hash.
No files in Experiments 1–3 are modified.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_model3_experiment4.py' -v
```

Small tests cover budget enforcement including preparation overhead,
constant baselines, exhaustive verification of a small schedule grid,
tail-window scoring and boundaries, capped search, seed/R independence,
product-mixture covariance dynamics against full density-matrix Kraus
channels and exact Lindblad evolution, R-prefix reproducibility,
serialization, plots, and preservation of numerical data on plot failure.
No default 500-step N=10 production result is claimed during implementation.
