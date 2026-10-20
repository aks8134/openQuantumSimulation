# Experiment 3: manual and automatic randomized Lie–Trotter schedules

This classical experiment follows the workflow of Model1/Experiment11's
`classical_randomized_lie_trotter.py`, using Model3's four boundary channels
and either a user-supplied timestep sequence or a reference-assisted
automatic steady-state controller. All times and step sizes on the CLI are
physical times, rather than dimensionless `J*t`.

Defaults: `N=3`, `J=gamma=1`, `p_left=0.9`, `p_right=0.1`, `delta=0`,
and exchange prefactor `c=0.5`:

```text
H = c J sum_j (X_j X_{j+1} + Y_j Y_{j+1})
```

`--J` and `--gamma` accept positive finite values. The common rate `gamma`
applies to both endpoints. Reservoir populations accept the full interval
`[0,1]`. Pump/loss rates are `gamma*p_left`, `gamma*(1-p_left)`,
`gamma*p_right`, and `gamma*(1-p_right)`. The initial preparations default
to all zero, all one, and a left-site excitation. Custom binary preparations
can be selected with `--initial-states` (leftmost bit = leftmost site).

## Run with a manual schedule

From the repository root:

```bash
.venv/bin/python Model3/Experiment3/classical_randomized_lie_trotter.py \
  --J 1 --gamma 1 --n-qubits 3 --trajectories 64 \
  --dt-schedule 2:0.01 10:0.05 20:0.1
```

Each `END:DT` pair sets the step size from the previous endpoint (initially
zero) up to `END`. This example uses 0.01 until time 2, then 0.05 until time
10, then 0.1 until time 20. Endpoints must strictly increase; step sizes
must be positive. A shorter final step in each interval reaches its endpoint
exactly. The final time is inferred from the last endpoint. An optional
`--t-final` must agree with it. Step sizes may increase or decrease.

Alternatively, specify every consecutive step directly:

```bash
.venv/bin/python Model3/Experiment3/classical_randomized_lie_trotter.py \
  --step-dts 0.01 0.01 0.02 0.05 0.1 --initial-states 100
```

This runs five steps through time 0.19. Explicit steps are not repeated.
The three schedule modes are mutually exclusive. For a constant-step
baseline, use:

```bash
.venv/bin/python Model3/Experiment3/classical_randomized_lie_trotter.py \
  --t-final 20 --trotter-delta-t 0.05
```

With no schedule flags, the script uses `t_final=20` and step size 0.1.
`--record-every 10` saves observations every tenth step, always including
time zero and the final time; it does not change any integration step.
The complete integration schedule and jump choices are saved even when
observations are thinned. `--step-cap` defaults to one million.

Other controls: `--seed-trajectories` (alias `--seed`, default 29),
`--p-left`, `--p-right`, `--exchange-prefactor 0.25`, `--sampling-policy
uniform`, `--steady-tolerance`, `--output-directory`, `--no-plots`, and
`--no-progress`.

## Automatic steady-state schedule

Enable the controller explicitly; manual scheduling remains the default:

```bash
.venv/bin/python Model3/Experiment3/classical_randomized_lie_trotter.py \
  --auto-schedule --n-qubits 10 --J 1 --gamma 1 --trajectories 64 \
  --steady-tolerance 0.01 \
  --initial-dt 0.5 --minimum-dt 0.00001 \
  --check-interval 5 --steady-window 10 --max-time 200
```

This is a two-pass method. First, a deterministic mean-channel calculation
plans the common schedule, without drawing any random choices. Second, the
usual sampled solver replays those exact time nodes. Changing `R` or the
seed leaves the planned schedule unchanged. Exact continuous trajectories
are generated during replay for comparison; the controller uses the
physical stationary covariance, not the exact transient trajectory.

The controller uses

```text
d(C,D) = max(max_j |n_j(C)-n_j(D)|,
             max_b |I_b(C)-I_b(D)| / J)
```

over all supplied preparations and all sites/bonds. For total tolerance
`epsilon`, the default budgets are `0.3*epsilon` for stationary timestep
bias, `0.3*epsilon` for mean-channel relaxation, and `0.4*epsilon` for
sampling. Set these explicitly with `--bias-tolerance`,
`--relaxation-tolerance`, and `--sampling-tolerance`. They must be positive
and sum to at most `--steady-tolerance`.

The planning pass proceeds as follows:

1. Starting at `--initial-dt`, solve the finite-step mean-channel stationary
   covariance and measure its bias against the physical stationary
   covariance. Halve until the first passing step, including the minimum
   step if necessary. This is the largest passing **tested** step, not a
   global optimum. Failed numerical solves and negligible endpoint-damping
   resonances are recorded and rejected.
2. Evolve the deterministic mean from every requested preparation, beginning
   at the largest valid candidate. At each physical-time checkpoint, refine
   to the next valid candidate when relaxation to the current channel's
   stationary covariance is no larger than
   `max(relaxation_tolerance, refine_factor*bias)`. The default
   `--refine-factor` is 1: refine when the stationary bias becomes comparable
   to or larger than the remaining relaxation error.
3. Once the bias budget passes, keep that nominal step and evolve longer
   until relaxation is within its own budget. Both this condition and the
   actual mean distance to the physical target (at most bias budget plus
   relaxation budget) must hold at **every integration node** for the full
   `--steady-window`. Stopping decisions occur at checkpoints.

The physical default initial step is `0.5/max(J,gamma)`, the nominal minimum
is `1e-5/max(J,gamma)`, the checking interval is `5/J`, the window is twice
the checking interval, and the time cap is `200/J`. The checking interval
must be at least the initial step, and the window must be at least one
checking interval. The last step into a checkpoint or time cap may be
shortened below the nominal minimum. Its actual channel is used in both
passes, and its effect is included in the window checks. Checkpoints at
multiples of the initial step avoid regular remainder steps during halving.

Automatic controls require `--auto-schedule`. That flag is mutually
exclusive with all three manual schedule flags. Use `--max-time` to bound
automatic evolution; `--t-final` is reserved for manual schedules.
`--step-cap` bounds the total number of planned/replayed integration steps.

After replay, the sampling assessment checks every integration node in the
passing tail window for all preparations/sites/bonds. It requires:

- Realized sampled-versus-mean discrepancy within the sampling budget.
- Descriptive two-standard-error bands within the sampling budget.
- Sampled distance to the physical stationary observables within the total
  tolerance.

`R=1` cannot pass the sampling check because uncertainty is unestimable.
The two-SE test is descriptive, not a simultaneous confidence bound. A
passing result covers the saved integration nodes in a finite window;
it does not establish permanent settling or full-state trace distance.
The controller does not change `R` automatically or tighten the timestep
in response to sampling fluctuations. A run can therefore have a converged
mean controller and `sampling_budget_not_met`, including with `R=64`.
Increasing `--trajectories` addresses that separate limitation.

In automatic mode, `--record-every` thins the transient output, but decision
nodes and **all** tail-window nodes are always saved for the sampling check.
Plots display up to three representative populations and currents; the
stationary-distance metric continues to include all sites/bonds.

## Channel and reference

Each step independently selects exactly one jump, with default probabilities
`(p_left, 1-p_left, p_right, 1-p_right)/2`. The selected two-level ancilla
dilation scales its jump by `1/sqrt(probability)`, and traces/resets the
ancilla after the step. Zero-rate jumps have zero sampling probability
under the rate-weighted policy. Signed cosines
`cos(sqrt(rate*dt/probability))` are retained, including for large steps.

To follow the Model1 reference, defaults are **Hamiltonian first, then
selected jump**, with **even then odd bonds** in the Hamiltonian factor.
`--order jump-first` and `--system-formula exact` reproduce the factor
choices used by default in Model3/Experiment2. The latter exponentiates
the full XX Hamiltonian factor; both choices still use the finite-step
randomized dilation for the reservoir.

The solver uses the shared `library/reservoir_scaling.py` covariance
channels. Pumping accesses multiple excitation sectors, so the Model1
single-excitation amplitude reduction does not apply. The `N x N`
covariance evolution is exact for these XX populations and currents for
each chosen channel sequence. It does not reconstruct the full, generally
non-Gaussian many-body state. Interactions (`delta != 0`) are unsupported.

The exact reference is continuous Lindblad evolution at the same recorded
physical times, with covariance generator `A = -i*h - G/2` and physical
stationary covariance obtained from `A C_* + C_* A^dagger + P = 0`.
Internally, the shared library uses a real gauge and dimensionless time:
`ratio=gamma/J`, `step=J*dt`. Physical currents include the factor of `J`.
For the defaults, stationary populations at `N=3` are `(0.58,0.5,0.42)`
and both stationary excitation currents are `0.32`.

## Outputs and interpretation

Each configuration saves a JSON file in `results/`, an observables PNG
and an executed-schedule PNG in `figures/`, under this experiment by default.
The filename hash includes the physical model, preparations, seed,
trajectory count, complete time grid, recording settings, factor choices,
and tolerance. Changing the schedule creates a different result file.
Repeating identical arguments regenerates the same deterministic result
and replaces that configuration's files. Numerical results are saved before
plotting, so a failed plot does not lose the simulation.

JSON includes:

- Every time node, actual step size, recorded step count, and each
  trajectory's selected jump (`0=left pump`, `1=left loss`, `2=right pump`,
  `3=right loss`). Jump choices have axes `(trajectory, step)`.
- Mean site populations, magnetizations `Z=1-2*n`, and excitation currents
  (positive left to right), with trajectory standard errors. Observable
  arrays have axes `(initial_state, recorded_time, site_or_bond)`.
- Deterministic mean-channel and exact Lindblad observables, stationary
  targets, and maximum sampled-time errors against each reference.
  These separate dilation/Trotter bias from finite-trajectory sampling error.
- Final per-path covariances in the documented real gauge. The same random
  jump sequence is shared across preparations for each path. Independent
  path streams preserve existing trajectories when `R` increases.

Automatic runs additionally save controller settings, candidate timestep
solves and their bias/residual checks, stage transitions, checkpoint
decisions, final targets, and the stopping reason. A separate `plans/`
JSON is written before trajectory replay. Controller settings and generated
nodes enter the result hash; manual configurations keep their original
schema and filename hashes. The final result reports `run_status`:

- `passed`: mean budgets and the replay sampling checks passed their finite
  window.
- `sampling_budget_not_met`: the mean controller converged; the replay
  sampling check gives the specific reasons.
- `minimum_dt_reached`: no tested step passed the stationary bias budget.
  The saved result contains the initial state only, and no figures are made.
- `max_time_reached` or `step_cap_reached`: the mean controller did not
  establish a passing window within the supplied limit.

Numerical failures during candidate stationary solves are recorded. A
bounded, nonconverged run is saved with its status rather than labeled
successful. `sampling_assessment` is `not_assessed` when the mean controller
has not converged. The result also records planning time separately and
the replay-versus-planned final observable discrepancies.

Plot solid lines show sampled means, shaded regions show one standard error,
dashed lines show exact Lindblad evolution, and dotted lines show the
deterministic mean channel. Faint horizontal lines show stationary targets.

The reported stationary distance is the largest site population deviation
or bond current deviation divided by `J`, against the physical stationary
observables, separately for each preparation. `first_sampled_time_within_tolerance`
is an observed threshold crossing. This metric is not full-state trace
distance, and a finite ensemble can leave the tolerance later. Large steps
can produce stationary bias. `R=1` saves zero standard-error placeholders
and marks uncertainty as unestimable. Saved maximum errors cover recorded
times only; thinning observations can miss peaks between them.

## Replot existing data

To regenerate clearer plots from existing saved JSON files, without running
any evolution, use:

```bash
.venv/bin/python Model3/Experiment3/replot_selected_observables.py
```

This saves full-range and late-time views in `figures/three_observables/`,
with links in `Plots.md`. The N=10 views display sites 0, 4, and 9 and
bonds 0–1, 4–5, and 8–9; smaller chains display up to three available
sites/bonds. The stationary-distance panel retains the saved metric over
all sites and bonds. `--sites`, `--bonds`, and `--late-fraction` customize
the view. Original results and plots are preserved.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_model3_manual_schedule.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_model3_auto_schedule.py' -v
```

Tests check interval remainders, explicit schedules, physical units, mean
channels and sampled variable-step sequences against full density-matrix
dilations, exact references against the existing Model3 solver, seed
reproducibility, endpoint reservoirs, recording independence, configuration
separation, and saved plots.
Automatic tests additionally cover lazy parsing, scaled defaults, budget
validation, halving and decision history, seed/R independence, replay of
checkpoint remainders, complete tail checks with thinned output, explicit
failure statuses, stationary-solve rejection, sampling diagnostics, and
saved plans/results. These are small validation cases; no N=10 automatic
production run is claimed here.
