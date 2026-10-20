# Experiment6: schedule control driven by a steady-state error budget

Experiment6 chooses timesteps automatically from the current deterministic
mean covariance and an accuracy budget. It accepts no timestep grid or prior
schedule. It runs one feedback policy, instead of comparing manually selected
multi-stage schedule configurations. The policy is reference-assisted: it
uses the known XX covariance channel and physical stationary covariance.
It plans one common schedule offline for all sampled trajectories; it is not
feedback from hardware measurements.

Defaults retain the recent study: N=10, J=gamma=1, reservoirs (0.9,0.1),
exchange prefactor 0.5, half_mixed preparation, Hamiltonian-first even-odd
splitting, total tolerance 0.02, mean allowance 0.016, sampling allowance
0.004, 500 preparation steps, and a 10-unit forward mean-validation window.
The sampled pump/loss dilation itself is unchanged.

## Run

From the repository root:

```bash
MPLCONFIGDIR=/tmp/model3_exp6_matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python Model3/Experiment6/adaptive_budget.py --seeds 29 30 31
```

The schedule is selected once, followed by independent R=8192 replays for
the three requested seeds. The default without `--seeds` checks seed 29.
R and seed do not affect any controller decisions. For planning alone:

```bash
.venv/bin/python Model3/Experiment6/adaptive_budget.py --plan-only
```

To assess sampling without changing the controller:

```bash
.venv/bin/python Model3/Experiment6/adaptive_budget.py --R-values 64 1024 8192
```

J, gamma, N, the two reservoir populations, preparation, tolerances, and
step/time caps are configurable on the command line. The sampling allowance
defaults to 20% of the total tolerance. `--minimum-dt` and `--maximum-dt`
specify bounds only, defaulting to 0.0001/max(J,gamma) and 1/max(J,gamma).
They do not specify a schedule or a list of candidates.

## How a timestep is chosen

At a decision point, let C be the current deterministic mean covariance,
C* the physical stationary covariance, and Phi_h the finite-step mean channel.
The metric is the same as Experiment4:

`d(A,B) = max(max_site |n(A)-n(B)|, max_bond |I(A)-I(B)|/J)`.

For an automatically proposed h:

1. Forecast H repeated mean steps from C, recording each local-observable
   error `e_k = d(Phi_h^k(C), C*)`. Default H=12.
2. Solve for the channel stationary covariance C_h* and compute its bias
   `b(h)=d(C_h*, C*)`. Reject invalid fixed points and negligible-damping
   resonances. The shared solver checks its residual and covariance positivity.
3. Use `max(e_H,b(h))` as a forecast error floor. A step that rapidly reduces
   error but has a large stationary bias receives a poor score.
4. Convert the forecast into a step-count proxy using the mean allowance
   epsilon and current error e_0. If the forecast floor fits epsilon, prefer
   the earliest forecast crossing, with a small residual-error tie-break.
   Otherwise extrapolate cost from progress per circuit step:

   ```text
   floor = max(e_H, b)
   score = k_first + 0.1*floor/epsilon                    if floor <= epsilon
           H*max(e_0-epsilon,epsilon)/(e_0-floor)
             + 0.1*floor/epsilon                         if floor < e_0
   ```

   A vanishing timestep with negligible progress receives a large cost. If
   none of the evaluated steps predicts progress within the remaining gate
   budget, double the prediction horizon automatically, up to 192 steps or
   the remaining preparation budget.
   If no progress is predicted even there, report a failure rather than
   silently spending the entire budget at the minimum timestep.

5. Minimize this score continuously in log(h) between the bounds. The bounds
   and previous h are also evaluated as safeguards. No hand-picked interior
   timestep list or retained winning schedule is used.
6. Commit at most four actual mean steps, checking for a valid output after
   every step. Then recompute h from the updated state. The timestep may grow
   or shrink; it is not restricted to a monotone three-stage schedule.

This is receding-horizon model control. Numerical trial evaluations are
necessary to choose h, but they are automatic local model predictions, not
manual experiments or a sweep of whole schedule configurations. The score
is a heuristic, not a proven bound on remaining preparation steps. The
numerical routine is a
[local bounded scalar optimizer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html),
so this does not establish globally minimal depth or guarantee that it beats
the separately optimized 117-step schedule.

Policy settings `--prediction-steps` (12), `--maximum-prediction-steps` (192), `--decision-steps` (4),
`--optimizer-iterations` (24), and `--optimizer-log-tolerance` (0.025) expose
the accuracy/work tradeoff in classical control. Their defaults are a single
policy configuration, not settings fitted by loading earlier schedules.
The current step count is a depth proxy; native depth can depend on gate
angles and routing and is not explicitly optimized here.

## Actual acceptance checks

A forecast is never enough for success. After an actual preparation step
reaches the mean allowance, independently evolve the mean from that output
at its final h for at least the requested physical-time window. The output
and **every future integration node** must fit the mean allowance. Failure
at an intermediate node rejects the check immediately. The hypothetical
future state is never committed to the preparation state. Passing the check
stops the circuit at the output.

Sampling is replayed only after a passing mean plan. At the output, require
R>1, sampled-to-mean discrepancy and maximum descriptive two-SE values below
the sampling allowance, and sampled physical error below total tolerance.
Initial product mixtures are analytically averaged per random jump path;
preparation-bit noise, measurement shots, and hardware noise are excluded.
Two-SE checks are not simultaneous confidence bounds. A finite-node window
does not prove continuous-time or permanent settling.

Prediction steps, rejected probes, and forward mean checks are classical
planning costs. Only the committed preparation prefix appears in replay
`schedule.step_dts`, trajectory choices, and `workload.per_trajectory_steps`.
Explicit failure statuses include `step_budget_reached`,
`maximum_time_reached`, `no_predicted_progress`, and `no_valid_timestep`. A failed mean plan does not
trigger sampled replay or a hidden fallback to an earlier experiment.

## Results and audit trail

Each configuration writes `Model3/Experiment6/runs/adaptive_<hash>/`:

- `plan.json`: every decision, automatically evaluated h values, scores,
  stationary-bias checks, committed states' errors, validation attempts,
  preparation schedule, and classical prediction/validation work counts.
- `study.json` and `Study.md`: controller status, selected schedule, and replay
  summaries. Plan-only or failed runs explicitly lack sampling validation.
- `replay_seed29_R8192.json` etc.: observables, uncertainty, exact reference,
  jump choices, output checks, and the separate forward mean check.
- `figures/controller.png`, representative observable plots, and output/forward
  validation plots.

Preparation and any failed forward check remain inspectable. The plan is saved
before sampled replay, and replay data is saved before plotting. The common
replay engine independently recomputes the actual forward mean errors.

## Circuits and depth

Experiment5 accepts the Experiment6 replay schema. To compile a successful
new controller result, use its actual path:

```bash
.venv/bin/python Model3/Experiment5/circuit_depth.py \
  --source Model3/Experiment6/runs/adaptive_HASH/replay_seed29_R8192.json
```

This is a separate offline compilation. Experiment6 does not run hardware
jobs or calculate native depth by itself. Exact-Hamiltonian replay is an
optional classical reference mode; Experiment5 supports the even-odd circuit
mode only.

## Validation

`tests/test_model3_experiment6.py` checks budget-dependent scoring, state-dependent
continuous timestep choices, independence from seeds and R, absence of beam
search/prior-schedule loading, actual-versus-speculative state evolution,
intermediate forward-window failures, explicit budget/time/resonance failures,
zero-step stationary preparation, replay and circuit-schema compatibility,
R-prefix preservation, plan-only behavior, persistence, and plotting.

## Executed N=10 result

The command at the top was executed using the v2 controller, with its default
policy settings and no grid, retained schedules, or prior run inputs. It
produced **273 preparation steps**, ending at physical time 15.3977123.
The timesteps vary with the state: there are 64 constant-dt blocks, with
chosen values ranging from 0.0001 to approximately 0.356742.

The maximum forward mean error is **0.01408550**, below the 0.016 allowance.
At R=8192, all requested seeds pass:

| Seed | Sampled output error | Sampled-to-mean distance | Maximum two SE | Status |
|---:|---:|---:|---:|---|
| 29 | 0.01307689 | 0.00198044 | 0.00139440 | passed |
| 30 | 0.01301596 | 0.00091622 | 0.00138721 | passed |
| 31 | 0.01239639 | 0.00131704 | 0.00139759 | passed |

The mean output error is 0.01251049. The schedule was fixed before replay and
is identical for all seeds. The source arrays and separate forward means
were verified against the saved assessments; Experiment5 accepts the replay
files. Fourteen controller tests passed, including a functional test that
checks schedule changes when the tolerance is tightened. The seven existing
Experiment5 circuit tests also passed.

The controller required about 68 seconds of classical planning in this
environment. It used 30,960 forecast mean steps and 103,913 validation mean
steps in total. Its last chosen timestep is the lower bound, 0.0001, so the
successful final forward check alone uses **100,000 mean-only steps** to cover
10 physical-time units. This is substantial classical work, explicitly
excluded from the 273-step executable circuit. Raising the lower bound changes
the policy's permitted schedules and requires a fresh acceptance check.

The result does **not** beat Experiment4's separately searched 117-step
schedule. It demonstrates autonomous selection and passing accuracy under the
same forward-mean/output-sampling protocol. Prediction efficiency and native
depth can be improved independently; no native-depth result is claimed here.

Saved files:

- [Study report](runs/adaptive_6ea9f15a44d8/Study.md).
- [Controller decisions and plan](runs/adaptive_6ea9f15a44d8/plan.json).
- [Independent metric verification](runs/adaptive_6ea9f15a44d8/VerifiedResults.json).
- [Controller and chosen timestep plot](runs/adaptive_6ea9f15a44d8/figures/controller.png).
- [Representative observables, seed 29](runs/adaptive_6ea9f15a44d8/figures/observables_seed29_R8192.png).
- [Sampled output and forward mean check](runs/adaptive_6ea9f15a44d8/figures/validation_seed29_R8192.png).

An initial v1 prototype minimized short-horizon error without penalizing weak
progress per circuit step. It shrank h to its lower bound, reached 500 steps
at time 6.9213, and failed with mean error 0.04985. That diagnostic output is
retained in [adaptive_a7cedf39cf03](runs/adaptive_a7cedf39cf03/Study.md).
The v2 score and automatic horizon extension above address that observed
failure mechanism. This was a controller-design correction, not a schedule
borrowed from the earlier 117-step experiment.
