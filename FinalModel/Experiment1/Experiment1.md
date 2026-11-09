# Experiment1: offline scaling of the three models through N=100

This study is feasible for the existing **XX chains** (Delta=0, Model1 h=0).
The continuous dynamics and ideal randomized-channel means close on N-by-N
one-body covariance matrices. A full 2^N density matrix, a circuit simulator
and hardware execution are unnecessary. Adding ZZ interactions invalidates
this reduction and requires a separate simulation strategy.

The experiment has four questions: how physical relaxation time scales with
N; how many actual digital collision/reset steps prepare an acceptable
output; how much the warped cosine family reduces that count; and how native
compiled resources and logical Clifford+T estimates scale. All stages are
implemented in [study.py](study.py), with explicit inputs in [config.json](config.json).
Running the experiment is staged: a completed reference study does not imply
that the schedule or resource studies have completed.

## Physical models and comparison policy

Use H=c J sum(XX+YY), with boundary injection gamma*p_b and loss
gamma*(1-p_b). N counts **system qubits**; compiled trajectories allocate
N+2 including the two reusable boundary ancillas. Physical time uses the
configured rates; Jt is the dimensionless time. Default J=1 for all cases.

| Model | c | gamma/J | p_left, p_right | Initial preparation |
|---|---:|---:|---|---|
| Model1 | 1/2 | 0.2 | 0, 0 | One excitation at zero-based site N//2 |
| Model2 | 1/4 | 2 | 0.25, 0.25 | Product half mixture |
| Model3 | 1/2 | 1 | 0.9, 0.1 | Product half mixture |

These are explicit representative choices from Model1/Experiment11,
Model2/Experiment4 and Model3/Experiment7. Model2's rates are divided by ten
relative to its J=10, gamma=20 convention, which multiplies physical times
by ten. Model3 uses its schedule-study gamma=1 rather than Experiment1's
gamma=2. We do not import a stored winning schedule.

The first panel measures **within-model size scaling**. Cross-model differences
also reflect hopping normalization, dissipation and preparation. A controlled
second panel should rerun a copied config with common c=1/2, gamma/J=1 and
half_mixed preparation. Keep both panels rather than silently redefining the
models. Sensitivity runs should use vacuum/filled preparations, additional
gamma/J values, tighter tolerances and an N-dependent continuation window.
Do not fit one universal exponent across these different protocols.

Sizes default to 2,4,6,8,10,15,20,30,40,50,60,80,100. `--sizes` can specify
any integers from 2 through 100, including every size in that range.

## Acceptance and error budget

The observable distance is the maximum over **all sites and all bonds** of
population error and excitation-current error divided by J. It is a local
observable criterion, not full many-body trace distance. The total target
0.02 is allocated as 0.016 deterministic mean, 0.003 trajectory/measurement
sampling, and 0.001 rotation synthesis. The runner validates the mean only;
sampling remains unspent, unvalidated headroom. Resource paths are a small
compilation sample, not a converged trajectory ensemble or shot estimate.

For the continuous reference, in the library's real gauge,

```
A = J * generator(model)
C(t) = C_star + exp(t A) (C(0)-C_star) exp(t A)^T
B(t) = max(1,4c) ||C(t)-C_star||_F
```

Since A+A^T is negative semidefinite, B(t) is nonincreasing and bounds the
local distance at **every later physical time**. Bisection finds its crossing
of 0.016, reporting `settling_time_upper_bound`, the numerical bracket, the
actual local error there, and the stationary-equation residual. This is an
exact-model calculation in floating point, but its crossing is a **sufficient
upper bound on local settling time**, not the earliest local threshold
crossing. Do not label a plot of this quantity "exact mixing time". The
Frobenius certificate can be conservative and can affect empirical scaling.

Schedule acceptance instead checks the output and **every node** of a
constant-final-step continuation covering at least 10 physical-time units.
It is finite-window digital settling; it does not prove permanent settling,
nor require agreement with the continuous transient. Continuation steps and
search evaluations are classical validation work and never enter the
reported executable preparation count. Equal error budgets do not make the
continuous permanent certificate and digital finite window identical tests.

## Actual preparation steps and warped cosine

One digital step applies even bonds, odd bonds, then **one sampled boundary
pump or loss dilation and reset**. Jumps are rate weighted; the selected
amplitude is strengthened by its selection probability. The collision
angle is sqrt(2 gamma dt); its loss probability is sin^2 of that angle,
not 1-exp(-2 gamma dt). For Model1 the pump branches have zero probability.
Large steps can have collision resonances; no monotonicity in dt or K is
assumed.

The methods are a fixed dt=0.05 baseline, searched constant dt, and

```
dt_k = dt_min + (dt_max-dt_min)/2 * (1+cos(pi*(k/(K-1))^q))
k = 0,...,K-1; K=1 uses dt_min.
```

Endpoint names denote schedule endpoints, not the global search limits.
Search bounds default to dt in [0.001,2], q in [0.1,8], K<=4096, and total
physical time <=1e7. Two differential-evolution seeds search K and the log
parameters jointly. A passing fixed baseline within the search bounds is
retained by the constant search. Constant endpoints remain admissible warped
schedules; the passing constant incumbent is retained. Each selected schedule is
independently rebuilt with the existing dense affine channel, including
every continuation node and output covariance physicality.

Report K_fixed, K_constant, K_warped, sum(dt), their acceptance status and
reductions 1-K_warped/K_baseline only for passing pairs. These are **smallest
passing counts found in bounded searches**, not proven minimum counts.
`ceil(t_continuous/dt)` is not a validated digital step count. Search failure
is explicitly `no_passing_schedule_found`; it is not physical infeasibility.
The initial K cap is a workload control, not an expectation that N=100 will
pass. Increase bounds and optimizer effort in a new config when needed.

The O(N^2) mean step avoids dense N^2-by-N^2 channel matrices and the existing
packed-covariance optimizer's O(N^4) storage. Long schedules and repeated
optimization can still be expensive. Profile the pilot before the full
schedule sweep. A selected schedule gets one independent dense replay;
large sampled ensembles are not required for this deterministic study.

## Compiled resources and logical fault-tolerant estimates

Compilation uses Qiskit locally, with no backend account. Each accepted
schedule is compiled for two fixed random paths on both all-to-all and a
bidirectional line. The boundary ancillas sit beside their system endpoints.
Optimization level 1 and transpiler seed 11 are fixed. Native basis is
RZ/SX/X/CX; report full depth, two-qubit depth, CX count, operation counts,
preparation measurements, resets, and terminal Z readout. These are abstract
dependency-layer depths, not pulse durations or an IBM-device forecast.
Additional X/Y observable settings and measurement repetitions are separate
workloads; the current resource table is per Z-readout trajectory.

Each source step has (N-1) Hamiltonian exchanges plus one bath exchange:
**NK logical exchange blocks** and K resets. Logical source count is
therefore O(NK). Compiled counts and depth are measured, not inferred from
this formula. Full materialized compilation is capped at 200,000 exchange
blocks per trajectory; capped rows have no invented compiled result.

A second local compilation lowers to Clifford gates plus RZ. Exact
Clifford-angle rotations cost zero T; odd multiples of pi/4 cost one T or
T-dagger. For R remaining generic rotations, allocate operator error

```
eta = synthesis_allowance / (2 * max(1,4c) * R)
T_estimate = exact_T + R * ceil(3*log2(1/eta))
```

The factor 2 and observable norm factor make the allocation consistent with
a telescoping unitary-error/expectation-error argument, also through ideal
measurements and resets. The T formula is only a **leading-order estimate**
from [Ross and Selinger](https://arxiv.org/abs/1403.2975); omitted asymptotic
terms mean rounding up does not make it a certified upper bound. No actual
Clifford+T approximation is synthesized here. The exact post-synthesis
Clifford count and T-depth remain null. Angle classification has floating
point tolerance. Validate representative rotations with a synthesis tool
before making publication claims about absolute fault-tolerant counts.

Ignoring special angles, this gives roughly
O(NK log(NK/epsilon_synthesis)) logical T scaling for this construction.
T-depth, magic-state factories, code distance, physical qubits and runtime
require an error-correction architecture and physical assumptions and are
not inferred from T count. Reset/measurement costs are retained separately.
Multiply per-path costs by the required paths, measurement settings and
shots only after an explicit sampling study.

## Commands and saved artifacts

Run from the repository root. The reference sweep through N=100 is the
recommended first run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/finalmodel_mpl \
  .venv/bin/python FinalModel/Experiment1/study.py exact
```

Then pilot schedule selection and resources with identical overrides:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python FinalModel/Experiment1/study.py schedules --sizes 2 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python FinalModel/Experiment1/study.py resources --sizes 2 4
```

After profiling, omit `--sizes` for the configured sweep. `--models model1`
restricts a stage to one model. A copied config can be passed with `--config`.
Configuration, source hashes and package versions identify a run directory.
Rerunning reuses completed case checkpoints; interruption restarts only the
incomplete case. Resources require schedule checkpoints with exactly the
same config and overrides. Source changes intentionally create a new run.

Each stage produces compact CSV/Markdown tables. References also produce
PNG/PDF log-log plots. Full local JSON checkpoints record parameters,
selected timesteps, search summaries, validation, resource samples and
provenance. No hardware job or cloud API is used by any stage. Fits should
be added only after accepted data cover a useful range, with fit ranges,
residuals and tolerance sensitivity; failed/capped rows are not data points.

Verification:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests -p 'test_finalmodel_scaling.py' -v
```
