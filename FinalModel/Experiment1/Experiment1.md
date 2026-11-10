# Experiment1: full offline scaling through N=100

All three models use the agreed normalization:

```
H = (J/2) sum_i (X_i X_(i+1) + Y_i Y_(i+1)), J=1, gamma=1.
```

Boundary injection/loss rates are gamma*p_b and gamma*(1-p_b). N counts
system qubits, excluding two reusable boundary ancillas.

| Model | p_left, p_right | Initial preparation |
|---|---|---|
| Model1 | 0, 0 | One excitation at zero-based site N//2 |
| Model2 | 0.25, 0.25 | Product half mixture |
| Model3 | 0.9, 0.1 | Product half mixture |

Preparations and reservoirs remain different. The earlier pilot used
different rates/normalizations and is preserved in [InitialPilot.md](InitialPilot.md).

Configured sizes: 2,4,6,8,10,15,20,30,40,50,60,80,100, giving 39 model/size
cases. [config.json](config.json) records physical and search inputs.
No hardware, account service, noisy simulator or mitigation stage is used.

For backend-specific resources from these same schedules, use the separate
[Phoenix compilation script and instructions](PhoenixResources.md). It
captures a full real backend target, then compiles complete circuits
locally. The generic resource figures below remain a separate baseline.

## Acceptance and continuous relaxation

The distance is the maximum over all sites/bonds of population error and
excitation-current error divided by J. This is local-observable accuracy,
not full many-body trace distance. Model1 may retain excitation after local
observables pass; its output total excitation is retained in the report.

The target 0.02 is allocated as 0.016 deterministic mean, 0.003 sampling and
0.001 synthesis. Only the mean is validated here. The sampling allowance is
headroom, not a finite-trajectory or measurement-shot guarantee.

For the exact XX model, N-by-N covariances evolve as

```
A = J * generator(model)
C(t) = C_star + exp(t A) (C(0)-C_star) exp(t A)^T
B(t) = max(1,4c) ||C(t)-C_star||_F, c=1/2.
```

A+A^T is negative semidefinite, making B(t) nonincreasing and a bound on
every later local error. Bisection finds its 0.016 crossing. Report this as
a permanent local-settling-time **upper bound**, not the earliest local
crossing or many-body mixing time. Numerical brackets and stationary
residuals are saved. The physical-time cap is 1e7.

Digital schedules must pass at output and at every constant-final-step
continuation node covering at least 10 physical-time units. This is a
finite-window mean test, not permanent digital convergence. Continuation
and search work are classical validation and do not add preparation steps.
These continuous and digital guarantees are different; their physical
times are not compared as equivalent mixing-time guarantees.
Schedules optimize stationary-observable preparation; no tolerance on
matching the continuous transient is imposed.

## Steps and search policy

K counts executable preparation steps. Each applies even and odd system
bonds, one rate-weighted sampled boundary pump/loss exchange, then a reset.
The collision angle is sqrt(2 gamma dt); it gives sin^2 damping, not exact
exponential damping. Zero-rate Model1 pump branches are never selected.

The fixed dt=0.05 baseline directly steps the actual mean channel, with
K<=262144. Searched constant schedules start with 12 logarithmic and 20
linear timestep probes, the fixed step and in-bounds swap/collision
landmarks, followed by joint differential evolution on K and log(dt).
Warped cosine jointly searches K, log endpoints and log(q):

```
dt_k = dt_min + (dt_max-dt_min)/2 * [1+cos(pi*(k/(K-1))^q)]
k=0,...,K-1; K=1 uses the smaller endpoint.
```

Bounds: dt in [0.001,2], q in [0.1,8], search K<=4096 and total time <=1e7.
The search range narrows to a passing incumbent where possible; a fixed
baseline exceeding the search cap is retained. Constant incumbents are
valid equal-endpoint warped schedules. Two seeds (1729,1730) get up to 24
generations and population multiplier 6. Selection uses a 0.5% tolerance
margin. Each selected schedule is independently replayed by the existing
dense affine channel, including all continuation nodes and physicality.

Counts are best passing results found, not proven minima. Optimizer
convergence flags are separate from physical acceptance. No failed or
capped case is converted into a passing result by extrapolation.

## Acceleration and compilation

[full_study.py](full_study.py) runs the sweep with four local worker
processes and one BLAS thread each. [accelerated.py](accelerated.py) compiles
[channel.cpp](channel.cpp) locally with g++ -O3 into a source-hashed temporary
directory. Its O(N^2) update introduces no new channel approximation and is
checked against the existing independent dense implementation. There is
no exponentially large state or dense N^2-by-N^2 channel matrix.

[stream_compile.py](stream_compile.py) transpiles symbolic exchange blocks
locally to RZ/SX/X/CX and separately Clifford+RZ, binds chosen angles and
composes blocks without inter-block optimization. Counts and dependency
depths are computed exactly by streaming the compiled dependency maps,
verified against materialized compositions. This declared block-preserving
compiler avoids a full-circuit size cap; it is not a globally optimized
transpilation forecast.

All gates lie on the line with boundary-local ancillas, so this compiler's
line and all-to-all costs coincide. Source steps have N exchanges and one
reset. Report compiled operations, depth, two-qubit depth and N+2 logical
qubits. Dependency depth is not pulse duration. Initial preparation
measurements and final Z measurements are included. Paths with seeds 29,30
sample resource differences, not trajectory convergence.

For R generic RZ rotations, allocate eta=0.001/[2 max(1,4c) R]. Exact
Clifford/T angles are detected numerically; remaining rotations use the
leading ceil(3 log2(1/eta)) T estimate from
[Ross and Selinger](https://arxiv.org/abs/1403.2975). Omitted terms mean this
is neither a certified upper bound nor actual synthesis. Report base
Clifford gates excluding RZ separately. Exact post-synthesis Clifford count,
T-depth, magic-state factories, code distance, physical qubits and runtime
are not computed.

## Run and reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/finalmodel_mpl \
  .venv/bin/python FinalModel/Experiment1/full_study.py --workers 4
```

Use the printed run directory to regenerate the audited report and
standalone PNG/PDF figures:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/finalmodel_mpl \
  .venv/bin/python FinalModel/Experiment1/analyze_full.py FinalModel/Experiment1/runs/full_<hash>
```

`--stage exact|schedules|resources` reruns a stage; `--config` accepts another
config. Resources require matching schedule checkpoints. Atomic per-case
checkpoints resume complete cases. Source hashes, installed package
versions and config identify each run. [study.py](study.py) retains the
original small-case helper/reference functions; the full runner is the
recommended entry point.

[Results.md](Results.md) links the complete report. Full numerical JSON
stays local; compact CSV/Markdown and figures are retained. Fits state their
range and residuals and are empirical, not asymptotic laws.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/finalmodel_mpl \
  .venv/bin/python -m unittest discover -s tests -p 'test_finalmodel_scaling.py' -v
```
