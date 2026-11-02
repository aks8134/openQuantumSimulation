# Experiment4: minimum-step cosine preparation for Model2

See [the updated parameter study](ParameterStudy.md): wider searches find
a 44-step warped cosine at the original rate ratio, compared with 45
constant steps. A tuned rate gives a 20-step constant candidate. The
[initial bounded search](Results.md) and its earlier 80-step result are
retained with the correction made explicit.

Search for short steady-state preparations using the same schedule idea as
Model3. The requested initial state is **half-mixed**, `(I/2)^tensor N`.
Defaults are N=10, J=1, gamma=2, equal reservoirs p=0.25, Delta=0 and
`H=(J/4) sum(XX+YY)`. This is Model2's normalization. J=1, gamma=2 is the
dimensionless equivalent of the earlier J=10, gamma=20 benchmark; physical
times differ by a factor of ten.

Each step independently selects one of four rate-weighted jump dilations,
with probabilities `(0.125,0.375,0.125,0.375)`, angle
`sqrt(2 gamma dt)` and its signed cosine. The ancilla is reset once per
step. This is the collision channel used in the existing shared numerical
library, rather than an exact-reservoir exponential. The default order is
jump then Hamiltonian, as in Model2. The default Hamiltonian uses even/odd
bond layers, as in the implementable Model3 circuits. `--system-formula exact`
restores the unsplit Hamiltonian factor of Model2 Experiments 2/3; results
from these two choices must be distinguished.

The exact stationary covariance is `C*=p I`: every population is 0.25 and
every current is zero. Half-mixture is not stationary at this reservoir
population. Rate-weighted mean steps preserve `p I` at every dt, including
with the Hamiltonian split. Thus the stationary bias that motivated late
refinement in Model3 is absent here. A constant-step baseline is included
to measure whether annealing actually helps.

## Schedules and bounded search

For M >= 2, both cosine families use

```
dt_k = dt_min + (dt_max-dt_min)/2 * [1+cos(pi*(k/(M-1))^q)]
k = 0,...,M-1
```

Ordinary cosine fixes q=1; warped cosine optimizes q in [0.1,4]. Constant
steps are optimized separately. The one-step cosine uses its smaller
endpoint. Endpoint steps are bounded to [0.01,2] in physical time, and the
preparation time is capped at 500. These are explicit search restrictions;
increasing the maximum dt can expose collision resonances and different
passing regions. Both endpoints are optimized in log coordinates. Equal
endpoints are allowed, so the cosine families contain constant schedules.

Seeded differential evolution jointly searches the integer step count
(1 through 256) and the continuous parameters. Independent optimizer seeds
1729 and 1730 are used. After the joint search, four consecutive smaller
counts are explicitly challenged with both seeds and local refinement.
The complete trial summaries and selected step list are saved. An optimizer
failure means “no passing schedule found”; it does not prove infeasibility.
This experiment finds the **smallest passing schedule found within its
declared search**, rather than a mathematically certified global minimum.

## Acceptance and independent replay

The metric checks all N populations and N-1 excitation currents:

```
d(C,C*) = max(max_site |n-n*|, max_bond |I-I*|/J).
```

Total tolerance is 0.02, allocated as **0.016 mean + 0.004 sampling**.
Selection uses deterministic means only. The output and every node of a
constant-final-step continuation lasting at least 10 physical-time units
must satisfy the mean allowance. The full selected schedule is recomputed
independently and checked for covariance physicality and target invariance.
Continuation is classical validation and adds no preparation steps.
No transient-matching tolerance is imposed.

A separate permanent local-observable upper bound is reported:
`max(1,4c) ||C_output-pI||_2`. Every homogeneous sampled branch is a
contraction and preserves `pI` after rate-weighted averaging. Therefore the
operator norm of the mean deviation cannot increase under subsequent steps.
Passing this stronger bound establishes all later **ideal mean node** errors
within the allowance; a failure does not invalidate a passing finite-window
check. It is evaluated in floating point, with no interval-arithmetic or
many-body trace-distance claim.

Selected schedules are replayed at fixed seeds 29, 30, 31, initially with
8192 paths. If any seed fails, all three are retried at 16384 and then 32768;
all attempts remain visible, and no seeds or schedules are selected from
sampling outcomes. Each replay requires output error <=0.02, sampled-minus-
mean distance <=0.004 and maximum descriptive two SE <=0.004. Per-path RNGs
preserve prefixes as R increases. Initial preparation bits are analytically
averaged. These are empirical checks, not simultaneous confidence bounds;
measurement shots and hardware noise are excluded.

Step count measures one collision/reset per trajectory step. Native CZ
counts and compiled depth are not measured here. With even/odd Hamiltonian
splitting, the logical exchange count is N per step, including the selected
bath interaction. Exact Hamiltonian factors have a different implementation
cost. Earlier strict-transient tolerance studies use different criteria,
preparations and time horizons and are not equal-protocol depth baselines.

## Run

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model2_cosine_mpl \
  .venv/bin/python Model2/Experiment4/cosine_search.py
```

Use `--plan-only` for deterministic search without sampling; `--families
cosine` for ordinary cosine alone. `--help` lists tolerances, preparation,
model parameters, search bounds, seeds and workload controls. The forward
window and dt bounds are physical times; rescale them when changing J and
gamma together if the same dimensionless optimization is intended.

Each configuration writes `runs/cosine_<hash>/study.json`, `Study.md` and
`comparison.png/pdf`. Source hashes, full selected schedules, actual mean
observables, forward nodes, all optimizer trials and replay summaries are
retained. Configuration changes create separate directories. Figures can
be suppressed with `--no-plots`.

Independently reconstruct a saved study and export schedule CSVs and actual
population/current figures, without rerunning optimization or sampling:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model2_cosine_mpl \
  .venv/bin/python Model2/Experiment4/analyze_results.py \
  Model2/Experiment4/runs/cosine_<hash>
```

This verifier builds the four branch matrices and their weighted sum
explicitly, without importing the search or shared channel implementation.
It writes `Verified.md`, `verification.json`, `<family>_schedule.csv`, and
`observables.png/pdf`. The CSVs contain every executable preparation step;
forward-validation work is excluded.

The verifier also tests a sharper permanent mean condition for c=1/4:
if the signed deviation `B=±(C_output-pI)` and its one-step decrease
`B-M(B)` are positive semidefinite, positivity implies
`0 <= M^(k+1)(B) <= M^k(B)`. Populations and currents/J are then bounded
by the largest diagonal of B for every later constant-final-step node.
Both minimum eigenvalues must exceed 1e-10 to accept the numerical check.
This can establish permanent local mean accuracy even when the coarser
spectral-norm bound fails. It does not extend the finite-trajectory replay
checks to permanent sampling guarantees.

For checkpointed warped-cosine challenges starting from a completed ordinary
cosine, use:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model2_cosine_mpl \
  .venv/bin/python Model2/Experiment4/refine_warped.py \
  Model2/Experiment4/runs/cosine_<hash>
```

The source ordinary cosine is a valid warped schedule at q=1. Four shorter
counts are challenged using the source optimizer seed list, 16 global
iterations per trial and two local starts. Each complete trial and each
replay is saved; repeat with `--resume` to reuse them. This follow-up has its
own configuration hash and provenance back to the exact source study. It
does not inherit acceptance for any shorter count. Replays begin only after
all shorter-count trials are complete.

## Broader rate and schedule-parameter study

The new [parameter-study report](ParameterStudy.md) corrects the earlier
80-step conclusion and includes comparison figures, validated candidate
CSVs, and the [20-step mean lower-bound argument](StepLowerBound.md).

Run the seven-rate sweep at fixed p=0.25 and half-mixture input:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model2_cosine_mpl \
  .venv/bin/python Model2/Experiment4/parameter_study.py
```

It scans constant timesteps, then fits cosine/warped schedules at explicit
fixed counts. The defaults include Gamma/J=pi/8, identified by aligning
the swap and collision angles. Equal-endpoint constants remain valid
members of the cosine families. Accelerated propagation uses
`C=pI+(0.5-p)K`; all selected schedules are recomputed with the original
affine channel. This stage evaluates deterministic means only. Full trial
summaries are saved per case; `--resume` reuses completed cases and restarts
an interrupted case. Key follow-up commands are:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model2_cosine_mpl \
  .venv/bin/python Model2/Experiment4/parameter_study.py \
  --ratios 2 --iterations 64 --optimizer-seeds 1729 1730 3141 --neighbor-counts 5

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model2_cosine_mpl \
  .venv/bin/python Model2/Experiment4/parameter_study.py \
  --ratios .5 2 8 --families warped_cosine --maximum-power 64 \
  --iterations 48 --optimizer-seeds 2718 3141 --neighbor-counts 3
```

Separate bound controls use `--ratios 2 --maximum-dt 2` or
`--ratios 2 --maximum-dt 4`, with `--maximum-power 64 --iterations 48
--neighbor-counts 3`. These distinguish an unfavorable bound from an
optimizer missing a good region.

Validate the best saved mean candidates and replay all three fixed seeds:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model2_cosine_mpl \
  .venv/bin/python Model2/Experiment4/validate_parameters.py \
  Model2/Experiment4/runs/parameter_study_520a41a9e6c1/study.json \
  Model2/Experiment4/runs/parameter_study_733a92f411cc/study.json \
  Model2/Experiment4/runs/parameter_study_ab1cf3c6666b/study.json
```

The validator saves each replay, supports `--resume`, and independently
reconstructs every preparation/forward node using explicit branch matrices.
It increases R through 16384,32768,65536,131072 only when needed, retaining
all failures. The first source is the initial sweep; a fresh run of the
current policy can have a different configuration hash. Supply the actual
generated paths when reproducing from scratch.

After all runs, `summarize_parameters.py` regenerates the compact summary,
report and standalone PNG/PDF comparison figures from saved studies.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m unittest discover -s tests -p 'test_model2_cosine_search.py' -v
```
