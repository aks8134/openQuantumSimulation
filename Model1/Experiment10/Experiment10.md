# Experiment 10: exact high-N, long-time boundary-loss dynamics

[`exact.py`](exact.py) solves the same Model 1 XY chain as Experiment 1:

\[
H=\frac J2\sum_{i=0}^{N-2}(X_iX_{i+1}+Y_iY_{i+1})
 +\frac h2\sum_{i=0}^{N-1}Z_i,
\quad V_L=\sqrt\gamma\,\sigma^-_0,
\quad V_R=\sqrt\gamma\,\sigma^-_{N-1}.
\]

There are exactly two boundary loss operators and no interior jumps. The
initial state is one localized excitation at site `N//2`, configurable with
`--initial-site`. Site 0 uses the same tensor-order convention as Experiment 1.

## Why the reduction is exact

The Hamiltonian conserves excitation number and the jumps only lower it. Thus
the initial state stays in the invariant space
\(\{|\mathrm{vac}\rangle,|0\rangle,\ldots,|N-1\rangle\}\), of dimension
\(N+1\). Here \(|i\rangle\) means one excitation at site i, not the vacuum.
This is the same invariant-sector idea used by Experiment 7's optimized
classical reference. Experiment 10 additionally exploits the chosen pure
initial state; it does not import circuit experiments or change their solvers.

Every boundary jump sends the excitation to the same absorbing vacuum. The
entire unconditional Lindblad solution can therefore be reconstructed from
an **unnormalized** no-jump amplitude vector \(a(t)\in\mathbb C^N\):

\[
\dot a=G a,\qquad
G=-iH_1-\frac\gamma2
 (|0\rangle\!\langle0|+|N-1\rangle\!\langle N-1|),
\quad a_i(0)=\delta_{i,\lfloor N/2\rfloor},
\]

where \(H_1\) has nearest-neighbor hopping J and diagonal \(-h\), relative
to the vacuum energy. The exact density matrix in the invariant space is

\[
\rho(t)=|a(t)\rangle\!\langle a(t)|
 +[1-\|a(t)\|^2]|\mathrm{vac}\rangle\!\langle\mathrm{vac}|.
\]

The vector is **not normalized after propagation**: its norm loss is precisely
the probability transferred to vacuum. Sparse `scipy.sparse.linalg.expm_multiply`
evaluates the exponential action between observation times. There is no
Trotter approximation, trajectory sampling, circuit, hardware backend, or
full/reduced density-vector Liouvillian. Exact here means unsplit Lindblad
evolution, subject to ordinary floating-point numerical accuracy.

Generator storage and the evolving state are O(N); saved observables are
O(N × number of saved times). No trajectory of density matrices is stored.
Large N, very long evolution, or many saved samples can still increase runtime
and output memory, but not exponentially with N as in the full solver.

This implementation is **not valid** for multiple initial excitations, gain
operators, excitation-creating Hamiltonians, or arbitrary mixed/coherent
vacuum-plus-excitation initial states. Those need a suitably enlarged or more
general solver; they must not be silently substituted into this one.

## Observables and diagnostics

For \(c_i=a_i a_{i+1}^*\), the script saves

\[
n_i=|a_i|^2,\qquad C_i^{XY}=4\operatorname{Re}c_i,
\qquad I_{i\to i+1}=2J\operatorname{Im}c_i.
\]

The write-up's current is also saved as \(j_{i,i+1}=-2I_{i\to i+1}\).
Analytic instantaneous derivatives \(2\operatorname{Re}(a_i^*(Ga)_i)\)
check the local continuity equation and
\(\dot P=-\gamma(n_0+n_{N-1})\), where \(P=\sum_i n_i\).
These checks do not use inaccurate finite differences on a coarse time grid.

For J ≠ 0 and γ > 0 on this connected finite chain, the vacuum is the unique
steady state. In this solution the trace distance is exactly

\[
D(\rho(t),\rho_{\mathrm{vac}})=\tfrac12\|\rho(t)-\rho_{\mathrm{vac}}\|_1=P(t).
\]

The final P is compared to `--steady-tolerance` (default 1e-8). A large final
time is **not** automatically declared converged. Convergence can be slow for
large N and strong boundary damping. γ=0 or J=0 are allowed special cases, but
the unique-vacuum assertion then does not generally apply.

## Commands

The requested N=10, γ=5, t=1000 calculation:

```bash
uv run python Model1/Experiment10/exact.py \
  --n-qubits 10 \
  --gamma 5 \
  --t-final 1000 \
  --time-points 1001
```

Change `--n-qubits` for larger chains. Optional `--until-steady` stops at the
first saved time meeting `--steady-tolerance`; `--t-final` is still a maximum
horizon, not a guarantee of convergence. There is no adaptive extension to
infinite time. `--times 0 0.1 1 10 100 1000` supplies an arbitrary strictly
increasing list instead of a uniform grid. A single value is supported and
does not automatically insert t=0. `--times` cannot be combined with
`--t-final` or `--time-points`.

Other options: `--J` (alias `--j`), `--h`, `--initial-site`,
`--output-directory`, `--no-progress`, and `--no-plots`. Observation spacing
affects the resolution of saved/visualized dynamics, not a Trotter error.
For example, 51 points over t=1000 miss early oscillations even though each
sample is evaluated by the exact propagator.

## Outputs

Results go to `results/` and a four-panel figure to `figures/` under Experiment
10 (or the configured output directory). The figure uses labelled time-series
line plots for every site population and every bond correlation/flow, and a
logarithmic excitation-decay line plot with the vacuum tolerance. A single
saved time is shown with markers. Zero log-plot values alone are floored
to the smallest positive float for display; stored data is not clipped.

Configuration-hashed filenames include N, γ and the initial site; changing J,
h, times, tolerance or stopping mode also changes the hash. An identical run
replaces its own outputs. Different configurations do not overwrite each
other, and no Experiment 1/7 results are modified. JSON is saved atomically
**before** plotting, with a figure-status marker so data remains recoverable
if rendering fails. The final amplitude, observables, convergence status,
continuity diagnostics, configuration and evolution wall time are retained.

## Verified baseline

The N=10, γ=5, J=1, h=0 run above completed and saved
[its result JSON](results/exact_N10_gamma5p0_site5_f629f0b49ff7.json) and
[observable figure](figures/exact_N10_gamma5p0_site5_f629f0b49ff7_observables.png).
At t=1000 the remaining excitation was 1.4701822e-15. The first saved sample
below 1e-8 was t=528 on the unit-spaced observation grid; this is not an exact
continuous-time crossing estimate. Evolution took about 2 seconds on the test
machine, with about 168 MiB peak process memory including plotting. These are
measured timings, not guarantees for other configurations or machines.

For this localized initial excitation, real hopping, and uniform field, the
adjacent amplitudes have alternating real/imaginary phases after removing
the common field phase. Consequently nearest-neighbor XY correlations are
zero up to numerical roundoff; their line plots overlap at zero and are
labelled accordingly, while imaginary coherences produce nonzero excitation
flows. For a clearer view of the early dynamics, use a shorter horizon such
as `--t-final 100 --time-points 1001`.

The test suite compares all observable families against the full many-qubit
Lindblad solver for N=2,3,4 at γ=5 with nonzero h, checks N=128 sparse scaling,
loss-free and disconnected limits, arbitrary/single times, stopping behavior,
and preservation of JSON if plotting fails:

```bash
python -m unittest tests.test_experiment10_exact -v
```
