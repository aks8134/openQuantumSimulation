# Analysis of the existing manually scheduled Model3 runs

The saved results support a useful but specific conclusion: reducing the
Trotter step later can substantially improve the late-time state, while
large early steps can leave a very inaccurate transient trajectory. The
small-step schedule gives the strongest trajectory accuracy among the saved
runs. The data do not yet establish that manual adaptation outperforms an
accurate constant-step method at equal cost.

**Scope:** ten existing JSON files in `../results/`, nine at `N=10` and one
at `N=3`. No simulations, new trajectories, channel evolution, stationary
solves, exact-reference calculations, or tests were executed for this
analysis. All numbers and figures below are postprocessing of saved arrays.
The result files and experiment implementation were left unchanged.

Supporting artifacts:

- [Comparison figure](existing_runs_analysis.png)
- [Full numerical summary and input SHA-256 hashes](existing_runs_summary.json)
- [Comparison table as CSV](existing_runs_comparison.csv)

## What is comparable

All ten runs use `J=gamma=1`, `R=64`, seed 29, reservoir populations
`(0.9,0.1)`, exchange prefactor `0.5`, `delta=0`, rate-weighted jump
sampling, Hamiltonian-first order, and even-then-odd Hamiltonian splitting.
The three preparations are all zero, all one, and a left-site excitation.
Every Trotter endpoint was recorded (`record_every=1`). Horizons and chain
lengths vary; a run ending at 20 must not be ranked against a run ending at
60 solely by its final error or total step count.

In the table, `END:DT` has the same meaning as the CLI. Run IDs are the last
part of each saved filename.

| Run ID | N | Final time | Steps | Schedule |
|---|---:|---:|---:|---|
| `754e0cbbdb98` | 10 | 20 | 1180 | `2:0.1 10:0.05 20:0.01` |
| `24233d21d16c` | 10 | 40 | 544 | `2:0.5 10:0.2 20:0.1 40:0.05` |
| `39cec2ef4427` | 10 | 40 | 140 | `20:1 30:0.5 40:0.1` |
| `e5257c026bae` | 10 | 60 | 668 | `2:1 10:0.5 20:0.2 40:0.1 60:0.05` |
| `1f4884fb6f5a` | 10 | 60 | 315 | `30:2 60:0.1` |
| `1b4da2a37311` | 10 | 60 | 150 | `50:1 60:0.1` |
| `7cbec7bdcc09` | 10 | 60 | 117 | `50:3 60:0.1` |
| `56299862d9d8` | 10 | 60 | 45 | `30:2 60:1` |
| `5c1d59d5d02a` | 10 | 60 | 30 | `60:2` |
| `91ce13a92679` | 3 | 20 | 1180 | `2:0.1 10:0.05 20:0.01` |

## Separating discretization bias from trajectory sampling

Let `S(t)` be the saved finite-64-trajectory mean, `M(t)` the saved
deterministic mean channel, and `E(t)` the saved exact Lindblad reference.
Then, component by component,

```text
S - E = (M - E) + (S - M)
          bias     sampling discrepancy
```

`M-E` measures accumulated finite-step error, including both the reservoir
dilation approximation and the even/odd Hamiltonian approximation. The
saved data cannot isolate those two contributions. `S-M` measures the
realized sampling discrepancy for seed 29. Increasing `R` addresses the
latter; it does not remove `M-E`.

Population errors below are absolute probability differences. Current
errors are absolute excitation-current differences divided by `J`.
All maxima include every saved preparation and site/bond. "Peak" means
the maximum over that run's saved time nodes, not a certified maximum
between them. "Final" uses its listed final time.

| N=10 run | Peak population bias | Peak current bias / J | Final population bias | Final current bias / J |
|---|---:|---:|---:|---:|
| `754e0cbbdb98` | 0.03541 | 0.03744 | 0.002621 | 0.002387 |
| `24233d21d16c` | 0.19096 | 0.19348 | 0.009644 | 0.007373 |
| `39cec2ef4427` | 0.37060 | 0.42216 | 0.051281 | 0.022098 |
| `e5257c026bae` | 0.37060 | 0.42216 | 0.012208 | 0.007631 |
| `1f4884fb6f5a` | 0.36784 | 1.16685 | 0.028613 | 0.019418 |
| `1b4da2a37311` | 0.37060 | 0.42216 | 0.058803 | 0.026410 |
| `7cbec7bdcc09` | 0.49966 | 0.98483 | 0.139781 | 0.169744 |
| `56299862d9d8` | 0.36784 | 1.16685 | 0.147608 | 0.214004 |
| `5c1d59d5d02a` | 0.36784 | 1.16685 | 0.119655 | 0.461831 |

The strongest saved N=10 trajectory is `754e0cbbdb98`. Its actual
finite-trajectory peak errors are 0.03983 in population and 0.04663 in
current/J. At time 20 those drop to 0.009319 and 0.005620. At the same
time 20, the mean-channel population/current biases are:

| Schedule up to time 20 | Population bias at 20 | Current bias / J at 20 |
|---|---:|---:|
| `0.1 -> 0.05 -> 0.01` (`754e0cbbdb98`) | 0.002621 | 0.002387 |
| `0.5 -> 0.2 -> 0.1` (`24233d21d16c`) | 0.018187 | 0.016564 |
| `1 -> 0.5 -> 0.2` (`e5257c026bae`) | 0.043020 | 0.033699 |
| constant 1 through 20 (`1b4da2a37311`, `39cec2ef4427`) | 0.169652 | 0.243014 |
| constant 2 through 20 (three runs sharing that prefix) | 0.260316 | 0.537867 |

This comparison uses directly saved values at the same physical time;
no interpolation or new reference evaluation is involved.

For the fine schedule, the time-weighted population RMS bias is 0.006680
and RMS sampling discrepancy is 0.004356; current/J values are 0.005076
and 0.005649. Thus both contributions matter for that run. In contrast,
the `3 -> 0.1` schedule has population RMS bias 0.34793 and RMS sampling
discrepancy 0.01214: its dominant problem is discretization.
RMS uses trapezoidal quadrature of saved squared errors on native time
grids, averaged over preparations/sites or bonds. These are approximate
integral summaries; coarse grids can miss intervening peaks. RMS values
from different horizons are not equal-horizon comparisons.

The numerical summary also saves the exact decomposition of this
quadrature-based mean squared total error into bias squared, sampling
squared, and their cross term. Bias and sampling can partially cancel;
their maxima should not be added as if they occurred at the same point.

## A controlled comparison: refinement after time 30

Three runs have exactly the same schedule, jump choices, and saved
observables through time 30: constant step 2, step 2 followed by step 1,
and step 2 followed by step 0.1. Their initial coarse history is identical,
so their saved mean-channel differences after 30 directly demonstrate the
effect of changing the later steps.

| Subsequent step after time 30 | Total steps to 60 | Final population bias | Final current bias / J |
|---|---:|---:|---:|
| 2 | 30 | 0.119655 | 0.461831 |
| 1 | 45 | 0.147608 | 0.214004 |
| 0.1 | 315 | 0.028613 | 0.019418 |

Changing the later step from 2 to 0.1 reduces final population bias by
4.18 times and final current bias by 23.8 times, at 10.5 times the total
step count. Changing it only to 1 improves the current but worsens the
final population error. Error is not guaranteed to improve monotonically
for every observable under these large-step changes.

All three still have a peak current bias of 1.16685, already attained at
time 2 on bond 1 for the left-excitation preparation. Later refinement
cannot correct the already recorded trajectory. It can relax the current
state toward a more accurate trajectory going forward.

Among the six N=10 runs ending at 60, gradual refinement
`e5257c026bae` gives the smallest final mean-channel errors: 0.012208
in population and 0.007631 in current/J. Its actual sampled final errors
are 0.016948 and 0.018902. Its large initial step nevertheless produces
peak biases of 0.37060 and 0.42216. It is a stronger late-time
approximation than a full-trajectory approximation.

The `3 -> 0.1` run also shows that refinement need not immediately smooth
the error: its maximum mean-channel current error is 0.98483 at time
50.2, immediately after the change at 50. Its sampled current error there
is 1.01219. The state accumulated during coarse evolution matters after
the timestep changes.

For rate-weighted sampling here, each active jump dilation uses
`theta=sqrt(2*gamma*dt)`. Steps 1, 2, and 3 therefore have angles about
1.414, 2, and 2.449, rather than small angles. Their signed cosines are
about 0.156, -0.416, and -0.770. These are valid finite-step channels, but
they are far from the small-step expansion underlying the Lindblad
approximation. The observed large bias is consistent with that fact;
the saved data do not identify a unique mechanism for each error peak.

## Stationary convergence is a separate question

The saved stationary metric is

```text
max(max site population deviation, max bond current deviation / J)
```

against the physical stationary observables. It is not full-state trace
distance. The stationary current is 0.32; endpoint populations are 0.58
and 0.42, and interior populations are 0.5.

**None of the N=10 runs has even a sampled crossing of the saved 0.01
stationary tolerance, for any of the three preparations.** The same is
true of their exact references. The worst exact stationary distances at
their final times are approximately:

| Final time | Worst exact local stationary distance |
|---:|---:|
| 20 | 0.12123 |
| 40 | 0.06686 |
| 60 | 0.03956 |

Flat-looking curves are insufficient evidence of settling to 0.01.
Even a timestep method that matched the exact dynamics perfectly would
retain the physical residual distance at these saved horizons. Conversely,
a small stationary distance can result from a cancellation between
physical transient displacement and algorithmic bias, so it cannot
replace a comparison against the exact trajectory.

For N=3, the fine schedule has a different limitation. At time 20:

| Component | Worst local stationary distance over preparations |
|---|---:|
| Exact Lindblad | 0.00001984 |
| Deterministic mean channel | 0.001643 |
| Sampled R=64 mean | 0.012793 |

The mean channel stays below 0.01 at all recorded points after times
10.34, 9.2, and 9.7 for `000`, `111`, and `100`, respectively. The saved
exact counterparts cross at 7.95, 7.95, and 7.6 and remain below through
20. These are finite saved-window statements, not infinite-time proofs.

The sampled curves cross 0.01 at 10.98, 7.7, and 7.9, then leave it again.
All three have their last recorded exit at 19.57 and finish above the
tolerance. At the final point, maximum population bias is only 0.001623,
whereas maximum sampling discrepancy is 0.011217. This identifies a
sampling limitation at R=64 rather than a large remaining step bias.
The three preparations use the same jump histories, so their similar
late fluctuations are correlated, not three independent confirmations.

## Uncertainty, integrity checks, and limits of the evidence

The stored standard errors describe sampling about the mean channel,
not uncertainty in the timestep bias. Descriptive pointwise checks show
that about 97.8% of fine N=10 population samples lie within two standard
errors of the mean channel, while only 85.7% lie within two standard
errors of the exact reference. Large bias cannot be excused by the plotted
sampling bands. These percentages exclude numerically zero-SE entries
and are not simultaneous confidence levels: times, sites, and
preparations are correlated. The summary separately counts negligible-SE
entries with substantive bias.

Checks on the saved files found:

- Saved maximum-error summaries agree with maxima recomputed from their
  arrays.
- Time nodes, step sizes, recorded counts, observable dimensions, and
  jump-choice dimensions are consistent.
- Final covariances are symmetric and have eigenvalues within `[0,1]`
  up to numerical precision. This checks the stored covariance sector,
  not reconstruction or validation of the full many-body density matrix.
- Exact reference arrays agree at shared time nodes across equal-model
  runs, with maximum disagreement below `9e-14`. This is a consistency
  check of stored references, not a new independent exact solve.
- The three coarse-prefix comparison runs agree exactly through time 30.
- Input SHA-256 hashes were unchanged after analysis.

Every run uses one seed and R=64. Different schedules consume random
draws at different physical times; matching seeds does not make their
full trajectories physically paired. There are no independent-seed
replicates or R sweeps in these saved runs, and no accurate constant-step
baseline at a matched horizon and error budget. Consequently, these data
support observed error reduction and the bias/sampling diagnosis, but not
a seed-independent confidence guarantee, optimal schedule, or adaptive
speedup over an equally accurate fixed-step method. Saved wall times
include the classical reference and mean-channel work; step counts are
the cleaner workload comparison here and are not hardware gate counts.

The practical interpretation of the existing results is therefore:
`0.1 -> 0.05 -> 0.01` is the strongest saved choice for accurate dynamics;
gradual refinement to 0.05 is the strongest saved N=10 endpoint choice at
time 60; and refinement after a coarse prefix is useful for recovery but
does not validate that prefix. These conclusions describe the tested
schedules, without claiming a global optimum or launching further runs.
