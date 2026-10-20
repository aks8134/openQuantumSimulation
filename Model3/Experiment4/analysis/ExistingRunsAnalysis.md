# Analysis of the saved Experiment4 runs

Only existing `study.json`, `search.json`, and `replay_R64.json` files were
read. No search, trajectories, time evolution, or stationary-state calculations
were run. The companion [comparison.json](comparison.json) records the numerical
metrics and SHA-256 hashes of all 12 source JSON files; source hashes were checked
again after analysis. [comparison.csv](comparison.csv) contains the main table.
The saved-array analysis can be reproduced with `analyze_saved_runs.py` in this
directory. Original results and simulator code are unchanged.

## Main conclusion

Adaptation improves the tested mean accuracy within a 500-step budget, but
**none of these runs achieves the requested steady-state accuracy**. Allowing
steps as large as 3 does not improve the selected schedule. The narrow beam
search also misses useful schedules: reducing the switch spacing from 5 or 10
steps to 2 returns a worse result. These runs do not establish that 500 steps
are insufficient in principle.

All runs use N=10, J=gamma=1, reservoirs (0.9,0.1), exchange prefactor 0.5,
R=64, seed 29, and a 500-step cap. The error is

`d = max(max_site |n - n*|, max_bond |I - I*| / J)`.

It covers all 10 populations and 9 currents. The reported tail error is the
maximum over the final 10 units of physical time, at every saved integration
node. Errors below are absolute values in this metric. The total tolerance is
0.01; the mean allowance is 0.006 and sampling allowance is 0.004.

## Which runs are present

| Run suffix | Largest allowed dt | Switch block | Selected preparation | Steps | Final time | Mean tail error | Sampled tail error |
|---|---:|---:|---|---:|---:|---:|---:|
| `0dd4af45561e` | 1 | 10 | half_mixed | 490 | 43.125 | 0.015683 | 0.038008 |
| `01eecf5ce053` | 1 | 5 | profile_mixed | 380 | 33.4375 | 0.016328 | 0.044127 |
| `67296f68e125` | 3 | 5 | profile_mixed | 380 | 33.4375 | 0.016328 | 0.044127 |
| `2920cf23bb0f` | 2 | 2 | half_mixed | 394 | 31.125 | 0.018265 | 0.042324 |

The default block-10 run is present alongside three of the four suggested
commands. There is no saved max-dt-2, block-5 run in this directory, so that
particular comparison is unavailable.

The block-10 default has the lowest mean tail error among the saved runs. The
380-step schedules use 110 fewer steps for a slightly larger error. None passes
even the full 0.01 tolerance at the deterministic mean level over its tail.

## Selected schedules and depth

Notation below is `number of steps x dt`:

| Run | Selected schedule |
|---|---|
| max dt 1, block 10 | 20 x 0.5; 20 x 0.25; 450 x 0.0625 |
| max dt 1, block 5 | 20 x 0.5; 5 x 0.25; 355 x 0.0625 |
| max dt 3, block 5 | Identical to max dt 1, block 5 |
| max dt 2, block 2 | 6 x 1; 2 x 0.5; 386 x 0.0625 |

The block-5 runs have identical preparations, time nodes, mean observables, and
sampled observables. They are a repeated outcome of the same chosen schedule
and seed, not independent confirmation. No selected winner uses dt above 1.

The final dt=0.0625 stage consumes 92%-98% of the selected steps. The last
10-time-unit validation window alone spans 160 such steps. Coarser burn-in
therefore addresses a relatively small part of the remaining depth cost.

These are counts of channel steps, not measured compiled circuit depths. Warm
preparations have one modeled parallel local preparation layer, and compiled
depth is absent from the saved data. Comparing 380-490 steps directly with the
previous 3530-step Experiment3 run would also mix different achieved accuracy,
preparation choices, and stopping criteria.

## Adaptation helps relative to the saved constant baselines

The best constant candidate across all tested preparations is dt=0.0625,
profile_mixed, 500 steps, with mean tail error 0.0204892. Relative to this:

| Adaptive candidate | Reduction in mean tail error | Fewer steps |
|---|---:|---:|
| block 10, max dt 1 | 23.46% | 10 |
| block 5, max dt 1 or 3 | 20.31% | 120 |
| block 2, max dt 2 | 10.86% | 106 |

These compare deterministic mean results; the constant candidates were not
replayed at R=64, so sampled superiority over constant steps cannot be inferred.

Best covered-window constant errors across preparations are approximately
0.20978 at dt=1, 0.09162 at 0.5, 0.03972 at 0.25, 0.02091 at 0.125,
0.02049 at 0.0625, and 0.05974 at 0.03125. At dt=0.015625,
500 steps reach only time 7.8125 and cannot cover the required 10-time-unit
window. The poor small-step result reflects insufficient physical evolution
within the budget as well as the window requirement; it is not evidence that
smaller steps intrinsically have larger discretization bias.

## Warm preparation is useful, but the exact population profile adds little

Best tested mean tail errors for each preparation:

| Preparation | max 1 / block 10 | max 1 / block 5 | max 3 / block 5 | max 2 / block 2 |
|---|---:|---:|---:|---:|
| vacuum | 0.026673 | 0.033645 | 0.039559 | 0.043056 |
| filled | 0.026673 | 0.033645 | 0.039559 | 0.043056 |
| left_excitation | 0.027372 | 0.029950 | 0.038517 | 0.032576 |
| neel | 0.024142 | 0.027487 | 0.024266 | 0.026074 |
| half_mixed | 0.015683 | 0.016987 | 0.016987 | 0.018265 |
| profile_mixed | 0.016135 | 0.016328 | 0.016328 | 0.018340 |

Starting near half occupation is more effective in these tests than starting
empty or full. The simple half_mixed state achieves the best result overall.
In the block-5 runs, reference-assisted profile_mixed improves mean error over
half_mixed by only about 3.9%. All these preparations initially have zero
current/coherence; profile_mixed does not supply the correlated steady state.
Preparation-bit fluctuations and measurement-shot noise are analytically
averaged and are not included in the saved trajectory standard errors.

## Both deterministic error and sampling remain significant

| Configuration | Exact-to-steady tail error | Mean-to-exact tail error | Sampled-to-mean tail distance | Maximum 2 SE |
|---|---:|---:|---:|---:|
| max 1, block 10 | 0.005673 | 0.014099 | 0.027903 | 0.026750 |
| max 1 or 3, block 5 | 0.008284 | 0.014786 | 0.034075 | 0.026653 |
| max 2, block 2 | 0.010903 | 0.016421 | 0.032734 | 0.026724 |

The saved exact reference evolves continuously from the same initial product
mixture. Its tail error is already below 0.01 in the block-5 and block-10 runs,
while the Trotter mean remains above 0.01. Discretization of the dynamics is
therefore materially affecting the achieved accuracy. Mean-to-exact distance
includes accumulated transient discretization error; it is not a separate
calculation of the discrete channel's stationary bias. Maxima in this table
can occur at different times and observables, so they cannot be added as an
exact error decomposition.

The two-SE values exceed the 0.004 sampling allowance by about 6.7 times.
Increasing R can reduce this sampling component without increasing the step
count in one circuit. It cannot correct the deterministic mean error.
Rough inverse-square-root scaling would require roughly 2800-2900 trajectories
to reduce these pointwise two-SE magnitudes to 0.004 at the same variance.
This is an extrapolation, not a confidence guarantee, and does not account for
the maximum over correlated times and observables. Only one seed is available;
no across-seed reproducibility conclusion is possible.

## Low endpoint error is not sustained convergence

The smallest saved mean errors are 0.008258 (block 10), 0.009176 (block 5), and
0.009462 (block 2). All are transient crossings below 0.01. Their respective
final mean errors are 0.014063, 0.012670, and 0.014930. None of the sampled
curves crosses below 0.01 at any recorded node.

For the 380-step schedule, mean tail population error is 0.016328 and current
error/J is 0.015728. The worst population error occurs at site 7, outside the
three-site subset displayed in the usual observable plot. Across the tail,
the largest mean observable peak-to-peak variation is 0.018853. The final
population error is 0.012670 while the final current error/J is 0.009448:
looking only at the final current could suggest success while populations
still fail.

See [steady_state_comparison.png](steady_state_comparison.png) for all-local
error curves and the assessed windows. The plotted sampled-to-mean distance
is a realized discrepancy, not a confidence interval.

## The search needs attention before a physical limit is claimed

No saved run hits its search-node cap. Increasing that cap alone would not
change these completed searches. Nevertheless, beam width remains only 2,
and candidate retention uses endpoint error rather than the final-window
objective. The search can discard temporarily less accurate burn-in paths
that would become useful after refinement.

The block-2, max-dt-2 schedule family includes the block-10 schedules (all their
switch counts are multiples of 2), yet it returns a worse result. Adding grid
values also worsens some individual preparations. This is direct evidence of
heuristic pruning affecting the result, not evidence that finer switching or
coarser allowed steps are physically harmful.

The next search improvement should retain the best schedules already found
across configurations, widen or diversify retained paths, and align retention
more closely with the sustained-window objective. Half_mixed is a sensible
preparation to prioritize. Improving the deterministic schedule should precede
a large R sweep. If the mean still cannot meet the target within the desired
depth after stronger search, then investigate a lower-bias channel construction
or explicitly choose a looser accuracy target. No such changes or new runs
were made for this analysis.
