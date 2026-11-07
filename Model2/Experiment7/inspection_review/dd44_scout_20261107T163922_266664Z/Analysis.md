# Model2 Experiment7: Phoenix schedule inspection

The saved schedules support a selective DD experiment. Across the four Z-basis
trajectories, 67.8% of the summed system-qubit evolution window has no drive
pulse. The end sites have the most idle exposure, about 75%. A first XX policy
targeting quiet windows longer than 0.8 microseconds offers a useful tradeoff:
it retains 84.4% of the observed idle duration with 40.4% fewer candidate pairs
than the default 0.16-microsecond cutoff. These are gap-based estimates, not
compiled DD counts or measured error reduction.

This review reads the supplied archive only. No circuit was built, transpiled,
simulated or submitted, and no IBM account or job was queried. Source results
and the ZIP were not modified. The derived CSVs beside this report contain the
per-circuit, per-site and threshold summaries.

The archive is `Model2/Experiment7/hardware_runs/dd44_scout_20261107T163922_266664Z.zip`,
SHA256 `e55871fb1d568df8deba6cfe5a47a065aff3fae7b7f0b53ecc921460862d73c8`.
Job `dbnl8soou4as73b4p090` is a completely collected **DD-off cloud dry-run**.
The immutable plan hash, all four supplied source/specification/calibration
file hashes, result hash, job identity and all 12 exported timing texts match.
All 15 PUBs have their requested shots, totaling 6,528. QPY files were excluded
from the transfer, so their contents/hashes were not independently checked.

The frozen experiment uses N=10, J=1, Gamma/J=2, equal reservoir occupation .25,
half-mixed preparation, 44 warped-cosine steps, trajectory seed 29, transpiler
seed 11 and optimization level 3. The archived SDK versions are Qiskit 2.6.2
and Runtime 0.51.0. The selected path, ordered bath--system--bath, is:

`[38, 48, 47, 57, 56, 66, 76, 86, 96, 97, 98, 108]`.

Each server trace contains 880 distinct CZ operations and 44 reset groups.
Each reset is represented by four consecutive drive pieces of 32, 20, 32 and
16 cycles; their combined 100 cycles agree with the saved 400 ns reset
duration. The two end baths receive all of these resets. The traces have only
the main branch; this particular compilation does not report broadcast/receive
instructions. Baseline X instructions are already present from native circuit
compilation; their presence does not mean DD was enabled. The manifest has DD
and twirling disabled, a single baseline arm, and no DD policy.

| Trajectory | Compiler circuit duration, Z (us) | System idle fraction during evolution | Longest phase-separated system gap (us) |
| --- | ---: | ---: | ---: |
| 0 | 76.056 | 68.02% | 2.748 |
| 1 | 82.052 | 69.60% | 3.732 |
| 2 | 76.444 | 68.23% | 2.708 |
| 3 | 70.176 | 64.92% | 3.260 |

Circuit duration is the compiler's `circuit_duration` metadata multiplied by
the saved dt=4 ns. It includes initialization and readout; it is not billing
time. Idle fraction uses each system qubit's post-preparation through
pre-final-measurement window, rather than the complete circuit duration. Idle
time summed across qubits is qubit-time, not elapsed wall-clock time.

The analysis excludes initialization and the initial mixture measurement. For
each qubit, its conservative post-preparation boundary is the initial
measurement start plus the saved target duration of 485 cycles (1.94 us). The
window ends when that qubit's final measurement starts. Positive-duration
drive instructions are merged into occupied intervals, and each remaining gap
is split at every interior zero-duration phase instruction. No sequence is
proposed across those phase changes. XY/YX windows also include their terminal
readout rotations; the Z windows supply the cleanest evolution-only comparison.

This distinction matters: the supplied raw gap CSVs include 0.74 us per system
qubit after the last drive piece of the preparation measurement, while
acquisition/guard time remains. That adds 7.4 qubit-us per circuit to the naive
gap sum. Those intervals were removed here. A drive-channel gap alone is not
proof of a legal client-side DD delay.

The following values average the four Z trajectories. The longest gap is the
maximum over those trajectories, not an average.

| Site | Physical qubit | Mean idle time (us) | Mean idle fraction | Longest gap (us) | Mean candidate gaps >0.8 us |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 48 | 47.932 | 74.51% | 3.732 | 21.00 |
| 2 | 47 | 41.768 | 64.81% | 3.436 | 21.50 |
| 3 | 57 | 42.908 | 66.60% | 3.260 | 22.75 |
| 4 | 56 | 41.874 | 66.47% | 2.924 | 27.50 |
| 5 | 66 | 41.514 | 65.89% | 3.692 | 25.50 |
| 6 | 76 | 41.874 | 66.60% | 2.704 | 23.50 |
| 7 | 86 | 41.514 | 66.02% | 2.664 | 23.00 |
| 8 | 96 | 41.638 | 66.46% | 3.260 | 22.50 |
| 9 | 97 | 40.208 | 64.17% | 2.924 | 21.50 |
| 10 | 98 | 46.988 | 75.04% | 3.180 | 20.75 |

All sites have meaningful idle exposure, so this inspection does not justify
restricting DD to the end sites alone. The saved system T2 values span
128.7--238.4 us. This supports investigating idle protection, but timing and T2
do not establish the noise spectrum or predict a DD benefit. Saved X-gate
error estimates span 0.000115--0.000298; extra pulses also have a cost.

The saved X duration is 40 ns, so an XX pair occupies 80 ns. The current
`min_idle_ratio=2` rule asks for a gap strictly longer than 160 ns. A ratio of
10 instead asks for more than 800 ns. The following comparison uses all 12
evolution traces, with one hypothetical XX pair per qualifying phase-separated
gap. Coverage means fraction of observed idle duration contained in those
gaps, not protected coherence or error reduction.

| Gap must exceed (us) | Minimum idle/XX ratio | Mean candidate pairs per circuit | Mean candidate added X pulses | Idle duration retained |
| --- | ---: | ---: | ---: | ---: |
| 0.16 | 2 | 384.75 | 769.5 | 98.52% |
| 0.40 | 5 | 283.50 | 567.0 | 92.23% |
| 0.80 | 10 | 229.50 | 459.0 | 84.39% |
| 1.00 | 12.5 | 203.00 | 406.0 | 79.01% |
| 2.00 | 25 | 39.75 | 79.5 | 21.55% |

I recommend testing one symmetric XX cycle on all ten system sites, with
`min_idle_ratio=10`, spacings `[0.25, 0.5, 0.25]`, no multiple cycles, and no
bath DD. Keep initialization, reservoir resets and final readout intact.
Actual eligible windows, exact X counts and controller timing still require
the saved native QPY on the user's computer, the paired compilation checks,
and the paired IBM timing result. The current archive contains no DD arm;
its empty `server_comparison.json` is expected. This recommendation has not
been installed as a policy or run.

The four-path scout is useful for scheduling inspection, not stationary-state
accuracy certification. Its saved noiseless finite-path target error is
0.1186 and its own classical tolerance checks fail, as expected for a tiny
subset. This does not revise the separate 16,384-path validation. A later
hardware pilot should compare to the ideal ensemble of its exact saved paths
and use paired population and current measurements.

The returned counts and controls are randomized dry-run outputs, so no physical
populations, currents, SPAM quality or mitigation improvement were inferred.
`usage()` reports 3 seconds for the real-hardware equivalent of this baseline
workload; dry-run usage is an uncharged estimate, even though the metrics key
is named `qpu_charge_time_seconds`. The IBM job timestamps span 7.36 seconds.
The archived timing export timestamps span 11:45:40--12:01:22 in the producing
computer's local time, consistent with substantial local rendering work after
the job was already DONE.

IBM documents randomized dry-run data and uncharged usage estimates in
[Dry-run jobs](https://quantum.cloud.ibm.com/docs/en/guides/dry-run-jobs).
Its dynamic DD pass documents the idle/sequence ratio and pulse-error tradeoff
in [PadDynamicalDecoupling](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/transpiler-passes-scheduling-pad-dynamical-decoupling).
