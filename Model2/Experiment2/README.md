# Experiment 2: randomized single-jump dilation

This tests the qDRIFT-type Lindbladian sampling in
[Chen, Li, Lu and Ying, arXiv:2407.06594](https://arxiv.org/html/2407.06594v5)
on the same equal-reservoir Model 2 benchmark. It keeps the Ding–Li–Lin
ancilla dilation and replaces four jump interactions per step with one
independently sampled, probability-scaled jump. It uses no exact-reservoir
exponential. “Experiment 2” here names the numerical-method comparison;
the physical reservoirs remain equal.

The Hamiltonian convention remains
`H = (J/4) sum(XX + YY + Delta ZZ)` by default. Accordingly the excitation
current is `I = (J/4)(YX - XY)`. The fast observable runner requires
`Delta=0`; it computes populations and excitation currents, not a full
density matrix. Both three- and seven-spin results use `J=10`, `Gamma=20`,
`p=0.25`, and `T=12`.

Run the complete comparison, including 64 independent random sequences:

```bash
MPLCONFIGDIR=/tmp/model2_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python Model2/Experiment2/qdrift.py --n-qubits 7
```

Use `--n-qubits 3` for the three-spin comparison, `--realizations 0` to skip
sequence sampling, or `--dt ...` to choose the sweep points. `T/dt` must be
an integer. Observation times always lie on actual step boundaries.

Saved reports:

- [N=7 comparison and sequence fluctuations](results/N7_J10_G20/Results.md)
- [N=3 comparison and sequence fluctuations](results/N3_J10_G20/Results.md)
- [Certified population/current bounds and step counts](analysis/Bound.md)
- [Derivation of the bound](analysis/Proof.md)
- [Population tolerance 0.1 to 0.003 versus certified steps](analysis/tolerance_sweep/Results.md)
- [N=10 simulation, certificates, and N=3/7/10 comparison](analysis/N10_tolerance_sweep/Results.md)

Reproduce the certified tolerance sweep for both sizes:

```bash
MPLCONFIGDIR=/tmp/model2_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python Model2/Experiment2/analysis/tolerance_sweep.py
```

The sweep checks every algorithm step using validated one-step blocks.
Each selected count has a passing upper bound and its preceding count has
a violating lower witness. It locates adjacent thresholds without proving
global optimality over all smaller counts. Saved CSV and JSON include the
time step and certificate bounds.

At the relaxed population tolerance 0.003, a computer-assisted certificate
covers every step through T=12 for both N=3 and N=7: 5,441 steps suffice,
and 5,500 gives more margin. The previously selected 6,000-step setting is
also certified. The certificate includes arithmetic uncertainty under its
stated IEEE rounding assumptions and applies to the averaged channel.

For N=7, rate-weighted jump-first sampling passes the population margin
at `dt=0.000375`: 32,000 steps and 64,000 logical factors to T=12, compared
with 48,000 steps and 240,000 factors for the recommended old Lie setting.
This is an average-channel result. The 64-sequence Monte Carlo mean still
has maximum population error about 0.0129, so a small fixed seed collection
does not meet the 0.001 target. Sequence resampling and measurement cost
must be considered separately from per-sequence depth.

`randomized.sampled_kraus` constructs the full two-level ancilla unitary
and its system Kraus operators, including for interacting Hamiltonians.
The covariance shortcut is separately tested against that full construction.
It evaluates the average of random circuits exactly to numerical precision;
that average still approximates the continuous Lindblad evolution.

Costs in the reports count logical unitary factors, ancillary qubits, and
reset cycles per sequence. They exclude transpilation and measurement shots.
To measure the average channel on hardware, independently resample the
sequence per shot, or explicitly average across multiple sequences.
