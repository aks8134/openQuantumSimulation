# Experiment 11: independent, configurable randomized boundary-loss simulation

`mitigated_randomized_lie_trotter.py` implements the randomized strategy and
mitigation pipeline without importing any other experiment. Its own modules
contain the model, schedule, circuit synthesis, observable reconstruction,
reference solver, plotting, and checkpoint logic. It uses the shared
`IBMRuntime` and native `qem` libraries, not `mitiq_qem` or Mitiq. The optional
full reference uses the shared `library.classical` model builders.

The physical parameters are explicit immutable model inputs; no imported
experiment globals or monkey-patched experiment functions are used.

## Model and tunable physical parameters

For an open chain of N >= 2 qubits,

\[
H=\frac{J}{2}\sum_{i=0}^{N-2}(X_iX_{i+1}+Y_iY_{i+1})
  +\frac{h}{2}\sum_{i=0}^{N-1}Z_i,
\qquad
L_L=\sqrt{\gamma}\,\sigma^-_0,
\qquad
L_R=\sqrt{\gamma}\,\sigma^-_{N-1},
\]

\[
\dot\rho=-i[H,\rho]+\mathcal D[L_L](\rho)+\mathcal D[L_R](\rho),
\qquad
\mathcal D[L](\rho)=L\rho L^\dagger-\tfrac12\{L^\dagger L,\rho\}.
\]

Here sigma-minus denotes loss, |0><1|, not pumping. Both boundaries have the
same gamma. J and h are finite real numbers; gamma is nonnegative. Setting
gamma=0 removes dissipation mathematically. The script still constructs the
zero-angle jump block and reset, which the native compiler may simplify.

| Option | Default | Meaning |
| --- | --- | --- |
| `--n-qubits` | 4 | System size, excluding the two ancillas |
| `--J` / `--j` | 1.0 | Exchange/hopping strength |
| `--h` | 0.0 | Uniform longitudinal field |
| `--gamma` | 0.2 | Loss rate at each of the two boundaries |
| `--initial-site` | N // 2 | Site containing the initial single excitation |
| `--trotter-delta-t` | 0.1 | Maximum substep duration |
| `--trajectories` | 4 | R sampled boundary-choice trajectories |

The initial state is one localized excitation at `initial-site`. Arbitrary
multi-excitation states, boundary gain, independently tunable boundary rates,
and nonuniform fields are not implemented in this experiment.

The same Model value supplies the circuit angles, current-observable
normalization, exact reference, filenames, and result/checkpoint signatures.
Changing gamma does **not** leave a reference at the old gamma.

## Randomized step and boundary-local ancillas

At each substep of duration delta, choose L or R with probability 1/2. Apply
the full system step, then only the selected boundary dilation, then reset
that boundary's ancilla. The selected jump is strengthened by sqrt(2):

\[
\widetilde L_b=\sqrt2 L_b,
\qquad
\mathcal E_b(\delta)
=\mathcal I+\delta(\mathcal L_H+2\mathcal D[L_b])+O(\delta^2).
\]

Consequently the average of the two maps has the desired first-order
Lindblad generator. A trajectory samples these choices independently across
substeps; R trajectories are averaged. It is not deterministic application
of both jumps in each step and not the three-component strategy.

Divide H into disjoint even bonds, disjoint odd bonds, and the field. With
Qiskit's convention

\[
\operatorname{XXPlusYY}(\theta,0)
=\exp[-i\theta(XX+YY)/4],
\]

the chronological substep is

```
even bonds: XXPlusYY(2 J delta)
    → odd bonds: XXPlusYY(2 J delta)
    → each system qubit: RZ(h delta), if h != 0
    → selected boundary exchange: XXPlusYY(2 sqrt(2 gamma delta))
    → reset selected boundary ancilla
```

The field commutes with the number-conserving hopping. Within each bond
parity layer all exchanges commute; the even and odd layers need not commute.
There is one even/odd split per substep, not an additional nested timestep loop.

Tracing/resetting the initially zero boundary ancilla gives amplitude damping
with loss probability sin²(sqrt(2 gamma delta)); its first-order expansion is
2 gamma delta. This finite-step map is not the exact exponential damping
probability 1-exp(-2 gamma delta). Larger gamma therefore requires smaller
delta for the same accuracy. The script warns when 2 gamma delta >= 0.1;
this warning is not a rigorous error threshold or guarantee.

The logical register contains N+2 qubits, ordered

```
a_R, s_(N-1), ..., s_1, s_0, a_L
```

Site i is logical qubit N-i; the two local ancillas are logical qubits 0 and
N+1. Only the selected ancilla participates in a given jump. It is reset
before its next use. Routing on a real device can still introduce SWAPs;
boundary-local logical organization is an opportunity for placement, not a
guarantee of optimal hardware routing. The implementation has reset, but no
mid-circuit measurement/feed-forward branch.

### Compact diagrams, explicit compilation inputs

Circuit diagrams retain the XX+YY blocks. Just before compilation, every
exchange is explicitly synthesized into the same RX/RZ/RZZ sequence:

```
RZ ⊗ RZ → RX ⊗ RX → RZ ⊗ RZ → RZZ
        → RX ⊗ RX → RZZ → RZ ⊗ RZ → RX ⊗ RX
```

Each block has six RZ, six RX, and two RZZ operations: 14 operations, depth
8, and depth 5 after excluding virtual RZ layers. The sequence equals the
exchange up to a global minus sign; these gates are not controlled, so that
phase does not affect the simulated channel. RZ remains in the circuit.
Native transpilation may subsequently merge gates or change basis.

## Observation times, circuit counts and shots

`--times` accepts a single time or a strictly increasing nonnegative list.
Only the requested times generate circuits. A circuit at time t includes all
substeps from zero to t; there are no separate sampled circuits at intermediate
substeps unless explicitly requested as observation times.

Each observation interval is split into substeps no larger than the maximum
delta. For example, `--times 0.7 --trotter-delta-t 0.2` uses durations
0.2, 0.2, 0.2, 0.1 in the same circuit. `--times 0.2` does not add a t=0
baseline. Without `--times`, the defaults are `--t-final 2 --time-points 11`.

If there are T requested times, R trajectories, S ZNE scales and F folding
repetitions, the submitted measurement-circuit count is

\[
C=3TRSF.
\]

`--shots` is the total budget **per time, basis, and scale**, divided equally
among R F realizations. It must be divisible by R F. With R=4, F=1 and
8192 shots, each submitted circuit gets 2048 shots. No-ZNE requires F=1 and
uses only scale 1, so the count is 3 T R.

`--batch-size` limits submitted variants, including scales/repetitions. All
S F variants of one base circuit remain together; the size must be at least
S F. The defaults are 300 variants on local sampling backends and 5 on real
IBM hardware. No automatic bisection, resubmission or adaptive retry occurs.
Progress bars count preparation; provider execution reports elapsed time and
completed batches, not an invented within-batch completion percentage.

The seeded boundary choices match the established R-by-substep random draw.
Changing the requested grid/horizon can change both step boundaries and seeded
trajectory prefixes, particularly after trajectory row zero. Every saved
record includes its actual durations and boundary choices; appended data do
not assume unchanged random prefixes.

## Three measurement bases and observables

The bases are Z, all-X, and alternating X/Y (X on even sites, Y on odd sites).
Their reconstructed observables are

\[
p_i=\tfrac12(1-\langle Z_i\rangle),\qquad
C_i=\langle X_iX_{i+1}+Y_iY_{i+1}\rangle=2\langle X_iX_{i+1}\rangle,
\]

\[
I_i=\frac J2\langle Y_iX_{i+1}-X_iY_{i+1}\rangle
=\begin{cases}
-J\langle X_iY_{i+1}\rangle,&i\text{ even},\\
 J\langle Y_iX_{i+1}\rangle,&i\text{ odd}.
\end{cases}
\]

The three-basis reduction uses the model's U(1) symmetry, including
XX=YY and YX=-XY. It is exact for the ideal single-excitation/vacuum model;
hardware noise may break these identities. Thus hardware outputs retain the
same symmetry-based reconstruction as before, not a general five-basis
measurement of arbitrary noisy states. Agreement with a classical reference
alone does not establish that noise respects this assumption.

## Mitigation pipeline

All four mitigations default to enabled. Disable each independently:

- `--no-zne`
- `--no-pauli-twirling`
- `--no-dynamical-decoupling`
- `--no-readout-mitigation`

The pipeline is explicit synthesis → native transpilation → folding →
Pauli twirling → scheduled DD → sampling → readout parity correction →
scale extrapolation. Mitigation is implemented by `qem`; there is no conversion
to Cirq, Mitiq worker, or post-folding optimizing transpilation.

- ZNE defaults: scales 1, 1.5, 2, linear inference, F=1. Options include
  `--zne-scale-factors`, `--zne-inference linear|polynomial|richardson`,
  `--zne-polynomial-order`, `--fold-repetitions`, and `--seed-folding`.
  Gates are folded at their native physical locations; resets, measurements,
  barriers, delays, and virtual RZ/P/U1/phase operations are excluded.
  Unsupported inverse gates fail rather than silently lowering the circuit.
  When SXdg is unavailable, the exact inverse of SX is represented by SX³.
- Twirling uses rotation-aware Pauli pairs for supported RXX/RYY/RZZ and
  Clifford twirls for CX/CZ. Selected and untwirled gates and seeds are
  recorded. `--seed-twirling` is configurable. The additional pulses can
  increase resource cost; twirling is not guaranteed to improve accuracy.
- DD sequences: `--dd-sequence xx|yy|xy4|xy8` (`xyxy` aliases `xy4`). Optional
  `--dd-spacing-fractions` must match the pulse count plus one and sum to one.
  Hardware/fake targets supply timings and alignment constraints. Plain Aer
  requires an explicit `--dd-durations FILE` or DD disabled. The supplied
  [aer_timings.json](aer_timings.json) is **synthetic scheduling data only**:
  it does not introduce noise or model a real device. Short idle windows may
  receive no DD pulses; inspect the recorded insertion count.
- Readout mitigation on noisy fake/real backends requires
  `--readout-calibration FILE`. The mapping uses actual terminal physical
  measurement locations, including routing, not just initial placement.
  Missing physical coverage or partially measured correlated blocks fail.
  Plain noiseless Aer without a calibration uses a labelled identity model,
  which is not measured calibration and has no corrective effect.

Requested ZNE scales are used in the fit, matching the prior experiment.
Actual physical-operation-count scales are separately recorded because
quantized folding and native inverse synthesis can differ from the requests.
These are operation-count proxies, **not calibrated physical noise factors**.
Large deviations deserve attention before interpreting the extrapolation.

Shot, folding/twirling-realization, trajectory, and total standard errors are
reported. The extrapolated variance uses squared fit weights and diagonal
per-scale variances; shared-trajectory correlations across scales are not
included. With R=1 or F=1, the corresponding ensemble component cannot be
estimated and is reported as zero, not proven absent. Calibration uncertainty
and drift are also excluded. Estimates are not silently clipped to physical
bounds. The raw-unit maximum error across all sites/bonds/families is a
diagnostic, not a dimensionless normalized error tolerance.

### Readout calibration file

Matrices have rows = observed outcome and columns = prepared outcome. Blocks
refer to physical device qubits. Supply calibrated data for the backend and
all qubits that the compiled circuits actually measure, e.g.:

```json
{
  "backend": "ibm_kingston",
  "source": "Measured assignment calibration, timestamp and job IDs",
  "blocks": [
    {"physical_qubits": [12], "assignment": [[0.97, 0.04], [0.03, 0.96]]},
    {"physical_qubits": [13], "assignment": [[0.98, 0.05], [0.02, 0.95]]}
  ]
}
```

The example numbers/qubits are illustrative, **not usable Kingston
calibrations**. Separate blocks assume independent readout across blocks;
joint blocks can represent within-block correlations. This driver does not
automatically submit calibration jobs.

## Exact references

`--optimized-classical-reference` uses the exact vacuum-plus-single-excitation
reduction. Let a_i be unnormalized no-jump amplitudes. Removing an irrelevant
constant energy,

\[
\dot a=Ga,\qquad
G_{ii}=ih-\frac\gamma2(\delta_{i0}+\delta_{i,N-1}),\qquad
G_{i,i+1}=G_{i+1,i}=-iJ.
\]

Sparse exponential actions evolve only N complex amplitudes. The density
operator is |a><a| plus vacuum weight 1-||a||². Therefore

\[
p_i=|a_i|^2,\qquad C_i=4\operatorname{Re}(a_i a^*_{i+1}),
\qquad I_i=2J\operatorname{Im}(a_i a^*_{i+1}).
\]

This is the exact Lindblad reference for this initial state and loss-only
model, not a Trotter/randomized approximation. It does not extend unchanged
to pumping or arbitrary multi-excitation inputs. Uniform h contributes a
common phase in this sector, so these observables are independent of h in the
ideal model, although the circuit still includes its field rotations.

`--classical-reference` explicitly selects the full many-qubit density-vector
solver. It evaluates observables incrementally but still constructs the
exponentially large Liouvillian; N >= 10 can require substantial memory.
It is not silently replaced by the optimized reference. The two flags are
mutually exclusive. Either can be added later while resuming a fully sampled
checkpoint, without resampling. On append/replace runs, the chosen reference
is regenerated at all archived times with the archived physical parameters.

## Persistence, recovery and figures

Each physical/mitigation/sampling configuration has one merged result JSON
and one checkpoint JSON. Filenames include N, J, h, gamma, R, mitigation
flags, and a short hash of the complete configuration (including initial
site, timings/calibrations, shots, and seeds). Different physical models are
not appended into the same result. Existing experiment archives are untouched.

New observation times append; rerunning an existing time replaces its complete
trajectory/basis/scale/fold group. Changing the batch size or plotting choices
does not create a new configuration. Time grids are recorded per invocation
and are deliberately not part of the archive's model signature.

Checkpoints are written atomically before sampling and after every successful
provider batch. They retain raw counts, job IDs, metrics, choices and mitigation
diagnostics. An incomplete checkpoint is not overwritten by a fresh run:
use `--resume` with the same original time grid/configuration, or a different
`--checkpoint-file`. Resume skips the saved ordered prefix. Provider work that
finishes before a batch is checkpointed may still need job-level recovery;
this is batch-level persistence, not a guarantee against all interruptions.

The full circuit workload is never retained across batches. Reconstructed
result JSON is saved **before** exact-reference calculation and figure
generation. A plotting failure therefore does not erase sampled results.

Figures are:

- Line plots for site populations, bond XY correlations, and excitation flow,
  with uncertainty bands and optional exact-reference dashed lines; a fourth
  panel shows error/uncertainty diagnostics.
- Operation-count and depth bars comparing the explicit source, native base,
  and mitigated submitted circuits, averaged by scale over archived times.
  Resets and measurements count as operations; scheduled delays also count.
- Final-time observable-versus-scale plots when ZNE is enabled.
- Separate compact left/right single-step circuit diagrams.
- A final-time Z-basis device layout with physical index labels, logical-role
  mapping table, and compact left/right step diagrams underneath.

`--no-layout` skips layout generation and its extra transpilation. It does
not disable observable or metric figures. `--no-circuit-plots` removes both
separate step drawings and the layout's circuit subfigures, retaining the
topology/table. `--no-plots` disables all figures.

`--plot-only RESULT.json` regenerates figures without a Sampler submission;
add `--no-layout` to avoid backend lookup/transpilation entirely. It restores
the saved configuration rather than taking model values from CLI defaults.
`--layout-only` compiles and draws the selected representative circuit without
sampling. For real devices this still reads the backend through the account;
a fake backend needs no account. The new driver does not implement the older
cross-archive `--reuse-zne-result` workflow or `--metadata-only`.

## Commands

### Classical randomized-channel simulation (no Aer)

[`classical_randomized_lie_trotter.py`](classical_randomized_lie_trotter.py)
computes the ideal circuit-channel expectations with N complex amplitudes per
sampled boundary-choice trajectory. It retains the same even/odd bond order,
uniform-field phase, seeded R-by-substep schedule, shorter remainder substeps,
and selected-boundary dilation/reset map as the unmitigated circuit driver.
It does not import the circuit driver or other experiments.

For a selected boundary b the update is

\[
a_b\leftarrow\cos(\sqrt{2\gamma\delta})a_b.
\]

The cosine's sign is retained, including at large angles. The lost norm becomes
vacuum weight; amplitudes are never renormalized or replaced by an exact
exponential-damping factor. Thus this is exact evaluation of the **approximate
finite-step randomized channel**, not replacement by exact Lindblad evolution.
Its observable means match ideal circuit expectations for the same trajectories,
without measurement-shot noise. It performs no transpilation, Aer/hardware
sampling, ZNE, DD, twirling, or readout correction.

Propagation costs O(R N K) for K substeps. State storage is O(R N); saved mean
observable curves cost O(N T), per-trajectory excitation curves O(R T), and
the recorded boundary-choice schedule O(R K). No exponentially sized circuit
state/density matrix is constructed. Trajectory standard errors use sample
standard deviation divided by sqrt(R). R=1 returns zero placeholders, explicitly
marked as not an estimated uncertainty. Increasing R does not remove finite-step
bias; smaller delta must be checked separately.

For the N=10, J=8, gamma=20 parameters with a maximum substep of 0.1:

```bash
uv run python Model1/Experiment11/classical_randomized_lie_trotter.py \
  --n-qubits 10 --J 8 --h 0 --gamma 20 \
  --trajectories 4 --t-final 20 --time-points 101 \
  --trotter-delta-t 0.1 --steady-tolerance 1e-3 \
  --optimized-classical-reference
```

The default initial excitation is at site N//2 (site 5 here). The four-panel
line figure compares randomized observables and excitation decay with the exact
reference. Both randomized and exact distances/crossings are saved and reported.
At these rates, 2 gamma delta = 4; the exact reference reaching tolerance by
t=20 does **not** guarantee convergence or accuracy of the randomized map.

Options include `--times` (also one time without a baseline), `--initial-site`,
`--seed-trajectories`, `--no-plots`, `--no-progress`, and `--output-directory`.
Results use distinct `classical_randomized_...` filenames with parameter hashes;
no Aer results are changed. Compatible new times append and repeated times
replace, with their actual schedules retained. The result is saved atomically
before reference/plot generation. Adding the reference flag later, or rerunning
after a plotting failure, is cheap because there are no provider jobs to repeat.
No classical driver checkpoint/resume is needed or implemented.

The classical-driver tests compare small compact and explicitly synthesized
circuits using density-matrix channel propagation (including resets), check
large-angle signed-cosine behavior, seeded schedule compatibility, trajectory
statistics, large-N storage, and persistence/reference generation:

```bash
.venv/bin/python -m unittest tests.test_experiment11_classical -v
```

Commands below are Linux/bash. In PowerShell, replace trailing backslashes
with backticks, or use the entire command on one line. Run from the repo root.
Use `uv run python ... --help` for the complete CLI.

### Ideal Aer, no mitigation, configurable model

```bash
uv run python Model1/Experiment11/mitigated_randomized_lie_trotter.py \
  --n-qubits 10 --J 1 --h 0 --gamma 0.2 \
  --backend aer --trajectories 4 --shots 8192 \
  --times 0 0.2 0.4 0.6 0.8 1 \
  --trotter-delta-t 0.1 --optimization-level 3 \
  --optimized-classical-reference \
  --no-zne --no-pauli-twirling \
  --no-dynamical-decoupling --no-readout-mitigation --no-layout
```

For gamma=5, start with a substantially smaller delta, e.g. 0.005, and verify
accuracy against the exact reference; 0.005 is not a certified tolerance.
The number of substeps/circuit depth grows accordingly.

### Ideal Aer, ZNE only

```bash
uv run python Model1/Experiment11/mitigated_randomized_lie_trotter.py \
  --n-qubits 4 --J 1 --h 0 --gamma 0.2 \
  --backend aer --trajectories 4 --shots 8192 \
  --times 0 0.2 0.4 0.6 0.8 1 --trotter-delta-t 0.1 \
  --zne-scale-factors 1 1.5 2 --zne-inference linear \
  --optimization-level 3 --optimized-classical-reference \
  --no-pauli-twirling --no-dynamical-decoupling \
  --no-readout-mitigation --no-layout --batch-size 6
```

Ideal Aer has no physical gate noise to extrapolate: ZNE here checks ideal
channel preservation, accounting and fitting, not noise-mitigation efficacy.

### Ideal Aer, all pipeline stages

```bash
uv run python Model1/Experiment11/mitigated_randomized_lie_trotter.py \
  --n-qubits 4 --J 1 --h 0 --gamma 0.2 \
  --backend aer --trajectories 4 --shots 8192 \
  --times 0 0.2 0.4 --trotter-delta-t 0.1 \
  --zne-scale-factors 1 1.5 2 --optimization-level 3 \
  --dd-sequence xy4 --dd-durations Model1/Experiment11/aer_timings.json \
  --optimized-classical-reference --no-layout --batch-size 6
```

DD timings are synthetic and readout correction is identity in this command.
No physical noise reduction should be inferred from it.

### Fake Kingston, no mitigation

```bash
uv run python Model1/Experiment11/mitigated_randomized_lie_trotter.py \
  --n-qubits 4 --J 1 --h 0 --gamma 0.2 \
  --backend fake_kingston --trajectories 4 --shots 8192 \
  --times 0 0.2 0.4 --trotter-delta-t 0.1 \
  --optimization-level 3 --optimized-classical-reference \
  --no-zne --no-pauli-twirling \
  --no-dynamical-decoupling --no-readout-mitigation --no-layout --batch-size 6
```

This uses local noisy Aer with the installed FakeKingston snapshot, no login
and no hardware job. It is not a live Kingston calibration. `fake_fez` is
supported for layout-only. The installed FakeMiami snapshot lacks native
reset support; requesting sampling must fail rather than inserting invented
ideal reset/noise. FakePhoenix is not provided by this driver.

### IBM hardware, ZNE only

```bash
uv run python Model1/Experiment11/mitigated_randomized_lie_trotter.py \
  --n-qubits 4 --J 1 --h 0 --gamma 0.2 \
  --backend ibm_kingston --trajectories 4 --shots 8192 \
  --times 0 0.2 0.4 --trotter-delta-t 0.1 \
  --zne-scale-factors 1 1.5 2 --optimization-level 3 \
  --optimized-classical-reference --no-pauli-twirling \
  --no-dynamical-decoupling --no-readout-mitigation --no-layout --batch-size 3
```

The default credentials path is `IBMRuntime/apikey.json`; change it with
`--account-file`. To enable hardware DD omit its off-switch. To enable
hardware readout mitigation omit its off-switch and supply measured
`--readout-calibration FILE`. Hardware access, quota, native/reset support,
timing, and Runtime limits are still required; no hardware jobs were submitted
as part of implementation validation.

### Resume or recover figures

Append `--resume` to the exact original simulation command (batch size and
plot preferences may change). It samples only missing checkpointed variants;
when sampling is complete it proceeds directly to reconstruction/reference/plots.

```bash
uv run python Model1/Experiment11/mitigated_randomized_lie_trotter.py \
  --plot-only Model1/Experiment11/results/YOUR_RESULT.json --no-layout
```

To compile a fake layout with no sampling:

```bash
uv run python Model1/Experiment11/mitigated_randomized_lie_trotter.py \
  --n-qubits 4 --J 1 --h 0 --gamma 0.2 \
  --backend fake_fez --times 0.2 --trotter-delta-t 0.1 --layout-only
```

## Verification

### Size, loss-ratio, timestep and trajectory-count study

[`study_randomized_scaling.py`](study_randomized_scaling.py) implements a
dimensionless classical study, independent of plotting/save intervals:

\[
r=\gamma/J,\qquad \tau=Jt,\qquad s=J\Delta t.
\]

The default size grid is N=2,4,7,10,15,20,30,50. At J=1, h=0 and initial
site N//2, the exact steady-state landscape is computed over r=0.25,0.5,1,
1.5,2,2.5,3,4,6,8. The best grid ratio is locally refined unless it lies on a
search boundary; the refined ratio and best grid ratio receive timestep sweeps.
This is a bounded numerical search, not a proof of global optimality.

The error criteria are explicitly distinct:

- Vacuum trace distance <= 1e-3.
- Maximum population or normalized-current error <= 0.01 over algorithmic
  nodes. Current error is divided by J; XY correlations are identically zero
  for these ideal inputs and are not used to make accuracy look better.
- Steady-state-time relative error <= 5%, using the conservative upper end
  of the step-crossing bracket, not an interpolated claim of exact convergence.

The averaged map is computed using an N-by-N excitation-sector covariance:

\[
M_{k+1}=\tfrac12D_LUM_kU^\dagger D_L^\dagger
       +\tfrac12D_RUM_kU^\dagger D_R^\dagger.
\]

U is the even-then-odd hopping step and D_b multiplies the selected boundary
by cos(sqrt(2rs)). This is exact averaging over the two random alternatives,
not sequential application of both jumps. A real phase gauge makes propagation
real-valued; a sparse N²-dimensional linear map evaluates this update without
full many-qubit density matrices. Once both remaining excitations bound every
future observable error below the allocated tolerance, that tail is certified
without repeated observable comparisons. Reported peaks are evaluated-node
maxima; unevaluated tails have an explicit upper bound.

This study uses a constant algorithmic timestep. The original classical/circuit
drivers instead split each observation interval, potentially adding short
remainder steps at every saved time. The study's passing checks apply to its
constant-step definition; arbitrary observation grids in another driver should
be checked separately rather than assumed to reproduce the same trajectories.

Both the largest tested infinite-R timestep (0.01 transient and 5% timing
limits) and a conservative recommended timestep are reported. Recommendations
reserve 0.005 transient error for finite-step bias and 0.005 for finite-R
fluctuations, with a 2.5% timing budget for the averaged map. Finite ensembles
are checked against both the averaged channel and the exact reference.

The initial R grid is 4,8,16,32,64,128,256 with five seeds: 29,101,307,509,701.
Candidates are checked in ascending order; a candidate is rejected as soon as
any seed permanently exceeds a criterion, and only promising candidates are
propagated to steady state. Every completed R case is separately saved. The
smallest tested R must pass every listed seed; this is empirical validation,
not a statistical confidence guarantee. Independent per-path SeedSequence
streams make R prefixes stable during extensions. They do not reproduce the
circuit driver's specific row-major seeded trajectory draw, although they
sample the same distribution and channel.

Among the promising ratios, the recommendation minimizes dimensionless
substep count on the coarse timestep grid, then locally refines both passing
timestep bounds at the selected ratio (two bisection rounds by default,
configurable with `--step-refinements`). Passing/failing refinement brackets
are saved. This does not globally minimize
the full R-times-substeps cost or compute a complete timestep/R Pareto frontier.
The rate conversion for target T=20 includes a 10% speed margin:

\[
J=1.1\tau_{ss}/20,\quad\gamma=rJ,\quad\Delta t=s/J.
\]

If a smaller user-selected rate margin would put a validated seed's crossing
after the target, J is raised to cover the largest observed crossing. With the
default 10% margin and 5% timing-error limit, the expression above already
covers every accepted seed. This is still not a guarantee for untested seeds.

Increasing both rates compresses physical time, not the required substep count.
Every completed exact, averaged-channel and ensemble case is atomically saved
in a configuration-specific `study/scaling_<hash>/study.json`. `Study.md`
contains the recommendations and search assumptions; line plots summarize the
size scaling, exact ratio landscape, and physical rates/workload for the target
time. Resume skips completed cases. Time
and step caps are recorded; failed/capped searches are not declared validated.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
  Model1/Experiment11/study_randomized_scaling.py
```

Resume with the same options plus `--resume`. To extend unresolved trajectory
counts without rerunning exact/timestep calculations:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
  Model1/Experiment11/study_randomized_scaling.py \
  --resume --extend-R-values 512 1024 2048 4096
```

The extension is explicitly recorded in the ensemble case. Other configurable
options include sizes, ratios, steps, seeds, tolerances, search caps, target
time, rate margin, and output directory. BLAS thread limits prevent overhead
from multithreading the small matrices; no Aer or hardware job is used.

Focused tests check independence from other experiment imports, default
circuit compatibility, the explicit synthesis up to its known global phase,
tuned reference agreement with the full Lindblad solver, remainder substeps,
parameter/configuration guards, three-basis signs and J normalization,
native folding/reset-channel preservation, combined folding/twirling/DD,
readout mapping, append/replace behavior, interrupted-batch resume, and
plot-only recovery without sampling.

```bash
.venv/bin/python -m unittest tests.test_experiment11_standalone -v
```

Implementation validation also completed small ideal-Aer runs with tuned
J=0.8, h=0.3, gamma=0.7, both without mitigation and with all four pipeline
stages, and an append/replace run. All requested figure types were generated;
a FakeFez layout-only check completed without login or sampling. These small
shot-budget checks validate execution and persistence, not mitigation efficacy
or a statistically meaningful benchmark. Their artifacts were kept in `/tmp`,
separate from experiment result archives. No real-device job was submitted.
