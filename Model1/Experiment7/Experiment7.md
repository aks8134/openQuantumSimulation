# Experiment 7: Explicit randomized Lie--Trotter synthesis

Experiment 7 uses the same randomized single-boundary qDRIFT strategy as
Experiment 5, including its two boundary-local reset ancillas, Aer/IBM
execution, checkpointing, append-or-replace result archive, classical
reference, circuit figures, layout figure, and transpilation metrics. Its
important difference is that every logical `XXPlusYY(theta, 0)` operation is
expanded into `RZ`, `RX`, and `RZZ` operations at the compiler boundary.
Logical construction and circuit diagrams retain compact `XXPlusYY` blocks;
Aer or IBM execution never receives those blocks because lowering happens
immediately before each execution batch is compiled.

The implementation is
[`randomized_lie_trotter.py`](randomized_lie_trotter.py). The default working
point selected by Experiment 6 is

\[
R=4,\qquad \Delta t_{\max}=0.1,\qquad 0\leq t\leq2,
\]

but all three settings remain command-line configurable.

## Randomized Lindblad splitting

For the boundary-damped chain,

\[
\mathcal L=\mathcal L_H+\mathcal D[V_L]+\mathcal D[V_R],
\]

Experiment 7 samples one of

\[
\mathcal L_L=\mathcal L_H+2\mathcal D[V_L],\qquad
\mathcal L_R=\mathcal L_H+2\mathcal D[V_R]
\]

with equal probability during each internal substep. Consequently,

\[
\mathbb E_B[\mathcal L_B]=\mathcal L.
\]

As in Experiment 5, a selected boundary jump therefore has exchange angle

\[
\theta_J=2\sqrt{2\gamma\,dt},
\]

while every system bond has angle \(\theta_H=2Jdt\). Each sampled substep
contains \(N-1\) system exchanges, one selected boundary exchange, and one
reset. It uses \(N+2\) qubits: \(N\) system qubits and one local ancilla at
each boundary.

## Explicit exchange decomposition

For every system or jump exchange, the lowering pass constructs the following
eight layers directly:

\[
\begin{aligned}
&R_Z(\pi/2)\otimes R_Z(-\pi/2)\\
\rightarrow{}&R_X(\pi/2)\otimes R_X(-\pi/2)\\
\rightarrow{}&R_Z(\pi/2)\otimes R_Z(-\pi/2)\\
\rightarrow{}&R_{ZZ}(\theta/2)\\
\rightarrow{}&R_X(-\pi/2)\otimes R_X(-\pi/2)\\
\rightarrow{}&R_{ZZ}(\theta/2)\\
\rightarrow{}&R_Z(-\pi/2)\otimes R_Z(-\pi/2)\\
\rightarrow{}&R_X(-\pi/2)\otimes R_X(-\pi/2).
\end{aligned}
\]

This circuit equals `XXPlusYY(theta, 0)` up to an irrelevant global phase.
The repository tests verify that identity at several positive and negative
angles using the complete two-qubit unitary.

| Operation | Count | Hardware interpretation |
|---|---:|---|
| `RZ` | 6 | virtual frame changes |
| `RX` | 6 | physical one-qubit pulses |
| `RZZ` | 2 | physical two-qubit pulses |

Thus one exchange block has:

- 14 total Qiskit operations;
- 6 virtual and 8 physical operations;
- standard circuit depth 8;
- physical pulse depth 5 after zero-duration `RZ` layers are omitted; and
- two-qubit depth 2.

The virtual `RZ` operations are not deleted. They change the phase frame so
the two physical `RZZ` interactions generate \(XX+YY\), rather than merely
\(ZZ\). “Virtual” means essentially zero pulse duration, not semantically
optional.

For one randomized substep, there are exactly \(N\) exchange blocks. Before
state preparation, terminal basis rotations, measurements, and reset are
counted, their aggregate explicit resources are therefore

\[
6N\;R_Z,\qquad 6N\;R_X,\qquad 2N\;R_{ZZ}.
\]

The backend compiler still performs placement, routing, cancellation, and
lowering of `RZZ` to the backend target when necessary. It is no longer asked
to choose the mathematical `XXPlusYY` synthesis.

The separation is deliberately:

```text
logical XXPlusYY circuits -> compact circuit diagrams
                         -> explicit RZ/RX/RZZ lowering -> compiler/backend
```

The `pre-transpilation` values in the metrics figure refer to the explicitly
lowered `RZ/RX/RZZ` circuit, not the compact logical diagram. On Aer, explicit
lowering is performed one execution batch at a time; the default Experiment 7
batch size is 20 circuits and remains configurable with `--batch-size`.

## Running on Aer

The defaults can be run with:

```bash
uv run python Model1/Experiment7/randomized_lie_trotter.py \
  --n-qubits 4 \
  --backend aer \
  --shots 8192 \
  --classical-reference
```

An explicit time list and non-default convergence parameters are accepted:

```bash
uv run python Model1/Experiment7/randomized_lie_trotter.py \
  --n-qubits 4 \
  --backend aer \
  --trajectories 8 \
  --shots 8192 \
  --times 0 0.2 0.4 0.6 0.8 1.0 \
  --trotter-delta-t 0.05 \
  --optimization-level 3 \
  --classical-reference
```

### Optimized exact classical reference

Experiment 7 additionally provides `--optimized-classical-reference`. It is
an exact reduction for this model, not an approximation. The initial state
contains one excitation, the $XY$ Hamiltonian conserves excitation number,
and the loss-only boundary jumps can only move that excitation into the
vacuum. The evolution therefore remains in

\[
\operatorname{span}\{|\mathrm{vac}\rangle,|0\rangle,\ldots,|N-1\rangle\},
\]

whose Hilbert-space dimension is $N+1$. The optimized Liouvillian has
dimension $(N+1)^2\times(N+1)^2$, rather than the full
$4^N\times4^N$ Liouvillian. Site populations, nearest-neighbor $XY$
correlations, and oriented excitation flows are evaluated directly in this
invariant basis. This makes an exact $N=10$ reference small while preserving
the same numerical answer as the full-space calculation.

The existing `--classical-reference` flag is unchanged and still invokes the
original full-space reference workflow. The two flags are mutually exclusive.
The optimized reduction is currently implemented only in Experiment 7 and is
valid specifically for the one-excitation, excitation-conserving,
boundary-loss problem used here.

For example, a new $N=10$ run can use:

```bash
uv run python Model1/Experiment7/randomized_lie_trotter.py \
  --n-qubits 10 \
  --backend aer \
  --trajectories 4 \
  --shots 8192 \
  --times 0 0.2 0.4 0.6 0.8 1.0 \
  --trotter-delta-t 0.1 \
  --optimization-level 3 \
  --optimized-classical-reference
```

If the matching sampled circuits are already complete in the checkpoint, add
`--resume`; the saved sampler results are reused and only the reference and
post-processing stages continue.

`--trotter-delta-t` is the maximum internal step. A saved time of 0.35 with
`--trotter-delta-t 0.1` is implemented by substeps
\((0.1,0.1,0.1,0.05)\) in the same circuit. It does not generate separate
circuits for the intermediate times unless those times are listed explicitly.

Circuit-rendering images can be disabled without disabling the numerical
results or resource metrics:

```bash
uv run python Model1/Experiment7/randomized_lie_trotter.py \
  --n-qubits 7 \
  --backend aer \
  --trajectories 4 \
  --shots 8192 \
  --times 0 0.2 0.4 0.6 0.8 1.0 \
  --trotter-delta-t 0.1 \
  --optimization-level 3 \
  --classical-reference \
  --no-circuit-plots
```

`--no-circuit-plots` retains the transpiled topology/layout and
logical-to-physical mapping table, but omits the one-step circuit panels below
that layout. It also skips the separate left/right one-step circuit figures
and explicit exchange-decomposition figure. The observable figure,
transpilation-metrics figure, result JSON, and checkpoint are still produced.

## Running on IBM hardware

Substitute an accessible backend and choose a batch size appropriate for the
account:

```bash
uv run python Model1/Experiment7/randomized_lie_trotter.py \
  --n-qubits 4 \
  --backend ibm_kingston \
  --trajectories 4 \
  --shots 8192 \
  --times 0.2 \
  --trotter-delta-t 0.1 \
  --batch-size 5 \
  --optimization-level 3
```

The IBM API key and CRN are read through the same account mechanism as
Experiments 4 and 5. `--shots` is the total shot budget per saved time and
measurement basis and must be divisible by `--trajectories`. Results are
checkpointed after every completed batch.

After sampling, the script immediately retains only the representative
final-time circuit and releases the complete circuit batch. The merged result
JSON is then written before any figure rendering starts. Its
`figure_generation_status` is `pending` during rendering and changes to
`complete` only after every figure is saved. Consequently, a plotting-stage
memory interruption cannot discard the sampled observables.

To inspect placement without submitting a sampler job:

```bash
uv run python Model1/Experiment7/randomized_lie_trotter.py \
  --n-qubits 4 \
  --backend fake_fez \
  --times 0.2 \
  --trotter-delta-t 0.1 \
  --layout-only
```

## Outputs

Outputs are written under `Model1/Experiment7` by default with an
`explicit_randomized_lie_trotter_*` stem. A normal run produces:

- the observable figure and merged JSON result archive;
- a transpilation-metrics figure;
- separate left- and right-boundary one-step circuit figures;
- a transpiled topology/layout figure with logical-to-physical mapping and
  both one-step circuits;
- an exchange-decomposition figure containing the explicit circuit, resource
  table, and virtual-aware depth/count chart; and
- one active checkpoint file for the current requested run.

The one-step circuit figures and the circuit panels embedded in the layout
use compact logical `XXPlusYY` blocks. They are intentionally not drawings of
the much larger lowered `RZ/RX/RZZ` representation sent to the compiler.

New saved times are appended to a compatible JSON archive, while a newly run
time replaces the old value at that same time. The explicit decomposition
metadata and resource accounting are stored in both results and checkpoints.
