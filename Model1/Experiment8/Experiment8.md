# Experiment 8: physical local-folding ZNE for Experiment 7

## Purpose

Experiment 8 applies zero-noise extrapolation (ZNE) to the explicit
randomized Lie--Trotter circuits from Experiment 7. It keeps the same physical
model, randomized single-boundary strategy, two resettable boundary-local
ancillas, explicit `RZ`/`RX`/`RZZ` synthesis, observables, time scheduling, and
classical-reference options. The only new approximation layer is
post-transpilation local unitary folding.

The executable is
[`zne_randomized_lie_trotter.py`](zne_randomized_lie_trotter.py).
It uses the repository's immutable `mitiq_qem` ZNE plan and inference layer.
The folding transform itself lives in Experiment 8 because it must run after
backend transpilation and must distinguish virtual from physical native gates;
a normal pre-transpilation Mitiq fold would not provide that guarantee.

## Where folding is applied

Let the Experiment 7 logical circuit be $C$, and let the compiler map it to a
backend-native circuit

\[
C_{\mathrm{ISA}}=\mathcal{T}(C).
\]

Experiment 8 transpiles each base circuit exactly once. It then constructs all
noise-scaled variants directly from $C_{\mathrm{ISA}}$. There is no second
optimizing transpilation after folding, because such a pass could cancel the
inserted inverse pairs.

For an eligible native gate $G$, one local fold is

\[
G\longmapsto G G^\dagger G
              =G\left(G^\dagger G\right).
\]

The ideal operation is unchanged, while the physical implementation contains
two additional gates. If $L$ physical gates are eligible and the algorithm
inserts $m$ folds, the realized gate-count scale is

\[
\lambda_{\mathrm{actual}}=\frac{L+2m}{L}
                          =1+\frac{2m}{L}.
\]

For a requested scale $\lambda$, every eligible gate first receives

\[
k=\left\lfloor\frac{\lambda-1}{2}\right\rfloor
\]

complete folds. A seeded random subset receives one additional fold to realize
the remaining fractional scale as closely as the discrete gate count allows.
The requested and realized scales are both stored in the result JSON.

### Physical-gate policy

The folding transform is deliberately backend aware:

- `RZ`, `P`, `U1`, and `Phase` are treated as virtual frame changes and are
  not folded.
- A nonvirtual unitary ISA gate is eligible only if its inverse is supported on
  the same physical qubits by the backend target.
- On IBM targets that provide `SX` but not `SXdg`, the inverse is emitted as
  `RZ(pi) SX RZ(pi)`: one physical `SX` pulse surrounded by virtual frame
  changes. This keeps the folded circuit native without under-scaling `SX`.
- `reset`, `measure`, `barrier`, `delay`, and classical `store` operations are
  never folded.
- Folding is local to each gate. It therefore never moves an inverse through a
  reset or crosses a reset boundary.
- The transpiled layout and routing are retained by copying the compiled
  circuit and inserting inverse pairs in place.

This realizes the agreed physical interpretation: `RZ` is virtual, whereas
physical single- and two-qubit pulses such as `RX` and `RZZ` contribute to the
noise scale.

## Three-basis symmetry reconstruction

Experiment 8 measures only `Z`, `X`, and one alternating `XY`
product basis. At system site \(n\), the alternating basis uses \(X\) for even
\(n\) and \(Y\) for odd \(n\).

The ideal Hamiltonian and loss-only jumps keep the density operator block
diagonal in excitation number and inside the vacuum-plus-one-excitation
subspace. In that subspace,

\[
\langle X_nX_{n+1}\rangle=\langle Y_nY_{n+1}\rangle
\]

and

\[
\langle Y_nX_{n+1}\rangle=-\langle X_nY_{n+1}\rangle.
\]

The measured observables are therefore reconstructed as

\[
C_n^{XY}=2\langle X_nX_{n+1}\rangle
\]

and

\[
I_n=
\begin{cases}
-J\langle X_nY_{n+1}\rangle, & n\text{ even},\\
\phantom{-}J\langle Y_nX_{n+1}\rangle, & n\text{ odd}.
\end{cases}
\]

The corresponding single-circuit shot variances are

\[
\operatorname{Var}(C_n^{XY})
=4\frac{1-\langle X_nX_{n+1}\rangle^2}{S_{\mathrm{circuit}}},
\]

and

\[
\operatorname{Var}(I_n)
=J^2\frac{1-\langle P_n^{XY}\rangle^2}{S_{\mathrm{circuit}}},
\]

where \(P_n^{XY}\) denotes the parity-appropriate mixed Pauli product. These
factors are included explicitly in the implementation.

This is an exact reconstruction for the intended model. On real hardware it
acts as an excitation-symmetry projection: hardware errors that violate the
symmetry are not measured independently as `XX-YY` or `YX+XY`, but
their effect can still appear as deviation from the classical reference.
Experiments 5 and 7 retain their original five-basis diagnostic scheme.

## Randomized dilation and shot accounting

For each saved time, measurement basis, and randomized qDRIFT trajectory,
Experiment 8 generates

\[
N_{\mathrm{variants}}=
N_{\lambda}F
\]

folded circuits, where $N_{\lambda}$ is the number of scale factors and
$F$ is `--fold-repetitions`. With three measurement bases and \(R\)
randomized trajectories, the total circuit count per saved time is

\[
N_{\mathrm{circuits/time}}=3RFN_{\lambda}.
\]

The `--shots` argument retains the Experiment 7 interpretation separately at
every noise scale: it is the total number of shots per saved time and
measurement basis at that scale. Each concrete circuit receives

\[
S_{\mathrm{circuit}}=\frac{S}{RF}.
\]

Here $S=\texttt{--shots}$ and $R=\texttt{--trajectories}$. Consequently,
the total number of shots across all ZNE scales per saved time and basis is
$N_{\lambda}S$, and the total over all three bases is
\(3N_{\lambda}S\). The script requires $S$ to be divisible by $RF$.

## Extrapolation

Let $E(\lambda_j)$ be the observable averaged over the randomized dilation
trajectories and independent fold realizations at noise scale $\lambda_j$.
Experiment 8 fits

\[
E(\lambda)=a_0+a_1\lambda+\cdots+a_d\lambda^d

\]

and reports the zero-noise estimate

\[
E_{\mathrm{ZNE}}=E(0)=a_0.
\]

The available inference choices are:

- `linear`: $d=1$, using all supplied scale factors in a least-squares fit;
- `richardson`: $d=N_{\lambda}-1$, interpolation through every scale;
- `polynomial`: user-selected degree `--zne-polynomial-order`, which must be
  smaller than $N_{\lambda}$.

The result file records the extrapolation weights $w_j$, so every estimate
can also be written as

\[
E_{\mathrm{ZNE}}=\sum_j w_j E(\lambda_j).
\]

## Uncertainty decomposition

The script keeps three estimated uncertainty components separate:

1. finite-shot uncertainty within each measured circuit;
2. random-fold-location uncertainty across the $F$ folded realizations;
3. randomized-boundary-trajectory uncertainty across the $R$ qDRIFT paths.

The components are propagated through the linear extrapolation weights and
combined in quadrature. With `--fold-repetitions 1`, the observable folding
variance cannot be estimated and is reported as zero; use $F>1$ when that
component is important.

## Outputs and recovery

Each run writes one merged result JSON. New time points are appended and a
matching time point is replaced, provided all physics, compiler, ZNE, seed,
shot, and reference settings are compatible. The checkpoint is updated after
every submitted batch and supports `--resume`.

The generated figures are:

- ZNE observables, optional exact reference, and uncertainty decomposition;
- final-time scale dependence and the zero-noise extrapolates;
- explicit pre-transpilation, base-ISA, and post-fold operation/depth metrics;
- the transpiled physical layout and logical-to-physical mapping;
- the two logical one-step circuits and the explicit exchange decomposition,
  unless `--no-circuit-plots` is supplied.

The JSON additionally stores raw counts, per-variant circuit metadata, actual
fold scales, eligible/excluded gate counts, all uncertainty arrays, provider
job IDs, and figure-generation status. Data are saved before figure generation
so a plotting failure does not discard completed hardware results.

## Example commands

Small local Aer validation:

```bash
uv run python Model1/Experiment8/zne_randomized_lie_trotter.py \
  --n-qubits 4 \
  --backend aer \
  --trajectories 4 \
  --fold-repetitions 2 \
  --shots 8192 \
  --times 0 0.2 0.4 0.6 0.8 1.0 \
  --trotter-delta-t 0.1 \
  --zne-scale-factors 1 1.5 2 \
  --zne-inference linear \
  --optimization-level 3 \
  --optimized-classical-reference
```

IBM hardware uses the same command with an accessible backend name, for
example `--backend ibm_kingston`. This submits real QPU work. `--batch-size`
is the maximum number of already-folded circuits in one provider job and must
be at least
$N_{\lambda}F$, because all variants of one compiled base circuit remain in
the same batch.

PowerShell line continuation uses the backtick:

```powershell
uv run python .\Model1\Experiment8\zne_randomized_lie_trotter.py `
  --n-qubits 4 `
  --backend aer `
  --trajectories 4 `
  --fold-repetitions 2 `
  --shots 8192 `
  --times 0 0.2 0.4 0.6 0.8 1.0 `
  --trotter-delta-t 0.1 `
  --zne-scale-factors 1 1.5 2 `
  --zne-inference linear `
  --optimization-level 3 `
  --optimized-classical-reference
```

## Important interpretation of Aer results

The repository's plain `--backend aer` path is an ideal simulator unless a
noise model is added elsewhere. It validates circuit equivalence, scheduling,
shot allocation, inference, checkpointing, and plotting, but ideal Aer does
not provide a physical noise trend for ZNE to remove. The intended mitigation
test is therefore an IBM-hardware run, or a future explicitly noisy Aer target.

## Main configurable options

| Option | Meaning |
| --- | --- |
| `--n-qubits N` | Number of system qubits; total circuit width is $N+2$. |
| `--backend NAME` | `aer`, `fake_fez` with `--layout-only`, or an IBM backend. |
| `--trajectories R` | Randomized single-boundary qDRIFT paths. |
| `--fold-repetitions F` | Independent seeded local-fold placements per scale. |
| `--shots S` | Total shots per time/basis/scale, divided over $RF$. |
| `--times ...` | Arbitrary requested saved times. |
| `--trotter-delta-t DT` | Maximum internal substep; the last remainder may be shorter. |
| `--zne-scale-factors ...` | Unique increasing scales beginning with 1. |
| `--zne-inference` | `linear`, `richardson`, or `polynomial`. |
| `--zne-polynomial-order D` | Degree for polynomial inference. |
| `--seed-folding SEED` | Reproducible random local-fold selection. |
| `--batch-size B` | Maximum submitted folded circuits per provider job. |
| `--classical-reference` | Full exact Lindblad calculation for manageable $N$. |
| `--optimized-classical-reference` | Exact $N+1$ invariant-subspace reference. |
| `--resume` | Resume a compatible checkpoint. |
| `--layout-only` | Compile/draw the layout without sampling. |
| `--no-circuit-plots` | Omit circuit panels but retain layout, metrics, and result plots. |
