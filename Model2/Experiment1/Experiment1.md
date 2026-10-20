# Experiment 1: Model 2, equal boundary reservoirs

This classical benchmark simulates relaxation from `000`, `111`, and `100`
to the reservoir-selected product state. The defaults are three system
qubits, $J=1$, $\Delta=0$, $\Gamma_L=\Gamma_R=2$, and $p_L=p_R=1/4$.
Arbitrary finite, nonnegative, strictly increasing observation times are
supported, including a single positive time and irregular grids. Preparation
always occurs at $t=0$.

## Hamiltonian and current normalization

The selected default is

$$
H=cJ\sum_{j=1}^{n-1}(X_jX_{j+1}+Y_jY_{j+1}+\Delta Z_jZ_{j+1}),
\qquad c=\frac14.
$$

Continuity fixes the excitation current to

$$
I_j=cJ(Y_jX_{j+1}-X_jY_{j+1}).
$$

Use `--exchange-prefactor 0.5` for $c=1/2$, which matches both Model 1's
Hamiltonian and the original Model 2 excerpt's $I_j=J(YX-XY)/2$.
Changing $c$ changes the transient dynamics, but not the equal-reservoir
stationary product state. The implementation evaluates the continuity
equations directly from $\mathcal L\rho$, without time-grid finite differences.

## Exact physical reference

The four jumps, in their dilation label order, are

$$
V_1=\sqrt{\Gamma p}\,\sigma^+_1,\quad
V_2=\sqrt{\Gamma(1-p)}\,\sigma^-_1,\quad
V_3=\sqrt{\Gamma p}\,\sigma^+_n,\quad
V_4=\sqrt{\Gamma(1-p)}\,\sigma^-_n.
$$

Here $\sigma^+=|1\rangle\langle0|$ injects an excitation and
$\sigma^-=|0\rangle\langle1|$ removes it. At the default rates the amplitudes
are $\sqrt{1/2}$ for pumping and $\sqrt{3/2}$ for loss.
Site 1 is the leftmost bit in a computational basis string, using Model 1's
classical tensor ordering.

Column-major vectorization gives

$$
\operatorname{vec}\rho(t)=e^{tL}\operatorname{vec}\rho(0),
$$

$$
L=-i(\mathbb I_d\otimes H-H^T\otimes\mathbb I_d)
+\sum_{k=1}^{4}\left[
V_k^*\otimes V_k
-\frac12\mathbb I_d\otimes V_k^\dagger V_k
-\frac12(V_k^\dagger V_k)^T\otimes\mathbb I_d
\right],\qquad d=2^n.
$$

The reference uses the existing `classical.build_liouvillian` and sparse
exponential action, with no Hamiltonian/dissipator split. It is exact up to
floating-point numerical error. For irregular times, successive unsplit
exponential actions advance by the requested time intervals; the first
positive time is evolved from preparation at zero.

The exact stationary target is

$$
\rho_*=\left(\frac34|0\rangle\langle0|+
\frac14|1\rangle\langle1|\right)^{\otimes n},\qquad
\langle Z_j\rangle_*=0.5,\qquad \langle I_j\rangle_*=0.
$$

## Shared-ancilla dilation

The same first-order construction as Model 1 is reused, now with five
ancilla levels: reset label $0$ and four jump labels. For a physical step
$\delta t$ the dimensionless Hermitian block matrix is

$$
K=\begin{pmatrix}
\delta t H & \sqrt{\delta t}V_1^\dagger & \sqrt{\delta t}V_2^\dagger & \sqrt{\delta t}V_3^\dagger & \sqrt{\delta t}V_4^\dagger\\
\sqrt{\delta t}V_1&0&0&0&0\\
\sqrt{\delta t}V_2&0&0&0&0\\
\sqrt{\delta t}V_3&0&0&0&0\\
\sqrt{\delta t}V_4&0&0&0&0
\end{pmatrix}.
$$

The unsplit dilation exponentiates $U=e^{-iK}$ exactly, then extracts
$A_k=\langle k|U|0\rangle$ and applies
$\Phi_{\delta t}(\rho)=\sum_{k=0}^{4}A_k\rho A_k^\dagger$.
Lie splitting applies $K_H,K_1,K_2,K_3,K_4$ in chronological order.
Symmetric splitting applies the four jump half-steps, the full system
step, then the jump half-steps in reverse order. The ancilla is traced/reset
only after the complete unitary, never between its factors.

Each requested interval is divided into equal positive substeps no longer
than `--max-dt`, default 0.05. Kraus operators are cached by substep length.
The code propagates system density matrices rather than full joint density
matrices. A different observation grid can change the dilation substep
schedule slightly; the exact physical reference has no such step-size choice.

**An exactly exponentiated dilation is still an approximate Lindblad channel.**
This first-order construction has local physical-step error
$O(\delta t^2)$ and global error $O(T\delta t)$ for fixed $T$ as the step
shrinks. Symmetric unitary splitting does not turn it into a second-order
Lindblad solver. Finite-step stationary populations can also shift away from
0.5 and develop spurious small currents. Increasing the final time alone
does not remove that bias; step refinement is required.

| Object | General $n$ | Three-qubit benchmark |
|---|---|---|
| System $H,\rho,V_k$ | $2^n\times2^n$ | $8\times8$ |
| Exact Liouvillian $L$ | $4^n\times4^n$ | $64\times64$ |
| Minimal dilation $K,U$ | $(5\,2^n)\times(5\,2^n)$ | $40\times40$ |
| Binary-encoded dilation unitary | $2^{n+3}\times2^{n+3}$ | $64\times64$ |

The five-level ancilla needs three qubits in a binary encoding, so a circuit
would need six total qubits for this benchmark. The binary unitary's
$64\times64$ size and the Liouvillian's $64\times64$ size refer to different
spaces. This experiment is a classical simulation; it does not compile or
run a hardware circuit. Storage and dense dilation exponentiation grow
exponentially with $n$.

## Run and inspect results

From the repository root:

```bash
# Exact reference plus unsplit, Lie and symmetric dilation, all three states.
.venv/bin/python Model2/Experiment1/simulation.py

# Exact physical evolution at arbitrary times; even a single time is accepted.
.venv/bin/python Model2/Experiment1/exact.py --times 0 0.17 1.3 7.3 20
.venv/bin/python Model2/Experiment1/exact.py --times 7.3

# Dilation with a finer step, compared to the exact reference.
.venv/bin/python Model2/Experiment1/lie_trotter.py --times 0 0.17 1.3 7.3 20 --max-dt 0.0125
.venv/bin/python Model2/Experiment1/strang_trotter.py --max-dt 0.0125

# Use the original excerpt's exchange/current normalization.
.venv/bin/python Model2/Experiment1/simulation.py --exchange-prefactor 0.5

# Reverse the sign of the reservoir-selected target.
.venv/bin/python Model2/Experiment1/exact.py --p 0.75
```

`--n-qubits`, `--J`, `--delta`, `--gamma`, `--p`, `--t-final`, `--num-times`,
and `--output-dir` are also configurable. This experiment enforces equal
reservoirs; unequal reservoirs belong to the transport experiment.

Each run writes a plot, JSON diagnostics and parameters, and a compressed
NPZ containing every sampled density matrix, local $Z$ expectation, and bond
current. Array shapes are `(initial_state, time, row, column)` for density
matrices and `(initial_state, time, site_or_bond)` for observables.
The filename includes mode, $n$, $p$, and exchange prefactor. Reusing these
with different times or step sizes overwrites that run's files; use separate
output directories to keep multiple schedules.

The saved default benchmark is
[`results/all_N3_p0.25_c0.25.json`](results/all_N3_p0.25_c0.25.json),
with its [plot](results/all_N3_p0.25_c0.25.png) and
[density matrices](results/all_N3_p0.25_c0.25.npz).
Results for the original $J/2$ normalization are saved separately with
`c0.5` in the filename. Step-refinement diagnostics are in
`results/refinement_N3_p0.25_c0.25.json` and
`results/refinement_N3_p0.25_c0.5.json`.

For the selected $c=1/4$ normalization, at $t=20$ the exact magnetizations
from all three preparations differ from 0.5 by less than $8\times10^{-9}$,
and the continuity residual over the trajectory is below $3\times10^{-15}$.
At dilation step 0.05, the unsplit dilation instead settles near
$(0.49368,0.50310,0.49368)$. The Lie dilation settles near
$(0.49370,0.49370,0.49370)$.

The maximum Frobenius density-matrix error against exact Lindblad evolution,
over all three initial states and observation times $0,0.1,\ldots,10$, is:

| Maximum step | Unsplit dilation | Lie dilation | Symmetric dilation |
|---|---:|---:|---:|
| 0.1 | 0.04686 | 0.04756 | 0.04544 |
| 0.05 | 0.02137 | 0.02220 | 0.02072 |
| 0.025 | 0.01025 | 0.01075 | 0.00994 |
| 0.0125 | 0.00502 | 0.00529 | 0.00487 |

The optional $p=3/4$ reference also reaches the predicted $Z=-0.5$ within
$8\times10^{-9}$ at $t=20$; that run uses the irregular observation times
$0.17,1.3,7.3,20$.

## Validation

The exact population settling-time curve for N=2 through 20, with tolerance
1e-3, is in [the scaling report](analysis/steady_state_scaling/Results.md).
It uses the exact XX covariance equations, which give populations without
constructing the full density matrix. The main rates are J=10 and Gamma=20;
the report also gives the original J=1, Gamma=2 times, which are ten times longer.
Convergence means every site's population stays within 1e-3 of p=0.25 for
all three benchmark preparations. The all-ones preparation bounds the
other two, and its maximum site error is monotone, so there is no later
threshold recrossing.

```bash
MPLCONFIGDIR=/tmp/model2_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python Model2/Experiment1/analysis/steady_state_scaling.py
```

The seven-qubit Lie/Strang timestep comparison for $J=10$, $\Gamma=20$,
$p=1/4$, $\Delta=0$, $c=1/4$ and $t\leq12$ is in
[`analysis/strang_step_selection/Comparison.md`](analysis/strang_step_selection/Comparison.md).
Across the three benchmark preparations, Strang has larger maximum transient
population error at each tested step. At $\delta t=0.0005$, Lie's population
error is 0.000997 and Strang's is 0.001291; at 0.00025 the errors are 0.000496
and 0.000645. Keep Lie with 0.00025 for the existing population-error margin.
Both methods remain first order for physical Lindblad evolution.

The comparison runner evaluates powers of the same discrete Kraus channel
in number blocks, validates early powers against direct iteration, and
records separate Krylov and sparsification error bounds. This accelerates
the classical calculation while preserving the original ancilla-reset
counts; the main Lie/Strang scripts continue to use direct Kraus iteration.

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_model2_experiment1.py' -v
```

Checks cover isolated reservoir rates, exact stationarity with interactions,
the continuity operator identity under both normalizations, arbitrary-time
agreement with independent dense exponentials, equivalence to Model 1's
existing four-jump-capable dilation engine, trace/Hermiticity/positivity,
and reduced errors under step refinement. Model 1's existing dilation
circuit tests are also retained as a regression check.
