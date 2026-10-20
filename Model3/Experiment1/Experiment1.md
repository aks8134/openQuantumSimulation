# Experiment 1: Model 3, unequal boundary reservoirs

This classical experiment implements the supplied nonuniform-reservoir
example. Three qubits relax from `000`, `111`, and `100` toward a
nonequilibrium stationary state carrying an excitation current. Exact
Lindblad evolution is compared with unsplit, Lie, and symmetric
shared-ancilla dilation, using Model 2's existing propagation engines.

## Parameters and conventions

The defaults are $N=3$, $J=1$, $\Delta=0$, $\Gamma_L=\Gamma_R=2$,
$p_L=0.9$, $p_R=0.1$, and

$$
H=cJ\sum_{j=1}^{N-1}(X_jX_{j+1}+Y_jY_{j+1}+\Delta Z_jZ_{j+1}),
\qquad c=\frac12.
$$

This explicitly selects Model 2's optional $J/2$ normalization, as requested,
to match the quoted numerical targets. Model 2's $J/4$ default would instead
give $Z_*=(-0.64,0,0.64)$ and $I_*=0.16$ for these reservoirs.

The only jump operators act on the endpoints, in this fixed order:

$$
V_{L,+}=\sqrt{\Gamma p_L}\sigma^+_1,\quad
V_{L,-}=\sqrt{\Gamma(1-p_L)}\sigma^-_1,\quad
V_{R,+}=\sqrt{\Gamma p_R}\sigma^+_N,\quad
V_{R,-}=\sqrt{\Gamma(1-p_R)}\sigma^-_N.
$$

Their default amplitudes are $\sqrt{1.8}$, $\sqrt{0.2}$,
$\sqrt{0.2}$, $\sqrt{1.8}$. Here $\sigma^+=|1\rangle\langle0|$,
$\sigma^-=|0\rangle\langle1|$, and $n_j=(1-Z_j)/2$.
Site 1 is the leftmost bit. The excitation current is

$$
I_j=cJ(Y_jX_{j+1}-X_jY_{j+1}),\qquad j_j=-2I_j,
$$

where $j_j$ is the magnetization current. Positive $I$ carries excitations
from left to right; the sign difference follows from $Z=1-2n$.

## Exact stationary targets

With hopping amplitude $a=2cJ$, equal boundary rates, and $\Delta=0$,
stationary continuity and the XX correlation equations give

$$
I_* = \frac{2a^2\Gamma}{\Gamma^2+4a^2}(p_L-p_R),\qquad
n_{1,*}=p_L-I_*/\Gamma,\qquad
n_{N,*}=p_R+I_*/\Gamma.
$$

Every interior site has $n_{j,*}=(p_L+p_R)/2$. For $c=1/2$, $a=J$,
which reproduces the supplied formula.

| Reservoirs $(p_L,p_R)$ | Populations $(n_1,n_2,n_3)_*$ | Magnetizations $(Z_1,Z_2,Z_3)_*$ | Bond currents $(I_1,I_2)_*$ |
|---|---|---|---|
| $(0.25,0.25)$ | $(0.25,0.25,0.25)$ | $(0.5,0.5,0.5)$ | $(0,0)$ |
| $(0.9,0.1)$ | $(0.7,0.5,0.3)$ | $(-0.4,0,0.4)$ | $(0.4,0.4)$ |
| $(0.1,0.9)$ | $(0.3,0.5,0.7)$ | $(0.4,0,-0.4)$ | $(-0.4,-0.4)$ |

The forward magnetization current is $j_1=j_2=-0.8$. Unequal reservoirs
produce coherences; their stationary density matrix is not a product of
the three local diagonal states. The code solves the full stationary
equation $L\operatorname{vec}(\rho_*)=0$ with $\operatorname{tr}\rho_*=1$,
independently of the analytic profile, and uses this state as the distance
target. For $\Delta\ne0$, this numerical stationary solve remains applicable;
the XX formulas and correlation checks are skipped.

## Evolution and independent checks

Exact evolution uses $e^{tL}$ with column-major vectorization and sparse
exponential action. Observation times may be irregular or start after zero;
every preparation is at $t=0$.

The stationary solve retains all density-matrix elements between states
with equal excitation number. This exact symmetry reduction preserves the
stationary coherences and reduces the $N=7$ linear system from 16,384 to
3,432 unknowns. Dilation construction applies sparse matrix exponentials
directly to the reset isometry, avoiding full joint-unitary exponentiation.
The physical channels and factor ordering are unchanged.

For $\Delta=0$, a separate $N\times N$ correlation calculation checks the
entire exact trajectory, using $C_{ij}=\langle c_j^\dagger c_i\rangle$:

$$
A=-ih-G/2,\qquad AC_*+C_*A^\dagger+P=0,
\qquad C(t)=C_*+e^{tA}(C(0)-C_*)e^{tA^\dagger}.
$$

Here $h_{j,j+1}=h_{j+1,j}=a$, $G=\operatorname{diag}(\Gamma,0,\ldots,0,\Gamma)$,
and $P=\operatorname{diag}(\Gamma p_L,0,\ldots,0,\Gamma p_R)$.
Then $n_j=C_{jj}$ and $I_j=2a\operatorname{Im}C_{j,j+1}$.
Exact XX dynamics with endpoint dissipation are also studied by
[Yamanaka and Sasamoto](https://arxiv.org/abs/2104.11479).

Continuity is checked directly from $L\rho$, with the left source
$\Gamma(p_L-n_1)$ and right source $\Gamma(p_R-n_N)$, avoiding
finite-difference errors. Trace, Hermiticity, positivity, stationary
residuals, and errors against exact evolution are saved as diagnostics.

The dilation uses a five-level ancilla: reset label 0 plus four jump labels.
For each step, $K_{00}=\delta t H$, $K_{k0}=\sqrt{\delta t}V_k$ and
$K_{0k}=\sqrt{\delta t}V_k^\dagger$. Unsplit evolution exponentiates $K$;
Lie splitting applies the system then the four jump factors; symmetric
splitting surrounds the full system factor with jump half-factors in
forward and reverse order. The ancilla is traced/reset only after a
complete step. For $N=3$, $K$ is $40\times40$; a binary encoding would
require three ancilla qubits and six total qubits.

This first-order dilation approximates the Lindblad channel even when its
unitary is exactly exponentiated. Finite-step stationary bias is measured
against the physical stationary state and decreases under step refinement.
Symmetric unitary splitting does not make this a second-order physical-time
solver. This experiment runs classically and produces no hardware jobs.

## Run

From the repository root:

```bash
# All four methods, three preparations, and both reservoir controls.
.venv/bin/python Model3/Experiment1/simulation.py --controls

# Exact physical reference only, including both controls.
.venv/bin/python Model3/Experiment1/exact.py --controls

# Individual splitting studies (also include the exact and unsplit reference).
.venv/bin/python Model3/Experiment1/lie_trotter.py --max-dt 0.0125
.venv/bin/python Model3/Experiment1/strang_trotter.py --max-dt 0.0125

# Larger-chain Lie run with a single BLAS thread and the exact reference.
# Omit the additional unsplit dilation reference to save work.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  Model3/Experiment1/lie_trotter.py --n-qubits 7 --max-dt 0.1 --skip-unsplit

# Reverse the reservoirs or use arbitrary observation times.
.venv/bin/python Model3/Experiment1/exact.py --p-left 0.1 --p-right 0.9
.venv/bin/python Model3/Experiment1/exact.py --times 0.17 1.3 7.3 20

# Retain Model 2's J/4 normalization and recompute all targets.
.venv/bin/python Model3/Experiment1/exact.py --exchange-prefactor 0.25
```

Other options include `--n-qubits`, `--J`, `--delta`, `--gamma`, `--mode`,
`--t-final`, `--num-times`, and `--output-dir`. Rates are equal at the two
endpoints; reservoir populations may range from 0 to 1 inclusive.
`--controls` adds $(0.25,0.25)$ and the reversal of the selected pair.

Each run saves PNG, JSON, and compressed NPZ files under `results/`.
The CLI prints progress and elapsed time for each computational stage.
JSON also records stage timings. Lie/Strang modes include an unsplit
dilation reference by default; `--skip-unsplit` omits that extra comparison.
NPZ includes full sampled density matrices, populations, $Z$, $I$, $j$,
and the stationary density matrix and observables. Density arrays have
shape `(initial_state, time, row, column)`; observable arrays have shape
`(initial_state, time, site_or_bond)`.
Filenames encode method, chain length, reservoirs, $J$, $\Delta$, $\Gamma$,
and exchange prefactor. Times and step sizes are recorded in JSON but not
the filename; use separate output directories for refinement schedules.
The full density-matrix and dilation computations grow exponentially with
chain length.

## Saved benchmark

The default run through $t=20$, with 201 observation times and dilation
step limit 0.05, has been saved for all three reservoir pairs. The forward
run includes its [plot](results/all_N3_pL0.9_pR0.1_J1_D0_G2_c0.5.png),
[diagnostics](results/all_N3_pL0.9_pR0.1_J1_D0_G2_c0.5.json), and
[density matrices and observables](results/all_N3_pL0.9_pR0.1_J1_D0_G2_c0.5.npz).
Across all controls, the stationary residual is below $2\times10^{-16}$
and the exact magnetizations agree with the independent correlation
calculation within $2\times10^{-15}$.

For the forward run, the final observables from preparation `000` are:

| Method | $(Z_1,Z_2,Z_3)$ at $t=20$ | $(I_1,I_2)$ at $t=20$ |
|---|---|---|
| Exact Lindblad | $(-0.400000,0.000000,0.400000)$ | $(0.400000,0.400000)$ |
| Unsplit dilation | $(-0.406353,0.000000,0.406353)$ | $(0.394370,0.394370)$ |
| Lie dilation | $(-0.424618,-0.000127,0.410402)$ | $(0.389080,0.389430)$ |
| Symmetric dilation | $(-0.401091,0.000047,0.404622)$ | $(0.394368,0.394289)$ |

The finite-step dilation deviations persist at long times. Reduce
`--max-dt` to study their convergence toward the physical stationary state.
Measured step limits for absolute population error 0.001 at $N=3$ and $N=7$
are reported in [PopulationStepLimits.md](analysis/PopulationStepLimits.md).

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_model3_experiment1.py' -v
```

Tests compare all quoted stationary targets against the full Lindblad solve,
check both Hamiltonian normalizations and multiple chain lengths, verify
reflection under reservoir reversal and the continuity operator identity,
compare full trajectories with independent correlations, and check physical
density matrices and decreasing dilation errors under step refinement.
