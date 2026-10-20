# Experiment 3: randomized-dilation scaling study

This is a **classical-only** study of the equal-reservoir XX model. It sweeps
chain length, the boundary/hopping rate ratio, the dimensionless dilation
timestep, and the number of independent random trajectories. It does not run
Aer or hardware and does not change Experiments 1 or 2 or their saved results.
The shared numerical implementation is [reservoir_scaling.py](../../library/reservoir_scaling.py).

## Physical model and randomized method

The default conventions match Experiment 2:

$$
H=cJ\sum_{j=0}^{N-2}(X_jX_{j+1}+Y_jY_{j+1}),\quad c=\tfrac14,
\qquad p_L=p_R=0.25,\qquad \Delta=0.
$$

The four jumps, in sampling order, are

$$
\sqrt{\Gamma p_L}\sigma^+_0,\quad
\sqrt{\Gamma(1-p_L)}\sigma^-_0,\quad
\sqrt{\Gamma p_R}\sigma^+_{N-1},\quad
\sqrt{\Gamma(1-p_R)}\sigma^-_{N-1}.
$$

Both boundary total rates are the same tunable $\Gamma$. The initial
preparations are all zero, all one, and a left-end excitation. They are **not**
restricted to Model 1's vacuum-plus-single-excitation subspace.

Each step samples exactly one jump $V_a$ with probability $\mu_a$ and uses
the two-level reset-zero dilation with coupling $V_a/\sqrt{\mu_a}$. The
Hamiltonian is applied once at its original strength. Defaults are
rate-weighted sampling, $\mu=(p_L,1-p_L,p_R,1-p_R)/2$, and **jump then H**.
`--policy uniform` and `--order hamiltonian-first` expose alternatives.

The boundary channel multiplies incident covariances by
$\cos\theta_a$ and supplies $\sin^2\theta_a$ only for a pumping branch, where
$\theta_a=\sqrt{\Gamma f_a\Delta t/\mu_a}$. Signed cosines are retained;
this is the sampled dilation, not an exact-reservoir exponential. Different
jumps are random alternatives, not four deterministic factors per step.

`--system-formula exact` (default) preserves Experiment 2's full system
factor. It is inexpensive to exponentiate the $N\times N$ hopping matrix
classically; this does **not** claim a constant-cost circuit realization.
`--system-formula even-odd` optionally applies even and then odd bond
rotations once per step, adding that product-formula error. The exact
Lindblad reference always uses the full Hamiltonian.

## Exact reference and settling criterion

The XX covariance $C_{ij}=\langle c_j^\dagger c_i\rangle$ obeys

$$
\dot C=AC+CA^\dagger+P,\qquad
A=-ih-\tfrac\Gamma2(\Pi_0+\Pi_{N-1}),\quad
h_{j,j+1}=2cJ,\quad
P=\Gamma(p_L\Pi_0+p_R\Pi_{N-1}).
$$

Solve $AC_*+C_*A^\dagger+P=0$ and evolve exactly as
$C(t)=C_*+e^{tA}(C(0)-C_*)e^{tA^\dagger}$. Equal reservoirs give
$C_*=pI$. The code uses an equivalent real gauge to reduce storage and
computation. Covariance closure is exact for these observables and sampled
endpoint channels, but does not reconstruct a generally non-Gaussian
many-body density matrix. Interactions $\Delta\ne0$ are deliberately not
offered by this reduced study.

Populations and excitation currents are

$$
n_j=C_{jj},\qquad I_j=4cJ\operatorname{Im}C_{j,j+1}.
$$

Unlike Model 1, the stationary state is not vacuum. For a common criterion
across equal and unequal reservoirs, this study defines a conservative
permanent observable-settling bound

$$
d(t)=q\max_{\rm preparation}\|C(t)-C_*\|_F,\qquad q=\max(1,4c).
$$

Every population and current/$J$ deviation is at most $d(t)$. Because
$A+A^\dagger\le0$, $e^{tA}$ contracts, so $d(t)$ is nonincreasing. The
reported exact settling time is the crossing of $d(t)\le\epsilon_{ss}$.
It is **not many-body trace distance**, and is stronger than the
population-only settling criterion in Experiment 1. Crossings from those
different criteria must not be compared as if they were identical.

The averaged dilation is an affine covariance map with homogeneous part
$\mathcal M(C)=\sum_a\mu_a A_aCA_a^\dagger$. Its own fixed point $C_s$
may differ from $C_*$. Its permanent settling bound is

$$
b+q\max_{\rm preparation}\|C_k-C_s\|_F,\qquad
b=\max_j\{|(C_s-C_*)_{jj}|,\ |I_j(C_s-C_*)|/J\}.
$$

Each $A_a$ is a contraction, so this is a nonincreasing upper bound. The
fixed-point residual and covariance physicality are checked and recorded.
Bounds use ordinary floating-point numerics, not interval certificates.

## Study axes and acceptance

Use dimensionless coordinates $r=\Gamma/J$, $\tau=Jt$, and $s=J\Delta t$.
The default grids match Model 1's study: sizes 2, 4, 7, 10, 15, 20, 30, 50;
ratios 0.25 through 8; steps 0.1 through 0.0005 (including smaller steps
needed for nonzero stationary bias); $R=4,8,\ldots,256$;
five seeds 29, 101, 307, 509, 701. All grids and tolerances are configurable.

The accumulated approximation-error metric is

$$
E=\max_{k,\,\mathrm{preparation},\,j}
\left\{|n_j^{\rm approx}(k\Delta t)-n_j^{\rm exact}(k\Delta t)|,
\frac{|I_j^{\rm approx}(k\Delta t)-I_j^{\rm exact}(k\Delta t)|}{J}\right\}.
$$

It is **not error per step**. Every algorithmic node is checked. Mean-channel
tails can be bounded by the stationary bias plus two contractive envelopes;
an unevaluated tail is accepted only when that bound is within the budget.
Reported numerical peaks cover evaluated nodes, and the tail bound is saved.
No claim is made about between-node interpolations.

Defaults are dynamic tolerance 0.01, steady tolerance 0.001, relative timing
tolerance 5%, and target physical time 20. Half the dynamic budget is
reserved for mean bias and half for trajectory sampling; the recommended
mean step also reserves half the timing budget.

Unlike the loss-only model, individual trajectories keep fluctuating under
random pumping and loss, even after their average approaches stationarity.
Finite-$R$ means are tested at every node to 1.5 times the exact settling
time, against the exact solution and the averaged channel. By default they
must also remain locally within the steady tolerance at all nodes after
the exact settling time plus its 5% allowance until the end of that window.
`--no-finite-R-steady-check` removes this **finite-window** tail test and
selects $R$ solely by approximation/sampling accuracy. Neither mode claims
permanent finite-$R$ convergence or a probabilistic confidence guarantee.

Ratios are ranked by exact settling time and optionally locally refined.
The best few are compared by mean settling-step count, then the selected
ratio's timestep is locally refined twice. $R$ is tested in ascending order
until all seeds pass. These are grid/local recommendations, not a globally
optimal joint $r,s,R$ solution; large-step resonances can make passing
regions nonmonotonic. An unresolved $R$ is reported, never guessed.

Recommended physical rates are

$$
J=\frac{\max(m\tau_{ss}^{\rm exact},\tau_{ss}^{\rm mean})}{T_{target}},
\quad \Gamma=rJ,\quad \Delta t=s/J,\qquad m=1.1\ \text{by default}.
$$

Increasing both rates compresses physical time but does not reduce the
dimensionless substep workload. Reported steps exclude measurement bases,
shots, routing, transpilation and hardware noise.

## Commands and outputs

Run the full study:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
  Model2/Experiment3/study_randomized_scaling.py
```

For steady tolerance 0.01 and approximation tolerance 0.1:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
  Model2/Experiment3/study_randomized_scaling.py \
  --steady-tolerance 0.01 --dynamic-tolerance 0.1
```

Start with a smaller study if desired:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
  Model2/Experiment3/study_randomized_scaling.py \
  --sizes 3 7 10 --ratios 1 2 3 4 \
  --steps 0.1 0.05 0.02 0.01 --R-values 4 16 64 256 \
  --steady-tolerance 0.01 --dynamic-tolerance 0.1
```

Output goes into a configuration-hashed `study/scaling_.../` directory:
atomic `study.json`, a self-contained `Study.md` report, and line figures
`scaling.png`, `ratio_landscape.png`, and `target_workload.png`. Changed
tolerances/models/grids create a different directory and leave old results
intact. Each completed case is saved; an interrupted in-progress case is
restarted, not resumed within its timestep loop.

Repeat **the same command** with `--resume` to use completed cases. Add
`--extend-R-values 512 1024 2048` to extend an unresolved $R$ search without
repeating exact or averaged cases. `--step-refinements` can also increase
on resume. `--no-plots` suppresses figures; a cached resume without it can
generate them later. `--progress-every 10000` prints in-case progress.
Time, step and memory caps are configurable and reported explicitly.
Default grids can be expensive, especially for large $N$ and large $R$;
the new experiment has been smoke-tested, not exhaustively executed.

Use `--help` for the complete interface. The two new studies share a new
functional numerical module, not another experiment's script.

Validation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
  -m unittest discover -s tests -p 'test_reservoir_scaling.py' -v
```
