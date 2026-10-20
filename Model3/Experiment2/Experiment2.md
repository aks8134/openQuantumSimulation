# Experiment 2: unequal-reservoir randomized-dilation scaling study

This classical-only experiment performs the same size/rate/timestep/trajectory
study as [Model 2 Experiment 3](../../Model2/Experiment3/Experiment3.md), but
with Model 3's unequal reservoirs and Hamiltonian convention:

$$
H=\tfrac J2\sum_{j=0}^{N-2}(X_jX_{j+1}+Y_jY_{j+1}),\qquad
p_L=0.9,\quad p_R=0.1,\quad \Gamma_L=\Gamma_R=\Gamma,\quad\Delta=0.
$$

`--p-left`, `--p-right`, and `--exchange-prefactor` are tunable. The default
$c=1/2$ is intentionally different from Model 2's $c=1/4$: hopping is $J$
here rather than $J/2$. Ratio optima and normalized-current errors therefore
depend on that convention. The default initial states are all zero, all
one, and a left-end excitation, as in Experiment 1.

## Construction and physical reference

The four jumps are left pump/loss and right pump/loss, with rates
$\Gamma p_L$, $\Gamma(1-p_L)$, $\Gamma p_R$, $\Gamma(1-p_R)$.
Default sampling probabilities are these rate fractions divided by two.
Every step draws one jump, scales its dilation coupling by $1/\sqrt{\mu_a}$,
traces/resets that two-level ancilla, and applies the full system factor.
The order is **jump then H** by default. It is neither deterministic
four-jump splitting nor an exact-reservoir exponential.

The covariance reduction is exact for the XX observables, even though
pumping populates multiple excitation sectors. It does **not** assume
Model 1's single-excitation initial-state reduction and does not reconstruct
the full generally non-Gaussian density matrix. Interactions are unsupported
by this efficient study; no `--delta` flag silently applies the wrong solver.

The physical stationary covariance is solved from

$$
AC_*+C_*A^\dagger+P=0,\quad
A=-ih-\tfrac\Gamma2(\Pi_0+\Pi_{N-1}),\quad
h_{j,j+1}=J,\quad P=\Gamma(p_L\Pi_0+p_R\Pi_{N-1}).
$$

It has nonzero nearest-neighbor coherences and current. In particular,

$$
I_* = \frac{2J^2\Gamma}{\Gamma^2+4J^2}(p_L-p_R),\quad
n_{0,*}=p_L-I_*/\Gamma,\quad
n_{N-1,*}=p_R+I_*/\Gamma,
$$

and interior populations are $(p_L+p_R)/2$. For $J=1,\Gamma=2$, this
reproduces $n_*=(0.7,0.5,0.3)$ and $I_*=0.4$ at $N=3$.

The exact reference is continuous Lindblad evolution,
$C(t)=C_*+e^{tA}(C(0)-C_*)e^{tA^\dagger}$. The study checks sampled-channel
stationary bias, not merely population convergence to a presumed product
state. Unlike the equal-reservoir rate-weighted construction, the physical
unequal-reservoir stationary covariance is generally **not** an exact fixed
point of the finite-step mean dilation.

## Metrics and important limitations

All settings, search logic, checkpointing, derivations, and limitations are
documented in [Model 2's study description](../../Model2/Experiment3/Experiment3.md).
In particular:

- $r=\Gamma/J$, $\tau=Jt$, and $s=J\Delta t$. The study seeks large passing
  $s$ and then the smallest tested $R$ at the chosen rate ratio.
- Approximation error is the maximum **accumulated** population error and
  current error/$J$ over preparations/sites/bonds at algorithmic nodes;
  it is not one-step error. Defaults: dynamic tolerance 0.01, steady
  tolerance 0.001, timing tolerance 5%.
- Permanent exact settling uses $q\max\|C-C_*\|_F$, here $q=2$.
  This bounds population and current/$J$ errors, but is **not many-body
  trace distance** or simply the local-observable threshold crossing.
  Mean-channel settling includes its stationary bias relative to $C_*$.
- Finite-$R$ random pumping/loss keeps fluctuating. Every seed is tested
  over a finite horizon (default 1.5 times exact settling) only. By default,
  the local steady tolerance must hold after the exact settling deadline
  plus the timing allowance until that horizon. The explicit
  `--no-finite-R-steady-check` disables that tail test, not the transient
  accuracy tests. There is no permanent finite-$R$ or confidence guarantee.
- Half the dynamic budget is allocated to averaged bias and half to
  sampling. Both errors against the exact evolution and against the
  averaged channel are saved.
- Exact full-system propagation is the default. `--system-formula even-odd`
  is a deliberate alternative approximate method, not an undocumented
  change to the reference. Signed dilation cosines are preserved.
- Recommendations are grid/local, not a global optimum. Large finite-$R$
  costs and unresolved searches remain visible. Classical step counts are
  not native circuit gate/depth predictions.

## Run

Full default study:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
  Model3/Experiment2/study_randomized_scaling.py
```

With the relaxed tolerances recently used for Model 1:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
  Model3/Experiment2/study_randomized_scaling.py \
  --steady-tolerance 0.01 --dynamic-tolerance 0.1
```

For a smaller first study or reversed reservoirs:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
  Model3/Experiment2/study_randomized_scaling.py \
  --sizes 3 7 10 --ratios 1 2 3 4 \
  --steps 0.1 0.05 0.02 0.01 --R-values 4 16 64 256 \
  --p-left 0.1 --p-right 0.9 \
  --steady-tolerance 0.01 --dynamic-tolerance 0.1
```

All outputs live under this experiment's `study/scaling_<configuration>/`:
`study.json`, `Study.md`, and the size/rate/workload line figures. New
configurations do not overwrite prior studies. Each completed case is
checkpointed; repeat identical arguments plus `--resume` to continue.
An interrupted incomplete case restarts. Extend unresolved ensembles with
`--extend-R-values 512 1024 2048`; increase `--step-refinements` for more
local refinement. Both options can be used with `--resume`.

The physical target horizon defaults to 20 and the rate margin to 10%.
`--target-time`, `--rate-margin`, `--horizon-factor`, `--time-cap`,
`--step-cap`, `--memory-limit-mb`, `--progress-every`, `--seeds`, and all
search grids are configurable. `--no-plots` skips figures. No account,
credentials, simulator, or hardware job is required.

The implementation lives in [reservoir_scaling.py](../../library/reservoir_scaling.py)
and does not import other experiment scripts. Only small validation studies
have been executed during implementation; no full grid result is claimed.
