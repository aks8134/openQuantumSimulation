# Population error and maximum physical step

Parameters are those selected for Model 3: $J=1$, $\Delta=0$,
$\Gamma_L=\Gamma_R=2$, $p_L=0.9$, $p_R=0.1$, and exchange prefactor
$c=1/2$. These limits concern the existing first-order shared-ancilla
dilation with Lie or symmetric (Strang) splitting.

## Error over the transient trajectory

The absolute population error is

$$
\epsilon_n(\delta t)=\max_{s,j,k:\,0\le k\delta t\le20}
\left|\langle n_j\rangle_{s,\mathrm{dilation}}(k\delta t)
-\langle n_j\rangle_{s,\mathrm{Lindblad}}(k\delta t)\right|.
$$

Preparations $s$ are all zeros, all ones, and only the leftmost site excited.
Each run uses a fixed physical step. Threshold checks inspect every step;
the coarse sweep initially sampled at approximately 0.02 time intervals.
Population error 0.001 corresponds to magnetization error 0.002.

| Chain length | Approximate largest Lie step | Approximate largest Strang step |
|---|---:|---:|
| $N=3$ | 0.00314 | 0.00573 |
| $N=7$ | 0.00303 | 0.00440 |

Values in this table are numerical threshold estimates, rounded from
pass/fail brackets. Slightly smaller steps should be used for a strict
inequality; 0.00303 for $N=7$ Lie exceeds 0.001 by about $1.45\times10^{-7}$.
The measured brackets are:

| Chain/method | Passing step | Error at passing step | Failing step | Error at failing step |
|---|---:|---:|---:|---:|
| $N=3$, Lie | 0.00314 | 0.000998705 | 0.00315 | 0.001001884 |
| $N=3$, Strang | 0.00572 | 0.000998362 | 0.00574 | 0.001001858 |
| $N=7$, Lie | 0.00300 | 0.000990248 | 0.00303 | 0.001000145 |
| $N=7$, Strang | 0.00439 | 0.000998854 | 0.00440 | 0.001001130 |

A practical common choice for both lengths is Lie `--max-dt 0.0030`
and Strang `--max-dt 0.0043`. The latter gives a measured $N=7$ error
of 0.000978361. For $N=3$ alone, Strang 0.0055 also leaves a margin
below its measured threshold.

The worst Lie error is at site 1 from the all-zero preparation, near
$t=1.13$ ($N=3$) or $t=1.26$ ($N=7$). The worst Strang error is from
the all-one preparation, at site 3 near $t=0.59$ for $N=3$, and at
site 2 near $t=3.47$ for $N=7$. This interior transient explains why
the longer-chain Strang limit is smaller despite similar stationary bias.

These steps refer to a fixed actual step. The main experiment subdivides
each observation interval equally, so `--max-dt` may result in smaller
actual steps. Thresholds are specific to the parameters, preparations,
time interval, and population observable checked here.

## Stationary populations only

The separate stationary error is
$\max_j|\langle n_j\rangle_{\Phi_{\delta t},*}-\langle n_j\rangle_{L,*}|$.
The fixed point of each finite-step channel was solved numerically, rather
than inferred from a possibly unrelaxed finite-time state. The 0.001 limits are:

| Chain length | Lie step limit | Strang step limit |
|---|---:|---:|
| $N=3$ | 0.00402473 | 0.02099058 |
| $N=7$ | 0.00402472 | 0.02155023 |

For a stationary-only study, steps 0.004 (Lie) and 0.020 (Strang) are
below both chain lengths' measured limits. Transient errors at these steps
can exceed 0.001. Both methods retain first-order physical-step error in
this dilation construction.

Reaching the physical stationary state also requires enough evolution time.
The exact $N=7$ solution's maximum population deviation from the stationary
profile over the three preparations is 0.0315403 at $t=20$, 0.00371905 at
$t=40$, 0.0000456790 at $t=80$, and $5.56\times10^{-7}$ at $t=120$.
Reducing the time step alone does not remove that relaxation error.

## Reproduction and validation

[population_steps.py](population_steps.py) exponentiates each individual
jump factor through its exact two-level rotation formula and propagates
density matrices in excitation-number blocks. It uses the same five-level
shared ancilla as the main experiment. No Gaussian approximation is applied
to the dilation. Exact reference populations come from the XX correlation
equations evaluated at the same physical times.

Three tests passed: every Kraus element agrees with the original dense
exponentiation for $N=3$ and $N=7$, block propagation agrees with full
density propagation on random number-conserving states, and batched exact
populations agree with the existing correlation solver through $t=120$.

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model3-matplotlib .venv/bin/python \
  Model3/Experiment1/analysis/population_steps.py \
  --dt 0.01 0.005 0.0025

OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model3-matplotlib .venv/bin/python \
  Model3/Experiment1/analysis/population_steps.py --n 7 --methods strang \
  --dt 0.0043 0.00439 0.0044 --sample-interval 0 --output /tmp/strang_N7_threshold.json

OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model3-matplotlib .venv/bin/python \
  Model3/Experiment1/analysis/population_steps.py --stationary-thresholds \
  --output Model3/Experiment1/analysis/stationary_thresholds.json

OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model3-matplotlib .venv/bin/python \
  -m unittest discover -s tests -p 'test_model3_population_steps.py' -v
```

Saved results: [coarse sweep](population_steps.json),
[$N=3$ threshold checks](threshold_N3_every_step.json),
[$N=7$ threshold checks](threshold_N7.json), and
[stationary limits](stationary_thresholds.json).
