# Model 3 Experiment 2: randomized XX reservoir scaling study

Status: `running`. Classical covariance calculation only; no circuit or hardware execution.

H = 0.5 J sum(XX+YY), Delta=0, pL=0.9, pR=0.1.
Sampling: rate-weighted; order: jump-first; Hamiltonian: exact.

## Definitions and acceptance

$r=\Gamma/J$, $\tau=Jt$, $s=J\Delta t$. Preparations are all-zero, all-one, and a left-end excitation.

Approximation error is $\max_{k,\mathrm{prep},j}\{|n_j^{app}-n_j^{ex}|,|I_j^{app}-I_j^{ex}|/J\}$, accumulated at every algorithmic node, not the one-step local error.
Dynamic tolerance=0.2; averaged bias and finite-R sampling budgets are each 0.1. All 5 seeds must pass.

The exact settling criterion is $q\max_{\mathrm{prep}}\|C(t)-C_*\|_F\le\epsilon_{ss}$, where $q=\max(1,4c)$ and $H=cJ\sum(XX+YY)$. It bounds both populations and current/J, including at every later time. This is a conservative covariance/observable criterion, NOT full-state trace distance, and NOT Model 2 Experiment 1's population-only criterion.
Steady tolerance=0.01; timing tolerance=5% (half that budget for the recommended averaged step).
For the mean dilation use $b+q\max\|C_k-C_s\|_F$, where $C_s$ is its affine fixed point and b is its local stationary bias relative to the physical C*. Affine-map residual/positivity checks are saved.

Finite-R trajectories do not settle permanently: pumping/loss choices keep fluctuating. They are tested to 1.5 times the exact dimensionless settling time. Additionally, every checked node after the exact settling time plus its timing allowance must meet the local steady tolerance through that horizon. No infinite-time or probabilistic guarantee follows for finite R.

## Recommendations

| N | Gamma/J | Exact Jt_ss bound | Largest tested s (infinite R) | Recommended s | Smallest passing tested R | J | Gamma | Delta t | Steps to target |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 0.64843504 | 0.006875 | 0.004 | unresolved | 0.035663927 | 0.28531142 | 0.11215815 | 179 |
| 4 | 3.9139319 | 7.2468724 | 0.01125 | 0.005625 | unresolved | 0.39857798 | 1.5600071 | 0.014112671 | 1418 |
| 7 | 3.0301499 | 41.607882 | 0.01 | 0.005625 | 256 | 2.2884335 | 6.9342966 | 0.0024580133 | 8137 |
| 10 | 2.6866798 | 123.39437 | 0.01 | 0.005 | 256 | 6.7866902 | 18.233664 | 0.00073673615 | 27147 |
| 15 | 2.4393332 | 420.25874 | 0.01 | 0.005 | unresolved | 23.114231 | 56.383311 | 0.00021631696 | 92457 |

Rates place the *exact covariance-bound* settling time before the target with the configured margin. They do not remove the dimensionless step cost. The mean settling time is also included in rate selection. Finite-R steady claims are limited to the validated horizon/seeds.

## Search audit and limitations

Sizes: [2, 4, 7, 10, 15, 20, 30, 50]; ratio grid: [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 8]; s grid: [0.1, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.003, 0.002, 0.001, 0.0005]; R grid: [4, 8, 16, 32, 64, 128, 256]; supplemental R: []; seeds: [29, 101, 307, 509, 701].
Ratios are ranked by exact settling time, optionally locally refined. The promising ratios are compared by mean step count; s is refined locally at the selected ratio. This is not a global optimization over all r,s,R or a proof of minimal total R-times-step cost. Dilation resonances may make accuracy nonmonotonic in s; bisection is a local heuristic only.
Cases save after completion; interruption restarts an unfinished case. Cap or fixed-point-solve failures are unresolved, not successful. Later seeds skipped after a rejection are explicitly untested.
The Frobenius norm contracts under the exact dissipative propagator and the mean homogeneous map (an average of A C A^T with each A a contraction). A bias-plus-two-decaying-envelopes bound can certify an unevaluated mean-error tail. Reported peaks then cover evaluated nodes only. Bounds and crossings are floating-point results, not interval-arithmetic certificates.
The covariance reduction is exact for these XX observables and endpoint channels, not for the full generally non-Gaussian many-body density matrix. Delta!=0 is deliberately unsupported. Signed dilation cosines are preserved; jumps are not replaced with exact reservoir exponentials.
The exact-H option preserves Model 2 Experiment 2's system factor. Even-odd is an explicit optional change of the approximate method; the physical reference always exponentiates the full H.
Seeds use independent path streams, different from other drivers' row-major random draws. Step boundaries do not depend on a plotting grid. No measurement shot noise, routing, hardware gate depth or pulse fidelity is included.

[Complete checkpoint and cases](study.json).
