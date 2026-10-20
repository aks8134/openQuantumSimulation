# N=20 final-population preparation with 100 DLL resets

Method: independently rate-weighted randomized single-jump Ding–Li–Lin dilation, jump first, followed by the exact XX Hamiltonian factor. p=1/4.

Only the final populations are optimized. No transient accuracy is required. The results describe the exact ensemble mean up to floating-point roundoff; they do not include finite random-sequence sampling or measurement shots.

The three preparations are all zeros, all ones, and a leftmost excitation. Positive-semidefinite ordering proves that the all-ones preparation realizes the largest population error at every step.

Best found: J*tau=3.14168147074, Gamma*tau=2.04896696474, cos(theta)=-0.438149577847, final error=0.074014261899.

Required error: 0.001. Pass: False.

Equivalent physical parameters at Gamma=20: tau=0.102448348237, J=30.6660041358. This setting fails the 100-step target.

Continuing this same channel reaches 0.001 at step 549. This is a measured count for that setting, not a minimum over all settings.

Reference implementation agreement: 1.61e-15.

| Upper J*tau | Seed | Best J*tau | Gamma*tau | Final population error |
|---:|---:|---:|---:|---:|
| 10 | 24 | 3.1416814 | 2.0489736 | 0.074014262 |
| 10 | 71 | 3.1416813 | 2.0489672 | 0.074014262 |
| 10 | 109 | 3.1416817 | 2.0489672 | 0.074014262 |
| 100 | 24 | 33.964141 | 1.9733084 | 0.075884092 |
| 100 | 71 | 52.640858 | 1.9815224 | 0.076279242 |
| 100 | 109 | 3.1416815 | 2.048967 | 0.074014262 |
| 1000 | 24 | 764.63061 | 2.0095975 | 0.07489346 |
| 1000 | 71 | 119.31863 | 2.0133949 | 0.074656261 |
| 1000 | 109 | 665.82244 | 2.021661 | 0.074665888 |

This search does not establish a global minimum or rule out other DLL formulas, nonconstant timesteps, or engineered schedules. These large rotation angles are a final-state preparation experiment, not a validated approximation to the Lindblad transient.

Uniformly scaling J and Gamma and inversely scaling tau leaves the channel unchanged and cannot improve this result.
