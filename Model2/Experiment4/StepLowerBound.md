# A 20-step lower bound when the equal boundary rate may be tuned

This argument applies to **N=10, p=1/4, half-mixed initialization, Delta=0,
one rate-weighted boundary collision per step**, with the fixed **0.016
mean allowance**. Both boundary total rates are equal. It allows arbitrary
timesteps and number-conserving one-body Hamiltonian unitaries, so it includes
the ordinary and warped cosines in this experiment. It does not prove that
20 steps are achievable at the earlier fixed Gamma/J=2.

Write the covariance as

\[
C_k=pI+\delta K_k,\qquad \delta=1/2-p=1/4,\qquad K_0=I.
\]

Because the rate-weighted equal-reservoir mean preserves pI, K evolves by
the homogeneous map. Up to a unitary conjugation before or after the bath,
the bath action is `K -> W circle K`, where

\[
W=\tfrac12 r_Lr_L^\dagger+\tfrac12 r_Rr_R^\dagger.
\]

The vector r_L is one at every site except the left boundary, where it is
`a=cos(sqrt(2 Gamma dt))`; r_R has the analogous right boundary entry.
Thus W is positive semidefinite, with diagonal one in the interior and
`(1+a^2)/2 >= 1/2` at each boundary. Signed cosines are allowed.

Oppenheim's determinant inequality gives
`det(W circle K) >= det(K) product_i W_ii`; Hadamard's inequality gives
`det(K) <= product_i K_ii`. Both statements for positive semidefinite
matrices are stated as Theorems 1.1–1.2 in
[Zhang and Ding, Czechoslovak Mathematical Journal 59 (2009), 197–206](https://www.dml.cz/bitstream/handle/10338.dmlcz/140473/CzechMathJ_59-2009-1_14.pdf).
The following application to Model2 is our derivation.

Unitary conjugation preserves the determinant. Therefore each step obeys

\[
\det K_{k+1}\ge\tfrac14\det K_k,
\qquad \det K_M\ge4^{-M}.
\]

Since K is positive semidefinite,

\[
(\max_i K_{ii})^N\ge\prod_i K_{ii}\ge\det K_M\ge4^{-M}.
\]

The maximum population error alone is consequently at least

\[
\max_i|n_i-p|\ge\delta\,4^{-M/N}.
\]

For M=19 this lower bound is approximately **0.01795**, above 0.016. Hence
**at least 20 steps are necessary** for the mean criterion, independently
of the cosine parametrization or optimizer. The argument concerns the
deterministic branch average; a fortuitous finite sample cannot substitute
for the mean acceptance condition.

## A matching construction

With J=1, choose

\[
\Gamma=\pi/8,\qquad \Delta t=\pi
\]

at every step. Model2's one-body exchange angle is
`2 c J dt=pi/2` for c=1/4, so the even/odd Hamiltonian layers transport
occupations by swaps, with phases irrelevant to diagonal covariances.
The bath angle is also `sqrt(2 Gamma dt)=pi/2`, making a=0. The mean bath
action halves the deviation at each endpoint.

For the ten-site even/odd permutation, every transported occupation visits
the two boundaries four times in twenty steps. The resulting covariance is
exactly

\[
K_{20}=I/16,\qquad C_{20}=\tfrac{17}{64}I=0.265625I.
\]

The maximum population error is **1/64=0.015625**, and every current is zero.
All later ideal mean deviations remain bounded by 1/64 under contractive
rate-weighted steps preserving pI. The original affine solver, the separate
explicit-branch verifier and the accelerated implementation reproduce this
construction to floating-point precision.

Thus **20 is the minimum mean preparation count in this specified channel
class when the equal boundary rate may be chosen**, under the fixed
0.016/0.004 budget allocation. The construction does not promise accurate
continuous-time transients, many-body state distance, finite-shot confidence
or hardware fidelity. Finite-path sampling is evaluated separately and
requires a larger ensemble than the earlier 80-step example.
