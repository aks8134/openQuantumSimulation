# A population-error bound for randomized boundary dilation

The bound below certifies the **average randomized channel**, with
independently sampled jumps at each step. It applies to the equal-reservoir
XX benchmark, the three preparations `000...`, `111...`, `100...`, and every
integer step through T=12. It includes the actual first-order Ding–Li–Lin
sine/cosine dilation error. It does not bound shot noise, individual random
sequences, circuit compilation errors, or the full density-matrix distance.

The parameters are J=10, Gamma=20, p=1/4, Delta=0, and exchange prefactor
c=1/4. Sampling probabilities are (1/8,3/8,1/8,3/8), in left-pump,
left-loss, right-pump, right-loss order. The Hamiltonian factor follows the
single sampled jump. This is the same algorithm used in the existing
Experiment 2 comparison.

## 1. Exact finite-dimensional maps

Let C_ij = <c_j^dagger c_i>. A diagonal gauge with D_jj=i^j transforms C
into a real covariance matrix for these preparations. Its single-particle
Hamiltonian generator K has upper diagonal +kappa and lower diagonal
-kappa, where kappa=2cJ=5. Write

$$
A=K-\frac{\Gamma}{2}(\Pi_L+\Pi_R),\qquad
G=A\otimes I+I\otimes A.
$$

For row-major vectorization x=vec(C-pI), the exact continuous dynamics is
x(t)=exp(tG)x(0). Hence the exact one-step map is Q=exp(tau G), with
tau=T/M. This exponential is a reference in the certificate, not a change
to the reservoir implementation.

The sampled dilation angle is theta=sqrt(2 Gamma tau). Define U=exp(tau K)
and diagonal R_L,R_R, each equal to identity except that its boundary entry
is cos(theta). Summing pumping and loss probabilities at each boundary
gives probability 1/2. The affine source term cancels when centering at pI,
so the implemented **mean** step is exactly

$$
P=\frac12\left[(U R_L)\otimes(U R_L)
                    +(U R_R)\otimes(U R_R)\right].
$$

Thus the population error after m steps is a selected diagonal component
of (P^m-Q^m)x(0). Normalized excitation current is -4c times the selected
upper off-diagonal entry; here -4c=-1. No Gaussian-state assumption is used:
closure of these second moments is an operator identity for this XX model.

Both P and Q contract the Euclidean vector norm, equivalently the
covariance Frobenius norm. For P, each U R_b has spectral norm <=1;
Kronecker products and their average therefore have norm <=1. For Q,
G+G^T is negative semidefinite. The corresponding maps also contract the
matrix spectral norm. Every centered initial covariance has spectral norm
<=3/4, so every entry of P^m x(0) and Q^m x(0) has magnitude <=3/4.

## 2. A bound on a whole block of steps

Take a block starting at m and of length L. Set N_P=P-I and N_Q=Q-I.
For an observable row r and integer offset 0<=k<=L, the binomial theorem
gives

$$
f(m+k)=\sum_{\ell=0}^{d}{k\choose\ell}a_\ell+R_k,
\qquad
a_\ell=r\left(N_P^\ell P^m-N_Q^\ell Q^m\right)x(0).
$$

Here binomial(k,l)=0 for l>k. With z_P >= ||N_P||_infinity and
z_Q >= ||N_Q||_infinity, a uniform remainder bound is

$$
|R_k|\le B_{\rm tail}
=\frac34\sum_{\ell=d+1}^{L}{L\choose\ell}
    \left(z_P^\ell+z_Q^\ell\right).
$$

The sum is empty when d>=L. It is evaluated with exact rational arithmetic
using outward upper bounds for z_P,z_Q. For currents multiply the bound
by |4c|, which is one here.

The truncated sum is a real polynomial q(k). Convert q(Lu) to Bernstein
form of degree D>=d:

$$
q(Lu)=\sum_{j=0}^{D}b_j{D\choose j}u^j(1-u)^{D-j}.
$$

For 0<=u<=1, the basis functions are nonnegative and sum to one. Therefore

$$
\boxed{\quad |f(m+k)|\le\max_j|b_j|+B_{\rm tail}\quad}
$$

for **every integer k in the block**. The continuous polynomial enclosure
is a mathematical tool; it does not assert an error bound for an additional
physical evolution between algorithm steps. Taking the maximum over blocks,
sites/bonds, and the three initial states gives the reported bound.

This retains signed contributions when forming a_l and the Bernstein
coefficients. A generic bound such as m||P-Q||||x(0)|| loses these
cancellations and is much larger. The block enclosure is not an empirical
fit or an extrapolation from the old simulation errors.

## 3. Enclosing arithmetic and propagating uncertainty

`population_bound.py` represents each scalar or matrix by a binary64 center
and a componentwise outward radius. Rational constants are converted by
comparing their exact Fraction values. Products use the standard
dot-product error bound gamma_k=k*u/(1-k*u), with u=2^-53 and conservative
operation counts. Positive radius calculations are padded and rounded
outwards; a 1e-300 absolute padding covers underflow in the stated range.
Norm bounds include rounding allowance.

The physical U and Q are enclosed with degree-24 matrix Taylor series. If
z>=||B||_infinity and z<d+2, their omitted matrix terms have bound

$$
\left\|e^B-\sum_{k=0}^{d}\frac{B^k}{k!}\right\|_\infty
\le \frac{z^{d+1}}{(d+1)!}\frac{1}{1-z/(d+2)}.
$$

The cosine is enclosed using the alternating expansion
cos(sqrt(x))=sum_k (-x)^k/(2k)! for positive x=2 Gamma tau, plus its next-term
remainder. At truncation degree d, the omitted terms decrease when
x < (2d+3)(2d+4); this is the enforced condition. It also permits the larger
time steps needed for the tolerance sweep. The certificate uses no numerical eigensolver, logarithm,
trigonometric library function, or unbounded exponential routine.

At block boundaries, states are propagated with enclosed block powers.
Known contraction prevents uncertainty from growing exponentially due to
interval wrapping. For true contractions A,B and approximations a,b with
Frobenius error radii e_a,e_b,

$$
\|AB-ab\|_F\le e_a+e_b+e_a e_b.
$$

The implementation adds the matrix-multiplication rounding bound. For a
true transition S, approximate transition s and state x_hat with error e_x,
the next propagated state radius is bounded by

$$
e_{\rm next}\le e_x+\|S-s\|_F\|\hat x\|_2
                         +e_{\rm dot}.
$$

That radius is included in the polynomial coefficients. For N=7, M=5441,
the final propagation radius is below 2.01e-10. All reported upper bounds
already include arithmetic and truncation uncertainty.

The certificate is conditional on standard IEEE binary64 rounding and
dot-product arithmetic. It is a model-specific computer-assisted bound,
not a machine-checked formal proof or a universal closed-form step-count
theorem for arbitrary Lindbladians.

## 4. A counterexample when the criterion fails

A Bernstein upper bound exceeding tolerance alone does not establish a
violation. The script also selects an actual integer step and encloses its
polynomial value and remainder. This gives a lower witness for the true
population error. At N=7, M=5440, initial all-ones, site 1, step 43,

$$
t=\frac{12\cdot43}{5440}\approx0.09485294,
\qquad |\Delta p_1(t)|\ge0.003000485621>0.003.
$$

At M=5441 the upper certificate is below 0.003 for every step and site.
These adjacent checks locate a local threshold. They do not prove
monotonicity in M or exclude every possible smaller count outside the
checked candidates.

## 5. Validation

Five tests check rational matrix-product enclosures, the independent
binomial-to-Bernstein polynomial identity, agreement of the real maps with
the original complex channel, coverage of all 6001 observations for both
chain sizes, and an independently evaluated violating witness. The
independent comparison uses the original complex eigensolver and exact XX
reference, while the certificate uses Taylor enclosures and polynomial
bounds.

The paper's general average-channel theorem remains applicable; its
lambda^2 T^2/epsilon scale is 76,800,000 here, before its unspecified
constant. The much tighter result comes from restricting the guarantee
to this model, these observables, and these preparations. See
[Theorem 4 and Corollary 5](https://arxiv.org/html/2407.06594v5).
