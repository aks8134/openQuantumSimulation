"""Computer-assisted population/current bound for randomized XX dilation.

Use real covariance matrices in the gauge D_jj=i^j. The exact homogeneous
map is Q=exp(tau G); the weighted jump-first map is
P=(U R_L tensor U R_L + U R_R tensor U R_R)/2.
Both preserve C*=pI and contract the Frobenius norm of C-pI.

For each block of L steps, expand (I+N)^k in falling factorials, k=0..L.
Convert the error polynomial to Bernstein form: its range is enclosed by
its coefficients. A positive binomial tail bounds the omitted terms. Thus
the bound covers EVERY integer step, rather than a sampled-time curve.

Arithmetic uses outward radius bounds for binary64 operations and matrix
products. No sin, cos, exp, log, or eigensolver is used in the certificate:
small matrix exponentials and cos(sqrt(x)) use bounded Taylor series.
Assumes standard IEEE binary64 rounding and dot-product error bounds;
underflow is conservatively covered by a 1e-300 absolute padding.
"""

import argparse
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import comb, factorial
from pathlib import Path
import json
import time

import numpy as np


UNIT_ROUNDOFF = 2.0**-53
UNDERFLOW_PAD = 1e-300


def gamma(k):
    return (k * UNIT_ROUNDOFF) / (1 - k * UNIT_ROUNDOFF)


def up(x):
    return np.nextafter(np.asarray(x, float), np.inf)


def positive_padding(value, operations=16):
    return up(np.asarray(value) * (1 + 4 * gamma(operations)) + UNDERFLOW_PAD)


def rational_upper(value):
    converted = float(value)
    if Fraction.from_float(converted) < value:
        converted = np.nextafter(converted, np.inf)
    return float(converted)


def rational_lower(value):
    converted = float(value)
    if Fraction.from_float(converted) > value:
        converted = np.nextafter(converted, -np.inf)
    return float(converted)


def upper_norm(values, axis=None):
    """Outward upper bound for Euclidean/Frobenius norms."""
    values = np.asarray(values)
    # All inputs here are small and finite; squaring cannot overflow.
    size = values.size if axis is None else values.shape[axis]
    square_sum = positive_padding(np.sum(values * values, axis=axis), 3 * size + 4)
    return positive_padding(np.sqrt(square_sum), 4)


@dataclass
class Ball:
    center: np.ndarray
    radius: np.ndarray

    def __post_init__(self):
        self.center = np.asarray(self.center, float)
        self.radius = np.broadcast_to(np.asarray(self.radius, float), self.center.shape).copy()
        if np.any(self.radius < 0) or not np.all(np.isfinite(self.center)) or not np.all(np.isfinite(self.radius)):
            raise ValueError("invalid interval radius or nonfinite value")

    @classmethod
    def exact(cls, value):
        return cls(value, 0.0)

    @classmethod
    def rational(cls, value):
        value = Fraction(value)
        center = float(value)
        difference = abs(Fraction.from_float(center) - value)
        return cls(center, rational_upper(difference))

    @property
    def absolute_upper(self):
        return up(np.abs(self.center) + self.radius)

    def __add__(self, other):
        center = self.center + other.center
        radius = self.radius + other.radius + gamma(2) * (np.abs(self.center) + np.abs(other.center))
        return Ball(center, positive_padding(radius))

    def __neg__(self):
        return Ball(-self.center, self.radius)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        center = self.center * other.center
        radius = (np.abs(self.center) * other.radius + self.radius * np.abs(other.center) +
                  self.radius * other.radius + gamma(2) * np.abs(self.center) * np.abs(other.center))
        return Ball(center, positive_padding(radius))

    def scale(self, rational):
        return self * Ball.rational(rational)

    def __matmul__(self, other):
        n = self.center.shape[-1]
        center = self.center @ other.center
        radius = (np.abs(self.center) @ other.radius + self.radius @ np.abs(other.center) +
                  self.radius @ other.radius + gamma(2 * n + 2) * (np.abs(self.center) @ np.abs(other.center)))
        return Ball(center, positive_padding(radius, 8 * n + 32))

    def kron(self, other):
        center = np.kron(self.center, other.center)
        radius = (np.kron(np.abs(self.center), other.radius) + np.kron(self.radius, np.abs(other.center)) +
                  np.kron(self.radius, other.radius) + gamma(2) * np.kron(np.abs(self.center), np.abs(other.center)))
        return Ball(center, positive_padding(radius))

    def take_rows(self, rows):
        return Ball(self.center[rows], self.radius[rows])


def norm_infinity_upper(matrix):
    return float(positive_padding(np.max(np.sum(matrix.absolute_upper, axis=1)), 2 * matrix.center.shape[1] + 4))


def exponential(matrix, degree=24):
    """Taylor matrix exponential with a rigorous induced-infinity tail."""
    n = matrix.center.shape[0]
    z = Fraction.from_float(norm_infinity_upper(matrix))
    if z >= degree + 2:
        raise ValueError("Taylor degree insufficient for this matrix")
    result = term = Ball.exact(np.eye(n))
    for k in range(1, degree + 1):
        term = (term @ matrix).scale(Fraction(1, k))
        result = result + term
    tail = z**(degree + 1) / factorial(degree + 1) / (1 - z / (degree + 2))
    return Ball(result.center, up(result.radius + rational_upper(tail)))


def cosine_sqrt(x, degree=24):
    """cos(sqrt(x)) from an alternating rational-coefficient series."""
    z = Fraction.from_float(float(x.absolute_upper))
    # Only the omitted terms need to decrease for the alternating-series
    # remainder. This also covers the larger steps used at loose tolerances.
    if float(x.center) - float(x.radius) < 0 or z >= (2 * degree + 3) * (2 * degree + 4):
        raise ValueError("positive x and decreasing omitted cosine terms required")
    result = term = Ball.exact(1.0)
    for k in range(1, degree + 1):
        term = (term * x).scale(Fraction(-1, (2 * k - 1) * (2 * k)))
        result = result + term
    tail = rational_upper(z**(degree + 1) / factorial(2 * degree + 2))
    return Ball(result.center, up(result.radius + tail))


def model_maps(n, steps, t_final=Fraction(12), J=Fraction(10), gamma_rate=Fraction(20),
               p=Fraction(1, 4), prefactor=Fraction(1, 4)):
    """Validated maps with exact rational model inputs and time step."""
    if n < 2 or steps < 1 or J <= 0 or gamma_rate <= 0 or not 0 < p < 1:
        raise ValueError("invalid model or step count")
    tau = Fraction(t_final, steps)
    hopping = 2 * prefactor * J
    # Real gauge: K has +hopping on the upper diagonal, -hopping below.
    pattern = np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)
    K = Ball.exact(pattern).scale(hopping)
    boundary = np.zeros(n)
    boundary[[0, -1]] = 1
    A = K - Ball.exact(np.diag(boundary)).scale(gamma_rate / 2)
    G = A.kron(Ball.exact(np.eye(n))) + Ball.exact(np.eye(n)).kron(A)
    Q = exponential(G.scale(tau))
    U = exponential(K.scale(tau))
    cosine = cosine_sqrt(Ball.rational(2 * gamma_rate * tau))
    interactions = []
    for site in (0, n - 1):
        R_center = np.eye(n)
        R_radius = np.zeros((n, n))
        R_center[site, site] = float(cosine.center)
        R_radius[site, site] = float(cosine.radius)
        UR = U @ Ball(R_center, R_radius)
        interactions.append(UR.kron(UR))
    P = (interactions[0] + interactions[1]).scale(Fraction(1, 2))
    labels = ["0" * n, "1" * n, "1" + "0" * (n - 1)]
    p_ball = Ball.rational(p)
    initial_center = np.zeros((n * n, 3))
    initial_radius = np.zeros_like(initial_center)
    for s, label in enumerate(labels):
        for j, bit in enumerate(label):
            value = Ball.exact(int(bit)) - p_ball
            initial_center[j * n + j, s] = float(value.center)
            initial_radius[j * n + j, s] = float(value.radius)
    initial = Ball(initial_center, initial_radius)
    # In this gauge I/J = -4*prefactor * C[j,j+1].
    rows = np.r_[np.arange(n) * (n + 1), [j * n + j + 1 for j in range(n - 1)]]
    weights = np.r_[np.ones(n), np.full(n - 1, -float(4 * prefactor))]
    return P, Q, initial, rows, weights, labels, tau


def rounded_matmul_error(left, right):
    n = left.shape[-1]
    entry = positive_padding(gamma(2 * n + 2) * (np.abs(left) @ np.abs(right)), 4 * n + 16)
    return upper_norm(entry, axis=0) if right.shape[1] == 3 else upper_norm(entry)


def contraction_power(matrix, count):
    """Power enclosure using known contraction, avoiding interval wrapping.

    For true contractions A,B and approximate a,b with Frobenius radii
    ra,rb, ||AB-ab||F <= ra+rb+ra*rb. Add dot-product roundoff.
    """
    size = matrix.center.shape[0]
    result, result_error = np.eye(size), 0.0
    factor, factor_error = matrix.center, float(upper_norm(matrix.radius))
    remaining = count
    while remaining:
        if remaining % 2:
            error = (result_error + factor_error + result_error * factor_error +
                     float(rounded_matmul_error(result, factor)))
            result = result @ factor
            result_error = float(positive_padding(error))
        remaining //= 2
        if remaining:
            error = 2 * factor_error + factor_error**2 + float(rounded_matmul_error(factor, factor))
            factor = factor @ factor
            factor_error = float(positive_padding(error))
    return result, result_error


@lru_cache(maxsize=128)
def bernstein_transform(length, degree, bernstein_degree):
    """Exact rational change from binomial(k,l) to Bernstein(k/L)."""
    center = np.zeros((bernstein_degree + 1, degree + 1))
    radius = np.zeros_like(center)
    # Coefficients of binomial(L*u,l) as polynomials in u.
    powers = [Fraction(1)]
    for ell in range(degree + 1):
        for j in range(bernstein_degree + 1):
            value = sum(powers[r] * Fraction(comb(j, r), comb(bernstein_degree, r))
                        for r in range(min(ell, j) + 1))
            ball = Ball.rational(value)
            center[j, ell], radius[j, ell] = ball.center, ball.radius
        next_powers = [Fraction(0)] * (ell + 2)
        for r, coefficient in enumerate(powers):
            next_powers[r] -= Fraction(ell, ell + 1) * coefficient
            next_powers[r + 1] += Fraction(length, ell + 1) * coefficient
        powers = next_powers
    return Ball(center, radius)


def binomial_tail(length, degree, z, state_entry_bound):
    """Exact rational evaluation of a positive finite remainder bound."""
    z = Fraction.from_float(z)
    tail = sum(Fraction(comb(length, ell)) * z**ell for ell in range(degree + 1, length + 1))
    return rational_upper(tail * Fraction(state_entry_bound))


def certificate(n, steps, t_final=Fraction(12), tolerance=0.003,
                block_steps=48, degree=24, bernstein_degree=128):
    started = time.perf_counter()
    if degree > bernstein_degree:
        raise ValueError("Bernstein degree must be at least the expansion degree")
    P, Q, initial, rows, weights, labels, tau = model_maps(n, steps, t_final)
    size = n * n
    centered_P = P - Ball.exact(np.eye(size))
    centered_Q = Q - Ball.exact(np.eye(size))
    zP, zQ = norm_infinity_upper(centered_P), norm_infinity_upper(centered_Q)
    powers_P = [Ball.exact(np.eye(size))]
    powers_Q = [Ball.exact(np.eye(size))]
    for _ in range(degree):
        powers_P.append(centered_P @ powers_P[-1])
        powers_Q.append(centered_Q @ powers_Q[-1])
    projected_P = [power.take_rows(rows) for power in powers_P]
    projected_Q = [power.take_rows(rows) for power in powers_Q]
    row_bounds_P = [upper_norm(power.absolute_upper, axis=1) for power in projected_P]
    row_bounds_Q = [upper_norm(power.absolute_upper, axis=1) for power in projected_Q]
    xP, xQ = initial.center.copy(), initial.center.copy()
    errorP = errorQ = upper_norm(initial.radius, axis=0)
    cache = {}
    maximum = np.zeros((len(rows), 3))
    lower_witness = dict(error=0.0)
    blocks = []
    for start in range(0, steps, block_steps):
        length = min(block_steps, steps - start)
        if length not in cache:
            change = bernstein_transform(length, degree, bernstein_degree)
            tail = positive_padding(binomial_tail(length, degree, zP, Fraction(3, 4)) +
                                    binomial_tail(length, degree, zQ, Fraction(3, 4)))
            cache[length] = (change, tail, contraction_power(P, length), contraction_power(Q, length))
        change, tail, (transitionP, transitionP_error), (transitionQ, transitionQ_error) = cache[length]
        coefficient_center, coefficient_radius = [], []
        for ell in range(degree + 1):
            a = projected_P[ell] @ Ball.exact(xP)
            b = projected_Q[ell] @ Ball.exact(xQ)
            difference = a - b
            radius = positive_padding(difference.radius + row_bounds_P[ell][:, None] * errorP +
                                      row_bounds_Q[ell][:, None] * errorQ)
            coefficient_center.append(difference.center)
            coefficient_radius.append(radius)
        coefficients = Ball(np.asarray(coefficient_center).reshape(degree + 1, -1),
                            np.asarray(coefficient_radius).reshape(degree + 1, -1))
        bernstein = change @ coefficients
        weighted = bernstein.absolute_upper.reshape(bernstein_degree + 1, len(rows), 3)
        local_bound = positive_padding((weighted.max(axis=0) + tail) * np.abs(weights[:, None]))
        maximum = np.maximum(maximum, local_bound)
        blocks.append(dict(start_step=start, end_step=start + length,
                           population_bound=float(local_bound[:n].max()),
                           normalized_current_bound=float(local_bound[n:].max()),
                           tail_bound=float(tail)))
        # A certified lower witness checks an actual integer step near the
        # largest control coefficient; no bound failure is called a violation.
        j, row, state = np.unravel_index(np.argmax(weighted[:, :n]), weighted[:, :n].shape)
        witness_offset = min(length, max(0, round(length * j / bernstein_degree)))
        binomial_entries = [Ball.rational(comb(witness_offset, ell) if ell <= witness_offset else 0)
                            for ell in range(degree + 1)]
        binomials = Ball(np.asarray([x.center for x in binomial_entries])[None, :],
                         np.asarray([x.radius for x in binomial_entries])[None, :])
        evaluated = binomials @ coefficients
        column = row * 3 + state
        lower = max(0, rational_lower(Fraction.from_float(abs(float(evaluated.center[0, column]))) -
                                      Fraction.from_float(float(evaluated.radius[0, column])) -
                                      Fraction.from_float(float(tail))))
        if lower > lower_witness["error"]:
            lower_witness = dict(error=lower, initial_state=labels[state], site=int(row + 1),
                                 step=start + witness_offset, time=float((start + witness_offset) * tau))
        errorP = positive_padding(errorP + transitionP_error * upper_norm(xP, axis=0) +
                                 rounded_matmul_error(transitionP, xP))
        errorQ = positive_padding(errorQ + transitionQ_error * upper_norm(xQ, axis=0) +
                                 rounded_matmul_error(transitionQ, xQ))
        xP, xQ = transitionP @ xP, transitionQ @ xQ
    pop_bound = float(maximum[:n].max())
    return dict(n=n, steps=steps, t_final=float(t_final), dt=float(tau), tolerance=tolerance,
                population_upper_bound=pop_bound,
                normalized_current_upper_bound=float(maximum[n:].max()),
                certified_population=bool(pop_bound <= tolerance),
                certified_violation=bool(lower_witness["error"] > tolerance),
                lower_witness=lower_witness, block_steps=block_steps, expansion_degree=degree,
                bernstein_degree=bernstein_degree, blocks=blocks,
                population_bounds_by_state_and_site=maximum[:n].T.tolist(),
                current_bounds_by_state_and_bond=maximum[n:].T.tolist(),
                diagnostics=dict(P_entry_radius=float(P.radius.max()), Q_entry_radius=float(Q.radius.max()),
                                 final_propagation_radius_F=float(max(errorP.max(), errorQ.max())),
                                 maximum_binomial_tail=float(max(b['tail_bound'] for b in blocks)),
                                 increment_P_infinity_bound=zP, increment_Q_infinity_bound=zQ),
                elapsed_seconds=time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, nargs="+", default=[3, 7])
    parser.add_argument("--steps", type=int, nargs="+", default=[6000, 5500, 5400])
    parser.add_argument("--tolerance", type=float, default=0.003)
    parser.add_argument("--block-steps", type=int, default=48)
    parser.add_argument("--degree", type=int, default=24)
    parser.add_argument("--bernstein-degree", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "certificates")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for n in args.n_qubits:
        for steps in args.steps:
            result = certificate(n, steps, tolerance=args.tolerance, block_steps=args.block_steps,
                                 degree=args.degree, bernstein_degree=args.bernstein_degree)
            records.append(result)
            (args.output_dir / f"N{n}_M{steps}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({key: result[key] for key in ['n', 'steps', 'dt', 'population_upper_bound',
                              'normalized_current_upper_bound', 'certified_population', 'certified_violation',
                              'lower_witness', 'diagnostics', 'elapsed_seconds']}), flush=True)
    # Keep prior certified counts when refining one candidate. Individual
    # files record the polynomial degrees used for that candidate.
    saved = [json.loads(path.read_text()) for path in sorted(args.output_dir.glob("N*_M*.json"))]
    (args.output_dir / "summary.json").write_text(json.dumps(saved, indent=2) + "\n")


if __name__ == "__main__":
    main()
