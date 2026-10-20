"""Single-jump randomized Ding--Li--Lin dilation for the XX boundary model.

The jump is sampled independently at each step; the ancilla is reset after
each step. This is NOT an exact-reservoir exponential. Covariances close for
the XX model, so populations/current can be evaluated without reconstructing
the exponentially large density matrix. A mean covariance is not a complete
description of the generally non-Gaussian ensemble density matrix.
"""

import numpy as np
from scipy.linalg import eig, expm


POLICIES = ("uniform", "rate-weighted")
ORDERS = ("hamiltonian-first", "jump-first")


def probabilities(p, policy):
    if not 0 < p < 1 or policy not in POLICIES:
        raise ValueError("require 0 < p < 1 and a supported sampling policy")
    return (np.full(4, 0.25) if policy == "uniform" else
            np.array([p, 1 - p, p, 1 - p]) / 2)


def sampled_kraus(H, jump, probability, dt, order="jump-first"):
    """Full-density Kraus operators from a genuine two-level ancilla dilation.

    exp(-i sqrt(dt) [[0, (V/sqrt(mu))^dagger], [V/sqrt(mu), 0]])
    acts on an ancilla prepared in |0>. Hamiltonian evolution is a separate
    factor. This small reference implementation works also for nonzero Delta.
    """
    if dt <= 0 or not 0 < probability <= 1 or order not in ORDERS:
        raise ValueError("invalid step, sampling probability, or factor order")
    H = H.toarray() if hasattr(H, "toarray") else np.asarray(H)
    jump = jump.toarray() if hasattr(jump, "toarray") else np.asarray(jump)
    V = jump / np.sqrt(probability)
    zero = np.zeros_like(H)
    dilation = expm(-1j * np.sqrt(dt) * np.block([[zero, V.conj().T], [V, zero]]))
    d = H.shape[0]
    kraus = dilation[:, :d].reshape(2, d, d)
    unitary = expm(-1j * dt * H)
    if order == "hamiltonian-first":
        return kraus @ unitary
    return unitary @ kraus


def covariance_branches(n, J, gamma, p, prefactor, dt, policy, order):
    """Return C -> A_a C A_a^dagger + B_a for each sampled spin channel.

    C_ij = <c_j^dagger c_i>, with Jordan--Wigner fermions and c=|0><1|.
    Each boundary spin dilation multiplies incident coherences by cos(theta)
    and relaxes that site's population with sin(theta)^2. Closure is exact
    for these bilinear observables even when the density matrix is not Gaussian.
    """
    if (n < 2 or not isinstance(n, (int, np.integer)) or J == 0 or
            gamma <= 0 or prefactor <= 0 or dt <= 0 or order not in ORDERS or
            not np.all(np.isfinite([J, gamma, p, prefactor, dt]))):
        raise ValueError("invalid model parameters, time step, or order")
    mu = probabilities(p, policy)
    rates = gamma * np.array([p, 1 - p, p, 1 - p])
    theta = np.sqrt(rates * dt / mu)
    hopping = 2 * prefactor * J
    h = np.diag(np.full(n - 1, hopping), 1) + np.diag(np.full(n - 1, hopping), -1)
    unitary = expm(-1j * h * dt)
    matrices, offsets = [], []
    for a, boundary in enumerate((0, 0, n - 1, n - 1)):
        R = np.eye(n)
        R[boundary, boundary] = np.cos(theta[a])
        B = np.zeros((n, n), complex)
        if a % 2 == 0:  # pumping, source |0>, destination |1>
            B[boundary, boundary] = np.sin(theta[a])**2
        if order == "hamiltonian-first":
            matrices.append(R @ unitary)
            offsets.append(B)
        else:
            matrices.append(unitary @ R)
            offsets.append(unitary @ B @ unitary.conj().T)
    transitions = mu * np.sin(theta)**2
    denominator = transitions[0] + transitions[1]
    if denominator < 1e-14:
        raise ValueError("step too close to a zero-transition dilation resonance")
    effective_p = transitions[0] / denominator
    return dict(mu=mu, theta=theta, A=np.asarray(matrices), B=np.asarray(offsets),
                unitary=unitary, effective_p=float(effective_p), hopping=hopping)


def apply_mean(C, branches):
    return sum(mu * (A @ C @ A.conj().T + B)
               for mu, A, B in zip(branches["mu"], branches["A"], branches["B"]))


def initial_covariances(n):
    labels = ["0" * n, "1" * n, "1" + "0" * (n - 1)]
    return labels, np.asarray([np.diag([int(bit) for bit in bits]) for bits in labels], complex)


def observables(C, hopping):
    population = np.diagonal(C, axis1=-2, axis2=-1).real
    current = 2 * hopping * np.diagonal(C, offset=1, axis1=-2, axis2=-1).imag
    return population, current


def exact_covariances(initial, n, J, gamma, p, prefactor, times):
    """Exact continuous XX Lindblad second moments, not a time discretization."""
    hopping = 2 * prefactor * J
    h = np.diag(np.full(n - 1, hopping), 1) + np.diag(np.full(n - 1, hopping), -1)
    rates = np.zeros(n)
    rates[[0, -1]] = gamma
    generator = -1j * h - np.diag(rates) / 2
    values, vectors = eig(generator)
    if np.linalg.cond(vectors) > 1e7:
        E = np.asarray([expm(generator * t) for t in times])
    else:
        E = np.einsum("ab,bt,bc->tac", vectors,
                      np.exp(values[:, None] * np.asarray(times)[None, :]),
                      np.linalg.inv(vectors), optimize=True)
    return np.asarray([p * np.eye(n) + E @ (C - p * np.eye(n)) @ E.conj().swapaxes(-1, -2)
                       for C in initial])


def mean_trajectory(initial, branches, counts):
    """Exact ensemble channel powers on the covariance sector (roundoff aside).

    Row-major vectorization: vec(A C A^dagger)=(A tensor A*) vec(C).
    Check the eigendecomposition against direct steps and its power residual.
    """
    counts = np.asarray(counts)
    if counts.ndim != 1 or np.any(counts < 0) or np.any(counts != counts.astype(int)):
        raise ValueError("counts must be nonnegative integers")
    n = initial.shape[-1]
    stationary = branches["effective_p"] * np.eye(n)
    matrix = sum(mu * np.kron(A, A.conj())
                 for mu, A in zip(branches["mu"], branches["A"]))
    eigenvalues, eigenvectors = eig(matrix)
    coefficients = np.linalg.solve(eigenvectors, (initial - stationary).reshape(len(initial), -1).T)
    powers = eigenvalues[:, None] ** counts[None, :]
    C = np.einsum("ab,bt,bs->sta", eigenvectors, powers, coefficients,
                  optimize=True).reshape(len(initial), len(counts), n, n) + stationary
    direct = initial.copy()
    check_powers = eigenvalues[:, None] ** np.arange(101)[None, :]
    check = np.einsum("ab,bt,bs->sta", eigenvectors, check_powers, coefficients,
                      optimize=True).reshape(len(initial), 101, n, n) + stationary
    agreement = 0.0
    for step in range(101):
        agreement = max(agreement, float(np.max(np.abs(direct - check[:, step]))))
        direct = apply_mean(direct, branches)
    radius = float(np.max(np.abs(eigenvalues)))
    residual = np.linalg.norm(matrix @ eigenvectors - eigenvectors * eigenvalues, 2)
    # ||M||_2 <= 1: each A is a contraction. Bound the telescoping power
    # residual in exact arithmetic, allowing tiny eigenvalue-radius excess.
    bound = float(counts.max(initial=0) * max(1, radius)**counts.max(initial=0) *
                  residual * np.linalg.norm(coefficients, 2))
    fixed_residual = float(np.max(np.abs(apply_mean(stationary, branches) - stationary)))
    if agreement > 1e-9 or bound > 1e-7 or fixed_residual > 1e-12 or radius > 1 + 1e-10:
        raise RuntimeError("covariance channel power validation failed")
    return C, dict(early_direct_agreement=agreement,
                   eigenvector_condition=float(np.linalg.cond(eigenvectors)),
                   power_residual_bound=bound, spectral_radius=radius,
                   stationary_covariance_residual=fixed_residual,
                   maximum_hermiticity_error=float(np.max(np.abs(C - C.conj().swapaxes(-1, -2)))))


def sample_trajectories(initial, branches, counts, realizations, seed, order):
    """Random sequences, tracing the ancilla deterministically per sequence.

    No measurement shot noise or stochastic pure-state unraveling is added.
    Independent realizations share their sampled labels across preparations.
    """
    if realizations < 2 or order not in ORDERS:
        raise ValueError("require at least two realizations and a supported order")
    counts = np.asarray(counts, int)
    if counts[0] != 0 or np.any(np.diff(counts) <= 0):
        raise ValueError("sample counts must start at zero and increase")
    generator = np.random.default_rng(seed)
    n = initial.shape[-1]
    C = np.repeat(initial[:, None], realizations, axis=1)
    population = np.empty((len(initial), realizations, len(counts), n))
    current = np.empty((len(initial), realizations, len(counts), n - 1))
    U = branches["unitary"]
    cosine = np.cos(branches["theta"])
    sine_squared = np.sin(branches["theta"])**2
    output_index = 0
    draws = np.zeros(4, dtype=int)
    for step in range(counts[-1] + 1):
        if step == counts[output_index]:
            population[:, :, output_index], current[:, :, output_index] = observables(C, branches["hopping"])
            output_index += 1
            if output_index == len(counts):
                break
        labels = generator.choice(4, realizations, p=branches["mu"])
        draws += np.bincount(labels, minlength=4)
        if order == "hamiltonian-first":
            C = U @ C @ U.conj().T
        for a, boundary in enumerate((0, 0, n - 1, n - 1)):
            selected = np.flatnonzero(labels == a)
            C[:, selected, boundary, :] *= cosine[a]
            C[:, selected, :, boundary] *= cosine[a]
            if a % 2 == 0:
                C[:, selected, boundary, boundary] += sine_squared[a]
        if order == "jump-first":
            C = U @ C @ U.conj().T
    return population, current, dict(seed=seed, realizations=realizations,
                                    realized_sampling_probabilities=(draws / draws.sum()).tolist())
