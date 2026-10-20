"""Certified Krylov evaluation of integer powers of a discrete channel.

The Arnoldi residual gives an a posteriori trace-norm bound in exact
arithmetic when the original map is CPTP. Floating-point error is assessed
separately against direct iteration; this is not a rigorous rounding bound.
"""

import numpy as np
from scipy.linalg import eig, solve


def channel_powers(operator, initial, powers, dt, sectors, *, tolerance=1e-9, trace_norm_excess=0.0):
    """initial has shape (state, number-block vector entry)."""
    powers = np.asarray(powers, int)
    number_of_states, dimension = initial.shape
    largest = int(powers.max())
    beta = np.linalg.norm(initial)
    checkpoints = [80, 160, 320, 600]
    limit = min(checkpoints[-1], number_of_states * dimension)
    basis = np.empty((initial.size, limit + 1), complex, order='F')
    hessenberg = np.zeros((limit + 1, limit), complex)
    basis[:, 0] = initial.ravel() / beta
    offsets = np.r_[0, np.cumsum([len(indices)**2 for indices in sectors])]
    for column in range(limit):
        vector = basis[:, column].reshape(number_of_states, dimension)
        image = ((operator @ vector.T).T - vector).ravel() / dt
        # Two passes keep the basis orthogonal despite a nearly invariant subspace.
        for _ in range(2):
            coefficients = basis[:, :column+1].conj().T @ image
            hessenberg[:column+1, column] += coefficients
            image -= basis[:, :column+1] @ coefficients
        residual_norm = np.linalg.norm(image)
        hessenberg[column+1, column] = residual_norm
        size = column + 1
        invariant = residual_norm < 1e-11
        basis[:, size] = image / residual_norm if residual_norm else 0
        if size not in checkpoints and not invariant and size != limit:
            continue
        values, vectors = eig(hessenberg[:size, :size])
        channel_values = 1 + dt * values
        start = np.zeros(size, complex)
        start[0] = beta
        weights = solve(vectors, start)
        final_weights = vectors[-1] * weights
        # Evaluate the residual coefficient at every integer step, not just outputs.
        maximum_coefficient = 0.0
        logarithms = np.log(channel_values)
        for first in range(0, largest, 1024):
            indices = np.arange(first, min(first+1024, largest))
            coefficients = final_weights @ np.exp(logarithms[:, None] * indices)
            maximum_coefficient = max(maximum_coefficient, float(np.max(np.abs(coefficients))))
        trace_norm = 0.0
        if residual_norm:
            residual = basis[:, size].reshape(number_of_states, dimension)
            for state in residual:
                norm = sum(np.linalg.svd(state[first:last].reshape(len(indices), len(indices)), compute_uv=False).sum()
                           for first, last, indices in zip(offsets[:-1], offsets[1:], sectors))
                trace_norm = max(trace_norm, float(norm))
        bound = np.exp(largest * trace_norm_excess) * largest * dt * residual_norm * trace_norm * maximum_coefficient
        print(f'Krylov dimension={size}, residual trace-norm bound={bound:.3g}', flush=True)
        if bound <= tolerance:
            coefficients = vectors @ (weights[:, None] * np.exp(logarithms[:, None] * powers))
            trajectory = (basis[:, :size] @ coefficients).reshape(number_of_states, dimension, len(powers))
            return trajectory.transpose(0, 2, 1), dict(dimension=size, residual_trace_norm_bound=float(bound))
    raise RuntimeError(f'Krylov power evaluation did not meet tolerance {tolerance:g}')
