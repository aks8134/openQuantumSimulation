"""Lie dilation formula for the N-qubit boundary-damped XY chain.

For each physical step, the shared qutrit ancilla is prepared in |0>, the
dilation terms are applied in the order

    K_H -> K_1 -> K_2,

and the ancilla is traced out. This is the cheaper first-order circuit below
equation (27) of ``writeup.pdf``. The unsplit Lindblad and unsplit dilation
channels are both retained as separate references.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import eye


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from library import classical
from library.plotter import plot_observables


# ============================================================
# Configuration
# ============================================================

DEFAULT_NUMBER_OF_QUBITS = 7

J = 1.0
h = 0.0
gamma = 0.2

T_FINAL = 10.0
NUM_TIMES = 51

# Each interval between two saved times can be divided into smaller
# Trotter steps. Increasing this reduces the splitting error without
# changing the output time grid.
TROTTER_STEPS_PER_INTERVAL = 1

# ============================================================
# Simulation
# ============================================================

def main(number_of_qubits=DEFAULT_NUMBER_OF_QUBITS):
    if number_of_qubits < 2:
        raise ValueError("number_of_qubits must be at least two")

    (
        H,
        jump_operators,
        X_ops,
        Y_ops,
        Z_ops,
    ) = classical.build_linear_chain(
        number_of_qubits=number_of_qubits,
        J=J,
        h=h,
        gamma=gamma,
    )

    hilbert_dimension = 2**number_of_qubits
    identity = eye(hilbert_dimension, dtype=complex, format="csr")

    psi0 = classical.computational_state(
        number_of_qubits=number_of_qubits,
        excited_sites=[number_of_qubits // 2],
    )
    rho0 = np.outer(psi0, psi0.conj())
    times = np.linspace(0.0, T_FINAL, NUM_TIMES)

    density_matrices = classical.hamiltonian_dilation_evolve(
        rho0,
        H,
        jump_operators,
        times,
        product_formula="lie",
        steps_per_interval=TROTTER_STEPS_PER_INTERVAL,
    )

    # This applies exp[-i(K_H + K_1 + K_2)] before tracing out the shared
    # ancilla. It isolates product-formula error from dilation error.
    exact_dilation_density_matrices = (
        classical.hamiltonian_dilation_evolve(
            rho0,
            H,
            jump_operators,
            times,
            product_formula="exact",
            steps_per_interval=TROTTER_STEPS_PER_INTERVAL,
        )
    )

    # Build the unsplit evolution only as a numerical reference. It does
    # not participate in the Lie--Trotter propagation above.
    full_liouvillian = classical.build_liouvillian(H, jump_operators)
    exact_density_matrices = classical.exact_evolve(
        rho0,
        full_liouvillian,
        times,
    )

    (
        population_operators,
        exchange_operators,
        flow_operators,
    ) = classical.build_observables(
        identity=identity,
        X_ops=X_ops,
        Y_ops=Y_ops,
        Z_ops=Z_ops,
        J=J,
    )

    site_populations = np.asarray([
        classical.expectation_series(density_matrices, operator)
        for operator in population_operators
    ])
    exchange_correlations = np.asarray([
        classical.expectation_series(density_matrices, operator)
        for operator in exchange_operators
    ])
    bond_flows = np.asarray([
        classical.expectation_series(density_matrices, operator)
        for operator in flow_operators
    ])

    exact_site_populations = np.asarray([
        classical.expectation_series(exact_density_matrices, operator)
        for operator in population_operators
    ])
    exact_dilation_site_populations = np.asarray([
        classical.expectation_series(
            exact_dilation_density_matrices,
            operator,
        )
        for operator in population_operators
    ])

    total_excitation = site_populations.sum(axis=0)
    exact_total_excitation = exact_site_populations.sum(axis=0)
    exact_dilation_total_excitation = (
        exact_dilation_site_populations.sum(axis=0)
    )

    density_matrix_errors = np.linalg.norm(
        density_matrices - exact_density_matrices,
        axis=(1, 2),
    )
    population_errors = site_populations - exact_site_populations
    dilation_trotter_errors = np.linalg.norm(
        density_matrices - exact_dilation_density_matrices,
        axis=(1, 2),
    )

    final_rho = density_matrices[-1]
    dt = (times[1] - times[0]) / TROTTER_STEPS_PER_INTERVAL
    ancilla_dimension = len(jump_operators) + 1
    ancilla_qubits = int(np.ceil(np.log2(ancilla_dimension)))

    print(f"Number of system qubits: {number_of_qubits}")
    print(f"Shared-ancilla levels: {ancilla_dimension}")
    print(f"Ancilla qubits in binary encoding: {ancilla_qubits}")
    print(
        "Total encoded circuit qubits: "
        f"{number_of_qubits + ancilla_qubits}"
    )
    print(f"Number of jump terms: {len(jump_operators)}")
    print(f"Lie--Trotter substep: {dt}")

    print("\nDensity-matrix checks")
    print("Final trace:", np.trace(final_rho))
    print(
        "Final Hermiticity error:",
        np.linalg.norm(final_rho - final_rho.conj().T),
    )
    print(
        "Smallest final eigenvalue:",
        np.linalg.eigvalsh(final_rho).min(),
    )

    print("\nError relative to unsplit evolution")
    print(
        "Maximum density-matrix Frobenius error:",
        density_matrix_errors.max(),
    )
    print(
        "Final density-matrix Frobenius error:",
        density_matrix_errors[-1],
    )
    print(
        "Maximum site-population error:",
        np.max(np.abs(population_errors)),
    )
    print(
        "Maximum product-formula error relative to exact dilation:",
        dilation_trotter_errors.max(),
    )

    plot_observables(
        times,
        {
            "Lie--Trotter site populations": {
                "values": site_populations,
                "labels": [
                    f"site {site}"
                    for site in range(number_of_qubits)
                ],
                "ylabel": r"$\langle n_i\rangle$",
                "legend_columns": 2,
            },
            "Nearest-neighbor exchange correlations": {
                "values": exchange_correlations,
                "labels": [
                    f"bond {bond}-{bond + 1}"
                    for bond in range(number_of_qubits - 1)
                ],
                "ylabel": r"$C_i^{XY}$",
                "legend_columns": 2,
            },
            "Nearest-neighbor excitation flows": {
                "values": bond_flows,
                "labels": [
                    f"{bond} → {bond + 1}"
                    for bond in range(number_of_qubits - 1)
                ],
                "ylabel": r"$I_{i\rightarrow i+1}$",
                "legend_columns": 2,
                "zero_line": True,
            },
            "Total excitation comparison": {
                "values": np.vstack([
                    total_excitation,
                    exact_dilation_total_excitation,
                    exact_total_excitation,
                ]),
                "labels": [
                    "Lie dilation",
                    "exact dilation",
                    "exact Lindblad",
                ],
                "ylabel": r"$\sum_i\langle n_i\rangle$",
                "styles": [
                    {"linewidth": 2},
                    {"linestyle": "--", "linewidth": 2},
                    {"linestyle": ":", "linewidth": 2},
                ],
            },
            "Site-population errors": {
                "values": population_errors,
                "labels": [
                    f"site {site}"
                    for site in range(number_of_qubits)
                ],
                "ylabel": "Trotter - exact",
                "legend_columns": 2,
                "zero_line": True,
            },
            "Density-matrix error": {
                "values": np.vstack([
                    density_matrix_errors,
                    dilation_trotter_errors,
                ]),
                "labels": [
                    "vs exact Lindblad",
                    "vs exact dilation",
                ],
                "ylabel": r"$\|\rho_T-\rho_{exact}\|_F$",
            },
        },
        output_path=(
            Path(__file__).resolve().parent
            / "figures"
            / f"obs_lie_trotter_{number_of_qubits}.png"
        ),
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Lie Hamiltonian-dilation evolution for the N-qubit XY chain."
        )
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=DEFAULT_NUMBER_OF_QUBITS,
        help=(
            "number of system qubits (default: "
            f"{DEFAULT_NUMBER_OF_QUBITS})"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments.n_qubits)
