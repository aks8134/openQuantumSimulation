"""First-order Lie--Trotter solver for the boundary-damped XY chain.

The Lindblad generator is split as

    L = L_system + sum_k L_jump[k],

where ``L_system`` contains only ``-1j * [H, rho]`` and every
``L_jump[k]`` contains the complete dissipator associated with one jump
operator.  One time step applies the system evolution first, followed by
the individual jump evolutions.
"""

import numpy as np

from scipy.sparse import eye

from library import classical
from library.plotter import plot_observables


# ============================================================
# Configuration
# ============================================================

N_QUBITS = 7

J = 1.0
h = 0.0
gamma = 0.2

T_FINAL = 10.0
NUM_TIMES = 51

# Each interval between two saved times can be divided into smaller
# Trotter steps. Increasing this reduces the splitting error without
# changing the output time grid.
TROTTER_STEPS_PER_INTERVAL = 1

INITIALLY_EXCITED_SITES = [N_QUBITS // 2]


# ============================================================
# Simulation
# ============================================================

def main():
    (
        H,
        jump_operators,
        X_ops,
        Y_ops,
        Z_ops,
    ) = classical.build_linear_chain(
        number_of_qubits=N_QUBITS,
        J=J,
        h=h,
        gamma=gamma,
    )

    (
        system_liouvillian,
        jump_liouvillians,
    ) = classical.build_lie_trotter_liouvillians(
        H,
        jump_operators,
    )

    hilbert_dimension = 2**N_QUBITS
    identity = eye(hilbert_dimension, dtype=complex, format="csr")

    psi0 = classical.computational_state(
        number_of_qubits=N_QUBITS,
        excited_sites=INITIALLY_EXCITED_SITES,
    )
    rho0 = np.outer(psi0, psi0.conj())
    times = np.linspace(0.0, T_FINAL, NUM_TIMES)

    density_matrices = classical.lie_trotter_evolve(
        rho0,
        system_liouvillian,
        jump_liouvillians,
        times,
        steps_per_interval=TROTTER_STEPS_PER_INTERVAL,
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

    total_excitation = site_populations.sum(axis=0)
    exact_total_excitation = exact_site_populations.sum(axis=0)

    density_matrix_errors = np.linalg.norm(
        density_matrices - exact_density_matrices,
        axis=(1, 2),
    )
    population_errors = site_populations - exact_site_populations

    final_rho = density_matrices[-1]
    dt = (times[1] - times[0]) / TROTTER_STEPS_PER_INTERVAL

    print(f"Number of qubits: {N_QUBITS}")
    print(f"Number of jump generators: {len(jump_liouvillians)}")
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

    plot_observables(
        times,
        {
            "Lie--Trotter site populations": {
                "values": site_populations,
                "labels": [
                    f"site {site}"
                    for site in range(N_QUBITS)
                ],
                "ylabel": r"$\langle n_i\rangle$",
                "legend_columns": 2,
            },
            "Nearest-neighbor exchange correlations": {
                "values": exchange_correlations,
                "labels": [
                    f"bond {bond}-{bond + 1}"
                    for bond in range(N_QUBITS - 1)
                ],
                "ylabel": r"$C_i^{XY}$",
                "legend_columns": 2,
            },
            "Nearest-neighbor excitation flows": {
                "values": bond_flows,
                "labels": [
                    f"{bond} → {bond + 1}"
                    for bond in range(N_QUBITS - 1)
                ],
                "ylabel": r"$I_{i\rightarrow i+1}$",
                "legend_columns": 2,
                "zero_line": True,
            },
            "Total excitation comparison": {
                "values": np.vstack([
                    total_excitation,
                    exact_total_excitation,
                ]),
                "labels": ["Lie--Trotter", "unsplit exact"],
                "ylabel": r"$\sum_i\langle n_i\rangle$",
                "styles": [
                    {"linewidth": 2},
                    {"linestyle": "--", "linewidth": 2},
                ],
            },
            "Site-population errors": {
                "values": population_errors,
                "labels": [
                    f"site {site}"
                    for site in range(N_QUBITS)
                ],
                "ylabel": "Trotter - exact",
                "legend_columns": 2,
                "zero_line": True,
            },
            "Density-matrix error": {
                "values": density_matrix_errors,
                "labels": ["Frobenius norm"],
                "ylabel": r"$\|\rho_T-\rho_{exact}\|_F$",
            },
        },
        output_path=(
            f"Model1/figures/obs_lie_trotter_{N_QUBITS}.png"
        ),
    )


if __name__ == "__main__":
    main()
