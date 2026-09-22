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

    liouvillian = classical.build_liouvillian(H, jump_operators)

    hilbert_dimension = 2**number_of_qubits

    identity = eye(
        hilbert_dimension,
        dtype=complex,
        format="csr",
    )

    psi0 = classical.computational_state(
        number_of_qubits=number_of_qubits,
        excited_sites=[number_of_qubits // 2],
    )

    rho0 = np.outer(psi0, psi0.conj())

    times = np.linspace(
        0.0,
        T_FINAL,
        NUM_TIMES,
    )

    density_matrices = classical.exact_evolve(
        rho0,
        liouvillian,
        times,
    )

    # ========================================================
    # Build and evaluate observables
    # ========================================================

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

    # n_i(t), shape: (number_of_qubits, NUM_TIMES)
    site_populations = np.asarray([
        classical.expectation_series(density_matrices, operator)
        for operator in population_operators
    ])

    # C_i^XY(t), shape: (number_of_qubits - 1, NUM_TIMES)
    exchange_correlations = np.asarray([
        classical.expectation_series(density_matrices, operator)
        for operator in exchange_operators
    ])

    # I_{i -> i+1}(t), shape: (number_of_qubits - 1, NUM_TIMES)
    bond_flows = np.asarray([
        classical.expectation_series(density_matrices, operator)
        for operator in flow_operators
    ])

    # Current convention used in the document:
    #
    # j_{i,i+1} = J <X_i Y_{i+1} - Y_i X_{i+1}>
    #
    # Therefore j_document = -2 * I_{i -> i+1}.
    document_currents = -2.0 * bond_flows

    # ========================================================
    # Global quantities
    # ========================================================

    total_excitation = site_populations.sum(axis=0)

    total_exchange_energy = (
        0.5 * J * exchange_correlations.sum(axis=0)
    )

    mean_exchange_correlation = (
        exchange_correlations.mean(axis=0)
    )

    mean_excitation_flow = bond_flows.mean(axis=0)

    # ========================================================
    # Continuity-equation check
    # ========================================================

    # Numerically estimate dn_i/dt from the sampled time points.
    population_derivatives = np.gradient(
        site_populations,
        times,
        axis=1,
        edge_order=2,
    )

    continuity_rhs = np.zeros_like(site_populations)

    # Left boundary:
    # dn_0/dt = -I_{0 -> 1} - gamma*n_0
    continuity_rhs[0] -= gamma * site_populations[0]
    continuity_rhs[0] -= bond_flows[0]

    # Interior sites:
    # dn_i/dt = I_{i-1 -> i} - I_{i -> i+1}
    for i in range(1, number_of_qubits - 1):
        continuity_rhs[i] += (
            bond_flows[i - 1]
            - bond_flows[i]
        )

    # Right boundary:
    # dn_{N-1}/dt = I_{N-2 -> N-1} - gamma*n_{N-1}
    continuity_rhs[-1] -= gamma * site_populations[-1]
    continuity_rhs[-1] += bond_flows[-1]

    # Exclude endpoints because numerical derivatives are less
    # accurate there.
    continuity_residual = (
        population_derivatives[:, 1:-1]
        - continuity_rhs[:, 1:-1]
    )

    # The Hamiltonian conserves total excitation. With boundary-only
    # damping, its loss rate is set by the populations at the two ends:
    #
    #   dN_exc/dt = -gamma * (n_0 + n_{N-1}).
    total_excitation_derivative = np.gradient(
        total_excitation,
        times,
        edge_order=2,
    )
    total_excitation_loss_rate = -gamma * (
        site_populations[0] + site_populations[-1]
    )
    total_excitation_residual = (
        total_excitation_derivative[1:-1]
        - total_excitation_loss_rate[1:-1]
    )

    # ========================================================
    # Numerical diagnostics
    # ========================================================

    final_rho = density_matrices[-1]

    print(f"Number of system qubits: {number_of_qubits}")
    print(f"Hilbert-space dimension: {hilbert_dimension}")
    print(f"Liouville-space dimension: {hilbert_dimension**2}")

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

    print("\nObservable checks")
    print(
        "Maximum total-excitation balance residual:",
        np.max(np.abs(total_excitation_residual)),
    )
    print(
        "Maximum continuity residual:",
        np.max(np.abs(continuity_residual)),
    )

    print("\nFinal site populations")
    for i in range(number_of_qubits):
        print(f"n_{i} = {site_populations[i, -1]:+.6f}")

    print("\nFinal exchange correlations")
    for i in range(number_of_qubits - 1):
        print(
            f"C_XY({i},{i + 1}) = "
            f"{exchange_correlations[i, -1]:+.6f}"
        )

    print("\nFinal excitation flows")
    for i in range(number_of_qubits - 1):
        print(
            f"I({i}->{i + 1}) = "
            f"{bond_flows[i, -1]:+.6f}"
        )

    plot_observables(
        times,
        {
            "Local excitation populations": {
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
            "Total excitation": {
                "values": total_excitation,
                "labels": ["simulation"],
                "ylabel": r"$\sum_i\langle n_i\rangle$",
                "styles": [{"linewidth": 2}],
            },
        },
        output_path=(
            Path(__file__).resolve().parent
            / "figures"
            / f"obs_exact_{number_of_qubits}.png"
        ),
    )

    # Variables available for further analysis:
    #
    # site_populations[i, t]
    # exchange_correlations[i, t]
    # bond_flows[i, t]
    # document_currents[i, t]
    # total_exchange_energy[t]
    # mean_exchange_correlation[t]
    # mean_excitation_flow[t]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Exact Lindblad reference for the N-qubit XY chain."
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
