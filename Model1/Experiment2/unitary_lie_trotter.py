"""Circuit implementation of Lie--Trotter evolution for Experiment 1.

The two system qubits form the boundary-damped XY chain

    H = J/2 (X0 X1 + Y0 Y1) + h/2 (Z0 + Z1),

with one amplitude-damping jump operator on each system qubit. The coherent
XY evolution and the unitary dilation of each jump channel are implemented
with Qiskit's ``XXPlusYYGate``.

One Lie--Trotter substep is

    system full-step -> left jump full-step -> right jump full-step.

The two jump channels use bath qubits initially in |0>. After an exchange
gate, resetting the corresponding bath traces it out and supplies a fresh
environment for the next substep. A control qubit enables both jump gates.

The unsplit reference calculation reuses ``library/classical.py``.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import XXPlusYYGate
from qiskit_aer import AerSimulator
from scipy.sparse import eye


# Permit both ``python path/to/script.py`` and ``python -m ...`` execution.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from library import classical


# ============================================================
# Configuration
# ============================================================

NUMBER_OF_QUBITS = 2

J = 1.0
h = 0.0
gamma = 0.2

T_FINAL = 10.0
NUMBER_OF_TIME_POINTS = 51
TROTTER_STEPS_PER_INTERVAL = 1

INITIALLY_EXCITED_SITES = [NUMBER_OF_QUBITS // 2]

# Qiskit writes basis states as |q1 q0>. To make its reduced two-qubit
# density matrix use the same ordering as classical.py (site 0 tensor site 1),
# physical site 0 maps to Qiskit qubit 1 and site 1 maps to qubit 0.
SYSTEM_QUBITS_BY_SITE = (1, 0)
BATH_QUBITS_BY_SITE = (2, 3)
JUMP_CONTROL_QUBIT = 4
TOTAL_CIRCUIT_QUBITS = 5

OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "figures"
    / "unitary_lie_trotter.png"
)


# ============================================================
# Aer circuit
# ============================================================

def build_lie_trotter_circuit(times):
    """Construct the controlled-dilation Lie--Trotter circuit."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must contain at least two values")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must contain only finite values")
    if TROTTER_STEPS_PER_INTERVAL < 1:
        raise ValueError("TROTTER_STEPS_PER_INTERVAL must be positive")

    intervals = np.diff(times)
    if np.any(intervals <= 0.0):
        raise ValueError("times must be strictly increasing")
    if not np.allclose(intervals, intervals[0]):
        raise ValueError("the Aer circuit requires a uniform time grid")

    dt = intervals[0] / TROTTER_STEPS_PER_INTERVAL

    # XXPlusYYGate(theta) = exp[-i theta (XX+YY) / 4]. Hence theta=2Jdt
    # implements exp[-i dt J(XX+YY)/2], the Experiment 1 XY Hamiltonian.
    system_exchange_gate = XXPlusYYGate(
        theta=2.0 * J * dt,
        beta=0.0,
        label="XY system",
    )

    # The exact amplitude-damping channel for a duration dt has
    # p = 1-exp(-gamma*dt). Coupling a system qubit to a |0> bath with an
    # XXPlusYYGate(theta_jump) gives p = sin(theta_jump/2)^2.
    damping_probability = 1.0 - np.exp(-gamma * dt)
    jump_angle = 2.0 * np.arcsin(np.sqrt(damping_probability))
    controlled_jump_gate = XXPlusYYGate(
        theta=jump_angle,
        beta=0.0,
        label="jump dilation",
    ).control(
        num_ctrl_qubits=1,
        ctrl_state=1,
        label="controlled jump",
    )

    circuit = QuantumCircuit(TOTAL_CIRCUIT_QUBITS)

    # The control remains in |1>, enabling both boundary jump dilations.
    circuit.x(JUMP_CONTROL_QUBIT)
    for site in INITIALLY_EXCITED_SITES:
        circuit.x(SYSTEM_QUBITS_BY_SITE[site])

    circuit.save_density_matrix(
        list(reversed(SYSTEM_QUBITS_BY_SITE)),
        label="rho_0",
    )

    for time_index in range(1, times.size):
        for _ in range(TROTTER_STEPS_PER_INTERVAL):
            # Full coherent system step.
            circuit.append(system_exchange_gate, SYSTEM_QUBITS_BY_SITE)
            for system_qubit in SYSTEM_QUBITS_BY_SITE:
                circuit.rz(h * dt, system_qubit)

            # Full left and right jump steps. Resetting each bath realizes the
            # partial trace and prepares |0> for the next substep.
            for system_qubit, bath_qubit in zip(
                SYSTEM_QUBITS_BY_SITE,
                BATH_QUBITS_BY_SITE,
            ):
                circuit.append(
                    controlled_jump_gate,
                    [JUMP_CONTROL_QUBIT, system_qubit, bath_qubit],
                )
                circuit.reset(bath_qubit)

        # Qargs [q0, q1] produce a matrix in |q1 q0> order, matching
        # classical.py's site-0 tensor site-1 convention under our mapping.
        circuit.save_density_matrix([0, 1], label=f"rho_{time_index}")

    return circuit, dt, damping_probability, jump_angle


def simulate_with_aer(circuit, number_of_times):
    """Run the dilation circuit and return reduced system density matrices."""
    simulator = AerSimulator(method="density_matrix")
    compiled_circuit = transpile(circuit, simulator, optimization_level=1)
    result = simulator.run(compiled_circuit).result()

    if not result.success:
        raise RuntimeError(f"Aer simulation failed: {result.status}")

    data = result.data(0)
    density_matrices = np.asarray([
        np.asarray(data[f"rho_{index}"], dtype=complex)
        for index in range(number_of_times)
    ])
    return density_matrices, compiled_circuit


# ============================================================
# Reference solution and analysis
# ============================================================

def calculate_exact_solution(times):
    """Build and evolve the model entirely with library.classical."""
    (
        hamiltonian,
        jump_operators,
        x_operators,
        y_operators,
        z_operators,
    ) = classical.build_linear_chain(
        number_of_qubits=NUMBER_OF_QUBITS,
        J=J,
        h=h,
        gamma=gamma,
    )

    liouvillian = classical.build_liouvillian(
        hamiltonian,
        jump_operators,
    )
    initial_state = classical.computational_state(
        number_of_qubits=NUMBER_OF_QUBITS,
        excited_sites=INITIALLY_EXCITED_SITES,
    )
    initial_density_matrix = np.outer(initial_state, initial_state.conj())
    density_matrices = classical.exact_evolve(
        initial_density_matrix,
        liouvillian,
        times,
    )

    identity = eye(2**NUMBER_OF_QUBITS, dtype=complex, format="csr")
    observables = classical.build_observables(
        identity=identity,
        X_ops=x_operators,
        Y_ops=y_operators,
        Z_ops=z_operators,
        J=J,
    )
    return density_matrices, initial_density_matrix, observables


def evaluate_observables(density_matrices, observables):
    """Evaluate populations, exchange correlation, and excitation flow."""
    population_operators, exchange_operators, flow_operators = observables
    populations = np.asarray([
        classical.expectation_series(density_matrices, operator)
        for operator in population_operators
    ])
    exchange_correlations = np.asarray([
        classical.expectation_series(density_matrices, operator)
        for operator in exchange_operators
    ])
    excitation_flows = np.asarray([
        classical.expectation_series(density_matrices, operator)
        for operator in flow_operators
    ])
    return populations, exchange_correlations, excitation_flows


def plot_results(
    times,
    aer_observables,
    exact_observables,
    density_matrix_errors,
):
    """Plot the Aer trajectory and the unsplit classical reference."""
    aer_populations, aer_correlations, aer_flows = aer_observables
    exact_populations, exact_correlations, exact_flows = exact_observables

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = ("tab:blue", "tab:orange")

    for site, color in enumerate(colors):
        axes[0, 0].plot(
            times,
            aer_populations[site],
            color=color,
            label=f"site {site}, Aer",
        )
        axes[0, 0].plot(
            times,
            exact_populations[site],
            color=color,
            linestyle="--",
            label=f"site {site}, exact",
        )
    axes[0, 0].set_title("Local excitation populations")
    axes[0, 0].set_ylabel(r"$\langle n_i\rangle$")
    axes[0, 0].legend(ncol=2)

    axes[0, 1].plot(times, aer_correlations[0], label="Aer")
    axes[0, 1].plot(
        times,
        exact_correlations[0],
        linestyle="--",
        label="exact",
    )
    axes[0, 1].set_title("Nearest-neighbor XY correlation")
    axes[0, 1].set_ylabel(r"$C_0^{XY}$")
    axes[0, 1].legend()

    axes[1, 0].plot(times, aer_flows[0], label="Aer")
    axes[1, 0].plot(
        times,
        exact_flows[0],
        linestyle="--",
        label="exact",
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Excitation flow")
    axes[1, 0].set_ylabel(r"$I_{0\rightarrow1}$")
    axes[1, 0].legend()

    displayed_errors = np.maximum(density_matrix_errors, np.finfo(float).eps)
    axes[1, 1].semilogy(times, displayed_errors, color="tab:red")
    axes[1, 1].set_title("Error relative to unsplit evolution")
    axes[1, 1].set_ylabel(r"$\|\rho_{Aer}-\rho_{exact}\|_F$")

    for axis in axes.flat:
        axis.set_xlabel("Time")
        axis.grid(alpha=0.3)

    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    times = np.linspace(0.0, T_FINAL, NUMBER_OF_TIME_POINTS)

    (
        exact_density_matrices,
        initial_density_matrix,
        observables,
    ) = calculate_exact_solution(times)

    circuit, dt, damping_probability, jump_angle = (
        build_lie_trotter_circuit(times)
    )
    aer_density_matrices, compiled_circuit = simulate_with_aer(
        circuit,
        times.size,
    )

    if not np.allclose(aer_density_matrices[0], initial_density_matrix):
        raise RuntimeError("Aer and classical initial-state orderings disagree")

    aer_observables = evaluate_observables(aer_density_matrices, observables)
    exact_observables = evaluate_observables(
        exact_density_matrices,
        observables,
    )
    density_matrix_errors = np.linalg.norm(
        aer_density_matrices - exact_density_matrices,
        axis=(1, 2),
    )

    traces = np.trace(aer_density_matrices, axis1=1, axis2=2)
    hermiticity_errors = np.linalg.norm(
        aer_density_matrices
        - aer_density_matrices.conj().transpose(0, 2, 1),
        axis=(1, 2),
    )
    minimum_eigenvalue = min(
        np.linalg.eigvalsh(density_matrix).min()
        for density_matrix in aer_density_matrices
    )

    plot_results(
        times,
        aer_observables,
        exact_observables,
        density_matrix_errors,
    )

    print("Two-qubit boundary-damped XY chain on Qiskit Aer")
    print(f"J = {J}, h = {h}, gamma = {gamma}")
    print(f"Initially excited sites: {INITIALLY_EXCITED_SITES}")
    print(f"System qubits: {SYSTEM_QUBITS_BY_SITE}")
    print(f"Bath qubits: {BATH_QUBITS_BY_SITE}")
    print(f"Jump control qubit: {JUMP_CONTROL_QUBIT}")
    print(f"Lie--Trotter substep: {dt:.6f}")
    print(f"Damping probability per substep: {damping_probability:.8f}")
    print(f"Controlled-jump angle: {jump_angle:.8f}")
    print(f"Transpiled circuit depth: {compiled_circuit.depth()}")

    print("\nDensity-matrix checks")
    print(f"Maximum trace error: {np.max(np.abs(traces - 1.0)):.3e}")
    print(f"Maximum Hermiticity error: {np.max(hermiticity_errors):.3e}")
    print(f"Smallest eigenvalue: {minimum_eigenvalue:.3e}")

    print("\nError relative to classical.exact_evolve")
    print(f"Maximum Frobenius error: {density_matrix_errors.max():.3e}")
    print(f"Final Frobenius error: {density_matrix_errors[-1]:.3e}")
    print(f"\nSaved plot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
