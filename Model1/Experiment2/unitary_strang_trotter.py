"""Four-qubit symmetric circuit for Model Problem I's dilation.

The two system qubits form the boundary-damped XY chain

    H = J/2 (X0 X1 + Y0 Y1) + h/2 (Z0 + Z1),

with amplitude-damping jump operators on both boundary sites. Two ancilla
qubits encode the shared dilation states |0>, |1>, and |2>; |11> is unused.

The symmetric step keeps the full system evolution in the middle and places
half of each boundary-jump dilation on either side:

    K_1/2 -> K_2/2 -> K_H -> K_2/2 -> K_1/2.

The Hamiltonian factors are controlled on ancilla |00>, and every jump
exchange has the complementary open control needed to exclude |11>. Both
ancilla qubits are reset only after the complete palindromic substep.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import eye


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from IBMRuntime import (
    draw_circuit,
    empty,
    fold_left,
    map_tuple,
    observable,
    pauli,
    pauli_term,
    pipe,
    scan_left,
    x,
)
from library import classical
from Model1.Experiment2 import unitary_lie_trotter as circuit_tools


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
ESTIMATOR_PRECISION = 1.0 / np.sqrt(8192)
SEED_SIMULATOR = 17

INITIALLY_EXCITED_SITES = (NUMBER_OF_QUBITS // 2,)

# Match classical.py's site-0 tensor site-1 convention to Qiskit's |q1 q0>.
SYSTEM_QUBITS_BY_SITE = (1, 0)
ANCILLA_LOW_QUBIT = 2
ANCILLA_HIGH_QUBIT = 3
ANCILLA_QUBITS = (ANCILLA_LOW_QUBIT, ANCILLA_HIGH_QUBIT)
TOTAL_CIRCUIT_QUBITS = 4

ESTIMATOR_OBSERVABLES = (
    observable(
        "population_0",
        pauli_term(0.5),
        pauli_term(-0.5, pauli("Z", SYSTEM_QUBITS_BY_SITE[0])),
    ),
    observable(
        "population_1",
        pauli_term(0.5),
        pauli_term(-0.5, pauli("Z", SYSTEM_QUBITS_BY_SITE[1])),
    ),
    observable(
        "exchange_correlation",
        pauli_term(
            1.0,
            pauli("X", SYSTEM_QUBITS_BY_SITE[0]),
            pauli("X", SYSTEM_QUBITS_BY_SITE[1]),
        ),
        pauli_term(
            1.0,
            pauli("Y", SYSTEM_QUBITS_BY_SITE[0]),
            pauli("Y", SYSTEM_QUBITS_BY_SITE[1]),
        ),
    ),
    observable(
        "excitation_flow",
        pauli_term(
            -0.5 * J,
            pauli("X", SYSTEM_QUBITS_BY_SITE[0]),
            pauli("Y", SYSTEM_QUBITS_BY_SITE[1]),
        ),
        pauli_term(
            0.5 * J,
            pauli("Y", SYSTEM_QUBITS_BY_SITE[0]),
            pauli("X", SYSTEM_QUBITS_BY_SITE[1]),
        ),
    ),
)

OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "figures"
    / "unitary_strang_trotter.png"
)
TRANSPILATION_METRICS_OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "figures"
    / "strang_transpilation_metrics.png"
)
ONE_STEP_CIRCUIT_OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "figures"
    / "one_step_strang_trotter_circuit.png"
)


# ============================================================
# Functional symmetric circuit
# ============================================================

def _strang_trotter_substep(
    circuit,
    system_exchange_angle,
    jump_half_step_angle,
    dt,
):
    """Apply the jump-system-jump palindrome, then reset the ancilla."""
    first_boundary_half_step = circuit_tools._jump_dilation_factor(
        circuit,
        0,
        jump_half_step_angle,
        label="K_1/2",
    )
    second_boundary_half_step = circuit_tools._jump_dilation_factor(
        first_boundary_half_step,
        1,
        jump_half_step_angle,
        label="K_2/2",
    )
    system_step = circuit_tools._hamiltonian_dilation_factor(
        second_boundary_half_step,
        system_exchange_angle,
        h * dt,
        label="K_H",
    )
    reflected_second_boundary_half_step = (
        circuit_tools._jump_dilation_factor(
            system_step,
            1,
            jump_half_step_angle,
            label="K_2/2",
        )
    )
    reflected_first_boundary_half_step = (
        circuit_tools._jump_dilation_factor(
            reflected_second_boundary_half_step,
            0,
            jump_half_step_angle,
            label="K_1/2",
        )
    )
    return circuit_tools._reset_shared_ancilla(
        reflected_first_boundary_half_step
    )


def _time_interval(
    circuit,
    system_exchange_angle,
    jump_half_step_angle,
    dt,
):
    return fold_left(
        lambda current, _: _strang_trotter_substep(
            current,
            system_exchange_angle,
            jump_half_step_angle,
            dt,
        ),
        circuit,
        range(TROTTER_STEPS_PER_INTERVAL),
    )


def build_strang_trotter_circuits(times):
    """Construct one immutable Strang evolution prefix per output time."""
    time_grid = np.asarray(times, dtype=float)
    if time_grid.ndim != 1 or time_grid.size < 2:
        raise ValueError("times must contain at least two values")
    if not np.all(np.isfinite(time_grid)):
        raise ValueError("times must contain only finite values")
    if TROTTER_STEPS_PER_INTERVAL < 1:
        raise ValueError("TROTTER_STEPS_PER_INTERVAL must be positive")

    intervals = np.diff(time_grid)
    if np.any(intervals <= 0.0):
        raise ValueError("times must be strictly increasing")
    if not np.allclose(intervals, intervals[0]):
        raise ValueError("the Aer circuit requires a uniform time grid")

    dt = intervals[0] / TROTTER_STEPS_PER_INTERVAL
    system_exchange_angle = 2.0 * J * dt
    jump_half_step_angle = np.sqrt(gamma * dt)
    half_step_exchange_probability = np.sin(
        0.5 * jump_half_step_angle
    ) ** 2

    initial_circuit = pipe(
        empty(
            TOTAL_CIRCUIT_QUBITS,
            0,
            name="unitary_strang_trotter",
        ),
        *map_tuple(
            lambda site: x(SYSTEM_QUBITS_BY_SITE[site]),
            INITIALLY_EXCITED_SITES,
        ),
    )
    evolution_circuits = scan_left(
        lambda current, _: _time_interval(
            current,
            system_exchange_angle,
            jump_half_step_angle,
            dt,
        ),
        initial_circuit,
        range(1, time_grid.size),
    )

    return (
        evolution_circuits,
        dt,
        half_step_exchange_probability,
        jump_half_step_angle,
    )


def build_one_step_circuit(
    dt,
    jump_half_step_angle,
):
    """Return one isolated Strang substep for transpilation metrics."""
    return _strang_trotter_substep(
        empty(
            TOTAL_CIRCUIT_QUBITS,
            0,
            name="one_strang_trotter_step",
        ),
        2.0 * J * dt,
        jump_half_step_angle,
        dt,
    )


def plot_one_step_circuit(
    dt,
    jump_half_step_angle,
):
    """Draw one untranspiled palindromic Strang substep."""
    ONE_STEP_CIRCUIT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure = draw_circuit(
        build_one_step_circuit(
            dt,
            jump_half_step_angle,
        )
    )
    figure.savefig(
        ONE_STEP_CIRCUIT_OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


# ============================================================
# Exact reference and observables
# ============================================================

def calculate_exact_solution(times):
    """Build and evolve the model with library.classical."""
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
    liouvillian = classical.build_liouvillian(hamiltonian, jump_operators)
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
    return density_matrices, observables


def evaluate_observables(density_matrices, observables):
    """Evaluate populations, exchange correlation, and flow."""
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


def main():
    times = np.linspace(0.0, T_FINAL, NUMBER_OF_TIME_POINTS)
    exact_density_matrices, observables = calculate_exact_solution(times)

    (
        evolution_circuits,
        dt,
        half_step_exchange_probability,
        jump_half_step_angle,
    ) = build_strang_trotter_circuits(times)
    (
        aer_observables,
        estimator_standard_errors,
        maximum_compiled_depth,
        full_circuit_metrics,
    ) = circuit_tools.simulate_with_aer(
        evolution_circuits,
        precision=ESTIMATOR_PRECISION,
        seed_simulator=SEED_SIMULATOR,
        estimator_observables=ESTIMATOR_OBSERVABLES,
    )

    exact_observables = evaluate_observables(
        exact_density_matrices,
        observables,
    )
    observable_errors = np.sqrt(
        sum(
            map(
                lambda aer_and_exact: np.sum(
                    (aer_and_exact[0] - aer_and_exact[1]) ** 2,
                    axis=0,
                ),
                zip(aer_observables, exact_observables),
            )
        )
    )
    maximum_standard_error = max(
        map(lambda values: float(np.max(values)), estimator_standard_errors)
    )

    one_step_metrics = circuit_tools.transpilation_metrics(
        build_one_step_circuit(
            dt,
            jump_half_step_angle,
        ),
        estimator_observables=ESTIMATOR_OBSERVABLES,
    )
    circuit_tools.plot_results(
        times,
        aer_observables,
        exact_observables,
        observable_errors,
        output_path=OUTPUT_PATH,
        simulation_label="Aer Strang",
    )
    circuit_tools.plot_transpilation_metrics(
        one_step_metrics,
        full_circuit_metrics,
        output_path=TRANSPILATION_METRICS_OUTPUT_PATH,
        method_name="Strang",
    )
    plot_one_step_circuit(
        dt,
        jump_half_step_angle,
    )

    print("Two-qubit boundary-damped XY chain: Aer Strang splitting")
    print(f"J = {J}, h = {h}, gamma = {gamma}")
    print(f"Initially excited sites: {INITIALLY_EXCITED_SITES}")
    print(f"System qubits by site: {SYSTEM_QUBITS_BY_SITE}")
    print(f"Ancilla qubits (low, high): {ANCILLA_QUBITS}")
    print(f"Strang substep: {dt:.6f}")
    print(
        "Half jump-factor exchange probability: "
        f"{half_step_exchange_probability:.8f}"
    )
    print(f"Jump-dilation half-step angle: {jump_half_step_angle:.8f}")
    print(f"Estimator target precision: {ESTIMATOR_PRECISION:.8f}")
    print(f"Maximum reported standard error: {maximum_standard_error:.8f}")
    print(f"Maximum transpiled circuit depth: {maximum_compiled_depth}")

    one_pre, one_post = one_step_metrics
    full_pre, full_post = full_circuit_metrics
    print("\nOne-step transpilation metrics")
    print(f"Pre-transpilation: gate count = {one_pre[0]}, depth = {one_pre[1]}")
    print(f"Post-transpilation: gate count = {one_post[0]}, depth = {one_post[1]}")
    print("\nFull-circuit transpilation metrics")
    print(f"Pre-transpilation: gate count = {full_pre[0]}, depth = {full_pre[1]}")
    print(f"Post-transpilation: gate count = {full_post[0]}, depth = {full_post[1]}")

    print("\nEstimator observable error relative to exact evolution")
    print(f"Maximum error: {observable_errors.max():.3e}")
    print(f"Final error: {observable_errors[-1]:.3e}")
    print(f"\nSaved plot: {OUTPUT_PATH}")
    print(f"Saved transpilation metrics: {TRANSPILATION_METRICS_OUTPUT_PATH}")
    print(f"Saved one-step circuit: {ONE_STEP_CIRCUIT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
