"""Dynamic four-qubit Lie circuit for Model Problem I.

At the start of every physical step, resets guarantee that the shared
ancilla encoding is |a_H a_L> = |00>. This lets the circuit replace the
controlled system factor and the first controlled jump by ordinary
``XXPlusYYGate`` operations. After the first jump, a mid-circuit measurement
of a_L selects the second jump through classical feed-forward:

    K_H -> K_1 -> measure(a_L) -> if c=0: K_2 -> reset(a_L, a_H).

The measurement must occur after K_1 and before K_2. Measuring after K_2
could not replace K_2's open quantum control. The intermediate measurement
does not change the reduced system channel because K_2 is block diagonal in
a_L and the ancilla is traced/reset immediately afterward.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from IBMRuntime import (
    classically_controlled_xx_plus_yy,
    draw_circuit,
    empty,
    fold_left,
    map_tuple,
    measure,
    pipe,
    reset,
    rz,
    scan_left,
    x,
    xx_plus_yy,
)
from library import classical
from Model1.Experiment2 import unitary_lie_trotter as circuit_tools


# ============================================================
# Configuration
# ============================================================

NUMBER_OF_QUBITS = circuit_tools.NUMBER_OF_QUBITS

J = circuit_tools.J
h = circuit_tools.h
gamma = circuit_tools.gamma

T_FINAL = circuit_tools.T_FINAL
NUMBER_OF_TIME_POINTS = circuit_tools.NUMBER_OF_TIME_POINTS
TROTTER_STEPS_PER_INTERVAL = circuit_tools.TROTTER_STEPS_PER_INTERVAL
ESTIMATOR_PRECISION = circuit_tools.ESTIMATOR_PRECISION
SEED_SIMULATOR = circuit_tools.SEED_SIMULATOR

INITIALLY_EXCITED_SITES = circuit_tools.INITIALLY_EXCITED_SITES
SYSTEM_QUBITS_BY_SITE = circuit_tools.SYSTEM_QUBITS_BY_SITE
ANCILLA_LOW_QUBIT = circuit_tools.ANCILLA_LOW_QUBIT
ANCILLA_HIGH_QUBIT = circuit_tools.ANCILLA_HIGH_QUBIT
ANCILLA_QUBITS = circuit_tools.ANCILLA_QUBITS
TOTAL_CIRCUIT_QUBITS = circuit_tools.TOTAL_CIRCUIT_QUBITS
ESTIMATOR_OBSERVABLES = circuit_tools.ESTIMATOR_OBSERVABLES

LOW_ANCILLA_MEASUREMENT_BIT = 0
TOTAL_CLASSICAL_BITS = 1

OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "figures"
    / "dynamic_lie_trotter.png"
)
TRANSPILATION_METRICS_OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "figures"
    / "dynamic_lie_transpilation_metrics.png"
)
ONE_STEP_CIRCUIT_OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "figures"
    / "one_step_dynamic_lie_trotter_circuit.png"
)


# ============================================================
# Dynamic Lie circuit
# ============================================================

def _uncontrolled_system_factor(
    circuit,
    system_exchange_angle,
    field_angle,
):
    """Apply K_H without ancilla controls, which start in |00>."""
    exchange_circuit = pipe(
        circuit,
        xx_plus_yy(
            system_exchange_angle,
            0.0,
            *SYSTEM_QUBITS_BY_SITE,
            label="K_H",
        ),
    )
    if field_angle == 0.0:
        return exchange_circuit

    return fold_left(
        lambda current, system_qubit: pipe(
            current,
            rz(field_angle, system_qubit),
        ),
        exchange_circuit,
        SYSTEM_QUBITS_BY_SITE,
    )


def _uncontrolled_first_jump(circuit, jump_angle):
    """Apply K_1 without the q3 control, since q3 is known to be zero."""
    return pipe(
        circuit,
        xx_plus_yy(
            jump_angle,
            0.0,
            SYSTEM_QUBITS_BY_SITE[0],
            ANCILLA_LOW_QUBIT,
            label="K_1",
        ),
    )


def _classically_controlled_second_jump(circuit, jump_angle):
    """Measure q2 and apply K_2 only for the measured-zero branch."""
    return pipe(
        circuit,
        measure(
            ANCILLA_LOW_QUBIT,
            LOW_ANCILLA_MEASUREMENT_BIT,
        ),
        classically_controlled_xx_plus_yy(
            jump_angle,
            0.0,
            LOW_ANCILLA_MEASUREMENT_BIT,
            0,
            SYSTEM_QUBITS_BY_SITE[1],
            ANCILLA_HIGH_QUBIT,
            label="K_2",
        ),
    )


def _reset_shared_ancilla(circuit):
    return pipe(
        circuit,
        reset(ANCILLA_LOW_QUBIT),
        reset(ANCILLA_HIGH_QUBIT),
    )


def _dynamic_lie_substep(
    circuit,
    system_exchange_angle,
    jump_angle,
    dt,
):
    return pipe(
        circuit,
        lambda current: _uncontrolled_system_factor(
            current,
            system_exchange_angle,
            h * dt,
        ),
        lambda current: _uncontrolled_first_jump(current, jump_angle),
        lambda current: _classically_controlled_second_jump(
            current,
            jump_angle,
        ),
        _reset_shared_ancilla,
    )


def _time_interval(
    circuit,
    system_exchange_angle,
    jump_angle,
    dt,
):
    return fold_left(
        lambda current, _: _dynamic_lie_substep(
            current,
            system_exchange_angle,
            jump_angle,
            dt,
        ),
        circuit,
        range(TROTTER_STEPS_PER_INTERVAL),
    )


def build_dynamic_lie_trotter_circuits(times):
    """Construct one immutable dynamic-circuit prefix per output time."""
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
    jump_angle = 2.0 * np.sqrt(gamma * dt)
    dilation_exchange_probability = np.sin(0.5 * jump_angle) ** 2

    initial_circuit = pipe(
        empty(
            TOTAL_CIRCUIT_QUBITS,
            TOTAL_CLASSICAL_BITS,
            name="dynamic_lie_trotter",
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
            jump_angle,
            dt,
        ),
        initial_circuit,
        range(1, time_grid.size),
    )

    return (
        evolution_circuits,
        dt,
        dilation_exchange_probability,
        jump_angle,
    )


def build_one_step_circuit(dt, jump_angle):
    """Return one isolated dynamic Lie step for verification and metrics."""
    return _dynamic_lie_substep(
        empty(
            TOTAL_CIRCUIT_QUBITS,
            TOTAL_CLASSICAL_BITS,
            name="one_dynamic_lie_trotter_step",
        ),
        2.0 * J * dt,
        jump_angle,
        dt,
    )


def plot_one_step_circuit(dt, jump_angle):
    ONE_STEP_CIRCUIT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure = draw_circuit(build_one_step_circuit(dt, jump_angle))
    figure.savefig(
        ONE_STEP_CIRCUIT_OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


# ============================================================
# Simulation and reporting
# ============================================================

def calculate_lie_dilation_solution(times):
    """Return the coherent Lie-dilation trajectory for verification."""
    hamiltonian, jump_operators, *_ = classical.build_linear_chain(
        number_of_qubits=NUMBER_OF_QUBITS,
        J=J,
        h=h,
        gamma=gamma,
    )
    initial_state = classical.computational_state(
        number_of_qubits=NUMBER_OF_QUBITS,
        excited_sites=INITIALLY_EXCITED_SITES,
    )
    initial_density_matrix = np.outer(
        initial_state,
        initial_state.conj(),
    )
    return classical.hamiltonian_dilation_evolve(
        initial_density_matrix,
        hamiltonian,
        jump_operators,
        times,
        product_formula="lie",
        steps_per_interval=TROTTER_STEPS_PER_INTERVAL,
    )


def _aggregate_observable_error(first, second):
    return np.sqrt(
        sum(
            map(
                lambda pair: np.sum((pair[0] - pair[1]) ** 2, axis=0),
                zip(first, second),
            )
        )
    )


def main():
    times = np.linspace(0.0, T_FINAL, NUMBER_OF_TIME_POINTS)
    exact_density_matrices, observables = (
        circuit_tools.calculate_exact_solution(times)
    )
    lie_dilation_density_matrices = calculate_lie_dilation_solution(times)

    (
        evolution_circuits,
        dt,
        dilation_exchange_probability,
        jump_angle,
    ) = build_dynamic_lie_trotter_circuits(times)
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

    exact_observables = circuit_tools.evaluate_observables(
        exact_density_matrices,
        observables,
    )
    lie_dilation_observables = circuit_tools.evaluate_observables(
        lie_dilation_density_matrices,
        observables,
    )
    lindblad_errors = _aggregate_observable_error(
        aer_observables,
        exact_observables,
    )
    coherent_lie_errors = _aggregate_observable_error(
        aer_observables,
        lie_dilation_observables,
    )
    maximum_standard_error = max(
        map(lambda values: float(np.max(values)), estimator_standard_errors)
    )

    one_step_metrics = circuit_tools.transpilation_metrics(
        build_one_step_circuit(dt, jump_angle),
        estimator_observables=ESTIMATOR_OBSERVABLES,
    )
    circuit_tools.plot_results(
        times,
        aer_observables,
        exact_observables,
        lindblad_errors,
        output_path=OUTPUT_PATH,
        simulation_label="Aer dynamic Lie",
    )
    circuit_tools.plot_transpilation_metrics(
        one_step_metrics,
        full_circuit_metrics,
        output_path=TRANSPILATION_METRICS_OUTPUT_PATH,
        method_name="Dynamic Lie",
    )
    plot_one_step_circuit(dt, jump_angle)

    print("Two-qubit boundary-damped XY chain: dynamic Lie circuit")
    print(f"J = {J}, h = {h}, gamma = {gamma}")
    print(f"Initially excited sites: {INITIALLY_EXCITED_SITES}")
    print(f"System qubits by site: {SYSTEM_QUBITS_BY_SITE}")
    print(f"Ancilla qubits (low, high): {ANCILLA_QUBITS}")
    print(f"Classical feed-forward bit: {LOW_ANCILLA_MEASUREMENT_BIT}")
    print(f"Lie--Trotter substep: {dt:.6f}")
    print(
        "Full jump-factor exchange probability: "
        f"{dilation_exchange_probability:.8f}"
    )
    print(f"Jump-dilation angle: {jump_angle:.8f}")
    print(f"Estimator target precision: {ESTIMATOR_PRECISION:.8f}")
    print(f"Maximum reported standard error: {maximum_standard_error:.8f}")
    print(f"Maximum transpiled circuit depth: {maximum_compiled_depth}")

    one_pre, one_post = one_step_metrics
    full_pre, full_post = full_circuit_metrics
    print("\nOne-step transpilation metrics")
    print(f"Pre-transpilation: operation count = {one_pre[0]}, depth = {one_pre[1]}")
    print(f"Post-transpilation: operation count = {one_post[0]}, depth = {one_post[1]}")
    print("\nFull-circuit transpilation metrics")
    print(f"Pre-transpilation: operation count = {full_pre[0]}, depth = {full_pre[1]}")
    print(f"Post-transpilation: operation count = {full_post[0]}, depth = {full_post[1]}")

    print("\nEstimator observable error relative to exact Lindblad evolution")
    print(f"Maximum error: {lindblad_errors.max():.3e}")
    print(f"Final error: {lindblad_errors[-1]:.3e}")
    print("\nEstimator observable error relative to coherent Lie dilation")
    print(f"Maximum error: {coherent_lie_errors.max():.3e}")
    print(f"Final error: {coherent_lie_errors[-1]:.3e}")
    print(f"\nSaved plot: {OUTPUT_PATH}")
    print(f"Saved transpilation metrics: {TRANSPILATION_METRICS_OUTPUT_PATH}")
    print(f"Saved one-step circuit: {ONE_STEP_CIRCUIT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
