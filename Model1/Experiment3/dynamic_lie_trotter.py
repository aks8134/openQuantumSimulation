"""Optimized general-N dynamic Lie circuit for Model Problem I.

At the beginning of every physical dilation step, the two shared-ancilla
qubits are known to be in ``|a_H a_L> = |00>``. Consequently, the even and
odd system layers need no quantum controls, and neither does the first
boundary jump. After the first jump, measuring ``a_L`` replaces the open
quantum control of the second jump with classical feed-forward:

    H_even -> H_odd -> H_field -> K_1 -> measure(a_L)
        -> if c=0: K_2 -> reset(a_L, a_H).

Every even or odd layer contains disjoint, mutually commuting bonds and is
implemented by ordinary ``XXPlusYYGate`` factors. As in the unitary script,
there is one even layer and one odd layer per outer dilation step and no
separate inner system-Trotter loop.
"""

import argparse
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
from Model1.Experiment2 import unitary_lie_trotter as plot_tools
from Model1.Experiment3 import unitary_lie_trotter as circuit_tools


# ============================================================
# Configuration
# ============================================================

DEFAULT_NUMBER_OF_QUBITS = circuit_tools.DEFAULT_NUMBER_OF_QUBITS

J = circuit_tools.J
h = circuit_tools.h
gamma = circuit_tools.gamma

T_FINAL = circuit_tools.T_FINAL
NUMBER_OF_TIME_POINTS = circuit_tools.NUMBER_OF_TIME_POINTS
TROTTER_STEPS_PER_INTERVAL = circuit_tools.TROTTER_STEPS_PER_INTERVAL
ESTIMATOR_PRECISION = circuit_tools.ESTIMATOR_PRECISION
SEED_SIMULATOR = circuit_tools.SEED_SIMULATOR

LOW_ANCILLA_MEASUREMENT_BIT = 0
TOTAL_CLASSICAL_BITS = 1


def output_paths(number_of_qubits):
    figure_directory = Path(__file__).resolve().parent / "figures"
    return (
        figure_directory / f"dynamic_lie_trotter_{number_of_qubits}.png",
        figure_directory
        / f"dynamic_lie_transpilation_metrics_{number_of_qubits}.png",
        figure_directory
        / f"one_step_dynamic_lie_trotter_circuit_{number_of_qubits}.png",
    )


# ============================================================
# General-N dynamic Lie circuit
# ============================================================

def _uncontrolled_bond_layer(
    circuit,
    bonds,
    system_exchange_angle,
    system_qubits,
    *,
    layer_name,
):
    """Apply one commuting parity layer without ancilla controls."""
    return fold_left(
        lambda current, left_site: pipe(
            current,
            xx_plus_yy(
                system_exchange_angle,
                0.0,
                system_qubits[left_site],
                system_qubits[left_site + 1],
                label=f"{layer_name}[{left_site},{left_site + 1}]",
            ),
        ),
        circuit,
        bonds,
    )


def _uncontrolled_field_layer(circuit, field_angle, system_qubits):
    if field_angle == 0.0:
        return circuit

    return fold_left(
        lambda current, system_qubit: pipe(
            current,
            rz(field_angle, system_qubit),
        ),
        circuit,
        system_qubits,
    )


def _uncontrolled_system_factor(
    circuit,
    number_of_qubits,
    system_exchange_angle,
    field_angle,
    system_qubits,
):
    """Apply one even layer and one odd layer while the ancilla is |00>."""
    even_evolved = _uncontrolled_bond_layer(
        circuit,
        circuit_tools.even_bonds(number_of_qubits),
        system_exchange_angle,
        system_qubits,
        layer_name="H_even",
    )
    odd_evolved = _uncontrolled_bond_layer(
        even_evolved,
        circuit_tools.odd_bonds(number_of_qubits),
        system_exchange_angle,
        system_qubits,
        layer_name="H_odd",
    )
    return _uncontrolled_field_layer(
        odd_evolved,
        field_angle,
        system_qubits,
    )


def _uncontrolled_first_jump(
    circuit,
    jump_angle,
    system_qubits,
    shared_ancilla_qubits,
):
    """Apply K_1 directly because a_H is known to be zero."""
    ancilla_low, _ = shared_ancilla_qubits
    return pipe(
        circuit,
        xx_plus_yy(
            jump_angle,
            0.0,
            system_qubits[0],
            ancilla_low,
            label="K_1",
        ),
    )


def _classically_controlled_second_jump(
    circuit,
    jump_angle,
    system_qubits,
    shared_ancilla_qubits,
):
    """Measure a_L and apply K_2 only on the measured-zero branch."""
    ancilla_low, ancilla_high = shared_ancilla_qubits
    return pipe(
        circuit,
        measure(ancilla_low, LOW_ANCILLA_MEASUREMENT_BIT),
        classically_controlled_xx_plus_yy(
            jump_angle,
            0.0,
            LOW_ANCILLA_MEASUREMENT_BIT,
            0,
            system_qubits[-1],
            ancilla_high,
            label="K_2",
        ),
    )


def _reset_shared_ancilla(circuit, shared_ancilla_qubits):
    return fold_left(
        lambda current, ancilla_qubit: pipe(
            current,
            reset(ancilla_qubit),
        ),
        circuit,
        shared_ancilla_qubits,
    )


def _dynamic_lie_substep(
    circuit,
    number_of_qubits,
    system_exchange_angle,
    jump_angle,
    dt,
    system_qubits,
    shared_ancilla_qubits,
):
    return pipe(
        circuit,
        lambda current: _uncontrolled_system_factor(
            current,
            number_of_qubits,
            system_exchange_angle,
            h * dt,
            system_qubits,
        ),
        lambda current: _uncontrolled_first_jump(
            current,
            jump_angle,
            system_qubits,
            shared_ancilla_qubits,
        ),
        lambda current: _classically_controlled_second_jump(
            current,
            jump_angle,
            system_qubits,
            shared_ancilla_qubits,
        ),
        lambda current: _reset_shared_ancilla(
            current,
            shared_ancilla_qubits,
        ),
    )


def _time_interval(
    circuit,
    number_of_qubits,
    substep_dts,
    system_qubits,
    shared_ancilla_qubits,
):
    return fold_left(
        lambda current, dt: _dynamic_lie_substep(
            current,
            number_of_qubits,
            2.0 * J * dt,
            2.0 * np.sqrt(gamma * dt),
            dt,
            system_qubits,
            shared_ancilla_qubits,
        ),
        circuit,
        substep_dts,
    )


def _single_interval_substeps(interval, trotter_delta_t):
    if trotter_delta_t is None:
        dt = interval / TROTTER_STEPS_PER_INTERVAL
        return tuple(
            float(dt) for _ in range(TROTTER_STEPS_PER_INTERVAL)
        )

    delta_t = float(trotter_delta_t)
    if not np.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("trotter_delta_t must be finite and positive")
    step_count = max(
        1,
        int(np.ceil(interval / delta_t - 1e-12)),
    )
    final_dt = interval - (step_count - 1) * delta_t
    return (
        *(float(delta_t) for _ in range(step_count - 1)),
        float(final_dt),
    )


def _interval_substep_schedule(intervals, trotter_delta_t):
    return tuple(
        _single_interval_substeps(interval, trotter_delta_t)
        for interval in intervals
    )


def build_dynamic_lie_trotter_circuits_with_schedule(
    times,
    number_of_qubits,
    trotter_delta_t=None,
):
    """Construct prefixes using the actual duration of every interval."""
    if number_of_qubits < 2:
        raise ValueError("number_of_qubits must be at least two")

    time_grid = np.asarray(times, dtype=float)
    if time_grid.ndim != 1 or time_grid.size < 2:
        raise ValueError("times must contain at least two values")
    if not np.all(np.isfinite(time_grid)):
        raise ValueError("times must contain only finite values")
    if not np.isclose(time_grid[0], 0.0):
        raise ValueError("times must start at zero")
    if TROTTER_STEPS_PER_INTERVAL < 1:
        raise ValueError("TROTTER_STEPS_PER_INTERVAL must be positive")

    intervals = np.diff(time_grid)
    if np.any(intervals <= 0.0):
        raise ValueError("times must be strictly increasing")
    interval_substep_dts = _interval_substep_schedule(
        intervals,
        trotter_delta_t,
    )
    interval_jump_angles = tuple(
        tuple(2.0 * np.sqrt(gamma * dt) for dt in substeps)
        for substeps in interval_substep_dts
    )
    interval_dilation_exchange_probabilities = tuple(
        tuple(np.sin(0.5 * angle) ** 2 for angle in angles)
        for angles in interval_jump_angles
    )
    system_qubits = circuit_tools.system_qubits_by_site(number_of_qubits)
    shared_ancilla_qubits = circuit_tools.ancilla_qubits(number_of_qubits)
    initially_excited_sites = (number_of_qubits // 2,)

    initial_circuit = pipe(
        empty(
            number_of_qubits + 2,
            TOTAL_CLASSICAL_BITS,
            name=f"dynamic_lie_trotter_N{number_of_qubits}",
        ),
        *map_tuple(
            lambda site: x(system_qubits[site]),
            initially_excited_sites,
        ),
    )
    evolution_circuits = scan_left(
        lambda current, interval_index: _time_interval(
            current,
            number_of_qubits,
            interval_substep_dts[interval_index],
            system_qubits,
            shared_ancilla_qubits,
        ),
        initial_circuit,
        range(len(intervals)),
    )

    return (
        evolution_circuits,
        interval_substep_dts,
        interval_dilation_exchange_probabilities,
        interval_jump_angles,
    )


def build_dynamic_lie_trotter_circuits(times, number_of_qubits):
    """Construct prefixes on the uniform grid used by Experiment 3."""
    time_grid = np.asarray(times, dtype=float)
    (
        evolution_circuits,
        substep_dts,
        dilation_exchange_probabilities,
        jump_angles,
    ) = build_dynamic_lie_trotter_circuits_with_schedule(
        time_grid,
        number_of_qubits,
    )
    flat_substep_dts = tuple(
        dt for interval in substep_dts for dt in interval
    )
    if not np.allclose(flat_substep_dts, flat_substep_dts[0]):
        raise ValueError("the Aer circuit requires a uniform time grid")
    return (
        evolution_circuits,
        substep_dts[0][0],
        dilation_exchange_probabilities[0][0],
        jump_angles[0][0],
    )


def build_one_step_circuit(number_of_qubits, dt, jump_angle):
    """Return one isolated general-N dynamic Lie step."""
    if number_of_qubits < 2:
        raise ValueError("number_of_qubits must be at least two")

    system_qubits = circuit_tools.system_qubits_by_site(number_of_qubits)
    shared_ancilla_qubits = circuit_tools.ancilla_qubits(number_of_qubits)
    return _dynamic_lie_substep(
        empty(
            number_of_qubits + 2,
            TOTAL_CLASSICAL_BITS,
            name=f"one_dynamic_lie_trotter_step_N{number_of_qubits}",
        ),
        number_of_qubits,
        2.0 * J * dt,
        jump_angle,
        dt,
        system_qubits,
        shared_ancilla_qubits,
    )


# ============================================================
# Figures and driver
# ============================================================

def plot_one_step_circuit(circuit, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = draw_circuit(circuit)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main(number_of_qubits=DEFAULT_NUMBER_OF_QUBITS):
    if number_of_qubits < 2:
        raise ValueError("number_of_qubits must be at least two")

    times = np.linspace(0.0, T_FINAL, NUMBER_OF_TIME_POINTS)
    system_qubits = circuit_tools.system_qubits_by_site(number_of_qubits)
    shared_ancilla_qubits = circuit_tools.ancilla_qubits(number_of_qubits)
    estimator_observables = circuit_tools.build_estimator_observables(
        number_of_qubits,
        system_qubits,
    )
    (
        exact_density_matrices,
        coherent_lie_density_matrices,
        reference_observables,
    ) = circuit_tools.calculate_reference_solutions(
        times,
        number_of_qubits,
    )
    exact_observables = circuit_tools.evaluate_observables(
        exact_density_matrices,
        reference_observables,
    )
    coherent_lie_observables = circuit_tools.evaluate_observables(
        coherent_lie_density_matrices,
        reference_observables,
    )

    (
        evolution_circuits,
        dt,
        dilation_exchange_probability,
        jump_angle,
    ) = build_dynamic_lie_trotter_circuits(times, number_of_qubits)
    (
        aer_observables,
        estimator_standard_errors,
        maximum_compiled_depth,
        full_circuit_metrics,
    ) = circuit_tools.simulate_with_aer(
        evolution_circuits,
        number_of_qubits,
        estimator_observables,
    )
    lindblad_errors = circuit_tools.aggregate_observable_error(
        aer_observables,
        exact_observables,
    )
    coherent_lie_errors = circuit_tools.aggregate_observable_error(
        aer_observables,
        coherent_lie_observables,
    )
    maximum_standard_error = max(
        map(lambda values: float(np.max(values)), estimator_standard_errors)
    )

    one_step_circuit = build_one_step_circuit(
        number_of_qubits,
        dt,
        jump_angle,
    )
    one_step_metrics = circuit_tools.transpilation_metrics(
        one_step_circuit,
        estimator_observables,
    )
    results_path, metrics_path, circuit_path = output_paths(
        number_of_qubits
    )
    circuit_tools.plot_results(
        times,
        number_of_qubits,
        aer_observables,
        exact_observables,
        lindblad_errors,
        coherent_lie_errors,
        results_path,
    )
    plot_tools.plot_transpilation_metrics(
        one_step_metrics,
        full_circuit_metrics,
        output_path=metrics_path,
        method_name=f"N={number_of_qubits} dynamic even/odd Lie",
    )
    plot_one_step_circuit(one_step_circuit, circuit_path)

    print("General-N boundary-damped XY chain: dynamic Lie circuit")
    print(f"Number of system qubits: {number_of_qubits}")
    print(f"Total circuit qubits: {number_of_qubits + 2}")
    print(f"System qubits by site: {system_qubits}")
    print(f"Ancilla qubits (low, high): {shared_ancilla_qubits}")
    print(f"Even bonds: {circuit_tools.even_bonds(number_of_qubits)}")
    print(f"Odd bonds: {circuit_tools.odd_bonds(number_of_qubits)}")
    print("System layers per dilation step: one even, then one odd")
    print(f"Classical feed-forward bit: {LOW_ANCILLA_MEASUREMENT_BIT}")
    print(f"Lie--Trotter substep: {dt:.6f}")
    print(
        "Full jump-factor exchange probability: "
        f"{dilation_exchange_probability:.8f}"
    )
    print(f"System-bond XXPlusYY angle: {2.0 * J * dt:.8f}")
    print(f"Jump-dilation XXPlusYY angle: {jump_angle:.8f}")
    print(f"Estimator target precision: {ESTIMATOR_PRECISION:.8f}")
    print(f"Maximum reported standard error: {maximum_standard_error:.8f}")
    print(f"Maximum transpiled circuit depth: {maximum_compiled_depth}")

    one_pre, one_post = one_step_metrics
    full_pre, full_post = full_circuit_metrics
    print("\nOne-step transpilation metrics")
    print(
        "Pre-transpilation: operation count = "
        f"{one_pre[0]}, depth = {one_pre[1]}"
    )
    print(
        "Post-transpilation: operation count = "
        f"{one_post[0]}, depth = {one_post[1]}"
    )
    print("\nFull-circuit transpilation metrics")
    print(
        "Pre-transpilation: operation count = "
        f"{full_pre[0]}, depth = {full_pre[1]}"
    )
    print(
        "Post-transpilation: operation count = "
        f"{full_post[0]}, depth = {full_post[1]}"
    )

    print("\nEstimator observable error relative to exact Lindblad evolution")
    print(f"Maximum error: {lindblad_errors.max():.3e}")
    print(f"Final error: {lindblad_errors[-1]:.3e}")
    print("\nEstimator observable error relative to coherent Lie dilation")
    print(f"Maximum error: {coherent_lie_errors.max():.3e}")
    print(f"Final error: {coherent_lie_errors[-1]:.3e}")
    print(f"\nSaved results: {results_path}")
    print(f"Saved transpilation metrics: {metrics_path}")
    print(f"Saved one-step circuit: {circuit_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Optimized general-N dynamic Lie circuit with one even/odd "
            "system split."
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
