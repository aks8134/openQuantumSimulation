"""General-N unitary Lie dilation with an even/odd XY-system split.

Model Problem I has an N-qubit open XY chain and two amplitude-damping
jump operators, one at each boundary. Two additional qubits encode the
single shared qutrit ancilla

    |0>_a = |00>, |1>_a = |01>, |2>_a = |10>.

For N > 2, the complete system Hamiltonian is not represented by one gate.
One physical Lie-dilation step therefore uses one internal even/odd product
formula,

    H_even -> H_odd -> H_field -> K_1 -> K_2 -> reset(a_L, a_H).

The disjoint bond terms inside each even or odd layer commute and are
implemented directly by controlled ``XXPlusYYGate`` factors. There is no
separate inner Trotter-step count: every physical dilation step contains
exactly one even layer and one odd layer.
"""

import argparse
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
    Aer,
    CompilerConfig,
    Err,
    Estimate,
    EstimateResult,
    Ok,
    controlled_xx_plus_yy,
    draw_circuit,
    empty,
    execution_plan,
    fold_left,
    map_tuple,
    multi_controlled_rz,
    multi_controlled_xx_plus_yy,
    observable,
    pauli,
    pauli_term,
    pipe,
    reset,
    run_sync,
    scan_left,
    x,
)
from library import classical
from library.plotter import plot_observables
from Model1.Experiment2 import unitary_lie_trotter as plot_tools


# ============================================================
# Configuration
# ============================================================

DEFAULT_NUMBER_OF_QUBITS = 4

J = 1.0
h = 0.0
gamma = 0.2

T_FINAL = 10.0
NUMBER_OF_TIME_POINTS = 51
TROTTER_STEPS_PER_INTERVAL = 1
ESTIMATOR_PRECISION = 1.0 / np.sqrt(8192)
SEED_SIMULATOR = 17


# ============================================================
# Qubit layout and observables
# ============================================================

def system_qubits_by_site(number_of_qubits):
    """Map site ordering to Qiskit's little-endian qubit ordering."""
    return tuple(reversed(range(number_of_qubits)))


def ancilla_qubits(number_of_qubits):
    return (number_of_qubits, number_of_qubits + 1)


def even_bonds(number_of_qubits):
    """Return bonds whose left site has even index."""
    return tuple(range(0, number_of_qubits - 1, 2))


def odd_bonds(number_of_qubits):
    """Return bonds whose left site has odd index."""
    return tuple(range(1, number_of_qubits - 1, 2))


def build_estimator_observables(number_of_qubits, system_qubits):
    populations = map_tuple(
        lambda site: observable(
            f"population_{site}",
            pauli_term(0.5),
            pauli_term(-0.5, pauli("Z", system_qubits[site])),
        ),
        range(number_of_qubits),
    )
    correlations = map_tuple(
        lambda bond: observable(
            f"exchange_correlation_{bond}_{bond + 1}",
            pauli_term(
                1.0,
                pauli("X", system_qubits[bond]),
                pauli("X", system_qubits[bond + 1]),
            ),
            pauli_term(
                1.0,
                pauli("Y", system_qubits[bond]),
                pauli("Y", system_qubits[bond + 1]),
            ),
        ),
        range(number_of_qubits - 1),
    )
    flows = map_tuple(
        lambda bond: observable(
            f"excitation_flow_{bond}_{bond + 1}",
            pauli_term(
                -0.5 * J,
                pauli("X", system_qubits[bond]),
                pauli("Y", system_qubits[bond + 1]),
            ),
            pauli_term(
                0.5 * J,
                pauli("Y", system_qubits[bond]),
                pauli("X", system_qubits[bond + 1]),
            ),
        ),
        range(number_of_qubits - 1),
    )
    return (*populations, *correlations, *flows)


def output_paths(number_of_qubits):
    figure_directory = Path(__file__).resolve().parent / "figures"
    return (
        figure_directory / f"unitary_lie_trotter_{number_of_qubits}.png",
        figure_directory
        / f"lie_transpilation_metrics_{number_of_qubits}.png",
        figure_directory
        / f"one_step_lie_trotter_circuit_{number_of_qubits}.png",
    )


# ============================================================
# General-N unitary circuit
# ============================================================

def _controlled_bond_layer(
    circuit,
    bonds,
    system_exchange_angle,
    system_qubits,
    shared_ancilla_qubits,
    *,
    layer_name,
):
    """Apply all commuting bonds in one even or odd system layer."""
    return fold_left(
        lambda current, left_site: pipe(
            current,
            multi_controlled_xx_plus_yy(
                system_exchange_angle,
                0.0,
                shared_ancilla_qubits,
                system_qubits[left_site],
                system_qubits[left_site + 1],
                control_state=0,
                label=(
                    f"{layer_name}[{left_site},{left_site + 1}]"
                ),
            ),
        ),
        circuit,
        bonds,
    )


def _controlled_field_layer(
    circuit,
    field_angle,
    system_qubits,
    shared_ancilla_qubits,
):
    if field_angle == 0.0:
        return circuit

    return fold_left(
        lambda current, system_qubit: pipe(
            current,
            multi_controlled_rz(
                field_angle,
                shared_ancilla_qubits,
                system_qubit,
                control_state=0,
                label="H_field",
            ),
        ),
        circuit,
        system_qubits,
    )


def _system_dilation_factor(
    circuit,
    number_of_qubits,
    system_exchange_angle,
    field_angle,
    system_qubits,
    shared_ancilla_qubits,
):
    """Apply one even layer and one odd layer, with no inner repetition."""
    even_evolved = _controlled_bond_layer(
        circuit,
        even_bonds(number_of_qubits),
        system_exchange_angle,
        system_qubits,
        shared_ancilla_qubits,
        layer_name="H_even",
    )
    odd_evolved = _controlled_bond_layer(
        even_evolved,
        odd_bonds(number_of_qubits),
        system_exchange_angle,
        system_qubits,
        shared_ancilla_qubits,
        layer_name="H_odd",
    )
    return _controlled_field_layer(
        odd_evolved,
        field_angle,
        system_qubits,
        shared_ancilla_qubits,
    )


def _boundary_jump_factor(
    circuit,
    jump_index,
    jump_angle,
    system_qubits,
    shared_ancilla_qubits,
):
    ancilla_low, ancilla_high = shared_ancilla_qubits
    boundary_layouts = (
        (
            ancilla_high,
            system_qubits[0],
            ancilla_low,
        ),
        (
            ancilla_low,
            system_qubits[-1],
            ancilla_high,
        ),
    )
    control, boundary_system_qubit, active_ancilla = (
        boundary_layouts[jump_index]
    )
    return pipe(
        circuit,
        controlled_xx_plus_yy(
            jump_angle,
            0.0,
            control,
            boundary_system_qubit,
            active_ancilla,
            control_state=0,
            label=f"K_{jump_index + 1}",
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


def _lie_trotter_substep(
    circuit,
    number_of_qubits,
    system_exchange_angle,
    jump_angle,
    dt,
    system_qubits,
    shared_ancilla_qubits,
):
    system_evolved = _system_dilation_factor(
        circuit,
        number_of_qubits,
        system_exchange_angle,
        h * dt,
        system_qubits,
        shared_ancilla_qubits,
    )
    jump_evolved = fold_left(
        lambda current, jump_index: _boundary_jump_factor(
            current,
            jump_index,
            jump_angle,
            system_qubits,
            shared_ancilla_qubits,
        ),
        system_evolved,
        range(2),
    )
    return _reset_shared_ancilla(
        jump_evolved,
        shared_ancilla_qubits,
    )


def _time_interval(
    circuit,
    number_of_qubits,
    system_exchange_angle,
    jump_angle,
    dt,
    system_qubits,
    shared_ancilla_qubits,
):
    return fold_left(
        lambda current, _: _lie_trotter_substep(
            current,
            number_of_qubits,
            system_exchange_angle,
            jump_angle,
            dt,
            system_qubits,
            shared_ancilla_qubits,
        ),
        circuit,
        range(TROTTER_STEPS_PER_INTERVAL),
    )


def build_lie_trotter_circuits(times, number_of_qubits):
    """Build every saved-time prefix for an arbitrary N >= 2."""
    if number_of_qubits < 2:
        raise ValueError("number_of_qubits must be at least two")

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
    system_qubits = system_qubits_by_site(number_of_qubits)
    shared_ancilla_qubits = ancilla_qubits(number_of_qubits)
    initially_excited_sites = (number_of_qubits // 2,)

    initial_circuit = pipe(
        empty(
            number_of_qubits + 2,
            0,
            name=f"unitary_lie_trotter_N{number_of_qubits}",
        ),
        *map_tuple(
            lambda site: x(system_qubits[site]),
            initially_excited_sites,
        ),
    )
    evolution_circuits = scan_left(
        lambda current, _: _time_interval(
            current,
            number_of_qubits,
            system_exchange_angle,
            jump_angle,
            dt,
            system_qubits,
            shared_ancilla_qubits,
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


def build_one_step_circuit(number_of_qubits, dt, jump_angle):
    system_qubits = system_qubits_by_site(number_of_qubits)
    shared_ancilla_qubits = ancilla_qubits(number_of_qubits)
    return _lie_trotter_substep(
        empty(
            number_of_qubits + 2,
            0,
            name=f"one_lie_trotter_step_N{number_of_qubits}",
        ),
        number_of_qubits,
        2.0 * J * dt,
        jump_angle,
        dt,
        system_qubits,
        shared_ancilla_qubits,
    )


# ============================================================
# Aer estimation
# ============================================================

def _run_estimator(
    circuit,
    estimator_observables,
    precision,
    seed_simulator,
):
    plan = execution_plan(
        circuit=circuit,
        target=Aer(),
        compiler=CompilerConfig(optimization_level=1),
        workload=Estimate(
            observables=estimator_observables,
            precision=precision,
            seed_simulator=seed_simulator,
        ),
    )

    match run_sync(plan):
        case Err(error):
            raise RuntimeError(f"Aer estimation failed: {error}")
        case Ok(result) if isinstance(result, EstimateResult):
            return result
        case Ok(result):
            raise RuntimeError(
                f"Unexpected runtime result: {type(result).__name__}"
            )


def _group_estimator_values(results, number_of_qubits, project):
    values_by_time = np.asarray(
        map_tuple(
            lambda result: map_tuple(project, result.values),
            results,
        ),
        dtype=float,
    )
    first_bond_index = number_of_qubits
    first_flow_index = 2 * number_of_qubits - 1
    return (
        values_by_time[:, :first_bond_index].T,
        values_by_time[:, first_bond_index:first_flow_index].T,
        values_by_time[:, first_flow_index:].T,
    )


def simulate_with_aer(
    evolution_circuits,
    number_of_qubits,
    estimator_observables,
):
    results = map_tuple(
        lambda indexed_circuit: _run_estimator(
            indexed_circuit[1],
            estimator_observables,
            ESTIMATOR_PRECISION,
            SEED_SIMULATOR + indexed_circuit[0],
        ),
        enumerate(evolution_circuits),
    )
    estimates = _group_estimator_values(
        results,
        number_of_qubits,
        lambda value: value.value,
    )
    standard_errors = _group_estimator_values(
        results,
        number_of_qubits,
        lambda value: value.standard_error,
    )
    maximum_compiled_depth = max(
        map(lambda result: result.compiled_depth, results)
    )
    full_result = results[-1]
    full_circuit_metrics = (
        (full_result.original_gate_count, full_result.original_depth),
        (full_result.compiled_gate_count, full_result.compiled_depth),
    )
    return (
        estimates,
        standard_errors,
        maximum_compiled_depth,
        full_circuit_metrics,
    )


def transpilation_metrics(circuit, estimator_observables):
    result = _run_estimator(
        circuit,
        estimator_observables,
        ESTIMATOR_PRECISION,
        SEED_SIMULATOR,
    )
    return (
        (result.original_gate_count, result.original_depth),
        (result.compiled_gate_count, result.compiled_depth),
    )


# ============================================================
# Classical references
# ============================================================

def calculate_reference_solutions(times, number_of_qubits):
    (
        hamiltonian,
        jump_operators,
        x_operators,
        y_operators,
        z_operators,
    ) = classical.build_linear_chain(
        number_of_qubits=number_of_qubits,
        J=J,
        h=h,
        gamma=gamma,
    )
    initial_state = classical.computational_state(
        number_of_qubits=number_of_qubits,
        excited_sites=(number_of_qubits // 2,),
    )
    initial_density_matrix = np.outer(
        initial_state,
        initial_state.conj(),
    )
    exact_density_matrices = classical.exact_evolve(
        initial_density_matrix,
        classical.build_liouvillian(hamiltonian, jump_operators),
        times,
    )
    coherent_lie_density_matrices = classical.hamiltonian_dilation_evolve(
        initial_density_matrix,
        hamiltonian,
        jump_operators,
        times,
        product_formula="lie",
        steps_per_interval=TROTTER_STEPS_PER_INTERVAL,
    )
    identity = eye(2**number_of_qubits, dtype=complex, format="csr")
    reference_observables = classical.build_observables(
        identity=identity,
        X_ops=x_operators,
        Y_ops=y_operators,
        Z_ops=z_operators,
        J=J,
    )
    return (
        exact_density_matrices,
        coherent_lie_density_matrices,
        reference_observables,
    )


def evaluate_observables(density_matrices, reference_observables):
    population_operators, correlation_operators, flow_operators = (
        reference_observables
    )
    return (
        np.asarray(
            map_tuple(
                lambda operator: classical.expectation_series(
                    density_matrices,
                    operator,
                ),
                population_operators,
            )
        ),
        np.asarray(
            map_tuple(
                lambda operator: classical.expectation_series(
                    density_matrices,
                    operator,
                ),
                correlation_operators,
            )
        ),
        np.asarray(
            map_tuple(
                lambda operator: classical.expectation_series(
                    density_matrices,
                    operator,
                ),
                flow_operators,
            )
        ),
    )


def aggregate_observable_error(first, second):
    return np.sqrt(
        sum(
            map(
                lambda pair: np.sum((pair[0] - pair[1]) ** 2, axis=0),
                zip(first, second),
            )
        )
    )


# ============================================================
# Figures
# ============================================================

def _paired_styles(curve_count):
    colors = map_tuple(lambda index: f"C{index % 10}", range(curve_count))
    return (
        *map_tuple(lambda color: {"color": color}, colors),
        *map_tuple(
            lambda color: {"color": color, "linestyle": "--"},
            colors,
        ),
    )


def plot_results(
    times,
    number_of_qubits,
    aer_observables,
    exact_observables,
    lindblad_errors,
    coherent_lie_errors,
    output_path,
):
    aer_populations, aer_correlations, aer_flows = aer_observables
    exact_populations, exact_correlations, exact_flows = exact_observables
    site_labels = map_tuple(
        lambda site: f"site {site}",
        range(number_of_qubits),
    )
    bond_labels = map_tuple(
        lambda bond: f"bond {bond}-{bond + 1}",
        range(number_of_qubits - 1),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_observables(
        times,
        {
            "Local excitation populations": {
                "values": np.vstack((aer_populations, exact_populations)),
                "labels": (
                    *map_tuple(lambda label: f"{label}, Aer", site_labels),
                    *map_tuple(lambda label: f"{label}, exact", site_labels),
                ),
                "styles": _paired_styles(number_of_qubits),
                "ylabel": r"$\langle n_i\rangle$",
                "legend_columns": 2,
            },
            "Nearest-neighbor XY correlations": {
                "values": np.vstack((
                    aer_correlations,
                    exact_correlations,
                )),
                "labels": (
                    *map_tuple(lambda label: f"{label}, Aer", bond_labels),
                    *map_tuple(lambda label: f"{label}, exact", bond_labels),
                ),
                "styles": _paired_styles(number_of_qubits - 1),
                "ylabel": r"$C_i^{XY}$",
                "legend_columns": 2,
            },
            "Nearest-neighbor excitation flows": {
                "values": np.vstack((aer_flows, exact_flows)),
                "labels": (
                    *map_tuple(lambda label: f"{label}, Aer", bond_labels),
                    *map_tuple(lambda label: f"{label}, exact", bond_labels),
                ),
                "styles": _paired_styles(number_of_qubits - 1),
                "ylabel": r"$I_{i\rightarrow i+1}$",
                "legend_columns": 2,
                "zero_line": True,
            },
            "Aggregate observable errors": {
                "values": np.vstack((
                    np.maximum(lindblad_errors, np.finfo(float).eps),
                    np.maximum(coherent_lie_errors, np.finfo(float).eps),
                )),
                "labels": (
                    "vs exact Lindblad",
                    "vs coherent Lie dilation",
                ),
                "styles": (
                    {"color": "tab:red"},
                    {"color": "tab:purple"},
                ),
                "ylabel": "Euclidean error",
                "yscale": "log",
            },
        },
        output_path=output_path,
        show=False,
        columns=2,
        figsize=(14, 10),
    )
    plt.close(figure)


def plot_one_step_circuit(circuit, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = draw_circuit(circuit)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


# ============================================================
# Driver
# ============================================================

def main(number_of_qubits=DEFAULT_NUMBER_OF_QUBITS):
    if number_of_qubits < 2:
        raise ValueError("number_of_qubits must be at least two")

    times = np.linspace(0.0, T_FINAL, NUMBER_OF_TIME_POINTS)
    system_qubits = system_qubits_by_site(number_of_qubits)
    shared_ancilla_qubits = ancilla_qubits(number_of_qubits)
    estimator_observables = build_estimator_observables(
        number_of_qubits,
        system_qubits,
    )
    (
        exact_density_matrices,
        coherent_lie_density_matrices,
        reference_observables,
    ) = calculate_reference_solutions(times, number_of_qubits)
    exact_observables = evaluate_observables(
        exact_density_matrices,
        reference_observables,
    )
    coherent_lie_observables = evaluate_observables(
        coherent_lie_density_matrices,
        reference_observables,
    )

    (
        evolution_circuits,
        dt,
        dilation_exchange_probability,
        jump_angle,
    ) = build_lie_trotter_circuits(times, number_of_qubits)
    (
        aer_observables,
        estimator_standard_errors,
        maximum_compiled_depth,
        full_circuit_metrics,
    ) = simulate_with_aer(
        evolution_circuits,
        number_of_qubits,
        estimator_observables,
    )
    lindblad_errors = aggregate_observable_error(
        aer_observables,
        exact_observables,
    )
    coherent_lie_errors = aggregate_observable_error(
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
    one_step_metrics = transpilation_metrics(
        one_step_circuit,
        estimator_observables,
    )
    results_path, metrics_path, circuit_path = output_paths(
        number_of_qubits
    )
    plot_results(
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
        method_name=f"N={number_of_qubits} even/odd Lie",
    )
    plot_one_step_circuit(one_step_circuit, circuit_path)

    print("General-N boundary-damped XY chain: unitary Lie dilation")
    print(f"Number of system qubits: {number_of_qubits}")
    print(f"Total circuit qubits: {number_of_qubits + 2}")
    print(f"System qubits by site: {system_qubits}")
    print(f"Ancilla qubits (low, high): {shared_ancilla_qubits}")
    print(f"Even bonds: {even_bonds(number_of_qubits)}")
    print(f"Odd bonds: {odd_bonds(number_of_qubits)}")
    print("System layers per dilation step: one even, then one odd")
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
    print(f"Pre-transpilation: gate count = {one_pre[0]}, depth = {one_pre[1]}")
    print(f"Post-transpilation: gate count = {one_post[0]}, depth = {one_post[1]}")
    print("\nFull-circuit transpilation metrics")
    print(f"Pre-transpilation: gate count = {full_pre[0]}, depth = {full_pre[1]}")
    print(f"Post-transpilation: gate count = {full_post[0]}, depth = {full_post[1]}")

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
            "General-N unitary Lie dilation with one even/odd system split."
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
