"""Four-qubit Lie circuit for Model Problem I's shared-ancilla dilation.

The two system qubits form the boundary-damped XY chain

    H = J/2 (X0 X1 + Y0 Y1) + h/2 (Z0 + Z1),

with one amplitude-damping jump operator on each boundary. Two additional
qubits encode the single shared dilation qutrit as

    |0>_a = |00>, |1>_a = |01>, |2>_a = |10>.

The unused |11> state is excluded by open controls on the jump exchanges.
The Hamiltonian block is controlled on the ancilla state |00>.

One Lie--Trotter substep is

    system full-step -> left jump full-step -> right jump full-step.

Both ancilla qubits are reset only after the complete substep, which traces
out the shared qutrit and prepares it for the next substep.

The unsplit reference calculation reuses ``library/classical.py``.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import eye


# Permit both ``python path/to/script.py`` and ``python -m ...`` execution.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from library import classical
from library.plotter import plot_observables
from IBMRuntime import (
    Aer,
    CompilerConfig,
    Err,
    Estimate,
    EstimateResult,
    Ok,
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
    controlled_xx_plus_yy,
)


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

# Qiskit labels two-qubit basis states as |q1 q0>. To match classical.py's
# site-0 tensor site-1 convention, physical site 0 maps to Qiskit qubit 1 and
# site 1 maps to qubit 0. Estimator observables use this logical mapping before
# the runtime interpreter applies the transpiler layout.
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
    / "unitary_lie_trotter.png"
)
TRANSPILATION_METRICS_OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "figures"
    / "one_step_transpilation_metrics.png"
)
ONE_STEP_CIRCUIT_OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "figures"
    / "one_step_lie_trotter_circuit.png"
)


# ============================================================
# Functional runtime circuit
# ============================================================

def _hamiltonian_dilation_factor(
    circuit,
    system_exchange_angle,
    field_angle,
    *,
    label,
):
    """Apply exp(-i K_H) controlled on the ancilla state |00>."""
    exchange_circuit = pipe(
        circuit,
        multi_controlled_xx_plus_yy(
            system_exchange_angle,
            0.0,
            ANCILLA_QUBITS,
            *SYSTEM_QUBITS_BY_SITE,
            control_state=0,
            label=label,
        ),
    )
    if field_angle == 0.0:
        return exchange_circuit

    return fold_left(
        lambda current, system_qubit: pipe(
            current,
            multi_controlled_rz(
                field_angle,
                ANCILLA_QUBITS,
                system_qubit,
                control_state=0,
                label="field |00>",
            ),
        ),
        exchange_circuit,
        SYSTEM_QUBITS_BY_SITE,
    )


def _jump_dilation_factor(circuit, jump_index, jump_angle, *, label):
    """Apply exp(-i K_j) without coupling to the unused ancilla state."""
    jump_layouts = (
        (
            ANCILLA_HIGH_QUBIT,
            SYSTEM_QUBITS_BY_SITE[0],
            ANCILLA_LOW_QUBIT,
        ),
        (
            ANCILLA_LOW_QUBIT,
            SYSTEM_QUBITS_BY_SITE[1],
            ANCILLA_HIGH_QUBIT,
        ),
    )
    control, system_qubit, active_ancilla = jump_layouts[jump_index]
    return pipe(
        circuit,
        controlled_xx_plus_yy(
            jump_angle,
            0.0,
            control,
            system_qubit,
            active_ancilla,
            control_state=0,
            label=label,
        ),
    )


def _reset_shared_ancilla(circuit):
    """Trace out the qutrit and prepare |00> for the next substep."""
    return pipe(
        circuit,
        reset(ANCILLA_LOW_QUBIT),
        reset(ANCILLA_HIGH_QUBIT),
    )


def _lie_trotter_substep(
    circuit,
    system_exchange_angle,
    jump_angle,
    dt,
):
    system_evolved_circuit = _hamiltonian_dilation_factor(
        circuit,
        system_exchange_angle,
        h * dt,
        label="K_H",
    )
    jump_evolved_circuit = fold_left(
        lambda current, jump_index: _jump_dilation_factor(
            current,
            jump_index,
            jump_angle,
            label=f"K_{jump_index + 1}",
        ),
        system_evolved_circuit,
        range(2),
    )
    return _reset_shared_ancilla(jump_evolved_circuit)


def _time_interval(
    circuit,
    system_exchange_angle,
    jump_angle,
    dt,
):
    return fold_left(
        lambda current, _: _lie_trotter_substep(
            current,
            system_exchange_angle,
            jump_angle,
            dt,
        ),
        circuit,
        range(TROTTER_STEPS_PER_INTERVAL),
    )


def build_lie_trotter_circuits(times):
    """Construct one immutable evolution prefix per requested time."""
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

    # XXPlusYYGate(theta) = exp[-i theta (XX+YY) / 4]. Hence theta=2Jdt
    # implements exp[-i dt J(XX+YY)/2], the Experiment 1 XY Hamiltonian.
    system_exchange_angle = 2.0 * J * dt

    # K_j = sqrt(gamma*dt)/2 (XX + YY) on its active system--ancilla
    # pair. Since XXPlusYYGate(theta) = exp[-i theta(XX+YY)/4], the
    # dilation angle is exactly 2*sqrt(gamma*dt), not the angle of an
    # independently traced amplitude-damping channel.
    jump_angle = 2.0 * np.sqrt(gamma * dt)
    dilation_exchange_probability = np.sin(0.5 * jump_angle) ** 2
    initial_circuit = pipe(
        empty(
            TOTAL_CIRCUIT_QUBITS,
            0,
            name="unitary_lie_trotter",
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
    """Return only one Lie--Trotter substep for compilation metrics."""
    return _lie_trotter_substep(
        empty(
            TOTAL_CIRCUIT_QUBITS,
            0,
            name="one_lie_trotter_step",
        ),
        2.0 * J * dt,
        jump_angle,
        dt,
    )


def _run_estimator(
    circuit,
    precision,
    seed_simulator,
    estimator_observables=ESTIMATOR_OBSERVABLES,
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
            raise RuntimeError(f"Unexpected runtime result: {type(result).__name__}")


def _group_estimator_values(results, project):
    values_by_time = map_tuple(
        lambda result: map_tuple(project, result.values),
        results,
    )
    return (
        np.asarray(map_tuple(lambda values: values[:2], values_by_time)).T,
        np.asarray(map_tuple(lambda values: (values[2],), values_by_time)).T,
        np.asarray(map_tuple(lambda values: (values[3],), values_by_time)).T,
    )


def simulate_with_aer(
    evolution_circuits,
    precision=ESTIMATOR_PRECISION,
    seed_simulator=SEED_SIMULATOR,
    estimator_observables=ESTIMATOR_OBSERVABLES,
):
    """Estimate populations, XY correlation, and flow with EstimatorV2."""
    results = map_tuple(
        lambda indexed_circuit: _run_estimator(
            indexed_circuit[1],
            precision,
            (
                None
                if seed_simulator is None
                else seed_simulator + indexed_circuit[0]
            ),
            estimator_observables,
        ),
        enumerate(evolution_circuits),
    )
    estimates = _group_estimator_values(
        results,
        lambda value: value.value,
    )
    standard_errors = _group_estimator_values(
        results,
        lambda value: value.standard_error,
    )
    maximum_compiled_depth = max(
        map(lambda result: result.compiled_depth, results)
    )
    full_circuit_result = results[-1]
    full_circuit_metrics = (
        (
            full_circuit_result.original_gate_count,
            full_circuit_result.original_depth,
        ),
        (
            full_circuit_result.compiled_gate_count,
            full_circuit_result.compiled_depth,
        ),
    )
    return (
        estimates,
        standard_errors,
        maximum_compiled_depth,
        full_circuit_metrics,
    )


def one_step_transpilation_metrics(dt, jump_angle):
    """Return (gate count, depth) before and after one-step transpilation."""
    return transpilation_metrics(build_one_step_circuit(dt, jump_angle))


def transpilation_metrics(
    circuit,
    estimator_observables=ESTIMATOR_OBSERVABLES,
):
    """Return (gate count, depth) before and after transpiling a circuit."""
    result = _run_estimator(
        circuit,
        ESTIMATOR_PRECISION,
        SEED_SIMULATOR,
        estimator_observables,
    )
    return (
        (result.original_gate_count, result.original_depth),
        (result.compiled_gate_count, result.compiled_depth),
    )


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
    return density_matrices, observables


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
    observable_errors,
    *,
    output_path=OUTPUT_PATH,
    simulation_label="Aer Lie",
):
    """Plot the Aer trajectory and the unsplit classical reference."""
    aer_populations, aer_correlations, aer_flows = aer_observables
    exact_populations, exact_correlations, exact_flows = exact_observables
    displayed_errors = np.maximum(observable_errors, np.finfo(float).eps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_observables(
        times,
        {
            "Local excitation populations": {
                "values": np.vstack((
                    aer_populations[0],
                    exact_populations[0],
                    aer_populations[1],
                    exact_populations[1],
                )),
                "labels": (
                    f"site 0, {simulation_label}",
                    "site 0, exact",
                    f"site 1, {simulation_label}",
                    "site 1, exact",
                ),
                "ylabel": r"$\langle n_i\rangle$",
                "legend_columns": 2,
                "styles": (
                    {"color": "tab:blue"},
                    {"color": "tab:blue", "linestyle": "--"},
                    {"color": "tab:orange"},
                    {"color": "tab:orange", "linestyle": "--"},
                ),
            },
            "Nearest-neighbor XY correlation": {
                "values": np.vstack((aer_correlations[0], exact_correlations[0])),
                "labels": (simulation_label, "exact"),
                "ylabel": r"$C_0^{XY}$",
                "styles": ({}, {"linestyle": "--"}),
            },
            "Excitation flow": {
                "values": np.vstack((aer_flows[0], exact_flows[0])),
                "labels": (simulation_label, "exact"),
                "ylabel": r"$I_{0\rightarrow1}$",
                "styles": ({}, {"linestyle": "--"}),
                "zero_line": True,
            },
            "Measured-observable error": {
                "values": displayed_errors,
                "ylabel": "Euclidean error",
                "styles": ({"color": "tab:red"},),
                "yscale": "log",
            },
        },
        output_path=output_path,
        show=False,
        columns=2,
        figsize=(13, 9),
    )
    plt.close(figure)


def plot_transpilation_metrics(
    one_step_metrics,
    full_circuit_metrics,
    *,
    output_path=TRANSPILATION_METRICS_OUTPUT_PATH,
    method_name="Lie--Trotter",
):
    """Plot pre/post metrics for one step and for the complete circuit."""
    labels = ("Gate count", "Circuit depth")
    positions = np.arange(len(labels))
    width = 0.34

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    metric_sets = (
        (f"One {method_name} substep", one_step_metrics),
        ("Complete final-time circuit", full_circuit_metrics),
    )

    for axis, (title, (pre_metrics, post_metrics)) in zip(axes, metric_sets):
        pre_bars = axis.bar(
            positions - width / 2,
            pre_metrics,
            width,
            label="Pre-transpilation",
            color="tab:blue",
        )
        post_bars = axis.bar(
            positions + width / 2,
            post_metrics,
            width,
            label="Post-transpilation",
            color="tab:orange",
        )
        axis.bar_label(pre_bars, fmt="%d", padding=3)
        axis.bar_label(post_bars, fmt="%d", padding=3)
        axis.set_xticks(positions, labels)
        axis.set_ylabel("Count")
        axis.set_title(title)
        axis.legend()
        axis.grid(axis="y", alpha=0.3)
        axis.set_ylim(
            0.0,
            1.15 * max((*pre_metrics, *post_metrics, 1)),
        )

    figure.suptitle(f"{method_name} transpilation metrics")
    figure.tight_layout()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_one_step_circuit(dt, jump_angle):
    """Draw one untranspiled Lie--Trotter substep."""
    ONE_STEP_CIRCUIT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure = draw_circuit(build_one_step_circuit(dt, jump_angle))
    figure.savefig(
        ONE_STEP_CIRCUIT_OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    times = np.linspace(0.0, T_FINAL, NUMBER_OF_TIME_POINTS)

    exact_density_matrices, observables = calculate_exact_solution(times)

    (
        evolution_circuits,
        dt,
        dilation_exchange_probability,
        jump_angle,
    ) = (
        build_lie_trotter_circuits(times)
    )
    (
        aer_observables,
        estimator_standard_errors,
        maximum_compiled_depth,
        full_circuit_metrics,
    ) = simulate_with_aer(
        evolution_circuits,
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
    one_step_metrics = one_step_transpilation_metrics(
        dt,
        jump_angle,
    )

    plot_results(
        times,
        aer_observables,
        exact_observables,
        observable_errors,
    )
    plot_transpilation_metrics(one_step_metrics, full_circuit_metrics)
    plot_one_step_circuit(dt, jump_angle)

    print("Two-qubit boundary-damped XY chain on Qiskit Aer")
    print(f"J = {J}, h = {h}, gamma = {gamma}")
    print(f"Initially excited sites: {INITIALLY_EXCITED_SITES}")
    print(f"System qubits: {SYSTEM_QUBITS_BY_SITE}")
    print(f"Ancilla qubits (low, high): {ANCILLA_QUBITS}")
    print(f"Lie--Trotter substep: {dt:.6f}")
    print(
        "Full jump-factor exchange probability: "
        f"{dilation_exchange_probability:.8f}"
    )
    print(f"Jump-dilation angle: {jump_angle:.8f}")
    print(f"Estimator target precision: {ESTIMATOR_PRECISION:.8f}")
    print(f"Maximum reported standard error: {maximum_standard_error:.8f}")
    print(f"Maximum transpiled circuit depth: {maximum_compiled_depth}")

    pre_metrics, post_metrics = one_step_metrics
    print("\nOne-step transpilation metrics")
    print(
        f"Pre-transpilation: gate count = {pre_metrics[0]}, "
        f"depth = {pre_metrics[1]}"
    )
    print(
        f"Post-transpilation: gate count = {post_metrics[0]}, "
        f"depth = {post_metrics[1]}"
    )

    full_pre_metrics, full_post_metrics = full_circuit_metrics
    print("\nFull-circuit transpilation metrics")
    print(
        f"Pre-transpilation: gate count = {full_pre_metrics[0]}, "
        f"depth = {full_pre_metrics[1]}"
    )
    print(
        f"Post-transpilation: gate count = {full_post_metrics[0]}, "
        f"depth = {full_post_metrics[1]}"
    )

    print("\nEstimator observable error relative to exact evolution")
    print(f"Maximum error: {observable_errors.max():.3e}")
    print(f"Final error: {observable_errors[-1]:.3e}")
    print(f"\nSaved plot: {OUTPUT_PATH}")
    print(
        "Saved transpilation metrics: "
        f"{TRANSPILATION_METRICS_OUTPUT_PATH}"
    )
    print(f"Saved one-step circuit: {ONE_STEP_CIRCUIT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
