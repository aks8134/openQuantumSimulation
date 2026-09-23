"""Run the general-N dynamic Lie circuit on Aer or IBM hardware.

IBM EstimatorV2 does not support dynamic circuits.  This experiment
therefore uses SamplerV2-compatible terminal-measurement circuits.  Five
measurement settings recover the Experiment 3 observables at every saved
time:

    Z, X, Y, alternating XY, and alternating YX.

The mid-circuit bit remains classical bit 0.  Terminal measurements of
system site n are stored in classical bit n + 1.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from math import pi, sqrt
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from IBMRuntime import (
    AER_METHODS,
    Aer,
    Circuit,
    CompilerConfig,
    Err,
    IBMHardware,
    Ok,
    RuntimeEnvironment,
    Sample,
    SampleResult,
    compile_circuit_batch_sync,
    counts_dict,
    draw_transpiled_circuit_layout_sync,
    h,
    load_ibm_account,
    measure,
    pipe,
    run_sample_batch_sync,
    rz,
)
from Model1.Experiment3 import dynamic_lie_trotter as dynamic_circuits
from Model1.Experiment3 import unitary_lie_trotter as reference_tools
from library import classical


DEFAULT_ACCOUNT_FILE = REPOSITORY_ROOT / "IBMRuntime" / "apikey.json"
DEFAULT_NUMBER_OF_QUBITS = dynamic_circuits.DEFAULT_NUMBER_OF_QUBITS
DEFAULT_BACKEND = "aer"
DEFAULT_SHOTS = 8192
DEFAULT_AER_BATCH_SIZE = 300
DEFAULT_HARDWARE_BATCH_SIZE = 5
DEFAULT_OPTIMIZATION_LEVEL = 1
DEFAULT_SEED_TRANSPILER = 11
DEFAULT_SEED_SIMULATOR = dynamic_circuits.SEED_SIMULATOR
CHECKPOINT_SCHEMA_VERSION = 1

MEASUREMENT_BASES = ("Z", "X", "Y", "XY", "YX")
MID_CIRCUIT_BIT = dynamic_circuits.LOW_ANCILLA_MEASUREMENT_BIT


@dataclass(frozen=True, slots=True)
class EvolutionSchedule:
    interval_substep_dts: tuple[tuple[float, ...], ...]
    interval_jump_probabilities: tuple[tuple[float, ...], ...]
    interval_jump_angles: tuple[tuple[float, ...], ...]
    requested_trotter_delta_t: float | None
    includes_initial_time: bool = True

    def prefix(self, time_count):
        interval_count = max(
            0,
            time_count - 1 if self.includes_initial_time else time_count,
        )
        return EvolutionSchedule(
            interval_substep_dts=(
                self.interval_substep_dts[:interval_count]
            ),
            interval_jump_probabilities=(
                self.interval_jump_probabilities[:interval_count]
            ),
            interval_jump_angles=(
                self.interval_jump_angles[:interval_count]
            ),
            requested_trotter_delta_t=self.requested_trotter_delta_t,
            includes_initial_time=self.includes_initial_time,
        )

    @property
    def substep_dts(self):
        return tuple(
            dt
            for interval in self.interval_substep_dts
            for dt in interval
        )

    @property
    def jump_probabilities(self):
        return tuple(
            probability
            for interval in self.interval_jump_probabilities
            for probability in interval
        )

    @property
    def jump_angles(self):
        return tuple(
            angle
            for interval in self.interval_jump_angles
            for angle in interval
        )


def _uniform_value(values):
    if not values:
        return None
    return (
        float(values[0])
        if np.allclose(values, values[0])
        else None
    )


def _schedule_payload(times, schedule):
    interval_start_times = (
        tuple(float(value) for value in times[:-1])
        if schedule.includes_initial_time
        else (0.0, *tuple(float(value) for value in times[:-1]))
    )
    interval_end_times = (
        tuple(float(value) for value in times[1:])
        if schedule.includes_initial_time
        else tuple(float(value) for value in times)
    )
    return {
        "requested_trotter_delta_t": (
            schedule.requested_trotter_delta_t
        ),
        "includes_initial_time": schedule.includes_initial_time,
        "uniform_substep_dt": _uniform_value(schedule.substep_dts),
        "uniform_system_exchange_angle": _uniform_value(
            tuple(2.0 * reference_tools.J * dt for dt in schedule.substep_dts)
        ),
        "uniform_jump_angle": _uniform_value(schedule.jump_angles),
        "uniform_jump_exchange_probability": _uniform_value(
            schedule.jump_probabilities
        ),
        "intervals": [
            {
                "start_time": interval_start_times[index],
                "end_time": interval_end_times[index],
                "substep_count": len(
                    schedule.interval_substep_dts[index]
                ),
                "substeps": [
                    {
                        "substep_index": substep_index,
                        "dt": dt,
                        "system_exchange_angle": float(
                            2.0 * reference_tools.J * dt
                        ),
                        "jump_angle": (
                            schedule.interval_jump_angles[index][
                                substep_index
                            ]
                        ),
                        "jump_exchange_probability": (
                            schedule.interval_jump_probabilities[index][
                                substep_index
                            ]
                        ),
                    }
                    for substep_index, dt in enumerate(
                        schedule.interval_substep_dts[index]
                    )
                ],
            }
            for index in range(len(schedule.interval_substep_dts))
        ],
    }


def _axis_for_site(basis, site):
    if basis in ("Z", "X", "Y"):
        return basis
    if basis == "XY":
        return "X" if site % 2 == 0 else "Y"
    if basis == "YX":
        return "Y" if site % 2 == 0 else "X"
    raise ValueError(f"unsupported measurement basis: {basis}")


def _append_terminal_measurements(
    circuit,
    number_of_qubits,
    basis,
):
    """Add basis rotations and terminal system measurements."""
    system_qubits = reference_tools.system_qubits_by_site(number_of_qubits)
    measured = Circuit(
        qubit_count=circuit.qubit_count,
        bit_count=number_of_qubits + 1,
        operations=circuit.operations,
        name=f"{circuit.name}_{basis}_measurement",
    )

    for site, qubit in enumerate(system_qubits):
        match _axis_for_site(basis, site):
            case "X":
                measured = pipe(measured, h(qubit))
            case "Y":
                measured = pipe(measured, rz(-pi / 2.0, qubit), h(qubit))
            case "Z":
                pass

    for site, qubit in enumerate(system_qubits):
        measured = pipe(measured, measure(qubit, site + 1))

    return measured


def build_sample_circuits(
    times,
    number_of_qubits,
    trotter_delta_t=None,
):
    """Build five terminal-measurement circuits per saved time."""
    requested_times = np.asarray(times, dtype=float)
    if requested_times.ndim != 1 or requested_times.size < 1:
        raise ValueError("times must contain at least one value")
    if not np.all(np.isfinite(requested_times)):
        raise ValueError("times must contain only finite values")

    single_target = requested_times.size == 1
    if single_target:
        target_time = float(requested_times[0])
        if target_time < 0.0:
            raise ValueError("a single time must be nonnegative")
        # The Experiment 3 builder requires an interval.  For t=0, the
        # arbitrary endpoint is never executed because only its initial
        # prefix is retained.
        evolution_times = np.asarray(
            (0.0, target_time if target_time > 0.0 else 1.0)
        )
        saved_circuit_indices = (1,) if target_time > 0.0 else (0,)
    else:
        evolution_times = requested_times
        saved_circuit_indices = tuple(range(requested_times.size))

    (
        evolution_circuits,
        interval_substep_dts,
        interval_dilation_exchange_probabilities,
        interval_jump_angles,
    ) = dynamic_circuits.build_dynamic_lie_trotter_circuits_with_schedule(
        evolution_times,
        number_of_qubits,
        trotter_delta_t=trotter_delta_t,
    )
    if single_target and np.isclose(requested_times[0], 0.0):
        interval_substep_dts = ()
        interval_dilation_exchange_probabilities = ()
        interval_jump_angles = ()
    saved_circuits = tuple(
        evolution_circuits[index] for index in saved_circuit_indices
    )
    entries = tuple(
        (
            time_index,
            basis,
            _append_terminal_measurements(
                circuit,
                number_of_qubits,
                basis,
            ),
        )
        for time_index, circuit in enumerate(saved_circuits)
        for basis in MEASUREMENT_BASES
    )
    return (
        tuple(entry[2] for entry in entries),
        tuple((entry[0], entry[1]) for entry in entries),
        EvolutionSchedule(
            interval_substep_dts=interval_substep_dts,
            interval_jump_probabilities=(
                interval_dilation_exchange_probabilities
            ),
            interval_jump_angles=interval_jump_angles,
            requested_trotter_delta_t=trotter_delta_t,
            includes_initial_time=not (
                single_target and requested_times[0] > 0.0
            ),
        ),
    )


def _chunks(values, size):
    return tuple(
        values[start : start + size]
        for start in range(0, len(values), size)
    )


def _runtime_target(backend_name, aer_method, account_file):
    if backend_name.lower() == "aer":
        return Aer(method=aer_method), RuntimeEnvironment()

    match load_ibm_account(account_file):
        case Err(error):
            raise RuntimeError(
                f"Could not load IBM account from {account_file}: "
                f"{error.message}"
            )
        case Ok(account):
            return (
                IBMHardware(backend_name),
                RuntimeEnvironment(ibm_account=account),
            )


def _runtime_error_message(error):
    if hasattr(error, "problems"):
        return "; ".join(error.problems)
    stage = getattr(error, "stage", "runtime")
    message = getattr(error, "message", str(error))
    return f"{stage}: {message}"


def execute_sample_circuits(
    circuits,
    options,
    on_batch_complete=None,
):
    """Execute circuit chunks and preserve their original order."""
    target, environment = _runtime_target(
        options.backend,
        options.aer_method,
        options.account_file,
    )
    compiler = CompilerConfig(
        optimization_level=options.optimization_level,
        seed_transpiler=options.seed_transpiler,
    )
    workload = Sample(
        shots=options.shots,
        seed_simulator=options.seed_simulator,
    )
    batch_size = options.batch_size or (
        DEFAULT_AER_BATCH_SIZE
        if options.backend.lower() == "aer"
        else DEFAULT_HARDWARE_BATCH_SIZE
    )
    circuit_batches = _chunks(circuits, batch_size)
    collected = ()

    for batch_index, circuit_batch in enumerate(circuit_batches, start=1):
        print(
            f"Executing batch {batch_index}/{len(circuit_batches)} "
            f"with {len(circuit_batch)} circuits on {options.backend}"
        )
        match run_sample_batch_sync(
            circuit_batch,
            target,
            compiler,
            workload,
            environment,
        ):
            case Ok(results):
                collected = (*collected, *results)
                if on_batch_complete is not None:
                    on_batch_complete(collected)
            case Err(error):
                message = _runtime_error_message(error)
                if (
                    options.backend.lower() != "aer"
                    and "6073" in message
                ):
                    raise RuntimeError(
                        f"{message}. Re-run with a smaller --batch-size."
                    )
                raise RuntimeError(message)

    return collected


def _classical_bit(bitstring, bit):
    compact = bitstring.replace(" ", "").replace("_", "")
    if bit >= len(compact):
        raise ValueError(
            f"bitstring {bitstring!r} does not contain classical bit {bit}"
        )
    return int(compact[-1 - bit])


def _expectation(counts, bits):
    shots = sum(counts.values())
    if shots <= 0:
        raise ValueError("sample counts must contain at least one shot")
    weighted_sum = sum(
        count
        * (
            -1.0
            if sum(_classical_bit(bitstring, bit) for bit in bits) % 2
            else 1.0
        )
        for bitstring, count in counts.items()
    )
    return weighted_sum / shots


def _pauli_standard_error(expectation, shots):
    return sqrt(max(0.0, 1.0 - expectation**2) / shots)


def _group_counts(results, metadata, time_count):
    grouped = tuple({} for _ in range(time_count))
    for result, (time_index, basis) in zip(
        results,
        metadata,
        strict=True,
    ):
        grouped[time_index][basis] = counts_dict(result)

    for time_index, values in enumerate(grouped):
        missing = tuple(
            basis for basis in MEASUREMENT_BASES if basis not in values
        )
        if missing:
            raise ValueError(
                f"time index {time_index} is missing bases {missing}"
            )
    return grouped


def observables_from_results(
    results,
    metadata,
    number_of_qubits,
    time_count,
):
    """Reconstruct populations, XY correlations, flows, and shot errors."""
    grouped = _group_counts(results, metadata, time_count)
    populations = np.zeros((number_of_qubits, time_count), dtype=float)
    correlations = np.zeros(
        (number_of_qubits - 1, time_count),
        dtype=float,
    )
    flows = np.zeros((number_of_qubits - 1, time_count), dtype=float)
    population_errors = np.zeros_like(populations)
    correlation_errors = np.zeros_like(correlations)
    flow_errors = np.zeros_like(flows)

    for time_index, basis_counts in enumerate(grouped):
        shots = sum(basis_counts["Z"].values())
        for site in range(number_of_qubits):
            z_value = _expectation(
                basis_counts["Z"],
                (site + 1,),
            )
            populations[site, time_index] = 0.5 * (1.0 - z_value)
            population_errors[site, time_index] = (
                0.5 * _pauli_standard_error(z_value, shots)
            )

        for bond in range(number_of_qubits - 1):
            terminal_bits = (bond + 1, bond + 2)
            xx_value = _expectation(
                basis_counts["X"],
                terminal_bits,
            )
            yy_value = _expectation(
                basis_counts["Y"],
                terminal_bits,
            )
            xy_basis = "XY" if bond % 2 == 0 else "YX"
            yx_basis = "YX" if bond % 2 == 0 else "XY"
            xy_value = _expectation(
                basis_counts[xy_basis],
                terminal_bits,
            )
            yx_value = _expectation(
                basis_counts[yx_basis],
                terminal_bits,
            )

            correlations[bond, time_index] = xx_value + yy_value
            flows[bond, time_index] = (
                0.5 * reference_tools.J * (yx_value - xy_value)
            )
            correlation_errors[bond, time_index] = sqrt(
                _pauli_standard_error(xx_value, shots) ** 2
                + _pauli_standard_error(yy_value, shots) ** 2
            )
            flow_errors[bond, time_index] = (
                0.5
                * abs(reference_tools.J)
                * sqrt(
                    _pauli_standard_error(xy_value, shots) ** 2
                    + _pauli_standard_error(yx_value, shots) ** 2
                )
            )

    return (
        (populations, correlations, flows),
        (population_errors, correlation_errors, flow_errors),
        grouped,
    )


def calculate_references(times, number_of_qubits, schedule=None):
    time_grid = np.asarray(times, dtype=float)
    if (
        schedule is None
        and time_grid.size >= 2
        and np.isclose(time_grid[0], 0.0)
    ):
        (
            exact_density_matrices,
            coherent_lie_density_matrices,
            reference_observables,
        ) = reference_tools.calculate_reference_solutions(
            time_grid,
            number_of_qubits,
        )
    else:
        (
            hamiltonian,
            jump_operators,
            x_operators,
            y_operators,
            z_operators,
        ) = classical.build_linear_chain(
            number_of_qubits=number_of_qubits,
            J=reference_tools.J,
            h=reference_tools.h,
            gamma=reference_tools.gamma,
        )
        initial_state = classical.computational_state(
            number_of_qubits=number_of_qubits,
            excited_sites=(number_of_qubits // 2,),
        )
        initial_density_matrix = np.outer(
            initial_state,
            initial_state.conj(),
        )
        liouvillian = classical.build_liouvillian(
            hamiltonian,
            jump_operators,
        )
        exact_density_matrices = np.asarray(
            tuple(
                initial_density_matrix
                if np.isclose(time, 0.0)
                else classical.exact_evolve(
                    initial_density_matrix,
                    liouvillian,
                    np.asarray((0.0, float(time))),
                )[-1]
                for time in time_grid
            )
        )

        coherent_density_matrix = initial_density_matrix
        coherent_values = [initial_density_matrix]
        interval_substeps_values = (
            schedule.interval_substep_dts
            if schedule is not None
            else tuple(
                (float(interval),)
                for interval in np.diff(
                    np.concatenate(((0.0,), time_grid))
                )
                if interval > 0.0
            )
        )
        for interval_substeps in interval_substeps_values:
            for dt in interval_substeps:
                coherent_density_matrix = (
                    classical.hamiltonian_dilation_evolve(
                        coherent_density_matrix,
                        hamiltonian,
                        jump_operators,
                        np.asarray((0.0, dt)),
                        product_formula="lie",
                        steps_per_interval=1,
                    )[-1]
                )
            coherent_values.append(coherent_density_matrix)
        coherent_lie_density_matrices = np.asarray(
            coherent_values
            if (
                schedule.includes_initial_time
                if schedule is not None
                else np.isclose(time_grid[0], 0.0)
            )
            else coherent_values[1:]
        )
        identity = reference_tools.eye(
            2**number_of_qubits,
            dtype=complex,
            format="csr",
        )
        reference_observables = classical.build_observables(
            identity=identity,
            X_ops=x_operators,
            Y_ops=y_operators,
            Z_ops=z_operators,
            J=reference_tools.J,
        )

    return (
        reference_tools.evaluate_observables(
            exact_density_matrices,
            reference_observables,
        ),
        reference_tools.evaluate_observables(
            coherent_lie_density_matrices,
            reference_observables,
        ),
    )


def _plot_observable_family(
    axis,
    times,
    values,
    title,
    ylabel,
    series_label,
    exact_values=None,
):
    for index, series in enumerate(values):
        line = axis.plot(
            times,
            series,
            label=series_label(index),
        )[0]
        if exact_values is not None:
            axis.plot(
                times,
                exact_values[index],
                linestyle="--",
                color=line.get_color(),
                label=f"{series_label(index)}, exact",
            )
    axis.set_title(title)
    axis.set_xlabel("Time")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small", ncol=2)


def plot_results(
    times,
    backend_name,
    measured,
    standard_errors,
    output_path,
    exact=None,
    coherent_lie=None,
):
    populations, correlations, flows = measured
    population_errors, correlation_errors, flow_errors = standard_errors
    exact_populations, exact_correlations, exact_flows = (
        (None, None, None) if exact is None else exact
    )
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)

    _plot_observable_family(
        axes[0, 0],
        times,
        populations,
        "Local excitation populations",
        r"$\langle n_i\rangle$",
        lambda site: f"site {site}, {backend_name}",
        exact_populations,
    )
    _plot_observable_family(
        axes[0, 1],
        times,
        correlations,
        "Nearest-neighbor XY correlations",
        r"$C_i^{XY}$",
        lambda bond: f"bond {bond}-{bond + 1}, {backend_name}",
        exact_correlations,
    )
    _plot_observable_family(
        axes[1, 0],
        times,
        flows,
        "Nearest-neighbor excitation flows",
        r"$I_{i\rightarrow i+1}$",
        lambda bond: f"bond {bond}-{bond + 1}, {backend_name}",
        exact_flows,
    )

    diagnostic_axis = axes[1, 1]
    if exact is not None and coherent_lie is not None:
        diagnostic_axis.plot(
            times,
            reference_tools.aggregate_observable_error(measured, exact),
            label="vs exact Lindblad",
        )
        diagnostic_axis.plot(
            times,
            reference_tools.aggregate_observable_error(
                measured,
                coherent_lie,
            ),
            label="vs coherent Lie dilation",
        )
        diagnostic_axis.set_title("Aggregate observable errors")
        diagnostic_axis.set_ylabel("Euclidean error")
    else:
        diagnostic_axis.plot(
            times,
            np.max(population_errors, axis=0),
            label="max population s.e.",
        )
        diagnostic_axis.plot(
            times,
            np.max(correlation_errors, axis=0),
            label="max correlation s.e.",
        )
        diagnostic_axis.plot(
            times,
            np.max(flow_errors, axis=0),
            label="max flow s.e.",
        )
        diagnostic_axis.set_title("Shot-noise standard errors")
        diagnostic_axis.set_ylabel("Standard error")

    diagnostic_axis.set_xlabel("Time")
    diagnostic_axis.grid(alpha=0.25)
    diagnostic_axis.legend()
    figure.suptitle(
        f"Dynamic Lie–Trotter sampling on {backend_name}",
        fontsize=15,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _metrics_pair(metrics):
    return (
        (metrics.original_gate_count, metrics.original_depth),
        (metrics.compiled_gate_count, metrics.compiled_depth),
    )


def compile_one_step_metrics(options, dt, jump_angle):
    """Compile one isolated dynamic Lie substep without executing it."""
    circuit = dynamic_circuits.build_one_step_circuit(
        options.n_qubits,
        dt,
        jump_angle,
    )
    target, environment = _runtime_target(
        options.backend,
        options.aer_method,
        options.account_file,
    )
    compiler = CompilerConfig(
        optimization_level=options.optimization_level,
        seed_transpiler=options.seed_transpiler,
    )
    match compile_circuit_batch_sync(
        (circuit,),
        target,
        compiler,
        environment,
    ):
        case Err(error):
            raise RuntimeError(_runtime_error_message(error))
        case Ok(metrics):
            return metrics[0]


def _representative_full_result(results, metadata, time_index):
    for result, entry in zip(results, metadata, strict=True):
        if entry == (time_index, "Z"):
            return result
    raise ValueError("could not locate the representative full circuit")


def _representative_full_circuit(circuits, metadata, time_index):
    for circuit, entry in zip(circuits, metadata, strict=True):
        if entry == (time_index, "Z"):
            return circuit
    raise ValueError("could not locate the representative full circuit")


def transpilation_summary(one_step_metrics, full_result):
    """Return pre/post operation-count and depth records for plotting."""
    full_pair = (
        (full_result.original_gate_count, full_result.original_depth),
        (full_result.compiled_gate_count, full_result.compiled_depth),
    )
    if one_step_metrics is None:
        labels = ("initial-state circuit",)
        pairs = (full_pair,)
    else:
        labels = (
            "one dynamic Lie substep",
            "complete final-time circuit (Z basis)",
        )
        pairs = (_metrics_pair(one_step_metrics), full_pair)
    return tuple(
        {
            "label": label,
            "pre": {
                "operation_count": pair[0][0],
                "depth": pair[0][1],
            },
            "post": {
                "operation_count": pair[1][0],
                "depth": pair[1][1],
            },
        }
        for label, pair in zip(labels, pairs, strict=True)
    )


def plot_transpilation_metrics(summary, output_path):
    """Plot pre/post operation count and depth for step and full circuit."""
    figure, axes = plt.subplots(
        1,
        len(summary),
        figsize=(6.5 * len(summary), 5),
    )
    axes = np.atleast_1d(axes)
    positions = np.arange(2)
    width = 0.36
    for axis, record in zip(axes, summary, strict=True):
        pre_values = (
            record["pre"]["operation_count"],
            record["pre"]["depth"],
        )
        post_values = (
            record["post"]["operation_count"],
            record["post"]["depth"],
        )
        pre_bars = axis.bar(
            positions - width / 2,
            pre_values,
            width,
            label="pre-transpilation",
        )
        post_bars = axis.bar(
            positions + width / 2,
            post_values,
            width,
            label="post-transpilation",
        )
        axis.bar_label(pre_bars, fmt="%d", padding=3)
        axis.bar_label(post_bars, fmt="%d", padding=3)
        axis.set_xticks(positions, ("Operations", "Depth"))
        axis.set_title(record["label"])
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(0, 1.18 * max((*pre_values, *post_values, 1)))
    axes[0].set_ylabel("Count")
    axes[-1].legend(fontsize="small")
    figure.suptitle("Dynamic Lie--Trotter transpilation metrics")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_one_step_circuit(number_of_qubits, dt, jump_angle, output_path):
    """Draw one pre-transpilation dynamic Lie substep."""
    circuit = dynamic_circuits.build_one_step_circuit(
        number_of_qubits,
        dt,
        jump_angle,
    )
    dynamic_circuits.plot_one_step_circuit(circuit, output_path)


def logical_qubit_roles(number_of_qubits):
    """Describe Experiment 4's little-endian logical circuit qubits."""
    return (
        *tuple(
            f"system site {number_of_qubits - 1 - qubit}"
            for qubit in range(number_of_qubits)
        ),
        "low ancilla a_L",
        "high ancilla a_H",
    )


def plot_transpiled_circuit_layout(circuit, options, output_path):
    """Compile and plot the representative circuit's backend placement."""
    target, environment = _runtime_target(
        options.backend,
        options.aer_method,
        options.account_file,
    )
    compiler = CompilerConfig(
        optimization_level=options.optimization_level,
        seed_transpiler=options.seed_transpiler,
    )
    match draw_transpiled_circuit_layout_sync(
        circuit,
        target,
        compiler,
        environment,
        view="virtual",
        logical_labels=logical_qubit_roles(options.n_qubits),
    ):
        case Err(error):
            raise RuntimeError(_runtime_error_message(error))
        case Ok(figure):
            pass
    title = f"Final-time Z-basis qubit layout on {options.backend}"
    if options.backend.lower() == "aer" and figure.axes:
        figure.axes[0].set_title(
            f"{title}\nunconstrained connectivity; identity placement",
            fontsize=14,
            pad=16,
        )
    else:
        figure.suptitle(title, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def output_paths(backend_name, number_of_qubits, output_directory=None):
    stem = (
        f"dynamic_lie_trotter_{_safe_name(backend_name)}_"
        f"{number_of_qubits}"
    )
    directory = (
        Path(__file__).resolve().parent
        if output_directory is None
        else Path(output_directory)
    )
    return (
        directory / "figures" / f"{stem}.png",
        directory / "results" / f"{stem}.json",
    )


def diagnostic_output_paths(
    backend_name,
    number_of_qubits,
    output_directory=None,
):
    results_figure, result_path = output_paths(
        backend_name,
        number_of_qubits,
        output_directory,
    )
    stem = result_path.stem
    return (
        results_figure.with_name(f"{stem}_transpilation_metrics.png"),
        results_figure.with_name(f"{stem}_one_step_circuit.png"),
        results_figure.with_name(f"{stem}_transpiled_layout.png"),
    )


def default_checkpoint_path(result_path):
    return result_path.with_name(
        f"{result_path.stem}_checkpoint.json"
    )


def _atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _metadata_file_specification(payload):
    configuration = payload.get("configuration")
    checkpoint_records = payload.get("results")
    if isinstance(configuration, dict) and isinstance(
        checkpoint_records,
        list,
    ):
        return {
            "kind": "checkpoint",
            "backend": configuration.get("backend"),
            "number_of_qubits": configuration.get(
                "number_of_system_qubits"
            ),
            "times": configuration.get("times"),
            "optimization_level": configuration.get(
                "optimization_level",
                DEFAULT_OPTIMIZATION_LEVEL,
            ),
            "seed_transpiler": configuration.get(
                "seed_transpiler",
                DEFAULT_SEED_TRANSPILER,
            ),
            "trotter_delta_t": configuration.get("trotter_delta_t"),
            "records": checkpoint_records,
        }

    result_records = payload.get("circuit_metadata")
    if isinstance(result_records, list):
        schedule = payload.get("evolution_schedule")
        return {
            "kind": "result",
            "backend": payload.get("backend"),
            "number_of_qubits": payload.get("number_of_system_qubits"),
            "times": payload.get("times"),
            "optimization_level": payload.get(
                "optimization_level",
                DEFAULT_OPTIMIZATION_LEVEL,
            ),
            "seed_transpiler": payload.get(
                "seed_transpiler",
                DEFAULT_SEED_TRANSPILER,
            ),
            "trotter_delta_t": (
                schedule.get("requested_trotter_delta_t")
                if isinstance(schedule, dict)
                else None
            ),
            "records": result_records,
        }
    raise ValueError(
        "metadata file is neither an Experiment 4 result nor checkpoint"
    )


def _next_metadata_backup_path(path):
    candidate = path.with_name(
        f"{path.stem}.before_metadata{path.suffix}"
    )
    index = 1
    while candidate.exists():
        candidate = path.with_name(
            f"{path.stem}.before_metadata_{index}{path.suffix}"
        )
        index += 1
    return candidate


def _backfilled_record(record, metrics, overwrite):
    replacements = {
        "original_operation_count": metrics.original_gate_count,
        "original_depth": metrics.original_depth,
        "compiled_operation_count": metrics.compiled_gate_count,
        "compiled_depth": metrics.compiled_depth,
    }
    return {
        **record,
        **{
            field: value
            for field, value in replacements.items()
            if overwrite
            or not isinstance(record.get(field), int)
            or record[field] < 0
        },
    }


def backfill_metadata_only(options):
    """Recompile recorded circuits and update metrics without submission."""
    if options.metadata_file is None:
        raise ValueError("--metadata-only requires --metadata-file PATH")
    metadata_path = Path(options.metadata_file)
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read metadata file {metadata_path}: {error}"
        ) from error

    specification = _metadata_file_specification(payload)
    backend_name = specification["backend"]
    number_of_qubits = specification["number_of_qubits"]
    records = specification["records"]
    if not isinstance(backend_name, str) or not backend_name:
        raise ValueError("metadata file has no valid backend name")
    if not isinstance(number_of_qubits, int) or number_of_qubits < 2:
        raise ValueError("metadata file has no valid system-qubit count")
    if not records:
        raise ValueError("metadata file contains no completed circuits")

    if specification["kind"] == "result" and "archive" in payload:
        circuit_values = []
        metadata_values = []
        histories = payload.get("run_history", ())
        rebuilt_runs = {}
        for time_index, time in enumerate(specification["times"]):
            matching_history_index = next(
                (
                    index
                    for index in reversed(range(len(histories)))
                    if any(
                        np.isclose(
                            time,
                            candidate,
                            rtol=1e-12,
                            atol=1e-12,
                        )
                        for candidate in histories[index].get("times", ())
                    )
                ),
                None,
            )
            if matching_history_index is None:
                run_times = (time,)
                run_delta_t = specification["trotter_delta_t"]
            else:
                history = histories[matching_history_index]
                run_times = history["times"]
                history_schedule = history.get("evolution_schedule")
                run_delta_t = (
                    history_schedule.get("requested_trotter_delta_t")
                    if isinstance(history_schedule, dict)
                    else specification["trotter_delta_t"]
                )
            cache_key = (
                matching_history_index,
                tuple(map(float, run_times)),
                run_delta_t,
            )
            if cache_key not in rebuilt_runs:
                rebuilt_runs[cache_key] = build_sample_circuits(
                    run_times,
                    number_of_qubits,
                    trotter_delta_t=run_delta_t,
                )[:2]
            run_circuits, run_metadata = rebuilt_runs[cache_key]
            run_time_index = next(
                index
                for index, candidate in enumerate(run_times)
                if np.isclose(
                    time,
                    candidate,
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
            selected = tuple(
                (circuit, basis)
                for circuit, (candidate_time_index, basis) in zip(
                    run_circuits,
                    run_metadata,
                    strict=True,
                )
                if candidate_time_index == run_time_index
            )
            circuit_values.extend(circuit for circuit, _ in selected)
            metadata_values.extend(
                (time_index, basis) for _, basis in selected
            )
        circuits = tuple(circuit_values)
        metadata = tuple(metadata_values)
    else:
        circuits, metadata, _ = build_sample_circuits(
            specification["times"],
            number_of_qubits,
            trotter_delta_t=specification["trotter_delta_t"],
        )
    if len(records) > len(circuits):
        raise ValueError("metadata file contains too many circuit records")
    for index, record in enumerate(records):
        expected_time_index, expected_basis = metadata[index]
        if (
            record.get("time_index") != expected_time_index
            or record.get("basis") != expected_basis
        ):
            raise ValueError(
                "metadata records do not match the rebuilt circuit order"
            )

    target, environment = _runtime_target(
        backend_name,
        options.aer_method,
        options.account_file,
    )
    compiler = CompilerConfig(
        optimization_level=int(specification["optimization_level"]),
        seed_transpiler=specification["seed_transpiler"],
    )
    print(
        f"Metadata-only: transpiling {len(records)} circuits for "
        f"{backend_name}; no Sampler job will be submitted"
    )
    match compile_circuit_batch_sync(
        circuits[: len(records)],
        target,
        compiler,
        environment,
    ):
        case Err(error):
            raise RuntimeError(_runtime_error_message(error))
        case Ok(metrics):
            pass

    updated_records = [
        _backfilled_record(record, metric, options.overwrite_metadata)
        for record, metric in zip(records, metrics, strict=True)
    ]
    changed_count = sum(
        original != updated
        for original, updated in zip(records, updated_records, strict=True)
    )
    if changed_count == 0:
        print("All circuit metrics are already present; no file was changed.")
        return

    backup_path = _next_metadata_backup_path(metadata_path)
    _atomic_write_json(backup_path, payload)
    record_key = (
        "results"
        if specification["kind"] == "checkpoint"
        else "circuit_metadata"
    )
    updated_payload = {
        **payload,
        record_key: updated_records,
        "metadata_backfill": {
            "backend": backend_name,
            "optimization_level": compiler.optimization_level,
            "seed_transpiler": compiler.seed_transpiler,
            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
            "circuit_count": len(metrics),
            "uses_current_backend_target": backend_name.lower() != "aer",
            "hardware_job_submitted": False,
        },
    }
    _atomic_write_json(metadata_path, updated_payload)
    print(f"Updated metrics for {changed_count} circuits: {metadata_path}")
    print(f"Backup: {backup_path}")
    print("Hardware jobs submitted: 0")


def _checkpoint_configuration(options, times, metadata):
    return {
        "backend": options.backend.lower(),
        "number_of_system_qubits": options.n_qubits,
        "shots_per_measurement_circuit": options.shots,
        "times": [float(value) for value in times],
        "measurement_bases": list(MEASUREMENT_BASES),
        "circuit_metadata": [
            {
                "time_index": time_index,
                "basis": basis,
            }
            for time_index, basis in metadata
        ],
        "optimization_level": options.optimization_level,
        "seed_transpiler": options.seed_transpiler,
        "seed_simulator": options.seed_simulator,
        "aer_method": options.aer_method,
        "trotter_delta_t": getattr(options, "trotter_delta_t", None),
    }


def _sample_result_record(index, result, metadata):
    time_index, basis = metadata[index]
    return {
        "circuit_index": index,
        "time_index": time_index,
        "basis": basis,
        "counts": counts_dict(result),
        "shots": result.shots,
        "backend_name": result.backend_name,
        "job_id": result.job_id,
        "original_operation_count": result.original_gate_count,
        "original_depth": result.original_depth,
        "compiled_operation_count": result.compiled_gate_count,
        "compiled_depth": result.compiled_depth,
    }


def save_checkpoint(
    checkpoint_path,
    options,
    times,
    schedule,
    results,
    metadata,
    status="partial",
):
    """Atomically persist a completed prefix of the sampling circuits."""
    if len(results) > len(metadata):
        raise ValueError("checkpoint contains more results than circuits")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment": "Model1/Experiment4 dynamic Lie-Trotter",
        "status": status,
        "completed_circuit_count": len(results),
        "total_circuit_count": len(metadata),
        "configuration": _checkpoint_configuration(
            options,
            times,
            metadata,
        ),
        "physics": _schedule_payload(times, schedule),
        "results": [
            _sample_result_record(index, result, metadata)
            for index, result in enumerate(results)
        ],
    }
    _atomic_write_json(Path(checkpoint_path), payload)


def _sample_result_from_record(record):
    counts = record.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("checkpoint result counts must be a JSON object")
    return SampleResult(
        counts=tuple(
            sorted(
                (str(bitstring), int(count))
                for bitstring, count in counts.items()
            )
        ),
        shots=int(record["shots"]),
        backend_name=str(record["backend_name"]),
        job_id=(
            None
            if record.get("job_id") is None
            else str(record["job_id"])
        ),
        original_gate_count=int(record["original_operation_count"]),
        original_depth=int(record["original_depth"]),
        compiled_gate_count=int(record["compiled_operation_count"]),
        compiled_depth=int(record["compiled_depth"]),
    )


def load_checkpoint(checkpoint_path, options, times, metadata):
    """Load and validate a checkpoint for an explicit resume operation."""
    path = Path(checkpoint_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read checkpoint {path}: {error}") from error

    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint {path} has an unsupported schema version"
        )
    expected_configuration = _checkpoint_configuration(
        options,
        times,
        metadata,
    )
    saved_configuration = payload.get("configuration")
    if (
        isinstance(saved_configuration, dict)
        and "trotter_delta_t" not in saved_configuration
    ):
        saved_configuration = {
            **saved_configuration,
            "trotter_delta_t": None,
        }
    if saved_configuration != expected_configuration:
        raise ValueError(
            "checkpoint configuration does not match this run; use the "
            "same backend, N, shots, time grid, and compiler settings"
        )

    records = payload.get("results")
    if not isinstance(records, list) or len(records) > len(metadata):
        raise ValueError("checkpoint has an invalid result list")
    for index, record in enumerate(records):
        expected_time_index, expected_basis = metadata[index]
        if (
            record.get("circuit_index") != index
            or record.get("time_index") != expected_time_index
            or record.get("basis") != expected_basis
        ):
            raise ValueError(
                "checkpoint results are not a consecutive circuit prefix"
            )
    return tuple(_sample_result_from_record(record) for record in records)


def _array_families_payload(values):
    return {
        "populations": values[0].tolist(),
        "xy_correlations": values[1].tolist(),
        "excitation_flows": values[2].tolist(),
    }


def _result_compatibility_signature(payload):
    schedule = payload.get("evolution_schedule")
    requested_delta_t = (
        schedule.get("requested_trotter_delta_t")
        if isinstance(schedule, dict)
        else None
    )
    effective_delta_t = (
        requested_delta_t
        if requested_delta_t is not None
        else (
            schedule.get("uniform_substep_dt")
            if isinstance(schedule, dict)
            else None
        )
    )
    return {
        "experiment": payload.get("experiment"),
        "backend": str(payload.get("backend", "")).lower(),
        "number_of_system_qubits": payload.get("number_of_system_qubits"),
        "shots_per_measurement_circuit": payload.get(
            "shots_per_measurement_circuit"
        ),
        "measurement_bases": tuple(payload.get("measurement_bases", ())),
        "optimization_level": payload.get("optimization_level"),
        "seed_transpiler": payload.get("seed_transpiler"),
        "seed_simulator": payload.get("seed_simulator"),
        "aer_method": payload.get("aer_method", "automatic"),
        "effective_trotter_delta_t": effective_delta_t,
        "has_classical_reference": all(
            key in payload
            for key in (
                "exact_reference_observables",
                "coherent_lie_reference_observables",
            )
        ),
    }


def _prospective_compatibility_signature(options, schedule):
    return {
        "experiment": "Model1/Experiment4 dynamic Lie-Trotter",
        "backend": options.backend.lower(),
        "number_of_system_qubits": options.n_qubits,
        "shots_per_measurement_circuit": options.shots,
        "measurement_bases": MEASUREMENT_BASES,
        "optimization_level": options.optimization_level,
        "seed_transpiler": options.seed_transpiler,
        "seed_simulator": options.seed_simulator,
        "aer_method": options.aer_method,
        "effective_trotter_delta_t": (
            options.trotter_delta_t
            if options.trotter_delta_t is not None
            else _uniform_value(schedule.substep_dts)
        ),
        "has_classical_reference": options.classical_reference,
    }


def _load_result_payload(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read existing result {path}: {error}"
        ) from error


def validate_existing_result_compatibility(output_path, options, schedule):
    """Reject unsafe archive mixtures before any sampling is submitted."""
    path = Path(output_path)
    if not path.exists():
        return None
    payload = _load_result_payload(path)
    existing = _result_compatibility_signature(payload)
    requested = _prospective_compatibility_signature(options, schedule)
    mismatches = tuple(
        key for key in requested if existing.get(key) != requested[key]
    )
    if mismatches:
        raise ValueError(
            "existing Experiment 4 result is incompatible in: "
            f"{', '.join(mismatches)}. Use matching options or a different "
            "--output-directory; no circuits were submitted."
        )
    return payload


def _merged_time_sources(existing_times, new_times):
    """Return sorted times and prefer the new run at matching times."""
    merged_times = [float(value) for value in existing_times]
    for value in map(float, new_times):
        if not any(
            np.isclose(value, current, rtol=1e-12, atol=1e-12)
            for current in merged_times
        ):
            merged_times.append(value)
    merged_times.sort()

    sources = []
    for value in merged_times:
        new_index = next(
            (
                index
                for index, current in enumerate(new_times)
                if np.isclose(value, current, rtol=1e-12, atol=1e-12)
            ),
            None,
        )
        if new_index is not None:
            sources.append(("new", new_index))
            continue
        old_index = next(
            index
            for index, current in enumerate(existing_times)
            if np.isclose(value, current, rtol=1e-12, atol=1e-12)
        )
        sources.append(("old", old_index))
    return merged_times, tuple(sources)


def _merge_array_on_time_axis(old_values, new_values, sources, axis):
    old_array = np.asarray(old_values)
    new_array = np.asarray(new_values)
    slices = tuple(
        np.take(
            new_array if source == "new" else old_array,
            index,
            axis=axis,
        )
        for source, index in sources
    )
    return np.stack(slices, axis=axis).tolist()


def _merge_family_payload(old_values, new_values, sources, axis=1):
    return {
        family: _merge_array_on_time_axis(
            old_values[family],
            new_values[family],
            sources,
            axis,
        )
        for family in (
            "populations",
            "xy_correlations",
            "excitation_flows",
        )
    }


def _records_by_time(records):
    grouped = {}
    for record in records:
        grouped.setdefault(int(record["time_index"]), []).append(record)
    return grouped


def _merge_indexed_records(
    old_records,
    new_records,
    sources,
    merged_times,
    *,
    include_time,
):
    old_grouped = _records_by_time(old_records)
    new_grouped = _records_by_time(new_records)
    merged = []
    for new_time_index, ((source, source_index), time) in enumerate(
        zip(sources, merged_times, strict=True)
    ):
        source_group = new_grouped if source == "new" else old_grouped
        for record in source_group[source_index]:
            updated = {**record, "time_index": new_time_index}
            if include_time:
                updated["time"] = float(time)
            merged.append(updated)
    return merged


def _run_history_record(payload):
    return {
        "completed_at_utc": payload.get("completed_at_utc"),
        "times": payload.get("times", ()),
        "job_ids": payload.get("job_ids", ()),
        "evolution_schedule": payload.get("evolution_schedule"),
        "transpilation_summary": payload.get(
            "transpilation_summary",
            (),
        ),
        "transpiled_layout": payload.get("transpiled_layout"),
    }


def merge_result_payloads(existing, new):
    """Merge time points, with the new run replacing matching times."""
    if _result_compatibility_signature(existing) != (
        _result_compatibility_signature(new)
    ):
        raise ValueError("cannot merge incompatible Experiment 4 results")
    merged_times, sources = _merged_time_sources(
        existing["times"],
        new["times"],
    )
    merged = {**existing, **new, "times": merged_times}
    for key in ("observables", "standard_errors"):
        merged[key] = _merge_family_payload(
            existing[key],
            new[key],
            sources,
        )
    if "exact_reference_observables" in new:
        for key in (
            "exact_reference_observables",
            "coherent_lie_reference_observables",
        ):
            merged[key] = _merge_family_payload(
                existing[key],
                new[key],
                sources,
            )
        for key in (
            "aggregate_observable_error_vs_exact",
            "aggregate_observable_error_vs_coherent_lie",
        ):
            merged[key] = _merge_array_on_time_axis(
                existing[key],
                new[key],
                sources,
                0,
            )
    merged["raw_counts"] = _merge_indexed_records(
        existing["raw_counts"],
        new["raw_counts"],
        sources,
        merged_times,
        include_time=True,
    )
    merged["circuit_metadata"] = _merge_indexed_records(
        existing["circuit_metadata"],
        new["circuit_metadata"],
        sources,
        merged_times,
        include_time=True,
    )
    uses_existing = any(source == "old" for source, _ in sources)
    merged["job_ids"] = tuple(
        dict.fromkeys(
            (
                *(existing.get("job_ids", ()) if uses_existing else ()),
                *new.get("job_ids", ()),
            )
        )
    )
    existing_history = existing.get("run_history")
    merged["run_history"] = (
        list(existing_history)
        if isinstance(existing_history, list)
        else [_run_history_record(existing)]
    )
    merged["run_history"].append(_run_history_record(new))
    merged["archive"] = {
        "schema_version": 1,
        "merge_policy": "new run replaces matching times",
        "saved_time_count": len(merged_times),
        "latest_run_times": new["times"],
    }
    return merged


def _family_arrays(values):
    return tuple(
        np.asarray(values[family], dtype=float)
        for family in (
            "populations",
            "xy_correlations",
            "excitation_flows",
        )
    )


def _payload_observables(payload, key):
    return _family_arrays(payload[key])


def save_results(
    output_path,
    options,
    times,
    schedule,
    results,
    metadata,
    measured,
    standard_errors,
    grouped_counts,
    extra_payload=None,
):
    schedule_payload = _schedule_payload(times, schedule)
    job_ids = tuple(
        dict.fromkeys(
            result.job_id
            for result in results
            if result.job_id is not None
        )
    )
    raw_counts = tuple(
        {
            "time_index": time_index,
            "time": float(times[time_index]),
            "basis": basis,
            "counts": grouped_counts[time_index][basis],
        }
        for time_index in range(len(times))
        for basis in MEASUREMENT_BASES
    )
    payload = {
        "experiment": "Model1/Experiment4 dynamic Lie-Trotter",
        "backend": options.backend,
        "number_of_system_qubits": options.n_qubits,
        "number_of_circuit_qubits": options.n_qubits + 2,
        "shots_per_measurement_circuit": options.shots,
        "measurement_bases": MEASUREMENT_BASES,
        "mid_circuit_classical_bit": MID_CIRCUIT_BIT,
        "system_terminal_bits_by_site": tuple(
            range(1, options.n_qubits + 1)
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "times": np.asarray(times, dtype=float).tolist(),
        # Scalar aliases remain populated for uniform grids and are null
        # for genuinely nonuniform schedules.
        "dt": schedule_payload["uniform_substep_dt"],
        "system_exchange_angle": schedule_payload[
            "uniform_system_exchange_angle"
        ],
        "jump_angle": schedule_payload["uniform_jump_angle"],
        "jump_exchange_probability": schedule_payload[
            "uniform_jump_exchange_probability"
        ],
        "evolution_schedule": schedule_payload,
        "optimization_level": options.optimization_level,
        "seed_transpiler": options.seed_transpiler,
        "seed_simulator": options.seed_simulator,
        "aer_method": options.aer_method,
        "job_ids": job_ids,
        "circuit_metadata": tuple(
            {
                "time_index": time_index,
                "basis": basis,
                "time": float(times[time_index]),
                "job_id": results[index].job_id,
                "original_operation_count": (
                    results[index].original_gate_count
                ),
                "original_depth": results[index].original_depth,
                "compiled_operation_count": (
                    results[index].compiled_gate_count
                ),
                "compiled_depth": results[index].compiled_depth,
            }
            for index, (time_index, basis) in enumerate(metadata)
        ),
        "observables": _array_families_payload(measured),
        "standard_errors": _array_families_payload(standard_errors),
        "raw_counts": raw_counts,
    }
    if extra_payload is not None:
        payload.update(extra_payload)
    path = Path(output_path)
    if path.exists():
        payload = merge_result_payloads(
            _load_result_payload(path),
            payload,
        )
    else:
        payload["archive"] = {
            "schema_version": 1,
            "merge_policy": "new run replaces matching times",
            "saved_time_count": len(payload["times"]),
            "latest_run_times": payload["times"],
        }
        payload["run_history"] = [_run_history_record(payload)]
    _atomic_write_json(path, payload)
    return payload


def main(options):
    if options.metadata_only:
        backfill_metadata_only(options)
        return
    if options.metadata_file is not None:
        raise ValueError("--metadata-file is only valid with --metadata-only")

    times = time_grid_from_options(options)
    (
        circuits,
        metadata,
        schedule,
    ) = build_sample_circuits(
        times,
        options.n_qubits,
        trotter_delta_t=options.trotter_delta_t,
    )
    print(
        f"Built {len(circuits)} circuits: {len(times)} saved times "
        f"x {len(MEASUREMENT_BASES)} measurement bases"
    )
    figure_path, result_path = output_paths(
        options.backend,
        options.n_qubits,
        options.output_directory,
    )
    (
        metrics_figure_path,
        circuit_figure_path,
        layout_figure_path,
    ) = diagnostic_output_paths(
        options.backend,
        options.n_qubits,
        options.output_directory,
    )
    existing_payload = validate_existing_result_compatibility(
        result_path,
        options,
        schedule,
    )
    if existing_payload is not None:
        print(
            f"Existing archive contains {len(existing_payload['times'])} "
            "saved time(s); new values will be appended and matching "
            "times replaced"
        )
    checkpoint_path = (
        default_checkpoint_path(result_path)
        if options.checkpoint_file is None
        else options.checkpoint_file
    )
    if options.resume:
        completed_results = load_checkpoint(
            checkpoint_path,
            options,
            times,
            metadata,
        )
        print(
            f"Resuming from {checkpoint_path}: "
            f"{len(completed_results)}/{len(circuits)} circuits complete"
        )
    else:
        completed_results = ()
        save_checkpoint(
            checkpoint_path,
            options,
            times,
            schedule,
            completed_results,
            metadata,
        )
        print(f"Checkpoint: {checkpoint_path}")

    def checkpoint_new_results(new_results):
        all_completed = (*completed_results, *new_results)
        save_checkpoint(
            checkpoint_path,
            options,
            times,
            schedule,
            all_completed,
            metadata,
        )
        print(
            f"Checkpointed {len(all_completed)}/{len(circuits)} "
            f"circuits: {checkpoint_path}"
        )

    remaining_circuits = circuits[len(completed_results) :]
    new_results = (
        execute_sample_circuits(
            remaining_circuits,
            options,
            on_batch_complete=checkpoint_new_results,
        )
        if remaining_circuits
        else ()
    )
    results = (*completed_results, *new_results)
    measured, standard_errors, grouped_counts = observables_from_results(
        results,
        metadata,
        options.n_qubits,
        len(times),
    )
    exact = coherent_lie = None
    if options.classical_reference:
        print("Calculating exact and coherent-Lie classical references")
        exact, coherent_lie = calculate_references(
            times,
            options.n_qubits,
            schedule,
        )

    full_result = _representative_full_result(
        results,
        metadata,
        len(times) - 1,
    )
    full_circuit = _representative_full_circuit(
        circuits,
        metadata,
        len(times) - 1,
    )
    representative_dt = (
        schedule.substep_dts[0] if schedule.substep_dts else None
    )
    one_step_metrics = (
        compile_one_step_metrics(
            options,
            representative_dt,
            schedule.jump_angles[0],
        )
        if representative_dt is not None
        else None
    )
    metrics_summary = transpilation_summary(
        one_step_metrics,
        full_result,
    )
    plot_transpilation_metrics(metrics_summary, metrics_figure_path)
    plot_transpiled_circuit_layout(
        full_circuit,
        options,
        layout_figure_path,
    )
    if representative_dt is not None:
        plot_one_step_circuit(
            options.n_qubits,
            representative_dt,
            schedule.jump_angles[0],
            circuit_figure_path,
        )

    extra_payload = {
        "transpilation_summary": metrics_summary,
        "transpiled_layout": {
            "backend": options.backend,
            "basis": "Z",
            "saved_time": float(times[-1]),
            "view": "virtual",
            "figure": layout_figure_path.name,
        },
    }
    if exact is not None:
        extra_payload.update(
            {
                "exact_reference_observables": (
                    _array_families_payload(exact)
                ),
                "coherent_lie_reference_observables": (
                    _array_families_payload(coherent_lie)
                ),
                "aggregate_observable_error_vs_exact": (
                    reference_tools.aggregate_observable_error(
                        measured,
                        exact,
                    ).tolist()
                ),
                "aggregate_observable_error_vs_coherent_lie": (
                    reference_tools.aggregate_observable_error(
                        measured,
                        coherent_lie,
                    ).tolist()
                ),
            }
        )

    combined_payload = save_results(
        result_path,
        options,
        times,
        schedule,
        results,
        metadata,
        measured,
        standard_errors,
        grouped_counts,
        extra_payload=extra_payload,
    )
    plot_results(
        np.asarray(combined_payload["times"], dtype=float),
        options.backend,
        _payload_observables(combined_payload, "observables"),
        _payload_observables(combined_payload, "standard_errors"),
        figure_path,
        (
            _payload_observables(
                combined_payload,
                "exact_reference_observables",
            )
            if "exact_reference_observables" in combined_payload
            else None
        ),
        (
            _payload_observables(
                combined_payload,
                "coherent_lie_reference_observables",
            )
            if "coherent_lie_reference_observables" in combined_payload
            else None
        ),
    )
    save_checkpoint(
        checkpoint_path,
        options,
        times,
        schedule,
        results,
        metadata,
        status="complete",
    )

    job_ids = tuple(
        dict.fromkeys(
            result.job_id
            for result in results
            if result.job_id is not None
        )
    )
    print(f"Backend: {options.backend}")
    print(f"System qubits: {options.n_qubits}")
    print(f"Total circuit qubits: {options.n_qubits + 2}")
    print(f"Shots per measurement circuit: {options.shots}")
    print(
        f"Archive saved times: {combined_payload['times']} "
        "(new run replaces matching times)"
    )
    if options.trotter_delta_t is None:
        print("Trotter resolution: one substep per saved-time interval")
    else:
        print(
            "Requested maximum Trotter step: "
            f"{options.trotter_delta_t:.8f}"
        )
    print(f"Total Trotter substeps: {len(schedule.substep_dts)}")
    uniform_dt = _uniform_value(schedule.substep_dts)
    if not schedule.substep_dts:
        print("Lie--Trotter substeps: none (initial-state-only run)")
    elif uniform_dt is None:
        print(
            "Lie--Trotter substep range: "
            f"{min(schedule.substep_dts):.8f}--"
            f"{max(schedule.substep_dts):.8f}"
        )
        print(
            "Jump XXPlusYY angle range: "
            f"{min(schedule.jump_angles):.8f}--"
            f"{max(schedule.jump_angles):.8f}"
        )
    else:
        print(f"Lie--Trotter step: {uniform_dt:.8f}")
        print(
            "System XXPlusYY angle: "
            f"{2.0 * reference_tools.J * uniform_dt:.8f}"
        )
        print(f"Jump XXPlusYY angle: {schedule.jump_angles[0]:.8f}")
    print(
        "Compiled operation-count range: "
        f"{min(result.compiled_gate_count for result in results)}--"
        f"{max(result.compiled_gate_count for result in results)}"
    )
    print(
        "Compiled depth range: "
        f"{min(result.compiled_depth for result in results)}--"
        f"{max(result.compiled_depth for result in results)}"
    )
    if job_ids:
        print(f"Provider job IDs: {', '.join(job_ids)}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved transpilation metrics: {metrics_figure_path}")
    print(f"Saved transpiled layout: {layout_figure_path}")
    if representative_dt is not None:
        print(f"Saved one-step circuit: {circuit_figure_path}")
    print(f"Saved data: {result_path}")
    print(f"Saved checkpoint: {checkpoint_path}")


def positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def at_least_two(value):
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("value must be at least two")
    return parsed


def positive_float(value):
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            "value must be finite and greater than zero"
        )
    return parsed


def time_grid_from_options(options):
    if options.times is None:
        return np.linspace(0.0, options.t_final, options.time_points)

    try:
        values = tuple(
            float(part.strip())
            for token in options.times
            for part in token.split(",")
            if part.strip()
        )
    except ValueError as error:
        raise ValueError(
            "--times must contain only comma- or space-separated numbers"
        ) from error
    time_grid = np.asarray(values, dtype=float)
    if time_grid.size == 1:
        if not np.isfinite(time_grid[0]) or time_grid[0] < 0.0:
            raise ValueError(
                "a single --times value must be finite and nonnegative"
            )
    if time_grid.size < 2:
        return time_grid
    if not np.all(np.isfinite(time_grid)):
        raise ValueError("--times must contain only finite values")
    if not np.isclose(time_grid[0], 0.0):
        raise ValueError("--times must start at 0")
    if np.any(np.diff(time_grid) <= 0.0):
        raise ValueError("--times values must be strictly increasing")
    return time_grid


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the general-N dynamic Lie open-system circuit on Aer or "
            "an IBM dynamic-circuit backend."
        )
    )
    parser.add_argument(
        "--n-qubits",
        type=at_least_two,
        default=DEFAULT_NUMBER_OF_QUBITS,
        help=(
            "number of system qubits "
            f"(default: {DEFAULT_NUMBER_OF_QUBITS})"
        ),
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help=(
            "use 'aer' for local simulation or an IBM backend name such "
            "as 'ibm_kingston' (default: aer)"
        ),
    )
    parser.add_argument(
        "--account-file",
        type=Path,
        default=DEFAULT_ACCOUNT_FILE,
        help=(
            "JSON file containing IBM 'apikey' and 'crn' fields "
            f"(default: {DEFAULT_ACCOUNT_FILE})"
        ),
    )
    parser.add_argument(
        "--shots",
        type=positive_integer,
        default=DEFAULT_SHOTS,
        help=f"shots per measurement circuit (default: {DEFAULT_SHOTS})",
    )
    parser.add_argument(
        "--t-final",
        type=positive_float,
        default=dynamic_circuits.T_FINAL,
        help=f"final physical time (default: {dynamic_circuits.T_FINAL})",
    )
    parser.add_argument(
        "--time-points",
        type=at_least_two,
        default=dynamic_circuits.NUMBER_OF_TIME_POINTS,
        help=(
            "number of saved times including zero "
            f"(default: {dynamic_circuits.NUMBER_OF_TIME_POINTS})"
        ),
    )
    parser.add_argument(
        "--times",
        nargs="+",
        default=None,
        metavar="T",
        help=(
            "explicit saved times; one nonnegative value T saves only T, "
            "while a list must start at 0; accepts space- or "
            "comma-separated values and overrides --t-final and "
            "--time-points"
        ),
    )
    parser.add_argument(
        "--trotter-delta-t",
        type=positive_float,
        default=None,
        help=(
            "maximum internal Lie--Trotter substep; each saved-time "
            "interval uses full steps of this size followed by a shorter "
            "remainder when necessary (default: one substep per interval)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=positive_integer,
        default=None,
        help=(
            "maximum circuits submitted in one provider job "
            f"(default: {DEFAULT_AER_BATCH_SIZE} on Aer and "
            f"{DEFAULT_HARDWARE_BATCH_SIZE} on IBM hardware)"
        ),
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=DEFAULT_OPTIMIZATION_LEVEL,
    )
    parser.add_argument(
        "--seed-transpiler",
        type=int,
        default=DEFAULT_SEED_TRANSPILER,
    )
    parser.add_argument(
        "--seed-simulator",
        type=int,
        default=DEFAULT_SEED_SIMULATOR,
        help="used only by Aer",
    )
    parser.add_argument(
        "--aer-method",
        choices=AER_METHODS,
        default="automatic",
        help="AerSimulator method used when --backend aer",
    )
    parser.add_argument(
        "--classical-reference",
        action="store_true",
        help=(
            "also calculate exact Lindblad and coherent-Lie references; "
            "use only for classically manageable N"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help=(
            "directory that receives figures/ and results/ "
            "(default: Model1/Experiment4)"
        ),
    )
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=None,
        help=(
            "checkpoint JSON path (default: a *_checkpoint.json file "
            "beside the final result)"
        ),
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "rebuild and transpile recorded circuits to fill unavailable "
            "operation/depth metrics; never submit a Sampler job"
        ),
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help=(
            "Experiment 4 result or checkpoint JSON updated by "
            "--metadata-only"
        ),
    )
    parser.add_argument(
        "--overwrite-metadata",
        action="store_true",
        help=(
            "replace existing nonnegative metrics in metadata-only mode; "
            "by default only missing or negative values are replaced"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "load the checkpoint and submit only its missing circuit "
            "suffix; configuration must match"
        ),
    )
    return parser.parse_args(arguments)


if __name__ == "__main__":
    main(parse_arguments())
