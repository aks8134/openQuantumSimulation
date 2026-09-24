"""Randomized single-boundary Lie--Trotter simulation on Aer or IBM QPUs.

For the boundary-damped XX chain,

    L = L_H + D[V_L] + D[V_R],

sample one of the two simple generators at every internal substep,

    L_L = L_H + 2 D[V_L],    L_R = L_H + 2 D[V_R].

Their uniform mean is the target generator.  The selected jump is therefore
scaled by sqrt(2), giving an XXPlusYY angle 2 sqrt(2 gamma dt).  A trajectory
uses two boundary-local ancilla wires, resets only the selected one, and
contains no measurement-based feed-forward.

``--shots`` is the total shot budget for one saved-time/measurement-basis
pair.  It is divided evenly among ``--trajectories`` independently sampled
boundary-choice sequences.  The same trajectory prefixes are reused at all
saved times and in all five measurement bases.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
from math import pi, sqrt
from pathlib import Path
import re
from types import SimpleNamespace
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
    Circuit,
    CompilerConfig,
    Err,
    Ok,
    SampleResult,
    compile_circuit_batch_sync,
    counts_dict,
    draw_circuit,
    draw_transpiled_circuit_layout_sync,
    empty,
    fold_left,
    h,
    map_tuple,
    measure,
    pipe,
    reset,
    rz,
    scan_left,
    x,
    xx_plus_yy,
)
from Model1.Experiment3 import unitary_lie_trotter as chain_tools
from Model1.Experiment4 import dynamic_lie_trotter as hardware_tools


DEFAULT_ACCOUNT_FILE = hardware_tools.DEFAULT_ACCOUNT_FILE
DEFAULT_NUMBER_OF_QUBITS = chain_tools.DEFAULT_NUMBER_OF_QUBITS
DEFAULT_BACKEND = "aer"
DEFAULT_TRAJECTORIES = 16
DEFAULT_SHOTS = 8192
DEFAULT_SEED_TRAJECTORIES = 29
DEFAULT_SEED_SIMULATOR = chain_tools.SEED_SIMULATOR
DEFAULT_OPTIMIZATION_LEVEL = hardware_tools.DEFAULT_OPTIMIZATION_LEVEL
DEFAULT_SEED_TRANSPILER = hardware_tools.DEFAULT_SEED_TRANSPILER
DEFAULT_AER_BATCH_SIZE = 300
DEFAULT_HARDWARE_BATCH_SIZE = 5
CHECKPOINT_SCHEMA_VERSION = 1
CIRCUIT_LAYOUT_VERSION = 2
ANCILLA_STRATEGY = "two boundary-local resettable ancillas"

J = chain_tools.J
h_field = chain_tools.h
gamma = chain_tools.gamma
T_FINAL = chain_tools.T_FINAL
NUMBER_OF_TIME_POINTS = chain_tools.NUMBER_OF_TIME_POINTS

MEASUREMENT_BASES = hardware_tools.MEASUREMENT_BASES
LEFT_BOUNDARY = 0
RIGHT_BOUNDARY = 1
BOUNDARY_NAMES = ("L", "R")


@dataclass(frozen=True, slots=True)
class RandomizedSchedule:
    """Physical substeps and all seeded boundary-choice trajectories."""

    interval_substep_dts: tuple[tuple[float, ...], ...]
    interval_jump_angles: tuple[tuple[float, ...], ...]
    interval_jump_probabilities: tuple[tuple[float, ...], ...]
    boundary_choices: tuple[tuple[int, ...], ...]
    requested_trotter_delta_t: float | None
    seed_trajectories: int
    includes_initial_time: bool = True

    @property
    def substep_dts(self):
        return tuple(
            dt
            for interval in self.interval_substep_dts
            for dt in interval
        )

    @property
    def jump_angles(self):
        return tuple(
            angle
            for interval in self.interval_jump_angles
            for angle in interval
        )

    @property
    def jump_probabilities(self):
        return tuple(
            probability
            for interval in self.interval_jump_probabilities
            for probability in interval
        )

    def prefix(self, time_count):
        interval_count = max(
            0,
            time_count - 1 if self.includes_initial_time else time_count,
        )
        substep_count = sum(
            len(interval)
            for interval in self.interval_substep_dts[:interval_count]
        )
        return RandomizedSchedule(
            interval_substep_dts=(
                self.interval_substep_dts[:interval_count]
            ),
            interval_jump_angles=(
                self.interval_jump_angles[:interval_count]
            ),
            interval_jump_probabilities=(
                self.interval_jump_probabilities[:interval_count]
            ),
            boundary_choices=tuple(
                choices[:substep_count]
                for choices in self.boundary_choices
            ),
            requested_trotter_delta_t=self.requested_trotter_delta_t,
            seed_trajectories=self.seed_trajectories,
            includes_initial_time=self.includes_initial_time,
        )


@dataclass(frozen=True, slots=True)
class ObservableUncertainties:
    shot: tuple[np.ndarray, np.ndarray, np.ndarray]
    trajectory: tuple[np.ndarray, np.ndarray, np.ndarray]
    total: tuple[np.ndarray, np.ndarray, np.ndarray]


def system_qubits_by_site(number_of_qubits):
    # Logical wire order is a_R, s_{N-1}, ..., s_0, a_L.  Consequently,
    # both boundary jumps are nearest-neighbour gates before transpilation.
    return tuple(
        number_of_qubits - site for site in range(number_of_qubits)
    )


def boundary_ancilla_qubits(number_of_qubits):
    """Return ancillas indexed by LEFT_BOUNDARY and RIGHT_BOUNDARY."""
    return (number_of_qubits + 1, 0)


def _single_interval_substeps(interval, trotter_delta_t):
    if trotter_delta_t is None:
        return (float(interval),)
    maximum_dt = float(trotter_delta_t)
    step_count = max(1, int(np.ceil(interval / maximum_dt - 1e-12)))
    final_dt = interval - (step_count - 1) * maximum_dt
    return (
        *(maximum_dt for _ in range(step_count - 1)),
        float(final_dt),
    )


def _validated_time_grid(times):
    time_grid = np.asarray(times, dtype=float)
    if time_grid.ndim != 1 or time_grid.size < 1:
        raise ValueError("times must contain at least one value")
    if not np.all(np.isfinite(time_grid)):
        raise ValueError("times must contain only finite values")
    if time_grid.size == 1:
        if time_grid[0] < 0.0:
            raise ValueError("a single time must be nonnegative")
        return time_grid
    if not np.isclose(time_grid[0], 0.0):
        raise ValueError("times must start at zero")
    if np.any(np.diff(time_grid) <= 0.0):
        raise ValueError("times must be strictly increasing")
    return time_grid


def build_randomized_schedule(
    times,
    trajectories,
    seed_trajectories,
    trotter_delta_t=None,
    *,
    includes_initial_time=True,
):
    """Create one common set of trajectory prefixes for the whole run."""
    time_grid = _validated_time_grid(times)
    if trajectories < 1:
        raise ValueError("trajectories must be positive")
    if trotter_delta_t is not None and (
        not np.isfinite(trotter_delta_t) or trotter_delta_t <= 0.0
    ):
        raise ValueError("trotter_delta_t must be finite and positive")

    interval_substep_dts = tuple(
        _single_interval_substeps(interval, trotter_delta_t)
        for interval in np.diff(time_grid)
    )
    interval_jump_angles = tuple(
        tuple(2.0 * np.sqrt(2.0 * gamma * dt) for dt in interval)
        for interval in interval_substep_dts
    )
    interval_jump_probabilities = tuple(
        tuple(np.sin(0.5 * angle) ** 2 for angle in interval)
        for interval in interval_jump_angles
    )
    substep_count = sum(map(len, interval_substep_dts))
    generator = np.random.default_rng(seed_trajectories)
    boundary_choices = tuple(
        tuple(map(int, row))
        for row in generator.integers(
            0,
            2,
            size=(trajectories, substep_count),
        )
    )
    return RandomizedSchedule(
        interval_substep_dts=interval_substep_dts,
        interval_jump_angles=interval_jump_angles,
        interval_jump_probabilities=interval_jump_probabilities,
        boundary_choices=boundary_choices,
        requested_trotter_delta_t=trotter_delta_t,
        seed_trajectories=seed_trajectories,
        includes_initial_time=includes_initial_time,
    )


def _system_factor(circuit, number_of_qubits, dt, system_qubits):
    def bond_layer(current, bonds, name):
        return fold_left(
            lambda value, left_site: pipe(
                value,
                xx_plus_yy(
                    2.0 * J * dt,
                    0.0,
                    system_qubits[left_site],
                    system_qubits[left_site + 1],
                    label=f"{name}[{left_site},{left_site + 1}]",
                ),
            ),
            current,
            bonds,
        )

    evolved = bond_layer(
        circuit,
        chain_tools.even_bonds(number_of_qubits),
        "H_even",
    )
    evolved = bond_layer(
        evolved,
        chain_tools.odd_bonds(number_of_qubits),
        "H_odd",
    )
    if h_field == 0.0:
        return evolved
    return fold_left(
        lambda value, qubit: pipe(value, rz(h_field * dt, qubit)),
        evolved,
        system_qubits,
    )


def _randomized_substep(
    circuit,
    number_of_qubits,
    dt,
    boundary,
    system_qubits,
    boundary_ancillas,
):
    if boundary not in (LEFT_BOUNDARY, RIGHT_BOUNDARY):
        raise ValueError("boundary must be LEFT_BOUNDARY or RIGHT_BOUNDARY")
    selected_site = 0 if boundary == LEFT_BOUNDARY else number_of_qubits - 1
    selected_ancilla = boundary_ancillas[boundary]
    jump_angle = 2.0 * np.sqrt(2.0 * gamma * dt)
    return pipe(
        _system_factor(circuit, number_of_qubits, dt, system_qubits),
        xx_plus_yy(
            jump_angle,
            0.0,
            system_qubits[selected_site],
            selected_ancilla,
            label=f"K_{BOUNDARY_NAMES[boundary]}",
        ),
        reset(selected_ancilla),
    )


def _trajectory_evolution_circuits(
    number_of_qubits,
    schedule,
    trajectory_index,
):
    system_qubits = system_qubits_by_site(number_of_qubits)
    boundary_ancillas = boundary_ancilla_qubits(number_of_qubits)
    choices = schedule.boundary_choices[trajectory_index]
    offsets = np.cumsum(
        (0, *map(len, schedule.interval_substep_dts)),
        dtype=int,
    )
    initial = pipe(
        empty(
            number_of_qubits + 2,
            0,
            name=(
                f"randomized_lie_N{number_of_qubits}_"
                f"trajectory_{trajectory_index}"
            ),
        ),
        *map_tuple(
            lambda site: x(system_qubits[site]),
            (number_of_qubits // 2,),
        ),
    )

    def interval(current, interval_index):
        start = int(offsets[interval_index])
        substeps = schedule.interval_substep_dts[interval_index]
        return fold_left(
            lambda value, local_index: _randomized_substep(
                value,
                number_of_qubits,
                substeps[local_index],
                choices[start + local_index],
                system_qubits,
                boundary_ancillas,
            ),
            current,
            range(len(substeps)),
        )

    return scan_left(
        interval,
        initial,
        range(len(schedule.interval_substep_dts)),
    )


def _axis_for_site(basis, site):
    if basis in ("Z", "X", "Y"):
        return basis
    if basis == "XY":
        return "X" if site % 2 == 0 else "Y"
    if basis == "YX":
        return "Y" if site % 2 == 0 else "X"
    raise ValueError(f"unsupported measurement basis: {basis}")


def _append_terminal_measurements(circuit, number_of_qubits, basis):
    system_qubits = system_qubits_by_site(number_of_qubits)
    measured = Circuit(
        qubit_count=circuit.qubit_count,
        bit_count=number_of_qubits,
        operations=circuit.operations,
        name=f"{circuit.name}_{basis}_measurement",
    )
    for site, qubit in enumerate(system_qubits):
        axis = _axis_for_site(basis, site)
        if axis == "X":
            measured = pipe(measured, h(qubit))
        elif axis == "Y":
            measured = pipe(measured, rz(-pi / 2.0, qubit), h(qubit))
    return fold_left(
        lambda value, site: pipe(
            value,
            measure(system_qubits[site], site),
        ),
        measured,
        range(number_of_qubits),
    )


def build_sample_circuits(
    times,
    number_of_qubits,
    trajectories=DEFAULT_TRAJECTORIES,
    seed_trajectories=DEFAULT_SEED_TRAJECTORIES,
    trotter_delta_t=None,
):
    """Build time-major, trajectory-major, basis-major sample circuits."""
    if number_of_qubits < 2:
        raise ValueError("number_of_qubits must be at least two")
    requested_times = _validated_time_grid(times)
    single_positive_target = (
        requested_times.size == 1 and requested_times[0] > 0.0
    )
    evolution_times = (
        np.asarray((0.0, float(requested_times[0])))
        if single_positive_target
        else requested_times
    )
    saved_circuit_indices = (
        (1,)
        if single_positive_target
        else tuple(range(requested_times.size))
    )
    schedule = build_randomized_schedule(
        evolution_times,
        trajectories,
        seed_trajectories,
        trotter_delta_t,
        includes_initial_time=not single_positive_target,
    )
    trajectory_circuits = tuple(
        _trajectory_evolution_circuits(
            number_of_qubits,
            schedule,
            trajectory_index,
        )
        for trajectory_index in range(trajectories)
    )
    entries = tuple(
        (
            time_index,
            trajectory_index,
            basis,
            _append_terminal_measurements(
                trajectory_circuits[trajectory_index][circuit_time_index],
                number_of_qubits,
                basis,
            ),
        )
        for time_index, circuit_time_index in enumerate(
            saved_circuit_indices
        )
        for trajectory_index in range(trajectories)
        for basis in MEASUREMENT_BASES
    )
    return (
        tuple(entry[3] for entry in entries),
        tuple(entry[:3] for entry in entries),
        schedule,
    )


def build_one_step_circuit(number_of_qubits, dt, boundary):
    """Return an isolated randomized step for circuit and metric figures."""
    if number_of_qubits < 2:
        raise ValueError("number_of_qubits must be at least two")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    return _randomized_substep(
        empty(
            number_of_qubits + 2,
            0,
            name=(
                f"one_randomized_lie_{BOUNDARY_NAMES[boundary]}_"
                f"N{number_of_qubits}"
            ),
        ),
        number_of_qubits,
        dt,
        boundary,
        system_qubits_by_site(number_of_qubits),
        boundary_ancilla_qubits(number_of_qubits),
    )


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
    return sum(
        count
        * (
            -1.0
            if sum(_classical_bit(bitstring, bit) for bit in bits) % 2
            else 1.0
        )
        for bitstring, count in counts.items()
    ) / shots


def _pauli_variance(expectation, shots):
    return max(0.0, 1.0 - expectation**2) / shots


def _group_counts(results, metadata, time_count, trajectories):
    grouped = tuple(
        tuple({} for _ in range(trajectories))
        for _ in range(time_count)
    )
    for result, (time_index, trajectory_index, basis) in zip(
        results,
        metadata,
        strict=True,
    ):
        grouped[time_index][trajectory_index][basis] = counts_dict(result)
    for time_index, trajectory_values in enumerate(grouped):
        for trajectory_index, values in enumerate(trajectory_values):
            missing = tuple(
                basis for basis in MEASUREMENT_BASES if basis not in values
            )
            if missing:
                raise ValueError(
                    f"time {time_index}, trajectory {trajectory_index} "
                    f"is missing bases {missing}"
                )
    return grouped


def _mean_and_uncertainty(values, shot_variances):
    """Separate shot error from corrected between-trajectory error."""
    trajectory_count = values.shape[-1]
    mean = np.mean(values, axis=-1)
    shot_variance_of_mean = np.sum(shot_variances, axis=-1) / (
        trajectory_count**2
    )
    if trajectory_count == 1:
        trajectory_variance_of_mean = np.zeros_like(mean)
    else:
        observed_variance = np.var(values, axis=-1, ddof=1)
        mean_shot_variance = np.mean(shot_variances, axis=-1)
        physical_variance = np.maximum(
            observed_variance - mean_shot_variance,
            0.0,
        )
        trajectory_variance_of_mean = physical_variance / trajectory_count
    return (
        mean,
        np.sqrt(shot_variance_of_mean),
        np.sqrt(trajectory_variance_of_mean),
    )


def observables_from_results(
    results,
    metadata,
    number_of_qubits,
    time_count,
    trajectories,
):
    """Average trajectories and report shot/trajectory uncertainties."""
    grouped = _group_counts(
        results,
        metadata,
        time_count,
        trajectories,
    )
    populations = np.zeros(
        (number_of_qubits, time_count, trajectories),
        dtype=float,
    )
    correlations = np.zeros(
        (number_of_qubits - 1, time_count, trajectories),
        dtype=float,
    )
    flows = np.zeros_like(correlations)
    population_shot_variances = np.zeros_like(populations)
    correlation_shot_variances = np.zeros_like(correlations)
    flow_shot_variances = np.zeros_like(flows)

    for time_index, trajectory_values in enumerate(grouped):
        for trajectory_index, basis_counts in enumerate(trajectory_values):
            z_shots = sum(basis_counts["Z"].values())
            for site in range(number_of_qubits):
                z_value = _expectation(basis_counts["Z"], (site,))
                populations[site, time_index, trajectory_index] = (
                    0.5 * (1.0 - z_value)
                )
                population_shot_variances[
                    site,
                    time_index,
                    trajectory_index,
                ] = 0.25 * _pauli_variance(z_value, z_shots)

            for bond in range(number_of_qubits - 1):
                bits = (bond, bond + 1)
                xx_counts = basis_counts["X"]
                yy_counts = basis_counts["Y"]
                xy_basis = "XY" if bond % 2 == 0 else "YX"
                yx_basis = "YX" if bond % 2 == 0 else "XY"
                xy_counts = basis_counts[xy_basis]
                yx_counts = basis_counts[yx_basis]
                xx_value = _expectation(xx_counts, bits)
                yy_value = _expectation(yy_counts, bits)
                xy_value = _expectation(xy_counts, bits)
                yx_value = _expectation(yx_counts, bits)
                correlations[bond, time_index, trajectory_index] = (
                    xx_value + yy_value
                )
                flows[bond, time_index, trajectory_index] = (
                    0.5 * J * (yx_value - xy_value)
                )
                correlation_shot_variances[
                    bond,
                    time_index,
                    trajectory_index,
                ] = (
                    _pauli_variance(xx_value, sum(xx_counts.values()))
                    + _pauli_variance(yy_value, sum(yy_counts.values()))
                )
                flow_shot_variances[
                    bond,
                    time_index,
                    trajectory_index,
                ] = 0.25 * J**2 * (
                    _pauli_variance(xy_value, sum(xy_counts.values()))
                    + _pauli_variance(yx_value, sum(yx_counts.values()))
                )

    family_values = (populations, correlations, flows)
    family_variances = (
        population_shot_variances,
        correlation_shot_variances,
        flow_shot_variances,
    )
    summaries = tuple(
        _mean_and_uncertainty(values, variances)
        for values, variances in zip(
            family_values,
            family_variances,
            strict=True,
        )
    )
    measured = tuple(summary[0] for summary in summaries)
    shot_errors = tuple(summary[1] for summary in summaries)
    trajectory_errors = tuple(summary[2] for summary in summaries)
    total_errors = tuple(
        np.sqrt(shot**2 + trajectory**2)
        for shot, trajectory in zip(
            shot_errors,
            trajectory_errors,
            strict=True,
        )
    )
    return (
        measured,
        ObservableUncertainties(
            shot=shot_errors,
            trajectory=trajectory_errors,
            total=total_errors,
        ),
        family_values,
        grouped,
    )


def calculate_exact_reference(times, number_of_qubits):
    """Return the exact Lindblad observables for manageable system sizes."""
    exact, _ = hardware_tools.calculate_references(
        np.asarray(times, dtype=float),
        number_of_qubits,
    )
    return exact


def _plot_observable_family(
    axis,
    times,
    values,
    uncertainties,
    title,
    ylabel,
    series_label,
    exact_values=None,
):
    for index, series in enumerate(values):
        line = axis.plot(times, series, label=series_label(index))[0]
        axis.fill_between(
            times,
            series - uncertainties[index],
            series + uncertainties[index],
            color=line.get_color(),
            alpha=0.14,
        )
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
    trajectories,
    measured,
    uncertainties,
    output_path,
    exact=None,
):
    populations, correlations, flows = measured
    total_population, total_correlation, total_flow = uncertainties.total
    exact_populations, exact_correlations, exact_flows = (
        (None, None, None) if exact is None else exact
    )
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
    _plot_observable_family(
        axes[0, 0],
        times,
        populations,
        total_population,
        "Local excitation populations",
        r"$\langle n_i\rangle$",
        lambda site: f"site {site}",
        exact_populations,
    )
    _plot_observable_family(
        axes[0, 1],
        times,
        correlations,
        total_correlation,
        "Nearest-neighbor XY correlations",
        r"$C_i^{XY}$",
        lambda bond: f"bond {bond}-{bond + 1}",
        exact_correlations,
    )
    _plot_observable_family(
        axes[1, 0],
        times,
        flows,
        total_flow,
        "Nearest-neighbor excitation flows",
        r"$I_{i\rightarrow i+1}$",
        lambda bond: f"bond {bond}-{bond + 1}",
        exact_flows,
    )

    diagnostic = axes[1, 1]
    diagnostic.plot(
        times,
        np.maximum.reduce(
            tuple(np.max(values, axis=0) for values in uncertainties.shot)
        ),
        label="max shot s.e.",
    )
    diagnostic.plot(
        times,
        np.maximum.reduce(
            tuple(
                np.max(values, axis=0)
                for values in uncertainties.trajectory
            )
        ),
        label="max trajectory s.e.",
    )
    if exact is not None:
        diagnostic.plot(
            times,
            chain_tools.aggregate_observable_error(measured, exact),
            label="observable error vs exact",
        )
    diagnostic.set_title("Error and uncertainty diagnostics")
    diagnostic.set_xlabel("Time")
    diagnostic.set_ylabel("Magnitude")
    diagnostic.grid(alpha=0.25)
    diagnostic.legend()
    figure.suptitle(
        f"Randomized single-boundary Lie simulation on {backend_name} "
        f"(R={trajectories})",
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


def compile_one_step_metrics(options, representative_dt):
    """Compile left and right one-step circuits without executing them."""
    circuits = tuple(
        build_one_step_circuit(options.n_qubits, representative_dt, boundary)
        for boundary in (LEFT_BOUNDARY, RIGHT_BOUNDARY)
    )
    target, environment = hardware_tools._runtime_target(
        options.backend,
        options.aer_method,
        options.account_file,
    )
    compiler = CompilerConfig(
        optimization_level=options.optimization_level,
        seed_transpiler=options.seed_transpiler,
    )
    match compile_circuit_batch_sync(
        circuits,
        target,
        compiler,
        environment,
    ):
        case Err(error):
            raise RuntimeError(hardware_tools._runtime_error_message(error))
        case Ok(metrics):
            return metrics


def _representative_full_result(results, metadata, time_index):
    for result, entry in zip(results, metadata, strict=True):
        if entry == (time_index, 0, "Z"):
            return result
    raise ValueError("could not locate the representative full circuit")


def _representative_full_circuit(circuits, metadata, time_index):
    for circuit, entry in zip(circuits, metadata, strict=True):
        if entry == (time_index, 0, "Z"):
            return circuit
    raise ValueError("could not locate the representative full circuit")


def transpilation_summary(one_step_metrics, full_result):
    full_pair = (
        (full_result.original_gate_count, full_result.original_depth),
        (full_result.compiled_gate_count, full_result.compiled_depth),
    )
    if one_step_metrics is None:
        labels = ("initial-state circuit",)
        pairs = (full_pair,)
    else:
        labels = (
            "one step, L",
            "one step, R",
            "full final, trajectory 0",
        )
        pairs = (
            _metrics_pair(one_step_metrics[0]),
            _metrics_pair(one_step_metrics[1]),
            full_pair,
        )
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
    figure, axes = plt.subplots(1, len(summary), figsize=(16, 5))
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
    figure.suptitle("Randomized single-boundary transpilation metrics")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_one_step_circuits(number_of_qubits, dt, left_path, right_path):
    for boundary, output_path in (
        (LEFT_BOUNDARY, left_path),
        (RIGHT_BOUNDARY, right_path),
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure = draw_circuit(
            build_one_step_circuit(number_of_qubits, dt, boundary)
        )
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)


def logical_qubit_roles(number_of_qubits):
    """Describe the boundary-bracketed logical wire order."""
    return (
        "right boundary ancilla a_R",
        *tuple(
            f"system site {number_of_qubits - 1 - qubit}"
            for qubit in range(number_of_qubits)
        ),
        "left boundary ancilla a_L",
    )


def _draw_circuit_image(axis, circuit, title):
    """Embed a normally sized Qiskit circuit rendering into an axis."""
    circuit_figure = draw_circuit(circuit, fold=-1)
    with BytesIO() as image_buffer:
        circuit_figure.savefig(
            image_buffer,
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
        image_buffer.seek(0)
        circuit_image = plt.imread(image_buffer, format="png")
    plt.close(circuit_figure)
    axis.imshow(circuit_image)
    axis.axis("off")
    axis.set_title(title, fontsize=13, pad=10)


def plot_transpiled_circuit_layout(
    circuit,
    options,
    output_path,
    *,
    one_step_circuits=None,
):
    """Plot placement, mapping, and both randomized one-step choices."""
    target, environment = hardware_tools._runtime_target(
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
        view="physical",
        logical_labels=logical_qubit_roles(options.n_qubits),
    ):
        case Err(error):
            raise RuntimeError(hardware_tools._runtime_error_message(error))
        case Ok(figure):
            pass

    width, height = figure.get_size_inches()
    figure.set_size_inches(
        max(float(width), 16.0),
        max(float(height) + 6.5, 13.5),
    )
    if figure.axes:
        figure.axes[0].set_position((0.025, 0.52, 0.63, 0.42))
    if len(figure.axes) > 1:
        figure.axes[1].set_position((0.69, 0.55, 0.285, 0.36))

    if one_step_circuits is None:
        circuit_axis = figure.add_axes((0.025, 0.035, 0.95, 0.40))
        circuit_axis.text(
            0.5,
            0.5,
            "No positive-time Trotter substep is present in this run.",
            ha="center",
            va="center",
            fontsize=12,
            transform=circuit_axis.transAxes,
        )
        circuit_axis.axis("off")
        circuit_axis.set_title(
            "Randomized one-step circuits before transpilation",
            fontsize=13,
            pad=10,
        )
    else:
        left_axis = figure.add_axes((0.025, 0.035, 0.46, 0.40))
        right_axis = figure.add_axes((0.515, 0.035, 0.46, 0.40))
        _draw_circuit_image(
            left_axis,
            one_step_circuits[LEFT_BOUNDARY],
            "One randomized left-boundary substep before transpilation",
        )
        _draw_circuit_image(
            right_axis,
            one_step_circuits[RIGHT_BOUNDARY],
            "One randomized right-boundary substep before transpilation",
        )

    title = (
        f"Final-time trajectory-0 Z-basis qubit layout on "
        f"{options.backend} (R={options.trajectories})"
    )
    if options.backend.lower() == "aer" and figure.axes:
        figure.axes[0].set_title(
            "Unconstrained connectivity; identity placement",
            fontsize=12,
            pad=10,
        )
    figure.suptitle(title, fontsize=16, y=0.985)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def output_paths(
    backend_name,
    number_of_qubits,
    trajectories,
    output_directory=None,
):
    stem = (
        f"randomized_lie_trotter_{_safe_name(backend_name)}_"
        f"N{number_of_qubits}_R{trajectories}"
    )
    directory = (
        Path(__file__).resolve().parent
        if output_directory is None
        else Path(output_directory)
    )
    return {
        "results_figure": directory / "figures" / f"{stem}.png",
        "metrics_figure": (
            directory / "figures" / f"{stem}_transpilation_metrics.png"
        ),
        "left_circuit_figure": (
            directory / "figures" / f"{stem}_one_step_left.png"
        ),
        "right_circuit_figure": (
            directory / "figures" / f"{stem}_one_step_right.png"
        ),
        "layout_figure": (
            directory / "figures" / f"{stem}_transpiled_layout.png"
        ),
        "result": directory / "results" / f"{stem}.json",
    }


def default_checkpoint_path(result_path):
    return result_path.with_name(f"{result_path.stem}_checkpoint.json")


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def shots_per_trajectory(total_shots, trajectories):
    if total_shots % trajectories != 0:
        raise ValueError(
            "--shots must be divisible by --trajectories so every "
            "randomized circuit receives the same number of shots"
        )
    per_trajectory = total_shots // trajectories
    if per_trajectory < 1:
        raise ValueError("--shots must be at least --trajectories")
    return per_trajectory


def _uniform_value(values):
    if not values:
        return None
    return float(values[0]) if np.allclose(values, values[0]) else None


def _schedule_payload(times, schedule):
    offsets = np.cumsum(
        (0, *map(len, schedule.interval_substep_dts)),
        dtype=int,
    )
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
        "requested_trotter_delta_t": schedule.requested_trotter_delta_t,
        "seed_trajectories": schedule.seed_trajectories,
        "includes_initial_time": schedule.includes_initial_time,
        "randomized_jump_rate_scale": 2.0,
        "uniform_substep_dt": _uniform_value(schedule.substep_dts),
        "uniform_system_exchange_angle": _uniform_value(
            tuple(2.0 * J * dt for dt in schedule.substep_dts)
        ),
        "uniform_randomized_jump_angle": _uniform_value(
            schedule.jump_angles
        ),
        "uniform_randomized_jump_exchange_probability": _uniform_value(
            schedule.jump_probabilities
        ),
        "boundary_choices": tuple(
            tuple(BOUNDARY_NAMES[value] for value in choices)
            for choices in schedule.boundary_choices
        ),
        "intervals": tuple(
            {
                "start_time": interval_start_times[index],
                "end_time": interval_end_times[index],
                "substeps": tuple(
                    {
                        "substep_index": int(offsets[index]) + local_index,
                        "dt": dt,
                        "system_exchange_angle": 2.0 * J * dt,
                        "randomized_jump_angle": (
                            schedule.interval_jump_angles[index][local_index]
                        ),
                        "randomized_jump_exchange_probability": (
                            schedule.interval_jump_probabilities[index][
                                local_index
                            ]
                        ),
                    }
                    for local_index, dt in enumerate(
                        schedule.interval_substep_dts[index]
                    )
                ),
            }
            for index in range(len(schedule.interval_substep_dts))
        ),
    }


def _checkpoint_configuration(options, times, metadata):
    return {
        "backend": options.backend.lower(),
        "number_of_system_qubits": options.n_qubits,
        "number_of_circuit_qubits": options.n_qubits + 2,
        "circuit_layout_version": CIRCUIT_LAYOUT_VERSION,
        "ancilla_strategy": ANCILLA_STRATEGY,
        "number_of_trajectories": options.trajectories,
        "total_shots_per_time_basis": options.shots,
        "shots_per_trajectory_circuit": shots_per_trajectory(
            options.shots,
            options.trajectories,
        ),
        "times": [float(value) for value in times],
        "measurement_bases": list(MEASUREMENT_BASES),
        "circuit_metadata": [
            {
                "time_index": time_index,
                "trajectory_index": trajectory_index,
                "basis": basis,
            }
            for time_index, trajectory_index, basis in metadata
        ],
        "optimization_level": options.optimization_level,
        "seed_transpiler": options.seed_transpiler,
        "seed_simulator": options.seed_simulator,
        "seed_trajectories": options.seed_trajectories,
        "aer_method": options.aer_method,
        "trotter_delta_t": options.trotter_delta_t,
    }


def _sample_result_record(index, result, metadata):
    time_index, trajectory_index, basis = metadata[index]
    return {
        "circuit_index": index,
        "time_index": time_index,
        "trajectory_index": trajectory_index,
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


def _sample_result_from_record(record):
    counts = record.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("checkpoint result counts must be a JSON object")
    return SampleResult(
        counts=tuple(
            sorted((str(key), int(value)) for key, value in counts.items())
        ),
        shots=int(record["shots"]),
        backend_name=str(record["backend_name"]),
        job_id=(
            None if record.get("job_id") is None else str(record["job_id"])
        ),
        original_gate_count=int(record["original_operation_count"]),
        original_depth=int(record["original_depth"]),
        compiled_gate_count=int(record["compiled_operation_count"]),
        compiled_depth=int(record["compiled_depth"]),
    )


def save_checkpoint(
    checkpoint_path,
    options,
    times,
    schedule,
    results,
    metadata,
    status="partial",
):
    if len(results) > len(metadata):
        raise ValueError("checkpoint contains more results than circuits")
    _atomic_write_json(
        checkpoint_path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "experiment": "Model1/Experiment5 randomized Lie-Trotter",
            "status": status,
            "completed_circuit_count": len(results),
            "total_circuit_count": len(metadata),
            "configuration": _checkpoint_configuration(
                options,
                times,
                metadata,
            ),
            "physics": _schedule_payload(times, schedule),
            "results": tuple(
                _sample_result_record(index, result, metadata)
                for index, result in enumerate(results)
            ),
        },
    )


def load_checkpoint(checkpoint_path, options, times, metadata):
    path = Path(checkpoint_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read checkpoint {path}: {error}") from error
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint has an unsupported schema version")
    if payload.get("configuration") != _checkpoint_configuration(
        options,
        times,
        metadata,
    ):
        raise ValueError(
            "checkpoint configuration does not match this run; use the "
            "same backend, N, R, shots, time grid, seeds, and compiler settings"
        )
    records = payload.get("results")
    if not isinstance(records, list) or len(records) > len(metadata):
        raise ValueError("checkpoint has an invalid result list")
    for index, record in enumerate(records):
        if record.get("circuit_index") != index or tuple(
            record.get(key)
            for key in ("time_index", "trajectory_index", "basis")
        ) != metadata[index]:
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
        "number_of_circuit_qubits": payload.get("number_of_circuit_qubits"),
        "circuit_layout_version": payload.get("circuit_layout_version"),
        "ancilla_strategy": payload.get("ancilla_strategy"),
        "number_of_trajectories": payload.get("number_of_trajectories"),
        "total_shots_per_time_basis": payload.get(
            "total_shots_per_time_basis"
        ),
        "measurement_bases": tuple(payload.get("measurement_bases", ())),
        "optimization_level": payload.get("optimization_level"),
        "seed_transpiler": payload.get("seed_transpiler"),
        "seed_simulator": payload.get("seed_simulator"),
        "seed_trajectories": payload.get("seed_trajectories"),
        "aer_method": payload.get("aer_method", "automatic"),
        "effective_trotter_delta_t": effective_delta_t,
        "has_exact_reference": "exact_reference_observables" in payload,
    }


def _prospective_compatibility_signature(options, schedule):
    return {
        "experiment": "Model1/Experiment5 randomized Lie-Trotter",
        "backend": options.backend.lower(),
        "number_of_system_qubits": options.n_qubits,
        "number_of_circuit_qubits": options.n_qubits + 2,
        "circuit_layout_version": CIRCUIT_LAYOUT_VERSION,
        "ancilla_strategy": ANCILLA_STRATEGY,
        "number_of_trajectories": options.trajectories,
        "total_shots_per_time_basis": options.shots,
        "measurement_bases": MEASUREMENT_BASES,
        "optimization_level": options.optimization_level,
        "seed_transpiler": options.seed_transpiler,
        "seed_simulator": options.seed_simulator,
        "seed_trajectories": options.seed_trajectories,
        "aer_method": options.aer_method,
        "effective_trotter_delta_t": (
            options.trotter_delta_t
            if options.trotter_delta_t is not None
            else _uniform_value(schedule.substep_dts)
        ),
        "has_exact_reference": options.classical_reference,
    }


def _load_result_payload(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read existing result {path}: {error}") from error


def validate_existing_result_compatibility(output_path, options, schedule):
    """Reject unsafe archive mixtures before any sampling is submitted."""
    path = Path(output_path)
    if not path.exists():
        return None
    payload = _load_result_payload(path)
    existing = _result_compatibility_signature(payload)
    requested = _prospective_compatibility_signature(options, schedule)
    mismatches = tuple(
        key
        for key in requested
        if existing.get(key) != requested[key]
    )
    if mismatches:
        raise ValueError(
            "existing Experiment 5 result is incompatible in: "
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
        for family in ("populations", "xy_correlations", "excitation_flows")
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
        source_group = (
            new_grouped if source == "new" else old_grouped
        )
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
        "circuit_layout_version": payload.get("circuit_layout_version"),
        "ancilla_strategy": payload.get("ancilla_strategy"),
        "evolution_schedule": payload.get("evolution_schedule"),
        "transpilation_summary": payload.get("transpilation_summary", ()),
    }


def merge_result_payloads(existing, new):
    """Merge time points, with the new run replacing matching times."""
    if _result_compatibility_signature(existing) != (
        _result_compatibility_signature(new)
    ):
        raise ValueError("cannot merge incompatible Experiment 5 results")
    merged_times, sources = _merged_time_sources(
        existing["times"],
        new["times"],
    )
    merged = {**existing, **new, "times": merged_times}
    merged["observables"] = _merge_family_payload(
        existing["observables"],
        new["observables"],
        sources,
    )
    merged["trajectory_observables"] = _merge_family_payload(
        existing["trajectory_observables"],
        new["trajectory_observables"],
        sources,
    )
    merged["standard_errors"] = {
        **new["standard_errors"],
        **{
            component: _merge_family_payload(
                existing["standard_errors"][component],
                new["standard_errors"][component],
                sources,
            )
            for component in ("shot", "trajectory", "total")
        },
    }
    if "exact_reference_observables" in new:
        merged["exact_reference_observables"] = _merge_family_payload(
            existing["exact_reference_observables"],
            new["exact_reference_observables"],
            sources,
        )
        merged["aggregate_observable_error_vs_exact"] = (
            _merge_array_on_time_axis(
                existing["aggregate_observable_error_vs_exact"],
                new["aggregate_observable_error_vs_exact"],
                sources,
                0,
            )
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
                *(
                    existing.get("job_ids", ())
                    if uses_existing
                    else ()
                ),
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
        for family in ("populations", "xy_correlations", "excitation_flows")
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
    uncertainties,
    trajectory_observables,
    grouped_counts,
    metrics_summary,
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
    payload = {
        "experiment": "Model1/Experiment5 randomized Lie-Trotter",
        "method": "uniform randomized single-boundary dilation",
        "backend": options.backend,
        "number_of_system_qubits": options.n_qubits,
        "number_of_circuit_qubits": options.n_qubits + 2,
        "circuit_layout_version": CIRCUIT_LAYOUT_VERSION,
        "ancilla_strategy": ANCILLA_STRATEGY,
        "system_qubits_by_site": system_qubits_by_site(options.n_qubits),
        "boundary_ancilla_qubits": {
            "L": boundary_ancilla_qubits(options.n_qubits)[LEFT_BOUNDARY],
            "R": boundary_ancilla_qubits(options.n_qubits)[RIGHT_BOUNDARY],
        },
        "number_of_trajectories": options.trajectories,
        "total_shots_per_time_basis": options.shots,
        "shots_per_trajectory_circuit": shots_per_trajectory(
            options.shots,
            options.trajectories,
        ),
        "measurement_bases": MEASUREMENT_BASES,
        "system_terminal_bits_by_site": tuple(range(options.n_qubits)),
        "uses_mid_circuit_measurement": False,
        "uses_classical_feed_forward": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "times": np.asarray(times, dtype=float).tolist(),
        "dt": schedule_payload["uniform_substep_dt"],
        "system_exchange_angle": schedule_payload[
            "uniform_system_exchange_angle"
        ],
        "randomized_jump_angle": schedule_payload[
            "uniform_randomized_jump_angle"
        ],
        "randomized_jump_exchange_probability": schedule_payload[
            "uniform_randomized_jump_exchange_probability"
        ],
        "evolution_schedule": schedule_payload,
        "optimization_level": options.optimization_level,
        "seed_transpiler": options.seed_transpiler,
        "seed_simulator": options.seed_simulator,
        "seed_trajectories": options.seed_trajectories,
        "aer_method": options.aer_method,
        "job_ids": job_ids,
        "transpilation_summary": metrics_summary,
        "circuit_metadata": tuple(
            {
                "time_index": time_index,
                "trajectory_index": trajectory_index,
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
            for index, (
                time_index,
                trajectory_index,
                basis,
            ) in enumerate(metadata)
        ),
        "observables": _array_families_payload(measured),
        "trajectory_observables": _array_families_payload(
            trajectory_observables
        ),
        "standard_errors": {
            "shot": _array_families_payload(uncertainties.shot),
            "trajectory": _array_families_payload(
                uncertainties.trajectory
            ),
            "total": _array_families_payload(uncertainties.total),
            "trajectory_component_note": (
                "between-trajectory sample variance with the mean analytic "
                "shot variance subtracted and clipped at zero"
            ),
        },
        "raw_counts": tuple(
            {
                "time_index": time_index,
                "time": float(times[time_index]),
                "trajectory_index": trajectory_index,
                "basis": basis,
                "counts": grouped_counts[time_index][trajectory_index][
                    basis
                ],
            }
            for time_index in range(len(times))
            for trajectory_index in range(options.trajectories)
            for basis in MEASUREMENT_BASES
        ),
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


def _execution_options(options):
    return SimpleNamespace(
        **{
            **vars(options),
            "shots": shots_per_trajectory(
                options.shots,
                options.trajectories,
            ),
        }
    )


def execute_sample_circuits(circuits, options, on_batch_complete=None):
    return hardware_tools.execute_sample_circuits(
        circuits,
        _execution_options(options),
        on_batch_complete=on_batch_complete,
    )


def _next_metadata_backup_path(path):
    candidate = path.with_name(f"{path.stem}.before_metadata{path.suffix}")
    index = 1
    while candidate.exists():
        candidate = path.with_name(
            f"{path.stem}.before_metadata_{index}{path.suffix}"
        )
        index += 1
    return candidate


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
            "number_of_circuit_qubits": configuration.get(
                "number_of_circuit_qubits"
            ),
            "circuit_layout_version": configuration.get(
                "circuit_layout_version"
            ),
            "ancilla_strategy": configuration.get("ancilla_strategy"),
            "trajectories": configuration.get("number_of_trajectories"),
            "seed_trajectories": configuration.get("seed_trajectories"),
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

    records = payload.get("circuit_metadata")
    schedule = payload.get("evolution_schedule")
    if isinstance(records, list) and isinstance(schedule, dict):
        return {
            "kind": "result",
            "backend": payload.get("backend"),
            "number_of_qubits": payload.get("number_of_system_qubits"),
            "number_of_circuit_qubits": payload.get(
                "number_of_circuit_qubits"
            ),
            "circuit_layout_version": payload.get(
                "circuit_layout_version"
            ),
            "ancilla_strategy": payload.get("ancilla_strategy"),
            "trajectories": payload.get("number_of_trajectories"),
            "seed_trajectories": payload.get("seed_trajectories"),
            "times": payload.get("times"),
            "optimization_level": payload.get(
                "optimization_level",
                DEFAULT_OPTIMIZATION_LEVEL,
            ),
            "seed_transpiler": payload.get(
                "seed_transpiler",
                DEFAULT_SEED_TRANSPILER,
            ),
            "trotter_delta_t": schedule.get(
                "requested_trotter_delta_t"
            ),
            "records": records,
        }
    raise ValueError(
        "metadata file is neither an Experiment 5 result nor checkpoint"
    )


def backfill_metadata_only(options):
    """Recompile saved circuits and update metrics without QPU execution."""
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
    expected_circuit_qubits = int(specification["number_of_qubits"]) + 2
    if (
        specification["number_of_circuit_qubits"]
        != expected_circuit_qubits
        or specification["circuit_layout_version"]
        != CIRCUIT_LAYOUT_VERSION
        or specification["ancilla_strategy"] != ANCILLA_STRATEGY
    ):
        raise ValueError(
            "metadata file uses the legacy one-ancilla Experiment 5 "
            "layout and cannot be backfilled with the current "
            "two-boundary-ancilla circuits"
        )
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
                    int(specification["number_of_qubits"]),
                    int(specification["trajectories"]),
                    int(specification["seed_trajectories"]),
                    run_delta_t,
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
                (circuit, trajectory_index, basis)
                for circuit, (
                    candidate_time_index,
                    trajectory_index,
                    basis,
                ) in zip(run_circuits, run_metadata, strict=True)
                if candidate_time_index == run_time_index
            )
            circuit_values.extend(circuit for circuit, _, _ in selected)
            metadata_values.extend(
                (time_index, trajectory_index, basis)
                for _, trajectory_index, basis in selected
            )
        circuits = tuple(circuit_values)
        metadata = tuple(metadata_values)
    else:
        circuits, metadata, _ = build_sample_circuits(
            specification["times"],
            int(specification["number_of_qubits"]),
            int(specification["trajectories"]),
            int(specification["seed_trajectories"]),
            specification["trotter_delta_t"],
        )
    records = specification["records"]
    if not records or len(records) > len(circuits):
        raise ValueError("metadata file has an invalid circuit record list")
    for index, record in enumerate(records):
        recorded = tuple(
            record.get(key)
            for key in ("time_index", "trajectory_index", "basis")
        )
        if recorded != metadata[index]:
            raise ValueError(
                "metadata records do not match the rebuilt circuit order"
            )

    target, environment = hardware_tools._runtime_target(
        specification["backend"],
        options.aer_method,
        options.account_file,
    )
    compiler = CompilerConfig(
        optimization_level=int(specification["optimization_level"]),
        seed_transpiler=specification["seed_transpiler"],
    )
    print(
        f"Metadata-only: transpiling {len(records)} circuits for "
        f"{specification['backend']}; no Sampler job will be submitted"
    )
    match compile_circuit_batch_sync(
        circuits[: len(records)],
        target,
        compiler,
        environment,
    ):
        case Err(error):
            raise RuntimeError(hardware_tools._runtime_error_message(error))
        case Ok(metrics):
            pass

    updated_records = []
    for record, metric in zip(records, metrics, strict=True):
        replacements = {
            "original_operation_count": metric.original_gate_count,
            "original_depth": metric.original_depth,
            "compiled_operation_count": metric.compiled_gate_count,
            "compiled_depth": metric.compiled_depth,
        }
        updated_records.append(
            {
                **record,
                **{
                    key: value
                    for key, value in replacements.items()
                    if options.overwrite_metadata
                    or not isinstance(record.get(key), int)
                    or record[key] < 0
                },
            }
        )
    changed_count = sum(
        old != new
        for old, new in zip(records, updated_records, strict=True)
    )
    if changed_count == 0:
        print("All circuit metrics are present; no file was changed.")
        return
    backup_path = _next_metadata_backup_path(metadata_path)
    _atomic_write_json(backup_path, payload)
    record_key = (
        "results"
        if specification["kind"] == "checkpoint"
        else "circuit_metadata"
    )
    _atomic_write_json(
        metadata_path,
        {
            **payload,
            record_key: updated_records,
            "metadata_backfill": {
                "backend": specification["backend"],
                "optimization_level": compiler.optimization_level,
                "seed_transpiler": compiler.seed_transpiler,
                "computed_at_utc": datetime.now(timezone.utc).isoformat(),
                "circuit_count": len(metrics),
                "uses_current_backend_target": (
                    str(specification["backend"]).lower() != "aer"
                ),
                "hardware_job_submitted": False,
            },
        },
    )
    print(f"Updated metrics for {changed_count} circuits: {metadata_path}")
    print(f"Backup: {backup_path}")
    print("Hardware jobs submitted: 0")


def main(options):
    if options.layout_only and options.metadata_only:
        raise ValueError("--layout-only cannot be combined with --metadata-only")
    if options.layout_only and options.resume:
        raise ValueError("--layout-only cannot be combined with --resume")
    if (
        not options.metadata_only
        and options.backend.lower() == "fake_fez"
        and not options.layout_only
    ):
        raise ValueError("--backend fake_fez requires --layout-only")
    if options.metadata_only:
        backfill_metadata_only(options)
        return
    if options.metadata_file is not None:
        raise ValueError("--metadata-file is only valid with --metadata-only")

    per_trajectory_shots = shots_per_trajectory(
        options.shots,
        options.trajectories,
    )
    times = hardware_tools.time_grid_from_options(options)
    circuits, metadata, schedule = build_sample_circuits(
        times,
        options.n_qubits,
        options.trajectories,
        options.seed_trajectories,
        options.trotter_delta_t,
    )
    print(
        f"Built {len(circuits)} circuits: {len(times)} saved times x "
        f"{options.trajectories} trajectories x "
        f"{len(MEASUREMENT_BASES)} measurement bases"
    )
    print(
        f"Shot allocation: {options.shots} total per time/basis = "
        f"{per_trajectory_shots} per randomized circuit"
    )
    paths = output_paths(
        options.backend,
        options.n_qubits,
        options.trajectories,
        options.output_directory,
    )
    representative_dt = (
        schedule.substep_dts[0] if schedule.substep_dts else None
    )
    one_step_circuits = (
        tuple(
            build_one_step_circuit(
                options.n_qubits,
                representative_dt,
                boundary,
            )
            for boundary in (LEFT_BOUNDARY, RIGHT_BOUNDARY)
        )
        if representative_dt is not None
        else None
    )
    if options.layout_only:
        full_circuit = _representative_full_circuit(
            circuits,
            metadata,
            len(times) - 1,
        )
        plot_transpiled_circuit_layout(
            full_circuit,
            options,
            paths["layout_figure"],
            one_step_circuits=one_step_circuits,
        )
        print(f"Saved transpiled layout: {paths['layout_figure']}")
        print("Sampler jobs submitted: 0 (layout-only mode)")
        return
    existing_payload = validate_existing_result_compatibility(
        paths["result"],
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
        default_checkpoint_path(paths["result"])
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
            f"Checkpointed {len(all_completed)}/{len(circuits)} circuits: "
            f"{checkpoint_path}"
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
    (
        measured,
        uncertainties,
        trajectory_observables,
        grouped_counts,
    ) = observables_from_results(
        results,
        metadata,
        options.n_qubits,
        len(times),
        options.trajectories,
    )
    exact = None
    if options.classical_reference:
        print("Calculating exact Lindblad reference")
        exact = calculate_exact_reference(times, options.n_qubits)

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
    one_step_metrics = (
        compile_one_step_metrics(options, representative_dt)
        if representative_dt is not None
        else None
    )
    metrics_summary = transpilation_summary(
        one_step_metrics,
        full_result,
    )
    plot_transpilation_metrics(
        metrics_summary,
        paths["metrics_figure"],
    )
    plot_transpiled_circuit_layout(
        full_circuit,
        options,
        paths["layout_figure"],
        one_step_circuits=one_step_circuits,
    )
    if representative_dt is not None:
        plot_one_step_circuits(
            options.n_qubits,
            representative_dt,
            paths["left_circuit_figure"],
            paths["right_circuit_figure"],
        )
    combined_payload = save_results(
        paths["result"],
        options,
        times,
        schedule,
        results,
        metadata,
        measured,
        uncertainties,
        trajectory_observables,
        grouped_counts,
        metrics_summary,
        extra_payload={
            "transpiled_layout": {
                "backend": options.backend,
                "basis": "Z",
                "trajectory_index": 0,
                "saved_time": float(times[-1]),
                "view": "physical",
                "figure": paths["layout_figure"].name,
                "includes_left_and_right_one_step_circuits": (
                    one_step_circuits is not None
                ),
            },
            **(
                {}
                if exact is None
                else {
                    "exact_reference_observables": (
                        _array_families_payload(exact)
                    ),
                    "aggregate_observable_error_vs_exact": (
                        chain_tools.aggregate_observable_error(
                            measured,
                            exact,
                        ).tolist()
                    ),
                }
            ),
        },
    )
    combined_uncertainties = ObservableUncertainties(
        shot=_family_arrays(
            combined_payload["standard_errors"]["shot"]
        ),
        trajectory=_family_arrays(
            combined_payload["standard_errors"]["trajectory"]
        ),
        total=_family_arrays(
            combined_payload["standard_errors"]["total"]
        ),
    )
    plot_results(
        np.asarray(combined_payload["times"], dtype=float),
        options.backend,
        options.trajectories,
        _payload_observables(combined_payload, "observables"),
        combined_uncertainties,
        paths["results_figure"],
        (
            _payload_observables(
                combined_payload,
                "exact_reference_observables",
            )
            if "exact_reference_observables" in combined_payload
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
    print(f"System/circuit qubits: {options.n_qubits}/{options.n_qubits + 2}")
    print(f"Randomized trajectories R: {options.trajectories}")
    print(f"Trajectory seed: {options.seed_trajectories}")
    print(
        f"Archive saved times: {combined_payload['times']} "
        "(new run replaces matching times)"
    )
    print(f"Total Trotter substeps: {len(schedule.substep_dts)}")
    if schedule.jump_angles:
        print(
            "Randomized jump-angle range: "
            f"{min(schedule.jump_angles):.8f}--"
            f"{max(schedule.jump_angles):.8f}"
        )
    else:
        print("Randomized jump angles: none (initial-state-only run)")
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
    generated_path_names = [
        "results_figure",
        "metrics_figure",
        "layout_figure",
        "result",
    ]
    if representative_dt is not None:
        generated_path_names.extend(
            ("left_circuit_figure", "right_circuit_figure")
        )
    for label in generated_path_names:
        print(f"Saved {label.replace('_', ' ')}: {paths[label]}")
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


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the randomized single-boundary Lie dilation on Aer or "
            "an IBM backend. Non-aer backends submit real QPU work."
        )
    )
    parser.add_argument(
        "--n-qubits",
        type=at_least_two,
        default=DEFAULT_NUMBER_OF_QUBITS,
        help=f"number of system qubits (default: {DEFAULT_NUMBER_OF_QUBITS})",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help=(
            "use 'aer' for local simulation, 'fake_fez' for offline "
            "--layout-only transpilation, or an IBM backend name "
            "(default: aer)"
        ),
    )
    parser.add_argument(
        "--account-file",
        type=Path,
        default=DEFAULT_ACCOUNT_FILE,
        help=f"IBM apikey/crn JSON (default: {DEFAULT_ACCOUNT_FILE})",
    )
    parser.add_argument(
        "--trajectories",
        "-R",
        type=positive_integer,
        default=DEFAULT_TRAJECTORIES,
        help=(
            "independent randomized boundary-choice paths "
            f"(default: {DEFAULT_TRAJECTORIES})"
        ),
    )
    parser.add_argument(
        "--shots",
        type=positive_integer,
        default=DEFAULT_SHOTS,
        help=(
            "total shots per saved-time/basis, divided evenly over R "
            f"trajectories (default: {DEFAULT_SHOTS})"
        ),
    )
    parser.add_argument(
        "--seed-trajectories",
        type=int,
        default=DEFAULT_SEED_TRAJECTORIES,
        help="seed for the reproducible L/R paths",
    )
    parser.add_argument(
        "--t-final",
        type=positive_float,
        default=T_FINAL,
    )
    parser.add_argument(
        "--time-points",
        type=at_least_two,
        default=NUMBER_OF_TIME_POINTS,
    )
    parser.add_argument(
        "--times",
        nargs="+",
        default=None,
        metavar="T",
        help=(
            "explicit saved times; one nonnegative value T saves only T, "
            "while a list must start at zero; comma- or "
            "space-separated and overrides --t-final/--time-points"
        ),
    )
    parser.add_argument(
        "--trotter-delta-t",
        type=positive_float,
        default=None,
        help=(
            "maximum internal substep; a shorter final remainder lands "
            "exactly on each saved time"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=positive_integer,
        default=None,
        help=(
            "maximum circuits per provider job "
            f"(default: {DEFAULT_AER_BATCH_SIZE} Aer, "
            f"{DEFAULT_HARDWARE_BATCH_SIZE} IBM)"
        ),
    )
    parser.add_argument(
        "--optimization-level",
        "--optimization_level",
        dest="optimization_level",
        type=int,
        choices=(0, 1, 2, 3),
        default=DEFAULT_OPTIMIZATION_LEVEL,
        help=(
            "Qiskit transpiler optimization level 0, 1, 2, or 3 "
            f"(default: {DEFAULT_OPTIMIZATION_LEVEL})"
        ),
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
    )
    parser.add_argument(
        "--aer-method",
        choices=AER_METHODS,
        default="automatic",
    )
    parser.add_argument(
        "--classical-reference",
        action="store_true",
        help="calculate the exact Lindblad reference for manageable N",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a matching checkpoint's missing circuit suffix",
    )
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help=(
            "transpile and draw only the final saved-time trajectory-0 "
            "Z-basis backend layout with both one-step choices; never "
            "submit a Sampler job or alter results"
        ),
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "rebuild/transpile saved circuits to fill metrics without a "
            "Sampler submission"
        ),
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--overwrite-metadata",
        action="store_true",
    )
    return parser.parse_args(arguments)


if __name__ == "__main__":
    main(parse_arguments())
