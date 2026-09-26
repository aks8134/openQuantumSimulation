"""Experiment 7: explicitly synthesized randomized Lie--Trotter circuits.

This experiment uses the same randomized single-boundary qDRIFT strategy as
Experiment 5, but expands every ``XXPlusYY(theta, 0)`` operation into an
explicit RZ/RX/RZZ circuit *before* compilation.  The compiler therefore
receives no logical XXPlusYY gates to decompose.

The default working point follows Experiment 6: R=4, maximum delta_t=0.1,
and saved times from zero through two.  All remain command-line configurable.
"""

from __future__ import annotations

from collections import Counter
from io import BytesIO
import json
from math import pi
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from IBMRuntime import (
    RX,
    RZ,
    RZZ,
    XXPlusYY,
    Circuit,
    draw_circuit,
)
from Model1.Experiment5 import randomized_lie_trotter as experiment5
from library import classical


EXPERIMENT_NAME = "Model1/Experiment7 explicit randomized Lie-Trotter"
METHOD_NAME = "uniform randomized single-boundary dilation with explicit RZ/RX/RZZ synthesis"
CIRCUIT_LAYOUT_VERSION = 1
EXCHANGE_SYNTHESIS_VERSION = 1
DEFAULT_TRAJECTORIES = 4
DEFAULT_TROTTER_DELTA_T = 0.1
DEFAULT_T_FINAL = 2.0
DEFAULT_TIME_POINTS = 11
DEFAULT_AER_EXPLICIT_BATCH_SIZE = 20
OPTIMIZED_REFERENCE_METHOD = "vacuum-plus-single-excitation exact Lindblad"

# Re-export the shared public constants and validators for recovery scripts and
# interactive use without duplicating Experiment 5's execution machinery.
DEFAULT_ACCOUNT_FILE = experiment5.DEFAULT_ACCOUNT_FILE
DEFAULT_SEED_TRAJECTORIES = experiment5.DEFAULT_SEED_TRAJECTORIES
DEFAULT_NUMBER_OF_QUBITS = experiment5.DEFAULT_NUMBER_OF_QUBITS
MEASUREMENT_BASES = experiment5.MEASUREMENT_BASES
LEFT_BOUNDARY = experiment5.LEFT_BOUNDARY
RIGHT_BOUNDARY = experiment5.RIGHT_BOUNDARY
BOUNDARY_NAMES = experiment5.BOUNDARY_NAMES
T_FINAL = DEFAULT_T_FINAL
NUMBER_OF_TIME_POINTS = DEFAULT_TIME_POINTS
positive_integer = experiment5.positive_integer
at_least_two = experiment5.at_least_two
positive_float = experiment5.positive_float
hardware_tools = experiment5.hardware_tools
chain_tools = experiment5.chain_tools
J = experiment5.J
h_field = experiment5.h_field
gamma = experiment5.gamma


_BASE_BUILD_SAMPLE_CIRCUITS = experiment5.build_sample_circuits
_BASE_BUILD_ONE_STEP_CIRCUIT = experiment5.build_one_step_circuit
_BASE_CHECKPOINT_CONFIGURATION = experiment5._checkpoint_configuration
_BASE_PROSPECTIVE_SIGNATURE = experiment5._prospective_compatibility_signature
_BASE_SAVE_RESULTS = experiment5.save_results
_BASE_CALCULATE_EXACT_REFERENCE = experiment5.calculate_exact_reference


def build_optimized_reference_model(number_of_qubits):
    """Build the exact vacuum-plus-single-excitation Lindblad model.

    The reduced basis is ``|vac>, |0>, ..., |N-1>``.  It is invariant for
    this experiment because the XY Hamiltonian conserves excitation number,
    while the two boundary jumps only remove an excitation.
    """

    if number_of_qubits < 2:
        raise ValueError("Use at least two system qubits.")

    dimension = number_of_qubits + 1
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)

    # The vacuum energy is removed as an irrelevant global energy shift.
    # A one-excitation state differs from the all-zero vacuum by -h_field.
    hamiltonian[1:, 1:] += -h_field * np.eye(
        number_of_qubits,
        dtype=complex,
    )
    for site in range(number_of_qubits - 1):
        left = site + 1
        right = site + 2
        hamiltonian[left, right] = J
        hamiltonian[right, left] = J

    left_jump = csr_matrix(
        (
            (np.sqrt(gamma),),
            ((0,), (1,)),
        ),
        shape=(dimension, dimension),
        dtype=complex,
    )
    right_jump = csr_matrix(
        (
            (np.sqrt(gamma),),
            ((0,), (number_of_qubits,)),
        ),
        shape=(dimension, dimension),
        dtype=complex,
    )

    initial_density_matrix = np.zeros(
        (dimension, dimension),
        dtype=complex,
    )
    initial_site = number_of_qubits // 2
    initial_density_matrix[initial_site + 1, initial_site + 1] = 1.0
    return (
        csr_matrix(hamiltonian),
        (left_jump, right_jump),
        initial_density_matrix,
    )


def _evolve_optimized_reference(initial_density_matrix, liouvillian, times):
    """Evolve from t=0 at arbitrary nonnegative requested times."""

    time_grid = np.asarray(times, dtype=float)
    if time_grid.ndim != 1 or time_grid.size == 0:
        raise ValueError("times must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(time_grid)) or np.any(time_grid < 0.0):
        raise ValueError("times must contain finite nonnegative values")
    if np.any(np.diff(time_grid) <= 0.0):
        raise ValueError("times must be strictly increasing")

    dimension = initial_density_matrix.shape[0]
    initial_vector = initial_density_matrix.reshape(-1, order="F")
    generator_trace = liouvillian.diagonal().sum()
    evolved = []
    density_vector = initial_vector.copy()
    previous_time = 0.0
    for time in time_grid:
        interval = float(time) - previous_time
        if interval > 0.0:
            density_vector = expm_multiply(
                interval * liouvillian,
                density_vector,
                traceA=interval * generator_trace,
            )
        evolved.append(
            density_vector.reshape(dimension, dimension, order="F")
        )
        previous_time = float(time)
    return np.asarray(evolved)


def calculate_optimized_classical_reference(times, number_of_qubits):
    """Return exact observables in the invariant N+1 dimensional space."""

    hamiltonian, jump_operators, initial_density_matrix = (
        build_optimized_reference_model(number_of_qubits)
    )
    density_matrices = _evolve_optimized_reference(
        initial_density_matrix,
        classical.build_liouvillian(hamiltonian, jump_operators),
        times,
    )

    time_count = density_matrices.shape[0]
    populations = np.empty((number_of_qubits, time_count), dtype=float)
    correlations = np.empty(
        (number_of_qubits - 1, time_count),
        dtype=float,
    )
    flows = np.empty((number_of_qubits - 1, time_count), dtype=float)

    for site in range(number_of_qubits):
        reduced_index = site + 1
        populations[site] = np.real(
            density_matrices[:, reduced_index, reduced_index]
        )
    for bond in range(number_of_qubits - 1):
        left = bond + 1
        right = bond + 2
        coherence = density_matrices[:, left, right]
        correlations[bond] = 4.0 * np.real(coherence)
        flows[bond] = 2.0 * J * np.imag(coherence)

    return populations, correlations, flows


def explicit_xx_plus_yy_operations(angle, first, second):
    """Return the fixed 14-operation synthesis of XXPlusYY(angle, 0).

    Operations are emitted in the requested eight layers.  The middle RZ
    layer commutes with RZZ and is deliberately placed after the second RZZ
    so the immutable circuit exactly reflects the physical-layer discussion.
    """

    theta = float(angle)
    return (
        # Layer 1: virtual RZ x RZ.
        RZ(pi / 2.0, first),
        RZ(-pi / 2.0, second),
        # Layer 2: physical RX x RX.
        RX(pi / 2.0, first),
        RX(-pi / 2.0, second),
        # Layer 3: virtual RZ x RZ.
        RZ(pi / 2.0, first),
        RZ(-pi / 2.0, second),
        # Layer 4: physical two-qubit interaction.
        RZZ(theta / 2.0, first, second),
        # Layer 5: physical RX x RX.
        RX(-pi / 2.0, first),
        RX(-pi / 2.0, second),
        # Layer 6: physical two-qubit interaction.
        RZZ(theta / 2.0, first, second),
        # Layer 7: virtual RZ x RZ.
        RZ(-pi / 2.0, first),
        RZ(-pi / 2.0, second),
        # Layer 8: physical RX x RX.
        RX(-pi / 2.0, first),
        RX(-pi / 2.0, second),
    )


def decompose_xx_plus_yy_circuit(circuit):
    """Expand every beta=0 XXPlusYY operation before transpilation."""

    operations = []
    for operation in circuit.operations:
        if not isinstance(operation, XXPlusYY):
            operations.append(operation)
            continue
        if not np.isclose(operation.phase, 0.0):
            raise ValueError(
                "the explicit Experiment 7 synthesis supports only "
                "XXPlusYY(theta, beta=0)"
            )
        operations.extend(
            explicit_xx_plus_yy_operations(
                operation.angle,
                operation.first,
                operation.second,
            )
        )
    return Circuit(
        qubit_count=circuit.qubit_count,
        bit_count=circuit.bit_count,
        operations=tuple(operations),
        name=f"{circuit.name}_explicit_rz_rx_rzz",
    )


def build_sample_circuits(
    times,
    number_of_qubits,
    trajectories=DEFAULT_TRAJECTORIES,
    seed_trajectories=DEFAULT_SEED_TRAJECTORIES,
    trotter_delta_t=DEFAULT_TROTTER_DELTA_T,
):
    """Build compact logical circuits; lowering occurs per compiler batch."""

    return _BASE_BUILD_SAMPLE_CIRCUITS(
        times,
        number_of_qubits,
        trajectories,
        seed_trajectories,
        trotter_delta_t,
    )


def build_one_step_circuit(number_of_qubits, dt, boundary):
    """Return the compact logical circuit used only for construction/plots."""

    return _BASE_BUILD_ONE_STEP_CIRCUIT(number_of_qubits, dt, boundary)


def prepare_circuit_for_compilation(circuit):
    """Apply the explicit synthesis immediately before compilation."""

    return decompose_xx_plus_yy_circuit(circuit)


def single_exchange_resource_summary():
    """Return virtual/physical accounting for one synthesized exchange."""

    operations = explicit_xx_plus_yy_operations(0.4, 0, 1)
    counts = Counter(type(operation).__name__.lower() for operation in operations)
    return {
        "qiskit_operation_count": len(operations),
        "gate_counts": {
            "rz": counts["rz"],
            "rx": counts["rx"],
            "rzz": counts["rzz"],
        },
        "virtual_gate_count": counts["rz"],
        "physical_gate_count": counts["rx"] + counts["rzz"],
        "standard_circuit_depth": 8,
        "physical_pulse_depth": 5,
        "two_qubit_depth": 2,
        "virtual_gate_types": ["rz"],
        "physical_gate_types": ["rx", "rzz"],
    }


def decomposition_payload(number_of_qubits):
    return {
        "version": EXCHANGE_SYNTHESIS_VERSION,
        "logical_gate": "XXPlusYY(theta, beta=0)",
        "performed_before_transpilation": True,
        "logical_circuit_retained_until_compiler_boundary": True,
        "execution_lowering_granularity": "one execution batch",
        "diagram_representation": "logical XXPlusYY blocks",
        "pre_transpilation_metric_representation": "explicit RZ/RX/RZZ",
        "contains_logical_xx_plus_yy_after_synthesis": False,
        "rzz_angles": ["theta/2", "theta/2"],
        "layer_order": [
            "RZ tensor RZ",
            "RX tensor RX",
            "RZ tensor RZ",
            "RZZ",
            "RX tensor RX",
            "RZZ",
            "RZ tensor RZ",
            "RX tensor RX",
        ],
        "single_exchange_resources": single_exchange_resource_summary(),
        "exchange_blocks_per_randomized_substep": number_of_qubits,
        "note": (
            "RZ operations are virtual frame changes; RX and RZZ require "
            "physical pulses. Virtual RZ layers remain semantically required."
        ),
    }


def output_paths(
    backend_name,
    number_of_qubits,
    trajectories,
    output_directory=None,
):
    stem = (
        f"explicit_randomized_lie_trotter_"
        f"{experiment5._safe_name(backend_name)}_"
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
        "decomposition_figure": (
            directory / "figures" / f"{stem}_exchange_decomposition.png"
        ),
        "result": directory / "results" / f"{stem}.json",
    }


def plot_exchange_decomposition(output_path, angle=0.4):
    circuit = Circuit(
        qubit_count=2,
        bit_count=0,
        operations=explicit_xx_plus_yy_operations(angle, 0, 1),
        name="explicit_xx_plus_yy",
    )
    circuit_figure = draw_circuit(circuit, fold=-1)
    with BytesIO() as buffer:
        circuit_figure.savefig(
            buffer,
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
        buffer.seek(0)
        circuit_image = plt.imread(buffer, format="png")
    plt.close(circuit_figure)

    summary = single_exchange_resource_summary()
    figure = plt.figure(figsize=(18, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.25, 1.0))
    circuit_axis = figure.add_subplot(grid[0, :])
    circuit_axis.imshow(circuit_image)
    circuit_axis.axis("off")
    circuit_axis.set_title(
        rf"Explicit $XX+YY$ synthesis before transpilation ($\theta={angle:g}$)",
        fontsize=14,
    )

    table_axis = figure.add_subplot(grid[1, 0])
    table_axis.axis("off")
    table = table_axis.table(
        cellText=(
            ("RZ", 6, "virtual frame changes"),
            ("RX", 6, "physical one-qubit pulses"),
            ("RZZ", 2, "physical two-qubit pulses"),
        ),
        colLabels=("Gate", "Count", "Implementation"),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.7)
    table_axis.set_title("Single exchange-gate accounting", fontsize=13)

    metric_axis = figure.add_subplot(grid[1, 1])
    labels = (
        "Total\noperations",
        "Virtual\ngates",
        "Physical\ngates",
        "Standard\ndepth",
        "Physical\npulse depth",
        "Two-qubit\ndepth",
    )
    values = (
        summary["qiskit_operation_count"],
        summary["virtual_gate_count"],
        summary["physical_gate_count"],
        summary["standard_circuit_depth"],
        summary["physical_pulse_depth"],
        summary["two_qubit_depth"],
    )
    bars = metric_axis.bar(labels, values)
    metric_axis.bar_label(bars, fmt="%d", padding=3)
    metric_axis.set_ylim(0, 16)
    metric_axis.set_ylabel("Count / layer count")
    metric_axis.grid(axis="y", alpha=0.25)
    metric_axis.set_title("Virtual-aware resource accounting", fontsize=13)

    figure.suptitle(
        "Experiment 7 explicit RZ/RX/RZZ exchange decomposition",
        fontsize=16,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _checkpoint_configuration(options, times, metadata):
    configuration = _BASE_CHECKPOINT_CONFIGURATION(options, times, metadata)
    return {
        **configuration,
        "experiment": EXPERIMENT_NAME,
        "circuit_layout_version": CIRCUIT_LAYOUT_VERSION,
        "exchange_synthesis_version": EXCHANGE_SYNTHESIS_VERSION,
    }


def _prospective_compatibility_signature(options, schedule):
    signature = _BASE_PROSPECTIVE_SIGNATURE(options, schedule)
    return {
        **signature,
        "experiment": EXPERIMENT_NAME,
        "circuit_layout_version": CIRCUIT_LAYOUT_VERSION,
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
    if len(results) > len(metadata):
        raise ValueError("checkpoint contains more results than circuits")
    experiment5._atomic_write_json(
        checkpoint_path,
        {
            "schema_version": experiment5.CHECKPOINT_SCHEMA_VERSION,
            "experiment": EXPERIMENT_NAME,
            "status": status,
            "completed_circuit_count": len(results),
            "total_circuit_count": len(metadata),
            "configuration": _checkpoint_configuration(
                options,
                times,
                metadata,
            ),
            "physics": experiment5._schedule_payload(times, schedule),
            "exchange_decomposition": decomposition_payload(
                options.n_qubits
            ),
            "results": tuple(
                experiment5._sample_result_record(index, result, metadata)
                for index, result in enumerate(results)
            ),
        },
    )


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
    optimized_reference_payload = (
        {
            "classical_reference_metadata": {
                "method": OPTIMIZED_REFERENCE_METHOD,
                "exact_for_this_model": True,
                "reduced_basis": [
                    "vacuum",
                    *(
                        f"one excitation at site {site}"
                        for site in range(options.n_qubits)
                    ),
                ],
                "hilbert_space_dimension": options.n_qubits + 1,
                "liouville_space_dimension": (options.n_qubits + 1) ** 2,
                "full_hilbert_space_dimension_avoided": 2**options.n_qubits,
                "validity": (
                    "one-excitation initial state, excitation-conserving "
                    "XY Hamiltonian, and loss-only boundary jumps"
                ),
            }
        }
        if getattr(options, "optimized_classical_reference", False)
        else {}
    )
    return _BASE_SAVE_RESULTS(
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
        extra_payload={
            **({} if extra_payload is None else extra_payload),
            "experiment": EXPERIMENT_NAME,
            "method": METHOD_NAME,
            "circuit_layout_version": CIRCUIT_LAYOUT_VERSION,
            "exchange_decomposition": decomposition_payload(
                options.n_qubits
            ),
            **optimized_reference_payload,
        },
    )


def install_experiment7_implementation():
    """Install Experiment 7 hooks into the shared Experiment 5 workflow."""

    experiment5.CIRCUIT_LAYOUT_VERSION = CIRCUIT_LAYOUT_VERSION
    experiment5.build_sample_circuits = build_sample_circuits
    experiment5.build_one_step_circuit = build_one_step_circuit
    experiment5.prepare_circuit_for_compilation = (
        prepare_circuit_for_compilation
    )
    experiment5.output_paths = output_paths
    experiment5._checkpoint_configuration = _checkpoint_configuration
    experiment5._prospective_compatibility_signature = (
        _prospective_compatibility_signature
    )
    experiment5.save_checkpoint = save_checkpoint
    experiment5.save_results = save_results


def main(options):
    optimized_reference = getattr(
        options,
        "optimized_classical_reference",
        False,
    )
    if optimized_reference and options.classical_reference:
        raise ValueError(
            "choose either --classical-reference or "
            "--optimized-classical-reference, not both"
        )

    install_experiment7_implementation()
    if (
        options.batch_size is None
        and options.backend.lower() == "aer"
        and not options.layout_only
        and not options.metadata_only
    ):
        options.batch_size = DEFAULT_AER_EXPLICIT_BATCH_SIZE
        print(
            "Experiment 7 Aer lowering/execution batch size: "
            f"{options.batch_size}"
        )
    requested_full_reference = options.classical_reference
    experiment5.calculate_exact_reference = (
        calculate_optimized_classical_reference
        if optimized_reference
        else _BASE_CALCULATE_EXACT_REFERENCE
    )
    if optimized_reference:
        # The shared Experiment 5 workflow already handles reference arrays,
        # plots, and archive merging.  Enable that path internally while
        # replacing only its calculator for this Experiment 7 invocation.
        options.classical_reference = True
        print(
            "Using exact vacuum-plus-single-excitation classical reference "
            f"(Hilbert dimension {options.n_qubits + 1}, Liouville "
            f"dimension {(options.n_qubits + 1) ** 2})"
        )
    try:
        experiment5.main(options)
    finally:
        experiment5.calculate_exact_reference = (
            _BASE_CALCULATE_EXACT_REFERENCE
        )
        options.classical_reference = requested_full_reference
    if getattr(options, "no_circuit_plots", False):
        print(
            "Skipped exchange-decomposition circuit figure "
            "(--no-circuit-plots)"
        )
        return
    paths = output_paths(
        options.backend,
        options.n_qubits,
        options.trajectories,
        options.output_directory,
    )
    plot_exchange_decomposition(paths["decomposition_figure"])
    print(f"Saved exchange decomposition: {paths['decomposition_figure']}")


def parse_arguments(arguments=None):
    def configure_parser(parser):
        parser.add_argument(
            "--optimized-classical-reference",
            action="store_true",
            help=(
                "calculate the exact Lindblad reference in the invariant "
                "vacuum-plus-single-excitation subspace; Experiment 7 only"
            ),
        )

    options = experiment5.parse_arguments(
        arguments,
        default_trajectories=DEFAULT_TRAJECTORIES,
        default_trotter_delta_t=DEFAULT_TROTTER_DELTA_T,
        default_t_final=DEFAULT_T_FINAL,
        default_time_points=DEFAULT_TIME_POINTS,
        description=(
            "Run Experiment 7: randomized single-boundary Lie dilation with "
            "every XXPlusYY gate explicitly synthesized into RZ/RX/RZZ "
            "before Aer or IBM compilation. Non-aer backends submit QPU work."
        ),
        configure_parser=configure_parser,
    )
    if options.classical_reference and options.optimized_classical_reference:
        raise ValueError(
            "choose either --classical-reference or "
            "--optimized-classical-reference, not both"
        )
    return options


if __name__ == "__main__":
    main(parse_arguments())
