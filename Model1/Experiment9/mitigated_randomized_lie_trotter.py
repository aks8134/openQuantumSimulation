"""Experiment 9: Experiment 8 plus native-qem twirling, timed DD and readout.

Transforms preserve native Qiskit circuits without conversion or re-lowering.
DD requires calibrated durations. Readout correction requires supplied physical
assignment matrices (identity only for the noiseless Aer execution path).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from IBMRuntime import (  # noqa: E402
    CompilerConfig,
    Err,
    Ok,
    Sample,
    run_sample_variants_batch_sync,
)
from Model1.Experiment5 import randomized_lie_trotter as experiment5  # noqa: E402
from Model1.Experiment7 import randomized_lie_trotter as experiment7  # noqa: E402
from Model1.Experiment8 import zne_randomized_lie_trotter as experiment8  # noqa: E402
from Model1.Experiment9.progress import BatchProgress, tqdm_factory  # noqa: E402
import qem  # noqa: E402


EXPERIMENT_NAME = "Model1/Experiment9 fully mitigated randomized Lie-Trotter"
METHOD_NAME = (
    "Experiment 8 native-qem ZNE with Pauli twirling, scheduled "
    "dynamical decoupling, and calibrated block readout mitigation"
)
MITIGATION_VERSION = 3
DEFAULT_SEED_TWIRLING = 314159
DEFAULT_DD_SEQUENCE = "xx"
DEFAULT_DD_SPACING = -1
SUPPORTED_TWIRLED_GATES = frozenset(("cz", "cx", "rxx", "ryy", "rzz"))

_BASE_CHECKPOINT_CONFIGURATION = experiment8._checkpoint_configuration
_BASE_COMPATIBILITY_SIGNATURE = experiment8._compatibility_signature
_BASE_PROSPECTIVE_SIGNATURE = experiment8._prospective_signature
_BASE_SAVE_RESULTS = experiment8.save_results
_BASE_LOAD_CHECKPOINT = experiment8.load_checkpoint
_EXPERIMENT8_ORIGINALS = {
    name: getattr(experiment8, name)
    for name in (
        "EXPERIMENT_NAME",
        "METHOD_NAME",
        "FoldDiagnostic",
        "execute_zne_circuits",
        "_diagnostic_from_record",
        "_raw_observable_arrays",
        "output_paths",
        "_checkpoint_configuration",
        "_compatibility_signature",
        "_prospective_signature",
        "save_results",
        "plot_results",
        "plot_transpilation_metrics",
        "_print_summary",
        "load_checkpoint",
    )
}

# The Experiment 8 driver calls analysis without passing its diagnostics.  The
# execution and checkpoint hooks below keep the corresponding immutable tuple
# here for the duration of one command.  It is cleared at the start/end of main.
_ACTIVE_DIAGNOSTICS = ()


@dataclass(frozen=True, slots=True)
class MitigationDiagnostic(experiment8.FoldDiagnostic):
    """Per-submitted-circuit diagnostics beyond Experiment 8 folding."""

    twirling_enabled: bool = False
    twirling_seed: int | None = None
    twirled_gate_count: int = 0
    twirled_gate_names: tuple[tuple[str, int], ...] = ()
    untwirled_two_qubit_gate_names: tuple[tuple[str, int], ...] = ()
    post_folding_operation_count: int = 0
    post_folding_depth: int = 0
    post_twirling_operation_count: int = 0
    dd_enabled: bool = False
    dd_applied: bool = False
    dd_sequence: str = "none"
    dd_inserted_x_count: int = 0
    dd_inserted_y_count: int = 0
    dd_inserted_identity_count: int = 0
    dd_inserted_delay_count: int = 0
    dd_skip_reason: str | None = None
    submitted_operation_count: int = 0
    submitted_depth: int = 0
    logical_to_physical: tuple[int, ...] = ()
    readout_blocks: tuple = ()
    readout_source: str = "disabled"


def pauli_twirl_native_circuit(circuit, seed, *, enabled=True, target=None):
    """Generate a native Pauli variant, including sign-adjusted rotations."""

    if not enabled:
        return circuit, {
            "seed": None,
            "twirled": (),
            "untwirled_two_qubit": (),
        }

    batch = experiment8._qem_value(qem.twirl(
        circuit,
        qem.TwirlingPlan(seed=seed, gate_names=tuple(sorted(SUPPORTED_TWIRLED_GATES))),
        target=target,
    ))
    twirled = batch.circuit
    counts = Counter(
        item.operation.name.lower()
        for item in circuit.data
        if len(item.qubits) == 2
        and item.operation.name.lower() in SUPPORTED_TWIRLED_GATES
    )
    skipped = Counter(
        item.operation.name.lower()
        for item in circuit.data
        if len(item.qubits) == 2
        and item.operation.name.lower() not in SUPPORTED_TWIRLED_GATES
        and hasattr(item.operation, "to_matrix")
    )
    return twirled, {
        "seed": int(seed),
        "twirled": tuple(sorted(counts.items())),
        "untwirled_two_qubit": tuple(sorted(skipped.items())),
    }


def apply_dynamical_decoupling(
    circuit,
    target=None,
    *,
    enabled=True,
    rule=DEFAULT_DD_SEQUENCE,
    spacing=DEFAULT_DD_SPACING,
    spacing_fractions=None,
    durations=None,
):
    """Schedule physical idle windows and pad them via qem.apply_dd."""

    if not enabled:
        return circuit, {
            "applied": False,
            "inserted_x": 0,
            "inserted_delay": 0,
            "skip_reason": "disabled by --no-dynamical-decoupling",
        }

    before_x = circuit.count_ops().get("x", 0)
    before_y = circuit.count_ops().get("y", 0)
    before_identity = circuit.count_ops().get("id", 0)
    if spacing != -1:
        raise ValueError("integer --dd-spacing is obsolete; use --dd-spacing-fractions")
    active = tuple(sorted({
        circuit.find_bit(bit).index for item in circuit.data
        if item.operation.name not in ("barrier", "delay") for bit in item.qubits
    }))
    padded = experiment8._qem_value(qem.apply_dd(
        circuit,
        qem.DDPlan(sequence=rule, qubits=active, spacing=spacing_fractions),
        target=target if durations is None else None,
        durations=durations,
    )).circuit
    if target is not None:
        for item in padded.data:
            if item.operation.name == "barrier":
                continue
            qargs = tuple(padded.find_bit(bit).index for bit in item.qubits)
            if not target.instruction_supported(operation_name=item.operation.name,
                                                qargs=qargs, parameters=item.operation.params):
                raise ValueError(f"DD emitted an unsupported native instruction: {item.operation.name} {qargs}")
    return padded, {
        "applied": True,
        "inserted_x": int(padded.count_ops().get("x", 0) - before_x),
        "inserted_y": int(padded.count_ops().get("y", 0) - before_y),
        "inserted_identity": int(
            padded.count_ops().get("id", 0) - before_identity
        ),
        "inserted_delay": int(padded.count_ops().get("delay", 0) - circuit.count_ops().get("delay", 0)),
        "skip_reason": None,
    }


def _logical_layout(circuit):
    layout = getattr(circuit, "layout", None)
    if layout is None:
        logical_to_physical = tuple(range(circuit.num_qubits))
    else:
        logical_to_physical = tuple(
            map(int, layout.initial_index_layout(filter_ancillas=True))
        )
    return logical_to_physical


def _readout_model(circuit, options):
    """Map physical calibration blocks to actual terminal classical outcomes.

    Measurement destinations, not the initial layout, capture routing SWAPs.
    Correlated blocks must be measured in full; independence is assumed between
    blocks. Calibration uncertainty is not included in the shot error bars.
    """
    if getattr(options, "no_readout_mitigation", True):
        return None
    measured = {}
    for item in circuit.data:
        if item.operation.name == "measure":
            physical = circuit.find_bit(item.qubits[0]).index
            classical = circuit.find_bit(item.clbits[0]).index
            if physical in measured or classical in measured.values():
                raise ValueError("readout correction requires unique terminal measurements")
            measured[physical] = classical
    payload = getattr(options, "readout_calibration_data", None)
    if payload is None:
        if options.backend.lower() != "aer":
            raise ValueError("hardware readout mitigation requires --readout-calibration FILE")
        return qem.ReadoutModel(tuple(
            qem.ReadoutBlock((bit,), ((1.0, 0.0), (0.0, 1.0)))
            for bit in sorted(measured.values())
        ), source="identity assignment matrices: noiseless Aer, not measured calibration")
    blocks = []
    for block in payload["blocks"]:
        physicals = tuple(block["physical_qubits"])
        if not set(physicals).intersection(measured):
            continue
        if not all(physical in measured for physical in physicals):
            raise ValueError("a correlated calibration block is only partially measured; use smaller calibration blocks")
        blocks.append(qem.ReadoutBlock(tuple(measured[physical] for physical in physicals),
                                      tuple(tuple(row) for row in block["assignment"])))
    model = qem.ReadoutModel(tuple(blocks), source=payload["source"])
    if set(model.bits) != set(measured.values()):
        raise ValueError("readout calibration does not cover every measured physical qubit")
    return model


def _dd_durations(options):
    payload = getattr(options, "dd_durations_data", None)
    if payload is None:
        return None
    from qiskit.transpiler import InstructionDurations
    return InstructionDurations(payload["instructions"], dt=payload["dt"])


def _twirling_seed(options, spec, base_index):
    return int(
        options.seed_twirling
        + (
            base_index * len(options.zne_scale_factors)
            + spec.scale_index
        )
        * options.fold_repetitions
        + spec.repetition
    )


def transform_native_variant(native, target, spec, base_index, options):
    """Apply the ordered Experiment 9 post-transpilation transform."""

    folded, fold_diagnostic = experiment8.fold_native_circuit(
        native,
        target,
        spec,
        base_index,
    )
    seed = _twirling_seed(options, spec, base_index)
    twirled, twirl = pauli_twirl_native_circuit(
        folded,
        seed,
        enabled=not options.no_pauli_twirling,
        target=target,
    )
    decoupled, dd = apply_dynamical_decoupling(
        twirled,
        target,
        enabled=not options.no_dynamical_decoupling,
        rule=options.dd_sequence,
        spacing=options.dd_spacing,
        spacing_fractions=getattr(options, "dd_spacing_fractions", None),
        durations=_dd_durations(options),
    )
    submitted = decoupled
    logical_to_physical = _logical_layout(native)
    readout_model = _readout_model(submitted, options)
    twirled_count = sum(value for _, value in twirl["twirled"])
    fold_values = asdict(fold_diagnostic)
    fold_values.update(
        {
            # Experiment 8 uses these fields for submitted-circuit metrics.
            "folded_operation_count": int(submitted.size()),
            "folded_depth": int(submitted.depth()),
            "twirling_enabled": not options.no_pauli_twirling,
            "twirling_seed": twirl["seed"],
            "twirled_gate_count": int(twirled_count),
            "twirled_gate_names": twirl["twirled"],
            "untwirled_two_qubit_gate_names": twirl[
                "untwirled_two_qubit"
            ],
            "post_folding_operation_count": int(folded.size()),
            "post_folding_depth": int(folded.depth()),
            "post_twirling_operation_count": int(twirled.size()),
            "dd_enabled": not options.no_dynamical_decoupling,
            "dd_applied": bool(dd["applied"]),
            "dd_sequence": (
                options.dd_sequence.upper()
                if not options.no_dynamical_decoupling
                else "none"
            ),
            "dd_inserted_x_count": int(dd["inserted_x"]),
            "dd_inserted_y_count": int(dd.get("inserted_y", 0)),
            "dd_inserted_identity_count": int(
                dd.get("inserted_identity", 0)
            ),
            "dd_inserted_delay_count": int(dd["inserted_delay"]),
            "dd_skip_reason": dd["skip_reason"],
            "submitted_operation_count": int(submitted.size()),
            "submitted_depth": int(submitted.depth()),
            "logical_to_physical": logical_to_physical,
            "readout_blocks": () if readout_model is None else tuple(
                (block.bits, block.assignment) for block in readout_model.blocks
            ),
            "readout_source": "disabled" if readout_model is None else readout_model.source,
        }
    )
    diagnostic = MitigationDiagnostic(**fold_values)
    return submitted, diagnostic


def execute_mitigated_circuits(
    base_circuits,
    options,
    *,
    base_index_offset=0,
    scale_indices=None,
    on_batch_complete=None,
):
    """Execute folded/twirled/DD-padded variants using Experiment 8 batching."""

    global _ACTIVE_DIAGNOSTICS
    factory = None if getattr(options, "no_progress", False) else tqdm_factory()
    target, environment = experiment5.hardware_tools._runtime_target(
        options.backend,
        options.aer_method,
        options.account_file,
    )
    compiler = CompilerConfig(
        optimization_level=options.optimization_level,
        seed_transpiler=options.seed_transpiler,
    )
    workload = Sample(
        shots=experiment8.shots_per_variant(options),
        seed_simulator=options.seed_simulator,
    )
    selected_scale_indices = (
        tuple(range(len(options.zne_scale_factors)))
        if scale_indices is None
        else tuple(map(int, scale_indices))
    )
    if not selected_scale_indices:
        raise ValueError("at least one scale index must be executed")
    variants_per_base = (
        len(selected_scale_indices) * options.fold_repetitions
    )
    submitted_batch_size = options.batch_size or (
        experiment5.DEFAULT_AER_BATCH_SIZE
        if options.backend.lower() in ("aer", "fake_kingston", "fake_miami")
        else experiment5.DEFAULT_HARDWARE_BATCH_SIZE
    )
    if submitted_batch_size < variants_per_base:
        raise ValueError(
            f"--batch-size must be at least {variants_per_base}, the number "
            "of scale/fold variants generated from one base circuit"
        )
    base_batch_size = max(1, submitted_batch_size // variants_per_base)
    indexed = tuple(enumerate(base_circuits, start=base_index_offset))
    batches = experiment8._chunks(indexed, base_batch_size)
    previously_active = _ACTIVE_DIAGNOSTICS
    collected = ()
    diagnostics = ()
    for batch_index, indexed_batch in enumerate(batches, start=1):
        indices = tuple(index for index, _ in indexed_batch)
        output_count = len(indexed_batch) * variants_per_base
        print(
            f"Executing Experiment 9 batch {batch_index}/{len(batches)}: "
            f"{len(indexed_batch)} base circuits -> {output_count} mitigated "
            f"variants on {options.backend}"
        )
        with BatchProgress(factory, label=f"Batch {batch_index}/{len(batches)}",
                           backend=options.backend, variant_count=output_count) as progress:
            progress.logical_start(len(indexed_batch))
            explicit = []
            for _, circuit in indexed_batch:
                explicit.append(experiment7.prepare_circuit_for_compilation(circuit))
                progress.advance()
            explicit_batch = tuple(explicit)
            index_iterator = iter(indices)
            batch_diagnostics = []

            def transform(native, backend_target):
                base_index = next(index_iterator)
                transformed = []
                for spec in experiment8.fold_specs(options, base_index, selected_scale_indices):
                    value = transform_native_variant(native, backend_target, spec, base_index, options)
                    transformed.append(value[0])
                    batch_diagnostics.append(value[1])
                    progress.advance()
                return tuple(transformed)

            result = run_sample_variants_batch_sync(
                explicit_batch,
                target,
                compiler,
                workload,
                environment,
                transform,
                progress_callback=progress.stage if factory is not None else None,
            )
        match result:
            case Ok(results):
                collected = (*collected, *results)
                diagnostics = (*diagnostics, *batch_diagnostics)
                _ACTIVE_DIAGNOSTICS = (*previously_active, *diagnostics)
                if on_batch_complete is not None:
                    on_batch_complete(collected, diagnostics)
            case Err(error):
                message = experiment5.hardware_tools._runtime_error_message(
                    error
                )
                if options.backend.lower() != "aer" and "6073" in message:
                    raise RuntimeError(
                        f"{message}. Re-run with a smaller --batch-size."
                    )
                raise RuntimeError(message)
    return collected, diagnostics


def _diagnostic_from_record(record):
    values = dict(record["fold_diagnostic"])
    for field in (
        "eligible_gate_counts",
        "excluded_unitary_gate_counts",
        "twirled_gate_names",
        "untwirled_two_qubit_gate_names",
    ):
        values[field] = tuple(tuple(item) for item in values.get(field, ()))
    values["logical_to_physical"] = tuple(
        values.get("logical_to_physical", ())
    )
    values["readout_blocks"] = tuple(
        (tuple(bits), tuple(tuple(row) for row in assignment))
        for bits, assignment in values.get("readout_blocks", ())
    )
    return MitigationDiagnostic(**values)


def load_checkpoint(path, options, times, metadata):
    global _ACTIVE_DIAGNOSTICS
    results, diagnostics = _BASE_LOAD_CHECKPOINT(
        path,
        options,
        times,
        metadata,
    )
    _ACTIVE_DIAGNOSTICS = diagnostics
    return results, diagnostics


def _diagnostic_lookup():
    return {
        (
            value.base_index,
            value.scale_index,
            value.fold_repetition,
        ): value
        for value in _ACTIVE_DIAGNOSTICS
    }


def _corrected_pauli(counts, bits, model, *, enabled, seed=None):
    raw = experiment5._expectation(counts, bits)
    shots = sum(counts.values())
    if not enabled:
        return raw, experiment5._pauli_variance(raw, shots)

    if model is None:
        raise ValueError("enabled readout mitigation requires recorded assignment matrices")
    corrected = experiment8._qem_value(qem.mitigate_parity(counts, model, tuple(bits)))
    variance = 0.0 if corrected.standard_error is None else corrected.standard_error**2
    return corrected.estimate, variance


def _raw_observable_arrays(grouped, options, time_count):
    """Experiment 8 reconstruction with qem calibrated parity correction."""

    scales = len(options.zne_scale_factors)
    shape = (
        time_count,
        scales,
        options.trajectories,
        options.fold_repetitions,
    )
    populations = np.zeros((options.n_qubits, *shape), dtype=float)
    correlations = np.zeros((options.n_qubits - 1, *shape), dtype=float)
    flows = np.zeros_like(correlations)
    population_shot = np.zeros_like(populations)
    correlation_shot = np.zeros_like(correlations)
    flow_shot = np.zeros_like(flows)
    lookup = _diagnostic_lookup()
    readout_enabled = not options.no_readout_mitigation

    for scale_index in range(scales):
        for time_index in range(time_count):
            for trajectory_index in range(options.trajectories):
                for repetition in range(options.fold_repetitions):
                    basis_counts = grouped[scale_index][time_index][
                        trajectory_index
                    ][repetition]
                    readout_seed_base = (
                        int(getattr(options, "seed_simulator", 0) or 0)
                        + 1000003 * scale_index
                        + 10007 * time_index
                        + 101 * trajectory_index
                        + repetition
                    ) % (2**32)

                    def model_for_basis(basis):
                        if not readout_enabled:
                            return None
                        basis_index = experiment8.MEASUREMENT_BASES.index(basis)
                        base_index = (
                            (
                                time_index * options.trajectories
                                + trajectory_index
                            )
                            * len(experiment8.MEASUREMENT_BASES)
                            + basis_index
                        )
                        diagnostic = lookup[
                            (base_index, scale_index, repetition)
                        ]
                        return qem.ReadoutModel(tuple(
                            qem.ReadoutBlock(bits, assignment)
                            for bits, assignment in diagnostic.readout_blocks
                        ), source=diagnostic.readout_source)

                    z_model = model_for_basis("Z")
                    for site in range(options.n_qubits):
                        value, variance = _corrected_pauli(
                            basis_counts["Z"],
                            (site,),
                            z_model,
                            enabled=readout_enabled,
                            seed=(readout_seed_base + site) % (2**32),
                        )
                        index = (
                            site,
                            time_index,
                            scale_index,
                            trajectory_index,
                            repetition,
                        )
                        populations[index] = 0.5 * (1.0 - value)
                        population_shot[index] = 0.25 * variance

                    x_model = model_for_basis("X")
                    mixed_model = model_for_basis("XY")
                    for bond in range(options.n_qubits - 1):
                        bits = (bond, bond + 1)
                        xx_value, xx_variance = _corrected_pauli(
                            basis_counts["X"],
                            bits,
                            x_model,
                            enabled=readout_enabled,
                            seed=(
                                readout_seed_base
                                + options.n_qubits
                                + 2 * bond
                            )
                            % (2**32),
                        )
                        mixed_value, mixed_variance = _corrected_pauli(
                            basis_counts["XY"],
                            bits,
                            mixed_model,
                            enabled=readout_enabled,
                            seed=(
                                readout_seed_base
                                + options.n_qubits
                                + 2 * bond
                                + 1
                            )
                            % (2**32),
                        )
                        index = (
                            bond,
                            time_index,
                            scale_index,
                            trajectory_index,
                            repetition,
                        )
                        correlations[index] = 2.0 * xx_value
                        flows[index] = (
                            -experiment5.J * mixed_value
                            if bond % 2 == 0
                            else experiment5.J * mixed_value
                        )
                        correlation_shot[index] = 4.0 * xx_variance
                        flow_shot[index] = experiment5.J**2 * mixed_variance
    return (
        (populations, correlations, flows),
        (population_shot, correlation_shot, flow_shot),
    )


def _mitigation_payload(options, diagnostics):
    dd_applied = sum(value.dd_applied for value in diagnostics)
    dd_skips = Counter(
        value.dd_skip_reason
        for value in diagnostics
        if value.dd_enabled and not value.dd_applied
    )
    return {
        "version": MITIGATION_VERSION,
        "transform_order": [
            "optimizing transpilation",
            "qem native local unitary folding (ZNE)",
            "qem native Pauli twirling",
            "qem duration-aware ALAP dynamical decoupling",
            "sampling",
            "qem calibrated block parity readout correction",
        ],
        "pauli_twirling": {
            "enabled": not options.no_pauli_twirling,
            "method": "qem.twirl",
            "supported_gate_names": sorted(SUPPORTED_TWIRLED_GATES),
            "seed": options.seed_twirling,
            "realizations_per_submitted_variant": 1,
            "ensemble": (
                "averaged jointly over qDRIFT trajectories and fold "
                "repetitions"
            ),
            "twirled_gate_count": int(
                sum(value.twirled_gate_count for value in diagnostics)
            ),
            "retranspiled_after_twirling": False,
            "post_mitigation_transpilation_optimization_level": None,
        },
        "dynamical_decoupling": {
            "enabled": not options.no_dynamical_decoupling,
            "sequence": options.dd_sequence.upper(),
            "spacing_fractions": options.dd_spacing_fractions,
            "schedule": "Qiskit duration-aware ALAP via qem.apply_dd",
            "circuits_padded": int(dd_applied),
            "circuits_not_padded": int(len(diagnostics) - dd_applied),
            "inserted_x_count": int(
                sum(value.dd_inserted_x_count for value in diagnostics)
            ),
            "inserted_y_count": int(
                sum(value.dd_inserted_y_count for value in diagnostics)
            ),
            "inserted_identity_count": int(
                sum(value.dd_inserted_identity_count for value in diagnostics)
            ),
            "skip_reasons": dict(dd_skips),
            "note": "calibrated target durations or explicit Aer timing file required; reset-initialized idle windows skipped",
        },
        "readout_mitigation": {
            "enabled": not options.no_readout_mitigation,
            "model": "calibrated assignment blocks, independent between blocks",
            "calibration_source": sorted(set(value.readout_source for value in diagnostics)),
            "implementation": "qem.mitigate_parity; adjoint block solve without dense global matrix",
            "quasi_expectations_clipped": False,
            "uncertainty": "empirical shot variance of corrected parity weights; calibration uncertainty excluded",
            "dense_2^N_confusion_matrix_constructed": False,
        },
    }


def output_paths(options):
    dd_state = (
        f"dd_{options.dd_sequence}"
        if not options.no_dynamical_decoupling
        else "dd_off"
    )
    states = "_".join(
        (
            "pauli_on" if not options.no_pauli_twirling else "pauli_off",
            dd_state,
            "readout_on" if not options.no_readout_mitigation else "readout_off",
        )
    )
    factors = "_".join(
        map(experiment8._safe_scale_name, options.zne_scale_factors)
    )
    zne = "zne" if experiment8._zne_enabled(options) else "no_zne"
    inference_name = (
        options.zne_inference if experiment8._zne_enabled(options) else "raw"
    )
    stem = (
        f"qem_{zne}_randomized_lie_trotter_"
        f"{experiment5._safe_name(options.backend)}_"
        f"N{options.n_qubits}_R{options.trajectories}_"
        f"F{options.fold_repetitions}_B3_{states}_{inference_name}_S{factors}"
    )
    directory = (
        Path(__file__).resolve().parent
        if options.output_directory is None
        else Path(options.output_directory)
    )
    return {
        "results_figure": directory / "figures" / f"{stem}.png",
        "scaling_figure": directory / "figures" / f"{stem}_zne_scaling.png",
        "metrics_figure": directory / "figures" / f"{stem}_transpilation_metrics.png",
        "layout_figure": directory / "figures" / f"{stem}_transpiled_layout.png",
        "left_circuit_figure": directory / "figures" / f"{stem}_one_step_left.png",
        "right_circuit_figure": directory / "figures" / f"{stem}_one_step_right.png",
        "decomposition_figure": directory / "figures" / f"{stem}_exchange_decomposition.png",
        "result": directory / "results" / f"{stem}.json",
    }


def _settings_payload(options):
    return {
        "version": MITIGATION_VERSION,
        "pauli_twirling": not options.no_pauli_twirling,
        "seed_twirling": options.seed_twirling,
        "dynamical_decoupling": not options.no_dynamical_decoupling,
        "dd_sequence": options.dd_sequence,
        "dd_spacing_fractions": (
            None if getattr(options, "dd_spacing_fractions", None) is None
            else list(options.dd_spacing_fractions)
        ),
        "dd_durations": getattr(options, "dd_durations_data", None),
        "readout_mitigation": not options.no_readout_mitigation,
        "readout_calibration": getattr(options, "readout_calibration_data", None),
    }


def _checkpoint_configuration(options, times, metadata):
    return {
        **_BASE_CHECKPOINT_CONFIGURATION(options, times, metadata),
        "experiment": EXPERIMENT_NAME,
        "additional_mitigation": _settings_payload(options),
    }


def _compatibility_signature(payload):
    return {
        **_BASE_COMPATIBILITY_SIGNATURE(payload),
        "additional_mitigation": payload.get("additional_mitigation_settings"),
    }


def _prospective_signature(options):
    return {
        **_BASE_PROSPECTIVE_SIGNATURE(options),
        "experiment": EXPERIMENT_NAME,
        "additional_mitigation": _settings_payload(options),
    }


def save_results(*args, **kwargs):
    payload = _BASE_SAVE_RESULTS(*args, **kwargs)
    options = args[1]
    diagnostics = args[6]
    path = Path(args[0])
    payload["experiment"] = EXPERIMENT_NAME
    payload["method"] = METHOD_NAME
    payload["additional_mitigation_settings"] = _settings_payload(options)
    payload["additional_mitigation"] = _mitigation_payload(
        options,
        diagnostics,
    )
    experiment5._atomic_write_json(path, payload)
    return payload


def plot_results(
    times,
    backend,
    trajectories,
    fold_repetitions,
    measured,
    uncertainties,
    output_path,
    exact=None,
    zne_enabled=True,
):
    """Draw Experiment 8 observables with an Experiment 9-specific title."""

    exact_values = (None, None, None) if exact is None else exact
    figure, axes = experiment8.plt.subplots(
        2,
        2,
        figsize=(16, 11),
        sharex=True,
    )
    families = (
        (
            measured[0],
            uncertainties.total[0],
            "Site populations",
            "Population",
            lambda index: rf"$n_{index}$",
            exact_values[0],
        ),
        (
            measured[1],
            uncertainties.total[1],
            "Nearest-neighbor XY correlations",
            r"$\langle X_nX_{n+1}+Y_nY_{n+1}\rangle$",
            lambda index: f"bond {index}-{index + 1}",
            exact_values[1],
        ),
        (
            measured[2],
            uncertainties.total[2],
            "Excitation flow",
            "Flow",
            lambda index: f"bond {index}-{index + 1}",
            exact_values[2],
        ),
    )
    for axis, family in zip(axes.flat[:3], families, strict=True):
        experiment5._plot_observable_family(
            axis,
            times,
            family[0],
            family[1],
            family[2],
            family[3],
            family[4],
            family[5],
        )
    diagnostic = axes[1, 1]

    def family_max(values):
        return np.maximum.reduce(
            tuple(np.max(value, axis=0) for value in values)
        )

    diagnostic.plot(
        times,
        family_max(uncertainties.shot),
        label="max propagated shot s.e.",
    )
    diagnostic.plot(
        times,
        family_max(uncertainties.folding),
        label="max fold/twirl realization s.e.",
    )
    diagnostic.plot(
        times,
        family_max(uncertainties.trajectory),
        label="max propagated trajectory s.e.",
    )
    diagnostic.plot(
        times,
        family_max(uncertainties.total),
        label="max total s.e.",
        linewidth=2,
    )
    diagnostic.set_title(
        "Mitigated ZNE uncertainty decomposition"
        if zne_enabled
        else "Mitigated baseline uncertainty decomposition"
    )
    diagnostic.set_xlabel("Time")
    diagnostic.set_ylabel("Standard error")
    diagnostic.grid(alpha=0.25)
    diagnostic.legend(fontsize="small")
    mode = "ZNE" if zne_enabled else "scale-one baseline"
    figure.suptitle(
        f"Experiment 9 combined mitigation on {backend}: {mode} "
        f"(R={trajectories}, F={fold_repetitions})",
        fontsize=15,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    experiment8.plt.close(figure)


def plot_transpilation_metrics(summary, output_path, *, zne_enabled=True):
    """Compare explicit, transpiled, and finally submitted resources."""

    factors = np.asarray(
        [value["requested_scale_factor"] for value in summary],
        dtype=float,
    )
    width = 0.25
    positions = np.arange(len(factors))
    figure, axes = experiment8.plt.subplots(1, 2, figsize=(14, 5.5))
    specifications = (
        (
            axes[0],
            "mean_original_explicit_operation_count",
            "mean_base_transpiled_operation_count",
            "mean_folded_operation_count",
            "Mean operation count",
        ),
        (
            axes[1],
            "mean_original_explicit_depth",
            "mean_base_transpiled_depth",
            "mean_folded_depth",
            "Mean circuit depth",
        ),
    )
    for axis, original_key, base_key, submitted_key, title in specifications:
        original = [value[original_key] for value in summary]
        base = [value[base_key] for value in summary]
        submitted = [value[submitted_key] for value in summary]
        first = axis.bar(
            positions - width,
            original,
            width,
            label="explicit pre-transpilation",
        )
        second = axis.bar(
            positions,
            base,
            width,
            label="base transpiled ISA",
        )
        third = axis.bar(
            positions + width,
            submitted,
            width,
            label="submitted after fold + twirl + DD",
        )
        axis.bar_label(first, fmt="%.0f", padding=2)
        axis.bar_label(second, fmt="%.0f", padding=2)
        axis.bar_label(third, fmt="%.0f", padding=2)
        axis.set_xticks(positions, tuple(map(str, factors)))
        axis.set_xlabel("Requested noise scale")
        axis.set_ylabel("Count")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.suptitle(
        "Experiment 9: final submitted-circuit resources",
        fontsize=14,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    experiment8.plt.close(figure)


def _print_summary(options, schedule, payload, results, paths, checkpoint_path):
    job_ids = tuple(
        dict.fromkeys(
            result.job_id for result in results if result.job_id is not None
        )
    )
    print(f"Backend: {options.backend}")
    print(f"System/circuit qubits: {options.n_qubits}/{options.n_qubits + 2}")
    print(f"Randomized trajectories R: {options.trajectories}")
    print(f"Measurement bases: {list(experiment8.MEASUREMENT_BASES)}")
    print(
        "Additional mitigation: "
        f"Pauli twirling={'off' if options.no_pauli_twirling else 'on'}, "
        f"DD={'off' if options.no_dynamical_decoupling else 'on'}, "
        f"readout={'off' if options.no_readout_mitigation else 'on'}"
    )
    if experiment8._zne_enabled(options):
        print(f"Fold realizations F: {options.fold_repetitions}")
        print(f"ZNE scale factors: {list(options.zne_scale_factors)}")
        print(f"ZNE inference: {options.zne_inference}")
        print(f"Extrapolation weights: {payload['zne']['extrapolation_weights']}")
    else:
        print("ZNE: disabled (unfurled scale-one baseline)")
    print(f"Shots per submitted circuit: {experiment8.shots_per_variant(options)}")
    print(f"Archive saved times: {payload['times']}")
    print(f"Total Trotter substeps: {len(schedule.substep_dts)}")
    print(
        "Submitted operation-count range: "
        f"{min(value.compiled_gate_count for value in results)}--"
        f"{max(value.compiled_gate_count for value in results)}"
    )
    print(
        "Submitted depth range: "
        f"{min(value.compiled_depth for value in results)}--"
        f"{max(value.compiled_depth for value in results)}"
    )
    if job_ids:
        print(f"Provider job IDs: {', '.join(job_ids)}")
    artifact_names = ["results_figure"]
    if experiment8._zne_enabled(options):
        artifact_names.append("scaling_figure")
    artifact_names.append("metrics_figure")
    if not getattr(options, "no_layout", False):
        artifact_names.append("layout_figure")
    artifact_names.append("result")
    for name in artifact_names:
        print(f"Saved {name.replace('_', ' ')}: {paths[name]}")
    print(f"Saved checkpoint: {checkpoint_path}")


def install_experiment9_hooks():
    """Install Experiment 9 behavior into the reusable Experiment 8 driver."""

    experiment8.EXPERIMENT_NAME = EXPERIMENT_NAME
    experiment8.METHOD_NAME = METHOD_NAME
    experiment8.FoldDiagnostic = MitigationDiagnostic
    experiment8.execute_zne_circuits = execute_mitigated_circuits
    experiment8._diagnostic_from_record = _diagnostic_from_record
    experiment8._raw_observable_arrays = _raw_observable_arrays
    experiment8.output_paths = output_paths
    experiment8._checkpoint_configuration = _checkpoint_configuration
    experiment8._compatibility_signature = _compatibility_signature
    experiment8._prospective_signature = _prospective_signature
    experiment8.save_results = save_results
    experiment8.plot_results = plot_results
    experiment8.plot_transpilation_metrics = plot_transpilation_metrics
    experiment8._print_summary = _print_summary
    experiment8.load_checkpoint = load_checkpoint


def restore_experiment8_hooks():
    for name, value in _EXPERIMENT8_ORIGINALS.items():
        setattr(experiment8, name, value)


def parse_arguments(arguments=None):
    def configure_parser(parser):
        parser.add_argument(
            "--no-layout", action="store_true",
            help="skip the transpiled topology/mapping figure and its circuit panels; keep other figures",
        )
        parser.add_argument(
            "--no-progress", action="store_true",
            help="disable tqdm preparation bars and elapsed-time batch wait indicators",
        )
        parser.add_argument(
            "--optimized-classical-reference",
            action="store_true",
            help="use the exact vacuum-plus-single-excitation reference",
        )
        parser.add_argument(
            "--no-zne",
            action="store_true",
            help="disable local folding and zero-noise extrapolation",
        )
        parser.add_argument(
            "--reuse-zne-result",
            type=Path,
            help=(
                "reserved for future Experiment 9 scale augmentation; "
                "currently rejected"
            ),
        )
        parser.add_argument(
            "--zne-scale-factors",
            nargs="+",
            type=float,
            default=experiment8.DEFAULT_SCALE_FACTORS,
            metavar="LAMBDA",
        )
        parser.add_argument(
            "--zne-inference",
            choices=("linear", "richardson", "polynomial"),
            default=experiment8.DEFAULT_ZNE_INFERENCE,
        )
        parser.add_argument(
            "--zne-polynomial-order",
            type=experiment5.positive_integer,
            default=2,
        )
        parser.add_argument(
            "--fold-repetitions",
            type=experiment5.positive_integer,
            default=experiment8.DEFAULT_FOLD_REPETITIONS,
        )
        parser.add_argument(
            "--seed-folding",
            type=int,
            default=experiment8.DEFAULT_SEED_FOLDING,
        )
        parser.add_argument(
            "--no-pauli-twirling",
            action="store_true",
            help="disable native qem CZ/CX/RXX/RYY/RZZ Pauli twirling",
        )
        parser.add_argument(
            "--seed-twirling",
            type=int,
            default=DEFAULT_SEED_TWIRLING,
        )
        parser.add_argument(
            "--no-dynamical-decoupling",
            action="store_true",
            help="disable duration-aware qem dynamical decoupling",
        )
        parser.add_argument(
            "--dd-sequence",
            choices=("xx", "yy", "xy4", "xy8", "xyxy"),
            default=DEFAULT_DD_SEQUENCE,
            help="timed DD sequence (xyxy is an alias for xy4)",
        )
        parser.add_argument(
            "--dd-spacing",
            type=int,
            default=DEFAULT_DD_SPACING,
            help="legacy option: only -1 accepted; use --dd-spacing-fractions",
        )
        parser.add_argument("--dd-spacing-fractions", nargs="+", type=float,
                            help="idle-time fractions before/between/after DD pulses; must sum to 1")
        parser.add_argument("--dd-durations", type=Path,
                            help="Aer timing JSON: dt (seconds) and InstructionDurations instructions")
        parser.add_argument("--readout-calibration", type=Path,
                            help="assignment-matrix JSON with backend, source and physical-qubit blocks")
        parser.add_argument(
            "--no-readout-mitigation",
            action="store_true",
            help="disable calibrated qem readout correction",
        )

    options = experiment5.parse_arguments(
        arguments,
        default_trajectories=experiment7.DEFAULT_TRAJECTORIES,
        default_trotter_delta_t=experiment7.DEFAULT_TROTTER_DELTA_T,
        default_t_final=experiment7.DEFAULT_T_FINAL,
        default_time_points=experiment7.DEFAULT_TIME_POINTS,
        description=(
            "Run Experiment 9: Experiment 8 ZNE plus Pauli twirling, "
            "timed dynamical decoupling, and calibrated readout mitigation. "
            "aer is noiseless local simulation; fake_kingston/fake_miami are local snapshot-noisy "
            "simulation without login. IBM backend names submit real QPU work."
        ),
        configure_parser=configure_parser,
    )
    if options.no_zne:
        if options.fold_repetitions != 1:
            raise ValueError("--no-zne requires --fold-repetitions 1")
        options.zne_scale_factors = (1.0,)
    else:
        options.zne_scale_factors = experiment8._validated_scale_factors(
            options.zne_scale_factors
        )
    if options.seed_folding < 0 or options.seed_twirling < 0:
        raise ValueError("folding and twirling seeds must be non-negative")
    if options.dd_spacing != -1:
        raise ValueError("--dd-spacing circuit-column counts are no longer used; use --dd-spacing-fractions")
    if options.dd_sequence == "xyxy":
        options.dd_sequence = "xy4"
    options.dd_spacing_fractions = (
        None if options.dd_spacing_fractions is None else tuple(options.dd_spacing_fractions)
    )
    qem.DDPlan(sequence=options.dd_sequence, spacing=options.dd_spacing_fractions)
    options.dd_durations_data = None
    if options.dd_durations is not None:
        if options.backend.lower() != "aer":
            raise ValueError("--dd-durations is for Aer only; hardware uses calibrated target timings")
        options.dd_durations_data = json.loads(options.dd_durations.read_text(encoding="utf-8"))
        if not np.isfinite(options.dd_durations_data["dt"]) or options.dd_durations_data["dt"] <= 0:
            raise ValueError("DD timing dt must be positive and finite")
        _dd_durations(options)
    options.readout_calibration_data = None
    if options.readout_calibration is not None:
        payload = json.loads(options.readout_calibration.read_text(encoding="utf-8"))
        if payload["backend"].lower() != options.backend.lower():
            raise ValueError("readout calibration backend does not match --backend")
        model = qem.ReadoutModel(tuple(
            qem.ReadoutBlock(tuple(block["physical_qubits"]),
                            tuple(tuple(row) for row in block["assignment"]))
            for block in payload["blocks"]
        ), source=payload["source"])
        for block in model.blocks:
            experiment8._qem_value(qem.mitigate_parity({"0": 2}, model, block.bits))
        options.readout_calibration_data = payload
    if options.reuse_zne_result is not None:
        raise ValueError(
            "--reuse-zne-result is not supported in Experiment 9 because "
            "the source must carry matching twirling/DD/readout diagnostics"
        )
    if experiment8._zne_enabled(options):
        experiment8._zne_plan(options)
    if options.classical_reference and options.optimized_classical_reference:
        raise ValueError(
            "choose either --classical-reference or "
            "--optimized-classical-reference, not both"
        )
    return options


def main(options):
    global _ACTIVE_DIAGNOSTICS
    plots_only = getattr(options, "layout_only", False) or getattr(options, "metadata_only", False)
    if not plots_only:
        if not options.no_dynamical_decoupling and options.backend.lower() == "aer" and options.dd_durations_data is None:
            raise ValueError("Aer has no calibrated durations: supply --dd-durations FILE or --no-dynamical-decoupling")
        if not options.no_readout_mitigation and options.backend.lower() != "aer" and options.readout_calibration_data is None:
            raise ValueError("hardware/fake-backend readout mitigation requires --readout-calibration FILE or --no-readout-mitigation; no circuits submitted")
        if not options.no_progress:
            tqdm_factory()  # Fail before starting work if the UI dependency is missing.
    _ACTIVE_DIAGNOSTICS = ()
    install_experiment9_hooks()
    print(
        "Experiment 9 mitigation: "
        f"ZNE={'off' if options.no_zne else 'on'}, "
        f"Pauli twirling={'off' if options.no_pauli_twirling else 'on'}, "
        f"DD={'off' if options.no_dynamical_decoupling else 'on'}, "
        f"readout={'off' if options.no_readout_mitigation else 'on'}"
    )
    try:
        experiment8.main(options)
    finally:
        restore_experiment8_hooks()
        _ACTIVE_DIAGNOSTICS = ()


if __name__ == "__main__":
    main(parse_arguments())
