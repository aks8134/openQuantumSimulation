"""Experiment 8: three-basis physical local-folding ZNE for Experiment 7.

The Experiment 7 circuit is explicitly lowered to RZ/RX/RZZ and transpiled
once.  Each transpiled circuit is then expanded into fixed-layout local-folded
ISA variants and submitted without a second optimizing transpilation.  Virtual
RZ operations, resets, measurements, delays, and barriers are never folded.
Passing ``--no-zne`` instead submits only the unfurled scale-one circuits.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gc
import json
from math import floor, pi
from pathlib import Path
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
    CompilerConfig,
    Err,
    Ok,
    Sample,
    SampleResult,
    counts_dict,
    run_sample_variants_batch_sync,
)
from Model1.Experiment5 import randomized_lie_trotter as experiment5
from Model1.Experiment7 import randomized_lie_trotter as experiment7
from mitiq_qem import combine_zne, inference, zne_plan


EXPERIMENT_NAME = "Model1/Experiment8 ZNE randomized Lie-Trotter"
METHOD_NAME = (
    "Experiment 7 randomized single-boundary dilation with post-transpilation "
    "physical local-folding zero-noise extrapolation and three-basis "
    "excitation-symmetry reconstruction"
)
FOLDING_VERSION = 1
MEASUREMENT_SCHEME_VERSION = 1
MEASUREMENT_BASES = ("Z", "X", "XY")
DEFAULT_SCALE_FACTORS = (1.0, 1.5, 2.0)
DEFAULT_ZNE_INFERENCE = "linear"
DEFAULT_FOLD_REPETITIONS = 1
DEFAULT_SEED_FOLDING = 8675309
VIRTUAL_GATE_NAMES = frozenset(("rz", "p", "u1", "phase"))
NONFOLDABLE_NAMES = frozenset(
    ("measure", "reset", "barrier", "delay", "store")
)


@dataclass(frozen=True, slots=True)
class FoldSpec:
    scale_index: int
    scale_factor: float
    repetition: int
    seed: int


@dataclass(frozen=True, slots=True)
class VariantMetadata:
    base_index: int
    time_index: int
    trajectory_index: int
    basis: str
    scale_index: int
    scale_factor: float
    fold_repetition: int
    fold_seed: int


@dataclass(frozen=True, slots=True)
class FoldDiagnostic:
    base_index: int
    scale_index: int
    scale_factor: float
    fold_repetition: int
    fold_seed: int
    base_operation_count: int
    base_depth: int
    folded_operation_count: int
    folded_depth: int
    eligible_physical_gate_count: int
    folded_gate_occurrences: int
    inserted_operation_count: int
    inserted_physical_gate_count: int
    inserted_virtual_gate_count: int
    actual_physical_scale_factor: float
    eligible_gate_counts: tuple[tuple[str, int], ...]
    excluded_unitary_gate_counts: tuple[tuple[str, int], ...]
    virtual_rz_count: int
    reset_count: int
    measurement_count: int


@dataclass(frozen=True, slots=True)
class ScaleUncertainties:
    shot: tuple[np.ndarray, ...]
    folding: tuple[np.ndarray, ...]
    trajectory: tuple[np.ndarray, ...]
    total: tuple[np.ndarray, ...]


def _safe_scale_name(value):
    return str(value).replace(".", "p").replace("-", "m")


def _zne_enabled(options):
    return not getattr(options, "no_zne", False) and len(
        options.zne_scale_factors
    ) > 1


def _validated_scale_factors(values):
    factors = tuple(map(float, values))
    if len(factors) < 2:
        raise ValueError("ZNE requires at least two scale factors")
    if not all(np.isfinite(value) and value >= 1.0 for value in factors):
        raise ValueError("ZNE scale factors must be finite and at least one")
    if not np.isclose(factors[0], 1.0):
        raise ValueError("the first ZNE scale factor must be 1.0")
    if tuple(sorted(factors)) != factors or len(set(factors)) != len(factors):
        raise ValueError("ZNE scale factors must be unique and increasing")
    return factors


def _zne_inference(options):
    return inference(
        options.zne_inference,
        order=(
            options.zne_polynomial_order
            if options.zne_inference == "polynomial"
            else None
        ),
    )


def _zne_plan(options):
    if not _zne_enabled(options):
        raise ValueError("a ZNE inference plan is unavailable in --no-zne mode")
    return zne_plan(
        tuple(options.zne_scale_factors),
        repetitions=1,
        inference=_zne_inference(options),
    )


def _variant_count_per_base(options):
    return len(options.zne_scale_factors) * options.fold_repetitions


def shots_per_variant(options):
    denominator = options.trajectories * options.fold_repetitions
    if options.shots % denominator:
        raise ValueError(
            "--shots must be divisible by --trajectories * "
            "--fold-repetitions so each submitted circuit gets equal shots"
        )
    value = options.shots // denominator
    if value < 1:
        raise ValueError(
            "--shots must be at least --trajectories * --fold-repetitions"
        )
    return value


def fold_specs(options, base_index, scale_indices=None):
    selected = (
        None if scale_indices is None else frozenset(map(int, scale_indices))
    )
    return tuple(
        FoldSpec(
            scale_index=scale_index,
            scale_factor=float(scale_factor),
            repetition=repetition,
            seed=(
                options.seed_folding
                + (
                    base_index * len(options.zne_scale_factors)
                    + scale_index
                )
                * options.fold_repetitions
                + repetition
            ),
        )
        for scale_index, scale_factor in enumerate(options.zne_scale_factors)
        if selected is None or scale_index in selected
        for repetition in range(options.fold_repetitions)
    )


def expand_metadata(base_metadata, options):
    return tuple(
        VariantMetadata(
            base_index=base_index,
            time_index=time_index,
            trajectory_index=trajectory_index,
            basis=basis,
            scale_index=spec.scale_index,
            scale_factor=spec.scale_factor,
            fold_repetition=spec.repetition,
            fold_seed=spec.seed,
        )
        for base_index, (time_index, trajectory_index, basis) in enumerate(
            base_metadata
        )
        for spec in fold_specs(options, base_index)
    )


def _target_properties(target, name, qargs):
    try:
        return target[name].get(tuple(qargs))
    except (KeyError, TypeError, AttributeError):
        return None


def _is_zero_duration(target, operation, qargs):
    properties = _target_properties(target, operation.name, qargs)
    duration = None if properties is None else properties.duration
    return duration is not None and np.isclose(float(duration), 0.0)


def _target_supports(target, operation_name, qargs):
    supported_names = frozenset(getattr(target, "operation_names", ()))
    if supported_names and operation_name not in supported_names:
        return False
    instruction_supported = getattr(target, "instruction_supported", None)
    if instruction_supported is None:
        return True
    try:
        return bool(
            instruction_supported(
                operation_name=operation_name,
                qargs=qargs,
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _inverse_sequence_if_foldable(circuit, item, target):
    from qiskit.circuit import Gate
    from qiskit.circuit.library import RZGate

    operation = item.operation
    name = operation.name.lower()
    qargs = tuple(circuit.find_bit(bit).index for bit in item.qubits)
    if (
        not isinstance(operation, Gate)
        or name in NONFOLDABLE_NAMES
        or name in VIRTUAL_GATE_NAMES
        or _is_zero_duration(target, operation, qargs)
    ):
        return None
    try:
        inverse = operation.inverse()
    except Exception:
        return None
    if _target_supports(target, inverse.name, qargs):
        return (inverse,)
    if (
        name == "sx"
        and _target_supports(target, "sx", qargs)
        and _target_supports(target, "rz", qargs)
    ):
        # IBM targets generally expose SX but not SXdg. Conjugating SX by
        # virtual pi Z rotations implements SXdg up to a global phase, so the
        # inverse retains one physical pulse and remains backend-native.
        return (RZGate(pi), operation, RZGate(pi))
    return None


def _fold_counts(scale_factor, gate_count, seed):
    if gate_count == 0:
        if np.isclose(scale_factor, 1.0):
            return np.zeros(0, dtype=int)
        raise ValueError("cannot noise-scale a circuit with no physical gates")
    complete_folds = int(floor((scale_factor - 1.0) / 2.0 + 1e-12))
    base_scale = 2 * complete_folds + 1
    residual_fraction = max(
        0.0,
        min(1.0, (scale_factor - base_scale) / 2.0),
    )
    residual_count = int(round(residual_fraction * gate_count))
    counts = np.full(gate_count, complete_folds, dtype=int)
    if residual_count:
        selected = np.random.default_rng(seed).choice(
            gate_count,
            size=residual_count,
            replace=False,
        )
        counts[selected] += 1
    return counts


def fold_native_circuit(circuit, target, spec, base_index=0):
    """Locally fold nonvirtual physical gates in a transpiled ISA circuit."""

    foldable = tuple(
        (index, _inverse_sequence_if_foldable(circuit, item, target))
        for index, item in enumerate(circuit.data)
    )
    eligible = tuple(
        (index, inverse_sequence)
        for index, inverse_sequence in foldable
        if inverse_sequence is not None
    )
    per_gate_folds = _fold_counts(
        spec.scale_factor,
        len(eligible),
        spec.seed,
    )
    folds_by_instruction = {
        instruction_index: int(per_gate_folds[position])
        for position, (instruction_index, _) in enumerate(eligible)
    }
    inverse_sequence_by_instruction = dict(eligible)

    folded = circuit.copy_empty_like()
    folded.global_phase = circuit.global_phase
    for instruction_index, item in enumerate(circuit.data):
        qargs = tuple(
            folded.qubits[circuit.find_bit(bit).index] for bit in item.qubits
        )
        cargs = tuple(
            folded.clbits[circuit.find_bit(bit).index] for bit in item.clbits
        )
        folded.append(item.operation, qargs, cargs)
        for _ in range(folds_by_instruction.get(instruction_index, 0)):
            for inverse_operation in inverse_sequence_by_instruction[
                instruction_index
            ]:
                folded.append(inverse_operation, qargs, cargs)
            folded.append(item.operation, qargs, cargs)

    eligible_names = Counter(
        circuit.data[index].operation.name for index, _ in eligible
    )
    excluded_names = Counter(
        item.operation.name
        for index, item in enumerate(circuit.data)
        if (
            getattr(item.operation, "name", "") not in NONFOLDABLE_NAMES
            and getattr(item.operation, "name", "") not in VIRTUAL_GATE_NAMES
            and foldable[index][1] is None
        )
    )
    folded_occurrences = int(np.sum(per_gate_folds))
    inserted_physical = 2 * folded_occurrences
    inserted = sum(
        folds_by_instruction[instruction_index]
        * (len(inverse_sequence) + 1)
        for instruction_index, inverse_sequence in eligible
    )
    actual_scale = (
        (len(eligible) + inserted_physical) / len(eligible)
        if eligible
        else 1.0
    )
    diagnostic = FoldDiagnostic(
        base_index=base_index,
        scale_index=spec.scale_index,
        scale_factor=spec.scale_factor,
        fold_repetition=spec.repetition,
        fold_seed=spec.seed,
        base_operation_count=int(circuit.size()),
        base_depth=int(circuit.depth()),
        folded_operation_count=int(folded.size()),
        folded_depth=int(folded.depth()),
        eligible_physical_gate_count=len(eligible),
        folded_gate_occurrences=folded_occurrences,
        inserted_operation_count=inserted,
        inserted_physical_gate_count=inserted_physical,
        inserted_virtual_gate_count=inserted - inserted_physical,
        actual_physical_scale_factor=float(actual_scale),
        eligible_gate_counts=tuple(sorted(eligible_names.items())),
        excluded_unitary_gate_counts=tuple(sorted(excluded_names.items())),
        virtual_rz_count=sum(
            item.operation.name.lower() in VIRTUAL_GATE_NAMES
            for item in circuit.data
        ),
        reset_count=sum(item.operation.name == "reset" for item in circuit.data),
        measurement_count=sum(
            item.operation.name == "measure" for item in circuit.data
        ),
    )
    return folded, diagnostic


def _chunks(values, size):
    return tuple(values[start : start + size] for start in range(0, len(values), size))


def execute_zne_circuits(
    base_circuits,
    options,
    *,
    base_index_offset=0,
    scale_indices=None,
    on_batch_complete=None,
):
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
        shots=shots_per_variant(options),
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
        if options.backend.lower() == "aer"
        else experiment5.DEFAULT_HARDWARE_BATCH_SIZE
    )
    if submitted_batch_size < variants_per_base:
        raise ValueError(
            f"--batch-size must be at least {variants_per_base}, the number "
            "of scale/fold variants generated from one base circuit"
        )
    base_batch_size = max(1, submitted_batch_size // variants_per_base)
    indexed_circuits = tuple(enumerate(base_circuits, start=base_index_offset))
    batches = _chunks(indexed_circuits, base_batch_size)
    collected = ()
    diagnostics = ()
    for batch_index, indexed_batch in enumerate(batches, start=1):
        indices = tuple(index for index, _ in indexed_batch)
        explicit_batch = tuple(
            experiment7.prepare_circuit_for_compilation(circuit)
            for _, circuit in indexed_batch
        )
        index_iterator = iter(indices)
        batch_diagnostics = []

        def transform(native, backend_target):
            base_index = next(index_iterator)
            transformed = tuple(
                fold_native_circuit(
                    native,
                    backend_target,
                    spec,
                    base_index,
                )
                for spec in fold_specs(
                    options,
                    base_index,
                    selected_scale_indices,
                )
            )
            batch_diagnostics.extend(item[1] for item in transformed)
            return tuple(item[0] for item in transformed)

        if _zne_enabled(options):
            description = (
                f"{len(indexed_batch)} base circuits -> "
                f"{len(indexed_batch) * variants_per_base} folded variants"
            )
        else:
            description = f"{len(indexed_batch)} unfurled circuits"
        print(
            f"Executing {'ZNE' if _zne_enabled(options) else 'baseline'} "
            f"batch {batch_index}/{len(batches)}: {description} "
            f"on {options.backend}"
        )
        match run_sample_variants_batch_sync(
            explicit_batch,
            target,
            compiler,
            workload,
            environment,
            transform,
        ):
            case Ok(results):
                collected = (*collected, *results)
                diagnostics = (*diagnostics, *batch_diagnostics)
                if on_batch_complete is not None:
                    on_batch_complete(collected, diagnostics)
            case Err(error):
                message = experiment5.hardware_tools._runtime_error_message(error)
                if options.backend.lower() != "aer" and "6073" in message:
                    raise RuntimeError(
                        f"{message}. Re-run with a smaller --batch-size."
                    )
                raise RuntimeError(message)
    return collected, diagnostics


def _group_counts(results, metadata, options, time_count):
    scale_count = len(options.zne_scale_factors)
    grouped = [
        [
            [
                [dict() for _ in range(options.fold_repetitions)]
                for _ in range(options.trajectories)
            ]
            for _ in range(time_count)
        ]
        for _ in range(scale_count)
    ]
    for result, item in zip(results, metadata, strict=True):
        grouped[item.scale_index][item.time_index][item.trajectory_index][
            item.fold_repetition
        ][item.basis] = counts_dict(result)
    for scale_index in range(scale_count):
        for time_index in range(time_count):
            for trajectory_index in range(options.trajectories):
                for repetition in range(options.fold_repetitions):
                    values = grouped[scale_index][time_index][trajectory_index][
                        repetition
                    ]
                    missing = tuple(
                        basis
                        for basis in MEASUREMENT_BASES
                        if basis not in values
                    )
                    if missing:
                        raise ValueError(
                            f"scale {scale_index}, time {time_index}, trajectory "
                            f"{trajectory_index}, fold {repetition} is missing "
                            f"bases {missing}"
                        )
    return grouped


def _raw_observable_arrays(grouped, options, time_count):
    scales = len(options.zne_scale_factors)
    shape = (time_count, scales, options.trajectories, options.fold_repetitions)
    populations = np.zeros((options.n_qubits, *shape), dtype=float)
    correlations = np.zeros((options.n_qubits - 1, *shape), dtype=float)
    flows = np.zeros_like(correlations)
    population_shot = np.zeros_like(populations)
    correlation_shot = np.zeros_like(correlations)
    flow_shot = np.zeros_like(flows)
    for scale_index in range(scales):
        for time_index in range(time_count):
            for trajectory_index in range(options.trajectories):
                for repetition in range(options.fold_repetitions):
                    basis_counts = grouped[scale_index][time_index][
                        trajectory_index
                    ][repetition]
                    z_shots = sum(basis_counts["Z"].values())
                    for site in range(options.n_qubits):
                        value = experiment5._expectation(
                            basis_counts["Z"],
                            (site,),
                        )
                        index = (
                            site,
                            time_index,
                            scale_index,
                            trajectory_index,
                            repetition,
                        )
                        populations[index] = 0.5 * (1.0 - value)
                        population_shot[index] = 0.25 * experiment5._pauli_variance(
                            value,
                            z_shots,
                        )
                    for bond in range(options.n_qubits - 1):
                        bits = (bond, bond + 1)
                        xx_counts = basis_counts["X"]
                        mixed_counts = basis_counts["XY"]
                        xx_value = experiment5._expectation(xx_counts, bits)
                        mixed_value = experiment5._expectation(
                            mixed_counts,
                            bits,
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
                        correlation_shot[index] = (
                            4.0
                            * experiment5._pauli_variance(
                                xx_value,
                                sum(xx_counts.values()),
                            )
                        )
                        flow_shot[index] = experiment5.J**2 * (
                            experiment5._pauli_variance(
                                mixed_value,
                                sum(mixed_counts.values()),
                            )
                        )
    return (
        (populations, correlations, flows),
        (population_shot, correlation_shot, flow_shot),
    )


def _scale_summary(values, shot_variances, trajectories, repetitions):
    mean = np.mean(values, axis=(-2, -1))
    shot_variance = np.sum(shot_variances, axis=(-2, -1)) / (
        trajectories * repetitions
    ) ** 2
    fold_means = np.mean(values, axis=-1)
    shot_variance_of_fold_means = np.sum(shot_variances, axis=-1) / repetitions**2
    if repetitions == 1:
        fold_variance_of_fold_means = np.zeros_like(fold_means)
    else:
        observed_fold_variance = np.var(values, axis=-1, ddof=1)
        mean_fold_shot_variance = np.mean(shot_variances, axis=-1)
        physical_fold_variance = np.maximum(
            observed_fold_variance - mean_fold_shot_variance,
            0.0,
        )
        fold_variance_of_fold_means = physical_fold_variance / repetitions
    folding_variance = np.sum(fold_variance_of_fold_means, axis=-1) / trajectories**2
    if trajectories == 1:
        trajectory_variance = np.zeros_like(mean)
    else:
        observed_trajectory_variance = np.var(fold_means, axis=-1, ddof=1)
        known_within_variance = np.mean(
            shot_variance_of_fold_means + fold_variance_of_fold_means,
            axis=-1,
        )
        physical_trajectory_variance = np.maximum(
            observed_trajectory_variance - known_within_variance,
            0.0,
        )
        trajectory_variance = physical_trajectory_variance / trajectories
    return (
        mean,
        np.sqrt(shot_variance),
        np.sqrt(folding_variance),
        np.sqrt(trajectory_variance),
        fold_means,
    )


def _zne_weights(plan):
    factors = np.asarray(plan.scale_factors, dtype=float)
    degree = (
        1
        if plan.inference.method == "linear"
        else len(factors) - 1
        if plan.inference.method == "richardson"
        else int(plan.inference.order)
    )
    design = np.vander(factors, degree + 1, increasing=True)
    return np.linalg.pinv(design)[0]


def _extrapolate_family(plan, scale_values):
    output = np.empty(scale_values.shape[:-1], dtype=float)
    for index in np.ndindex(output.shape):
        output[index] = combine_zne(
            plan,
            tuple(map(float, scale_values[index])),
        ).estimate
    return output


def calculate_zne_observables(results, metadata, options, time_count):
    grouped = _group_counts(results, metadata, options, time_count)
    raw_values, raw_shot_variances = _raw_observable_arrays(
        grouped,
        options,
        time_count,
    )
    summaries = tuple(
        _scale_summary(
            values,
            shot_variances,
            options.trajectories,
            options.fold_repetitions,
        )
        for values, shot_variances in zip(
            raw_values,
            raw_shot_variances,
            strict=True,
        )
    )
    scale_values = tuple(item[0] for item in summaries)
    scale_shot = tuple(item[1] for item in summaries)
    scale_folding = tuple(item[2] for item in summaries)
    scale_trajectory = tuple(item[3] for item in summaries)
    baseline_trajectory_values = tuple(item[4][:, :, 0, :] for item in summaries)
    if _zne_enabled(options):
        plan = _zne_plan(options)
        weights = _zne_weights(plan)
        measured = tuple(
            _extrapolate_family(plan, values) for values in scale_values
        )
    else:
        weights = np.asarray((1.0,))
        measured = tuple(values[..., 0] for values in scale_values)

    def propagate(errors):
        return tuple(
            np.sqrt(np.sum((values * weights) ** 2, axis=-1))
            for values in errors
        )

    shot = propagate(scale_shot)
    folding = propagate(scale_folding)
    trajectory = propagate(scale_trajectory)
    total = tuple(
        np.sqrt(shot_value**2 + fold_value**2 + trajectory_value**2)
        for shot_value, fold_value, trajectory_value in zip(
            shot,
            folding,
            trajectory,
            strict=True,
        )
    )
    scale_total = tuple(
        np.sqrt(shot_value**2 + fold_value**2 + trajectory_value**2)
        for shot_value, fold_value, trajectory_value in zip(
            scale_shot,
            scale_folding,
            scale_trajectory,
            strict=True,
        )
    )
    return {
        "measured": measured,
        "uncertainties": ScaleUncertainties(shot, folding, trajectory, total),
        "scale_values": scale_values,
        "scale_uncertainties": ScaleUncertainties(
            scale_shot,
            scale_folding,
            scale_trajectory,
            scale_total,
        ),
        "baseline_trajectory_values": baseline_trajectory_values,
        "grouped_counts": grouped,
        "weights": tuple(map(float, weights)),
    }


def output_paths(options):
    if _zne_enabled(options):
        factors = "_".join(map(_safe_scale_name, options.zne_scale_factors))
        stem = (
            f"zne_explicit_randomized_lie_trotter_"
            f"{experiment5._safe_name(options.backend)}_"
            f"N{options.n_qubits}_R{options.trajectories}_"
            f"F{options.fold_repetitions}_B3_"
            f"{options.zne_inference}_S{factors}"
        )
    else:
        stem = (
            f"no_zne_explicit_randomized_lie_trotter_"
            f"{experiment5._safe_name(options.backend)}_"
            f"N{options.n_qubits}_R{options.trajectories}_B3"
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


def default_checkpoint_path(result_path):
    return result_path.with_name(f"{result_path.stem}_checkpoint.json")


def _metadata_payload(item):
    return asdict(item)


def _folding_payload(options):
    enabled = _zne_enabled(options)
    return {
        "version": FOLDING_VERSION,
        "enabled": enabled,
        "method": (
            "post-transpilation local unitary folding"
            if enabled
            else "disabled; unfurled scale-one circuits"
        ),
        "scale_factors": list(options.zne_scale_factors),
        "fold_repetitions": options.fold_repetitions,
        "seed_folding": options.seed_folding,
        "virtual_gate_names_not_folded": sorted(VIRTUAL_GATE_NAMES),
        "nonunitary_operations_not_folded": sorted(NONFOLDABLE_NAMES),
        "physical_gate_selection": (
            "unitary transpiled ISA instructions with nonzero or unknown "
            "duration and a target-supported inverse"
        ),
        "reset_boundary_policy": "folding never crosses reset operations",
        "retranspiled_after_folding": False,
        "layout_preserved": True,
    }


def _checkpoint_configuration(options, times, metadata):
    return {
        "experiment": EXPERIMENT_NAME,
        "backend": options.backend.lower(),
        "number_of_system_qubits": options.n_qubits,
        "number_of_circuit_qubits": options.n_qubits + 2,
        "circuit_layout_version": experiment7.CIRCUIT_LAYOUT_VERSION,
        "exchange_synthesis_version": experiment7.EXCHANGE_SYNTHESIS_VERSION,
        "number_of_trajectories": options.trajectories,
        "fold_repetitions": options.fold_repetitions,
        "total_shots_per_time_basis_per_scale": options.shots,
        "shots_per_variant_circuit": shots_per_variant(options),
        "times": list(map(float, times)),
        "measurement_bases": list(MEASUREMENT_BASES),
        "measurement_scheme_version": MEASUREMENT_SCHEME_VERSION,
        "optimization_level": options.optimization_level,
        "seed_transpiler": options.seed_transpiler,
        "seed_simulator": options.seed_simulator,
        "seed_trajectories": options.seed_trajectories,
        "aer_method": options.aer_method,
        "trotter_delta_t": options.trotter_delta_t,
        "reuse_zne_result": (
            None
            if getattr(options, "reuse_zne_result", None) is None
            else str(Path(options.reuse_zne_result).resolve())
        ),
        "zne": {
            "enabled": _zne_enabled(options),
            "scale_factors": list(options.zne_scale_factors),
            "inference": (
                options.zne_inference if _zne_enabled(options) else None
            ),
            "polynomial_order": (
                options.zne_polynomial_order
                if _zne_enabled(options)
                and options.zne_inference == "polynomial"
                else None
            ),
        },
        "folding": _folding_payload(options),
        # Use JSON-native containers so a freshly-created configuration and
        # the same configuration loaded from disk compare equal on resume.
        "circuit_metadata": list(map(_metadata_payload, metadata)),
    }


def _result_record(index, result, metadata, diagnostic):
    return {
        "circuit_index": index,
        **_metadata_payload(metadata[index]),
        "counts": counts_dict(result),
        "shots": result.shots,
        "backend_name": result.backend_name,
        "job_id": result.job_id,
        "original_operation_count": result.original_gate_count,
        "original_depth": result.original_depth,
        "compiled_operation_count": result.compiled_gate_count,
        "compiled_depth": result.compiled_depth,
        "fold_diagnostic": asdict(diagnostic[index]),
    }


def _result_from_record(record):
    return SampleResult(
        counts=tuple(
            sorted((str(key), int(value)) for key, value in record["counts"].items())
        ),
        shots=int(record["shots"]),
        backend_name=str(record["backend_name"]),
        job_id=None if record.get("job_id") is None else str(record["job_id"]),
        original_gate_count=int(record["original_operation_count"]),
        original_depth=int(record["original_depth"]),
        compiled_gate_count=int(record["compiled_operation_count"]),
        compiled_depth=int(record["compiled_depth"]),
    )


def _diagnostic_from_record(record):
    values = dict(record["fold_diagnostic"])
    values["eligible_gate_counts"] = tuple(
        tuple(item) for item in values["eligible_gate_counts"]
    )
    values["excluded_unitary_gate_counts"] = tuple(
        tuple(item) for item in values["excluded_unitary_gate_counts"]
    )
    return FoldDiagnostic(**values)


def _matching_scale_index(scales, value):
    matches = tuple(
        index
        for index, scale in enumerate(scales)
        if np.isclose(float(scale), float(value), rtol=0.0, atol=1e-12)
    )
    if len(matches) > 1:
        raise ValueError(f"ambiguous matching noise scale for {value}")
    return None if not matches else matches[0]


def _scale_reuse_signature(payload):
    schedule = payload.get("evolution_schedule", {})
    folding = payload.get("folding", {})
    return {
        "backend": str(payload.get("backend", "")).lower(),
        "number_of_system_qubits": payload.get("number_of_system_qubits"),
        "number_of_circuit_qubits": payload.get("number_of_circuit_qubits"),
        "number_of_trajectories": payload.get("number_of_trajectories"),
        "fold_repetitions": payload.get("fold_repetitions"),
        "shots": payload.get("total_shots_per_time_basis_per_scale"),
        "measurement_bases": tuple(payload.get("measurement_bases", ())),
        "measurement_scheme_version": payload.get(
            "measurement_scheme_version"
        ),
        "optimization_level": payload.get("optimization_level"),
        "seed_transpiler": payload.get("seed_transpiler"),
        "seed_simulator": payload.get("seed_simulator"),
        "seed_trajectories": payload.get("seed_trajectories"),
        "seed_folding": folding.get("seed_folding"),
        "folding_version": folding.get("version"),
        "aer_method": payload.get("aer_method"),
        "trotter_delta_t": schedule.get("requested_trotter_delta_t"),
        "has_exact_reference": "exact_reference_observables" in payload,
    }


def _prospective_scale_reuse_signature(options):
    return {
        "backend": options.backend.lower(),
        "number_of_system_qubits": options.n_qubits,
        "number_of_circuit_qubits": options.n_qubits + 2,
        "number_of_trajectories": options.trajectories,
        "fold_repetitions": options.fold_repetitions,
        "shots": options.shots,
        "measurement_bases": MEASUREMENT_BASES,
        "measurement_scheme_version": MEASUREMENT_SCHEME_VERSION,
        "optimization_level": options.optimization_level,
        "seed_transpiler": options.seed_transpiler,
        "seed_simulator": options.seed_simulator,
        "seed_trajectories": options.seed_trajectories,
        "seed_folding": options.seed_folding,
        "folding_version": FOLDING_VERSION,
        "aer_method": options.aer_method,
        "trotter_delta_t": options.trotter_delta_t,
        "has_exact_reference": options.classical_reference,
    }


def load_reused_scale_variants(
    path,
    options,
    times,
    schedule,
    metadata,
):
    """Load matching raw scale variants and identify scales still to execute."""
    path = Path(path)
    source = experiment5._load_result_payload(path)
    existing = _scale_reuse_signature(source)
    requested = _prospective_scale_reuse_signature(options)
    mismatches = tuple(
        key for key, value in requested.items() if existing.get(key) != value
    )
    if mismatches:
        raise ValueError(
            "reused Experiment 8 result is incompatible in: "
            f"{', '.join(mismatches)}; no circuits were submitted"
        )
    if list(map(float, source.get("times", ()))) != list(map(float, times)):
        raise ValueError(
            "reused Experiment 8 result must contain exactly the requested "
            "saved times; no circuits were submitted"
        )
    requested_schedule = json.loads(
        json.dumps(experiment5._schedule_payload(times, schedule))
    )
    if source.get("evolution_schedule") != requested_schedule:
        raise ValueError(
            "reused Experiment 8 result has a different evolution schedule; "
            "no circuits were submitted"
        )

    source_scales = tuple(
        map(float, source.get("zne", {}).get("scale_factors", ()))
    )
    reused_scale_indices = tuple(
        index
        for index, scale in enumerate(options.zne_scale_factors)
        if _matching_scale_index(source_scales, scale) is not None
    )
    missing_scale_indices = tuple(
        index
        for index in range(len(options.zne_scale_factors))
        if index not in reused_scale_indices
    )
    if not reused_scale_indices:
        raise ValueError(
            "reused result contains none of the requested ZNE scales; "
            "no circuits were submitted"
        )
    if not missing_scale_indices:
        raise ValueError(
            "reused result already contains every requested ZNE scale; "
            "no circuits were submitted"
        )

    raw_records = source.get("raw_counts")
    circuit_records = source.get("circuit_metadata")
    if not isinstance(raw_records, list) or not isinstance(
        circuit_records,
        list,
    ) or len(raw_records) != len(circuit_records):
        raise ValueError("reused result has incomplete raw circuit records")

    source_records = {}
    for raw, circuit in zip(raw_records, circuit_records, strict=True):
        scale = float(raw["scale_factor"])
        key = (
            int(raw["time_index"]),
            int(raw["trajectory_index"]),
            str(raw["basis"]),
            scale,
            int(raw["fold_repetition"]),
        )
        if key in source_records:
            raise ValueError(f"reused result contains duplicate variant {key}")
        source_records[key] = (raw, circuit)

    adjusted_metadata = []
    reused = {}
    for index, item in enumerate(metadata):
        source_scale_index = _matching_scale_index(
            source_scales,
            item.scale_factor,
        )
        if source_scale_index is None:
            adjusted_metadata.append(item)
            continue
        source_scale = source_scales[source_scale_index]
        key = (
            item.time_index,
            item.trajectory_index,
            item.basis,
            source_scale,
            item.fold_repetition,
        )
        if key not in source_records:
            raise ValueError(f"reused result is missing raw variant {key}")
        raw, circuit = source_records[key]
        if int(raw["base_index"]) != item.base_index:
            raise ValueError(
                "reused result base-circuit ordering does not match the "
                "requested run"
            )
        adjusted_item = replace(item, fold_seed=int(raw["fold_seed"]))
        adjusted_metadata.append(adjusted_item)
        result = SampleResult(
            counts=tuple(
                sorted(
                    (str(bitstring), int(count))
                    for bitstring, count in raw["counts"].items()
                )
            ),
            shots=sum(map(int, raw["counts"].values())),
            backend_name=str(source["backend"]),
            job_id=(
                None
                if circuit.get("job_id") is None
                else str(circuit["job_id"])
            ),
            original_gate_count=int(circuit["original_operation_count"]),
            original_depth=int(circuit["original_depth"]),
            compiled_gate_count=int(circuit["compiled_operation_count"]),
            compiled_depth=int(circuit["compiled_depth"]),
        )
        diagnostic = replace(
            _diagnostic_from_record(circuit),
            base_index=item.base_index,
            scale_index=item.scale_index,
            scale_factor=item.scale_factor,
            fold_repetition=item.fold_repetition,
            fold_seed=adjusted_item.fold_seed,
        )
        reused[index] = (result, diagnostic)

    expected_reused = (
        len(times)
        * options.trajectories
        * len(MEASUREMENT_BASES)
        * options.fold_repetitions
        * len(reused_scale_indices)
    )
    if len(reused) != expected_reused:
        raise ValueError(
            f"expected {expected_reused} reusable variants but found "
            f"{len(reused)}"
        )
    return {
        "source": source,
        "source_path": path,
        "metadata": tuple(adjusted_metadata),
        "reused": reused,
        "reused_scale_indices": reused_scale_indices,
        "missing_scale_indices": missing_scale_indices,
    }


def combine_reused_and_executed_variants(
    metadata,
    reused,
    executed_metadata,
    executed_results,
    executed_diagnostics,
):
    if len(executed_metadata) != len(executed_results) or len(
        executed_results
    ) != len(executed_diagnostics):
        raise ValueError("executed scale-augmentation records are incomplete")
    executed = {
        item: (result, diagnostic)
        for item, result, diagnostic in zip(
            executed_metadata,
            executed_results,
            executed_diagnostics,
            strict=True,
        )
    }
    if len(executed) != len(executed_metadata):
        raise ValueError("executed scale-augmentation metadata are not unique")
    combined_results = []
    combined_diagnostics = []
    for index, item in enumerate(metadata):
        value = reused.get(index, executed.get(item))
        if value is None:
            raise ValueError(f"no result is available for target variant {item}")
        combined_results.append(value[0])
        combined_diagnostics.append(value[1])
    if len(reused) + len(executed) != len(metadata):
        raise ValueError("reused and executed variants do not partition the run")
    return tuple(combined_results), tuple(combined_diagnostics)


def save_checkpoint(
    path,
    options,
    times,
    schedule,
    results,
    metadata,
    diagnostics,
    status="partial",
):
    if len(results) != len(diagnostics) or len(results) > len(metadata):
        raise ValueError("checkpoint result/diagnostic lengths are inconsistent")
    experiment5._atomic_write_json(
        path,
        {
            "schema_version": 1,
            "experiment": EXPERIMENT_NAME,
            "status": status,
            "completed_circuit_count": len(results),
            "total_circuit_count": len(metadata),
            "configuration": _checkpoint_configuration(options, times, metadata),
            "physics": experiment5._schedule_payload(times, schedule),
            "exchange_decomposition": experiment7.decomposition_payload(
                options.n_qubits
            ),
            "results": tuple(
                _result_record(index, result, metadata, diagnostics)
                for index, result in enumerate(results)
            ),
        },
    )


def load_checkpoint(path, options, times, metadata):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("checkpoint has an unsupported schema version")
    if payload.get("configuration") != _checkpoint_configuration(
        options,
        times,
        metadata,
    ):
        raise ValueError("checkpoint configuration does not match this ZNE run")
    records = payload.get("results")
    if not isinstance(records, list) or len(records) > len(metadata):
        raise ValueError("checkpoint has an invalid result list")
    for index, record in enumerate(records):
        expected = _metadata_payload(metadata[index])
        actual = {key: record.get(key) for key in expected}
        if record.get("circuit_index") != index or actual != expected:
            raise ValueError("checkpoint results are not a consecutive circuit prefix")
    return (
        tuple(map(_result_from_record, records)),
        tuple(map(_diagnostic_from_record, records)),
    )


def _family_payload(values):
    return {
        "populations": values[0].tolist(),
        "xy_correlations": values[1].tolist(),
        "excitation_flows": values[2].tolist(),
    }


def _family_arrays(payload):
    return tuple(
        np.asarray(payload[name], dtype=float)
        for name in ("populations", "xy_correlations", "excitation_flows")
    )


def _metric_summary(results, diagnostics, options):
    return tuple(
        {
            "scale_index": scale_index,
            "requested_scale_factor": float(scale),
            "mean_actual_physical_scale_factor": float(
                np.mean(
                    [
                        value.actual_physical_scale_factor
                        for value in diagnostics
                        if value.scale_index == scale_index
                    ]
                )
            ),
            "mean_original_explicit_operation_count": float(
                np.mean(
                    [
                        result.original_gate_count
                        for result, value in zip(
                            results,
                            diagnostics,
                            strict=True,
                        )
                        if value.scale_index == scale_index
                    ]
                )
            ),
            "mean_original_explicit_depth": float(
                np.mean(
                    [
                        result.original_depth
                        for result, value in zip(
                            results,
                            diagnostics,
                            strict=True,
                        )
                        if value.scale_index == scale_index
                    ]
                )
            ),
            "mean_base_transpiled_operation_count": float(
                np.mean(
                    [
                        value.base_operation_count
                        for value in diagnostics
                        if value.scale_index == scale_index
                    ]
                )
            ),
            "mean_folded_operation_count": float(
                np.mean(
                    [
                        value.folded_operation_count
                        for value in diagnostics
                        if value.scale_index == scale_index
                    ]
                )
            ),
            "mean_base_transpiled_depth": float(
                np.mean(
                    [
                        value.base_depth
                        for value in diagnostics
                        if value.scale_index == scale_index
                    ]
                )
            ),
            "mean_folded_depth": float(
                np.mean(
                    [
                        value.folded_depth
                        for value in diagnostics
                        if value.scale_index == scale_index
                    ]
                )
            ),
            "mean_eligible_physical_gate_count": float(
                np.mean(
                    [
                        value.eligible_physical_gate_count
                        for value in diagnostics
                        if value.scale_index == scale_index
                    ]
                )
            ),
        }
        for scale_index, scale in enumerate(options.zne_scale_factors)
    )


def _compatibility_signature(payload):
    schedule = payload.get("evolution_schedule", {})
    zne = payload.get("zne", {})
    folding = payload.get("folding", {})
    scale_factors = tuple(zne.get("scale_factors", ()))
    return {
        "experiment": payload.get("experiment"),
        "backend": str(payload.get("backend", "")).lower(),
        "number_of_system_qubits": payload.get("number_of_system_qubits"),
        "number_of_circuit_qubits": payload.get("number_of_circuit_qubits"),
        "number_of_trajectories": payload.get("number_of_trajectories"),
        "fold_repetitions": payload.get("fold_repetitions"),
        "shots": payload.get("total_shots_per_time_basis_per_scale"),
        "measurement_bases": tuple(payload.get("measurement_bases", ())),
        "measurement_scheme_version": payload.get(
            "measurement_scheme_version"
        ),
        "optimization_level": payload.get("optimization_level"),
        "seed_transpiler": payload.get("seed_transpiler"),
        "seed_simulator": payload.get("seed_simulator"),
        "seed_trajectories": payload.get("seed_trajectories"),
        "aer_method": payload.get("aer_method"),
        "trotter_delta_t": schedule.get("requested_trotter_delta_t"),
        "zne_enabled": zne.get("enabled", len(scale_factors) > 1),
        "scale_factors": scale_factors,
        "zne_inference": zne.get("inference"),
        "polynomial_order": zne.get("polynomial_order"),
        "seed_folding": folding.get("seed_folding"),
        "folding_version": folding.get("version"),
        "has_exact_reference": "exact_reference_observables" in payload,
    }


def _prospective_signature(options):
    return {
        "experiment": EXPERIMENT_NAME,
        "backend": options.backend.lower(),
        "number_of_system_qubits": options.n_qubits,
        "number_of_circuit_qubits": options.n_qubits + 2,
        "number_of_trajectories": options.trajectories,
        "fold_repetitions": options.fold_repetitions,
        "shots": options.shots,
        "measurement_bases": MEASUREMENT_BASES,
        "measurement_scheme_version": MEASUREMENT_SCHEME_VERSION,
        "optimization_level": options.optimization_level,
        "seed_transpiler": options.seed_transpiler,
        "seed_simulator": options.seed_simulator,
        "seed_trajectories": options.seed_trajectories,
        "aer_method": options.aer_method,
        "trotter_delta_t": options.trotter_delta_t,
        "zne_enabled": _zne_enabled(options),
        "scale_factors": tuple(options.zne_scale_factors),
        "zne_inference": (
            options.zne_inference if _zne_enabled(options) else None
        ),
        "polynomial_order": (
            options.zne_polynomial_order
            if _zne_enabled(options)
            and options.zne_inference == "polynomial"
            else None
        ),
        "seed_folding": options.seed_folding,
        "folding_version": FOLDING_VERSION,
        "has_exact_reference": options.classical_reference,
    }


def validate_existing_result(path, options):
    path = Path(path)
    if not path.exists():
        return None
    payload = experiment5._load_result_payload(path)
    existing = _compatibility_signature(payload)
    requested = _prospective_signature(options)
    mismatches = tuple(key for key in requested if existing.get(key) != requested[key])
    if mismatches:
        raise ValueError(
            "existing Experiment 8 result is incompatible in: "
            f"{', '.join(mismatches)}. Use matching options or a different "
            "--output-directory; no circuits were submitted."
        )
    return payload


def _merge_results(existing, new):
    merged = experiment5.merge_result_payloads(existing, new)
    _, sources = experiment5._merged_time_sources(existing["times"], new["times"])
    merged["standard_errors"]["folding"] = experiment5._merge_family_payload(
        existing["standard_errors"]["folding"],
        new["standard_errors"]["folding"],
        sources,
    )
    merged["scale_observables"] = experiment5._merge_family_payload(
        existing["scale_observables"],
        new["scale_observables"],
        sources,
    )
    merged["scale_standard_errors"] = {
        component: experiment5._merge_family_payload(
            existing["scale_standard_errors"][component],
            new["scale_standard_errors"][component],
            sources,
        )
        for component in ("shot", "folding", "trajectory", "total")
    }
    return merged


def save_results(
    path,
    options,
    times,
    schedule,
    results,
    metadata,
    diagnostics,
    analysis,
    exact,
    paths,
):
    schedule_payload = experiment5._schedule_payload(times, schedule)
    job_ids = tuple(
        dict.fromkeys(
            result.job_id for result in results if result.job_id is not None
        )
    )
    grouped = analysis["grouped_counts"]
    raw_counts = tuple(
        {
            **_metadata_payload(item),
            "time": float(times[item.time_index]),
            "counts": grouped[item.scale_index][item.time_index][
                item.trajectory_index
            ][item.fold_repetition][item.basis],
        }
        for item in metadata
    )
    payload = {
        "experiment": EXPERIMENT_NAME,
        "method": (
            METHOD_NAME
            if _zne_enabled(options)
            else (
                "Experiment 7 randomized single-boundary dilation without "
                "zero-noise extrapolation and with three-basis "
                "excitation-symmetry reconstruction"
            )
        ),
        "backend": options.backend,
        "number_of_system_qubits": options.n_qubits,
        "number_of_circuit_qubits": options.n_qubits + 2,
        "circuit_layout_version": experiment7.CIRCUIT_LAYOUT_VERSION,
        "exchange_synthesis_version": experiment7.EXCHANGE_SYNTHESIS_VERSION,
        "ancilla_strategy": experiment5.ANCILLA_STRATEGY,
        "system_qubits_by_site": experiment5.system_qubits_by_site(options.n_qubits),
        "boundary_ancilla_qubits": {
            "L": experiment5.boundary_ancilla_qubits(options.n_qubits)[
                experiment5.LEFT_BOUNDARY
            ],
            "R": experiment5.boundary_ancilla_qubits(options.n_qubits)[
                experiment5.RIGHT_BOUNDARY
            ],
        },
        "number_of_trajectories": options.trajectories,
        "fold_repetitions": options.fold_repetitions,
        # Retain the shared Experiment 5 field as well as the explicit ZNE
        # field so archive merging checks shot compatibility correctly.
        "total_shots_per_time_basis": options.shots,
        "total_shots_per_time_basis_per_scale": options.shots,
        "total_shots_across_zne_scales_per_time_basis": (
            options.shots * len(options.zne_scale_factors)
        ),
        "total_shots_across_bases_and_zne_scales_per_time": (
            options.shots
            * len(MEASUREMENT_BASES)
            * len(options.zne_scale_factors)
        ),
        "shots_per_variant_circuit": shots_per_variant(options),
        "measurement_bases": MEASUREMENT_BASES,
        "measurement_scheme_version": MEASUREMENT_SCHEME_VERSION,
        "measurement_reconstruction": {
            "version": MEASUREMENT_SCHEME_VERSION,
            "scheme": "vacuum-plus-one-excitation three-basis",
            "bases": list(MEASUREMENT_BASES),
            "assumption": (
                "the density operator is block diagonal in excitation number "
                "and supported on the vacuum-plus-one-excitation subspace"
            ),
            "xy_correlation": "<XX+YY> = 2 <XX>",
            "flow_even_bond": "J/2 <YX-XY> = -J <XY>",
            "flow_odd_bond": "J/2 <YX-XY> = J <YX>",
            "alternating_basis": (
                "site n uses X for even n and Y for odd n"
            ),
        },
        "uses_mid_circuit_measurement": False,
        "uses_reset": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "times": list(map(float, times)),
        "dt": schedule_payload["uniform_substep_dt"],
        "system_exchange_angle": schedule_payload["uniform_system_exchange_angle"],
        "randomized_jump_angle": schedule_payload["uniform_randomized_jump_angle"],
        "evolution_schedule": schedule_payload,
        "optimization_level": options.optimization_level,
        "seed_transpiler": options.seed_transpiler,
        "seed_simulator": options.seed_simulator,
        "seed_trajectories": options.seed_trajectories,
        "aer_method": options.aer_method,
        "job_ids": job_ids,
        "zne": {
            "enabled": _zne_enabled(options),
            "scale_factors": list(options.zne_scale_factors),
            "inference": (
                options.zne_inference if _zne_enabled(options) else None
            ),
            "polynomial_order": (
                options.zne_polynomial_order
                if _zne_enabled(options)
                and options.zne_inference == "polynomial"
                else None
            ),
            "extrapolation_weights": list(analysis["weights"]),
            "noise_zero_point": 0.0 if _zne_enabled(options) else None,
        },
        "folding": _folding_payload(options),
        "exchange_decomposition": experiment7.decomposition_payload(
            options.n_qubits
        ),
        "transpilation_summary": _metric_summary(
            results,
            diagnostics,
            options,
        ),
        "observables": _family_payload(analysis["measured"]),
        "scale_observables": _family_payload(analysis["scale_values"]),
        "trajectory_observables": _family_payload(
            analysis["baseline_trajectory_values"]
        ),
        "standard_errors": {
            "shot": _family_payload(analysis["uncertainties"].shot),
            "folding": _family_payload(analysis["uncertainties"].folding),
            "trajectory": _family_payload(analysis["uncertainties"].trajectory),
            "total": _family_payload(analysis["uncertainties"].total),
        },
        "scale_standard_errors": {
            "shot": _family_payload(analysis["scale_uncertainties"].shot),
            "folding": _family_payload(analysis["scale_uncertainties"].folding),
            "trajectory": _family_payload(
                analysis["scale_uncertainties"].trajectory
            ),
            "total": _family_payload(analysis["scale_uncertainties"].total),
        },
        "circuit_metadata": tuple(
            {
                **_metadata_payload(item),
                "time": float(times[item.time_index]),
                "job_id": results[index].job_id,
                "original_operation_count": results[index].original_gate_count,
                "original_depth": results[index].original_depth,
                "compiled_operation_count": results[index].compiled_gate_count,
                "compiled_depth": results[index].compiled_depth,
                "fold_diagnostic": asdict(diagnostics[index]),
            }
            for index, item in enumerate(metadata)
        ),
        "raw_counts": raw_counts,
        "transpiled_layout": {
            "backend": options.backend,
            "basis": "Z",
            "trajectory_index": 0,
            "saved_time": float(times[-1]),
            "view": "physical",
            "figure": paths["layout_figure"].name,
        },
        "figure_generation_status": "pending",
    }
    if getattr(options, "reuse_zne_result", None) is not None:
        payload["scale_augmentation"] = {
            "source_result": str(Path(options.reuse_zne_result).resolve()),
            "reused_scale_factors": list(options.reused_scale_factors),
            "executed_scale_factors": list(options.executed_scale_factors),
            "caveat": (
                "reused and newly executed scales may sample different "
                "hardware calibration windows"
            ),
        }
    if exact is not None:
        payload["exact_reference_observables"] = _family_payload(exact)
        payload["aggregate_observable_error_vs_exact"] = (
            experiment5.chain_tools.aggregate_observable_error(
                analysis["measured"],
                exact,
            ).tolist()
        )
        payload["classical_reference_metadata"] = (
            {
                "method": experiment7.OPTIMIZED_REFERENCE_METHOD,
                "exact_for_this_model": True,
                "reduced_basis": "vacuum plus one excitation",
                "hilbert_space_dimension": options.n_qubits + 1,
                "liouville_space_dimension": (options.n_qubits + 1) ** 2,
            }
            if options.optimized_classical_reference
            else {
                "method": "full-Hilbert-space exact Lindblad evolution",
                "exact_for_this_model": True,
                "hilbert_space_dimension": 2**options.n_qubits,
                "liouville_space_dimension": 4**options.n_qubits,
            }
        )
    path = Path(path)
    if path.exists():
        payload = _merge_results(experiment5._load_result_payload(path), payload)
    else:
        payload["archive"] = {
            "schema_version": 1,
            "merge_policy": "new run replaces matching times",
            "saved_time_count": len(payload["times"]),
            "latest_run_times": payload["times"],
        }
        payload["run_history"] = [experiment5._run_history_record(payload)]
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
    exact_values = (None, None, None) if exact is None else exact
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
    families = (
        (measured[0], uncertainties.total[0], "Site populations", "Population", lambda i: rf"$n_{i}$", exact_values[0]),
        (measured[1], uncertainties.total[1], "Nearest-neighbor XY correlations", r"$\langle X_nX_{n+1}+Y_nY_{n+1}\rangle$", lambda i: f"bond {i}-{i + 1}", exact_values[1]),
        (measured[2], uncertainties.total[2], "Excitation flow", "Flow", lambda i: f"bond {i}-{i + 1}", exact_values[2]),
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
        return np.maximum.reduce(tuple(np.max(value, axis=0) for value in values))

    diagnostic.plot(times, family_max(uncertainties.shot), label="max propagated shot s.e.")
    diagnostic.plot(
        times,
        family_max(uncertainties.folding),
        label="max propagated folding s.e.",
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
        "ZNE uncertainty decomposition"
        if zne_enabled
        else "Baseline uncertainty decomposition"
    )
    diagnostic.set_xlabel("Time")
    diagnostic.set_ylabel("Standard error")
    diagnostic.grid(alpha=0.25)
    diagnostic.legend(fontsize="small")
    title = (
        f"Experiment 8 physical local-folding ZNE on {backend} "
        f"(R={trajectories}, F={fold_repetitions})"
        if zne_enabled
        else f"Experiment 8 unfurled baseline on {backend} (R={trajectories})"
    )
    figure.suptitle(title, fontsize=15)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_zne_scaling(
    scale_factors,
    scale_values,
    measured,
    times,
    weights,
    output_path,
):
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    titles = ("Populations", "XY correlations", "Excitation flows")
    ylabels = ("Population", "Correlation", "Flow")
    final_index = len(times) - 1
    for axis, values, extrapolated, title, ylabel in zip(
        axes,
        scale_values,
        measured,
        titles,
        ylabels,
        strict=True,
    ):
        for series_index in range(values.shape[0]):
            scale_series = values[series_index, final_index]
            label = (
                f"site {series_index}"
                if title == "Populations"
                else f"bond {series_index}-{series_index + 1}"
            )
            line = axis.plot(
                scale_factors,
                scale_series,
                marker="o",
                label=label,
            )[0]
            axis.scatter(
                (0.0,),
                (extrapolated[series_index, final_index],),
                marker="*",
                s=110,
                color=line.get_color(),
            )
            axis.plot(
                (0.0, *scale_factors),
                (extrapolated[series_index, final_index], *scale_series),
                linestyle=":",
                alpha=0.35,
                color=line.get_color(),
            )
        axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.4)
        axis.set_title(title)
        axis.set_xlabel("Requested physical noise scale")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")
    figure.suptitle(
        f"Final saved time t={float(times[-1]):g}: raw scales and "
        f"zero-noise extrapolates (weights={np.round(weights, 4)})",
        fontsize=14,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_transpilation_metrics(summary, output_path, *, zne_enabled=True):
    factors = np.asarray(
        [value["requested_scale_factor"] for value in summary],
        dtype=float,
    )
    width = 0.25
    positions = np.arange(len(factors))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for axis, original_key, base_key, folded_key, title in (
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
    ):
        original = [value[original_key] for value in summary]
        base = [value[base_key] for value in summary]
        folded = [value[folded_key] for value in summary]
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
            folded,
            width,
            label="post-fold ISA" if zne_enabled else "submitted ISA",
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
        (
            "Experiment 8: fixed-layout post-transpilation folding metrics"
            if zne_enabled
            else "Experiment 8: unfurled baseline transpilation metrics"
        ),
        fontsize=14,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _representative_full_circuit(circuits, metadata, final_time_index):
    return next(
        circuit
        for circuit, (time_index, trajectory_index, basis) in zip(
            circuits,
            metadata,
            strict=True,
        )
        if time_index == final_time_index
        and trajectory_index == 0
        and basis == "Z"
    )


def _print_summary(options, schedule, payload, results, paths, checkpoint_path):
    job_ids = tuple(
        dict.fromkeys(
            result.job_id for result in results if result.job_id is not None
        )
    )
    print(f"Backend: {options.backend}")
    print(f"System/circuit qubits: {options.n_qubits}/{options.n_qubits + 2}")
    print(f"Randomized trajectories R: {options.trajectories}")
    print(f"Measurement bases: {list(MEASUREMENT_BASES)}")
    if _zne_enabled(options):
        print(f"Fold realizations F: {options.fold_repetitions}")
        print(f"ZNE scale factors: {list(options.zne_scale_factors)}")
        print(f"ZNE inference: {options.zne_inference}")
        print(f"Extrapolation weights: {payload['zne']['extrapolation_weights']}")
        print(f"Shots per folded variant: {shots_per_variant(options)}")
    else:
        print("ZNE: disabled (unfurled scale-one baseline)")
        print(f"Shots per trajectory circuit: {shots_per_variant(options)}")
    print(f"Archive saved times: {payload['times']}")
    print(f"Total Trotter substeps: {len(schedule.substep_dts)}")
    print(
        "Folded operation-count range: "
        f"{min(result.compiled_gate_count for result in results)}--"
        f"{max(result.compiled_gate_count for result in results)}"
    )
    print(
        "Folded depth range: "
        f"{min(result.compiled_depth for result in results)}--"
        f"{max(result.compiled_depth for result in results)}"
    )
    if job_ids:
        print(f"Provider job IDs: {', '.join(job_ids)}")
    artifact_names = ["results_figure"]
    if _zne_enabled(options):
        artifact_names.append("scaling_figure")
    artifact_names.extend(("metrics_figure", "layout_figure", "result"))
    for name in artifact_names:
        print(f"Saved {name.replace('_', ' ')}: {paths[name]}")
    print(f"Saved checkpoint: {checkpoint_path}")


def main(options):
    if options.metadata_only:
        raise ValueError("Experiment 8 does not yet support --metadata-only")
    if options.metadata_file is not None:
        raise ValueError("--metadata-file is not supported by Experiment 8")
    if options.layout_only and options.resume:
        raise ValueError("--layout-only cannot be combined with --resume")
    if options.layout_only and options.reuse_zne_result is not None:
        raise ValueError(
            "--layout-only cannot be combined with --reuse-zne-result"
        )
    if (
        options.backend.lower() == "fake_fez"
        and not options.layout_only
    ):
        raise ValueError("--backend fake_fez requires --layout-only")
    shots_per_variant(options)
    if _zne_enabled(options):
        _zne_plan(options)
    experiment7.install_experiment7_implementation()

    optimized_reference = options.optimized_classical_reference
    if optimized_reference and options.classical_reference:
        raise ValueError(
            "choose either --classical-reference or "
            "--optimized-classical-reference, not both"
        )
    requested_full_reference = options.classical_reference
    experiment5.calculate_exact_reference = (
        experiment7.calculate_optimized_classical_reference
        if optimized_reference
        else experiment7._BASE_CALCULATE_EXACT_REFERENCE
    )
    if optimized_reference:
        options.classical_reference = True
        print(
            "Using exact vacuum-plus-single-excitation classical reference "
            f"(Hilbert dimension {options.n_qubits + 1})"
        )

    try:
        times = experiment5.hardware_tools.time_grid_from_options(options)
        base_circuits, base_metadata, schedule = experiment7.build_sample_circuits(
            times,
            options.n_qubits,
            options.trajectories,
            options.seed_trajectories,
            options.trotter_delta_t,
            measurement_bases=MEASUREMENT_BASES,
        )
        metadata = expand_metadata(base_metadata, options)
        reuse = None
        if options.reuse_zne_result is not None:
            reuse = load_reused_scale_variants(
                options.reuse_zne_result,
                options,
                times,
                schedule,
                metadata,
            )
            metadata = reuse["metadata"]
            execution_scale_indices = reuse["missing_scale_indices"]
            options.reused_scale_factors = tuple(
                options.zne_scale_factors[index]
                for index in reuse["reused_scale_indices"]
            )
            options.executed_scale_factors = tuple(
                options.zne_scale_factors[index]
                for index in execution_scale_indices
            )
            print(
                "Reusing raw variants at scales "
                f"{list(options.reused_scale_factors)} from "
                f"{reuse['source_path']}"
            )
            print(
                "Only missing scales will be submitted: "
                f"{list(options.executed_scale_factors)}"
            )
        else:
            execution_scale_indices = tuple(
                range(len(options.zne_scale_factors))
            )
        execution_metadata = tuple(
            item
            for item in metadata
            if item.scale_index in execution_scale_indices
        )
        variants_per_base = (
            len(execution_scale_indices) * options.fold_repetitions
        )
        if _zne_enabled(options):
            print(
                f"Built {len(base_circuits)} Experiment 7 base circuits; "
                f"the target analysis contains {len(metadata)} variants "
                f"({len(options.zne_scale_factors)} scales x "
                f"{options.fold_repetitions} folds per base), of which "
                f"{len(execution_metadata)} will be executed"
            )
        else:
            print(
                f"Built {len(base_circuits)} unfurled Experiment 7 circuits "
                "for the no-ZNE baseline"
            )
        print(
            "Three-basis reconstruction: "
            f"{', '.join(MEASUREMENT_BASES)}"
        )
        if _zne_enabled(options):
            print(
                f"Shot allocation per scale/time/basis: {options.shots} "
                f"total = {shots_per_variant(options)} per "
                "trajectory/fold circuit"
            )
        else:
            print(
                f"Shot allocation per time/basis: {options.shots} total = "
                f"{shots_per_variant(options)} per trajectory circuit"
            )
        paths = output_paths(options)
        representative_dt = schedule.substep_dts[0] if schedule.substep_dts else None
        circuit_plots_enabled = not options.no_circuit_plots
        one_step_circuits = (
            tuple(
                experiment7.build_one_step_circuit(
                    options.n_qubits,
                    representative_dt,
                    boundary,
                )
                for boundary in (experiment5.LEFT_BOUNDARY, experiment5.RIGHT_BOUNDARY)
            )
            if representative_dt is not None and circuit_plots_enabled
            else None
        )
        full_circuit = _representative_full_circuit(
            base_circuits,
            base_metadata,
            len(times) - 1,
        )
        if options.layout_only:
            experiment5.plot_transpiled_circuit_layout(
                full_circuit,
                options,
                paths["layout_figure"],
                one_step_circuits=one_step_circuits,
                include_circuit_panels=circuit_plots_enabled,
            )
            print(f"Saved transpiled layout: {paths['layout_figure']}")
            print("Sampler jobs submitted: 0 (layout-only mode)")
            return

        existing = validate_existing_result(paths["result"], options)
        if existing is not None:
            print(
                f"Existing archive contains {len(existing['times'])} saved "
                "time(s); new values will replace matches and append new times"
            )
        checkpoint_path = (
            default_checkpoint_path(paths["result"])
            if options.checkpoint_file is None
            else options.checkpoint_file
        )
        if options.resume:
            completed_results, completed_diagnostics = load_checkpoint(
                checkpoint_path,
                options,
                times,
                execution_metadata,
            )
            if len(completed_results) % variants_per_base:
                raise ValueError(
                    "checkpoint stops inside a base circuit's ZNE variant group"
                )
            print(
                f"Resuming {len(completed_results)}/"
                f"{len(execution_metadata)} submitted variants from "
                f"{checkpoint_path}"
            )
        else:
            completed_results, completed_diagnostics = (), ()
            save_checkpoint(
                checkpoint_path,
                options,
                times,
                schedule,
                (),
                execution_metadata,
                (),
            )
            print(f"Checkpoint: {checkpoint_path}")

        completed_base_count = len(completed_results) // variants_per_base

        def checkpoint_new(new_results, new_diagnostics):
            all_results = (*completed_results, *new_results)
            all_diagnostics = (*completed_diagnostics, *new_diagnostics)
            save_checkpoint(
                checkpoint_path,
                options,
                times,
                schedule,
                all_results,
                execution_metadata,
                all_diagnostics,
            )
            print(
                f"Checkpointed {len(all_results)}/"
                f"{len(execution_metadata)} submitted variants: "
                f"{checkpoint_path}"
            )

        remaining_base_circuits = base_circuits[completed_base_count:]
        if remaining_base_circuits:
            new_results, new_diagnostics = execute_zne_circuits(
                remaining_base_circuits,
                options,
                base_index_offset=completed_base_count,
                scale_indices=execution_scale_indices,
                on_batch_complete=checkpoint_new,
            )
        else:
            new_results, new_diagnostics = (), ()
        executed_results = (*completed_results, *new_results)
        executed_diagnostics = (*completed_diagnostics, *new_diagnostics)
        if len(executed_results) != len(execution_metadata):
            raise RuntimeError(
                "Experiment 8 execution returned an incomplete result set"
            )
        if reuse is None:
            results = executed_results
            diagnostics = executed_diagnostics
        else:
            results, diagnostics = combine_reused_and_executed_variants(
                metadata,
                reuse["reused"],
                execution_metadata,
                executed_results,
                executed_diagnostics,
            )
        analysis = calculate_zne_observables(
            results,
            metadata,
            options,
            len(times),
        )

        del base_circuits, remaining_base_circuits
        gc.collect()
        print(
            "Released all nonrepresentative logical circuit references after "
            "sampling and observable reconstruction"
        )

        exact = None
        if options.classical_reference:
            print("Calculating exact Lindblad reference")
            exact = experiment5.calculate_exact_reference(times, options.n_qubits)

        combined_payload = save_results(
            paths["result"],
            options,
            times,
            schedule,
            results,
            metadata,
            diagnostics,
            analysis,
            exact,
            paths,
        )
        print(f"Saved data before figure generation: {paths['result']}")
        summary = combined_payload["transpilation_summary"]
        plot_transpilation_metrics(
            summary,
            paths["metrics_figure"],
            zne_enabled=_zne_enabled(options),
        )
        experiment5.plot_transpiled_circuit_layout(
            full_circuit,
            options,
            paths["layout_figure"],
            one_step_circuits=one_step_circuits,
            include_circuit_panels=circuit_plots_enabled,
        )
        if circuit_plots_enabled and representative_dt is not None:
            experiment5.plot_one_step_circuits(
                options.n_qubits,
                representative_dt,
                paths["left_circuit_figure"],
                paths["right_circuit_figure"],
            )
            experiment7.plot_exchange_decomposition(paths["decomposition_figure"])

        combined_measured = _family_arrays(combined_payload["observables"])
        combined_uncertainties = ScaleUncertainties(
            shot=_family_arrays(combined_payload["standard_errors"]["shot"]),
            folding=_family_arrays(
                combined_payload["standard_errors"]["folding"]
            ),
            trajectory=_family_arrays(
                combined_payload["standard_errors"]["trajectory"]
            ),
            total=_family_arrays(combined_payload["standard_errors"]["total"]),
        )
        combined_exact = (
            _family_arrays(combined_payload["exact_reference_observables"])
            if "exact_reference_observables" in combined_payload
            else None
        )
        combined_times = np.asarray(combined_payload["times"], dtype=float)
        plot_results(
            combined_times,
            options.backend,
            options.trajectories,
            options.fold_repetitions,
            combined_measured,
            combined_uncertainties,
            paths["results_figure"],
            combined_exact,
            zne_enabled=_zne_enabled(options),
        )
        if _zne_enabled(options):
            plot_zne_scaling(
                tuple(options.zne_scale_factors),
                _family_arrays(combined_payload["scale_observables"]),
                combined_measured,
                combined_times,
                combined_payload["zne"]["extrapolation_weights"],
                paths["scaling_figure"],
            )
        combined_payload["figure_generation_status"] = "complete"
        experiment5._atomic_write_json(paths["result"], combined_payload)
        save_checkpoint(
            checkpoint_path,
            options,
            times,
            schedule,
            executed_results,
            execution_metadata,
            executed_diagnostics,
            status="complete",
        )
        _print_summary(
            options,
            schedule,
            combined_payload,
            results,
            paths,
            checkpoint_path,
        )
    finally:
        experiment5.calculate_exact_reference = (
            experiment7._BASE_CALCULATE_EXACT_REFERENCE
        )
        options.classical_reference = requested_full_reference


def parse_arguments(arguments=None):
    def configure_parser(parser):
        parser.add_argument(
            "--optimized-classical-reference",
            action="store_true",
            help=(
                "use Experiment 7's exact vacuum-plus-single-excitation "
                "reference"
            ),
        )
        parser.add_argument(
            "--no-zne",
            action="store_true",
            help=(
                "submit only the unfurled scale-one circuits; disable local "
                "folding and zero-noise extrapolation"
            ),
        )
        parser.add_argument(
            "--reuse-zne-result",
            type=Path,
            help=(
                "reuse matching raw scales from an existing Experiment 8 "
                "result and submit only requested scales absent from it"
            ),
        )
        parser.add_argument(
            "--zne-scale-factors",
            nargs="+",
            type=float,
            default=DEFAULT_SCALE_FACTORS,
            metavar="LAMBDA",
            help="increasing physical noise scales beginning with 1",
        )
        parser.add_argument(
            "--zne-inference",
            choices=("linear", "richardson", "polynomial"),
            default=DEFAULT_ZNE_INFERENCE,
        )
        parser.add_argument(
            "--zne-polynomial-order",
            type=experiment5.positive_integer,
            default=2,
            help="polynomial degree when --zne-inference polynomial is used",
        )
        parser.add_argument(
            "--fold-repetitions",
            type=experiment5.positive_integer,
            default=DEFAULT_FOLD_REPETITIONS,
            help="independent random local-fold realizations per noise scale",
        )
        parser.add_argument(
            "--seed-folding",
            type=int,
            default=DEFAULT_SEED_FOLDING,
        )

    options = experiment5.parse_arguments(
        arguments,
        default_trajectories=experiment7.DEFAULT_TRAJECTORIES,
        default_trotter_delta_t=experiment7.DEFAULT_TROTTER_DELTA_T,
        default_t_final=experiment7.DEFAULT_T_FINAL,
        default_time_points=experiment7.DEFAULT_TIME_POINTS,
        description=(
            "Run Experiment 8: post-transpilation physical local-folding ZNE "
            "on the Experiment 7 randomized Lie-Trotter circuits. Non-aer "
            "backends submit real QPU work."
        ),
        configure_parser=configure_parser,
    )
    if options.no_zne:
        if options.reuse_zne_result is not None:
            raise ValueError("--no-zne cannot be combined with --reuse-zne-result")
        if options.fold_repetitions != 1:
            raise ValueError(
                "--no-zne requires --fold-repetitions 1 because no fold "
                "realizations are generated"
            )
        options.zne_scale_factors = (1.0,)
    else:
        options.zne_scale_factors = _validated_scale_factors(
            options.zne_scale_factors
        )
    if options.seed_folding < 0:
        raise ValueError("--seed-folding must be non-negative")
    if options.reuse_zne_result is not None and not options.reuse_zne_result.is_file():
        raise ValueError(
            f"--reuse-zne-result does not exist: {options.reuse_zne_result}"
        )
    if _zne_enabled(options):
        _zne_plan(options)
    if options.classical_reference and options.optimized_classical_reference:
        raise ValueError(
            "choose either --classical-reference or "
            "--optimized-classical-reference, not both"
        )
    return options


if __name__ == "__main__":
    main(parse_arguments())
