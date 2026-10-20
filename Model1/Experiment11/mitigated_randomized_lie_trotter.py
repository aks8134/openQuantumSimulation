"""Standalone randomized boundary-dilation simulation with native qem mitigation.

Imports only Experiment 11's own modules and shared libraries. No experiment
hooks, copied global model parameters, or imports of Experiments 1--10.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import qem
from IBMRuntime import (AER_METHODS, Aer, CompilerConfig, Err, FakeIBMBackend, IBMHardware,
    RuntimeEnvironment, Sample, draw_circuit, draw_transpiled_circuit_layout_sync,
    load_ibm_account, run_sample_variants_batch_sync)
from Model1.Experiment11 import analysis, mitigation, physics
from Model1.Experiment11.progress import BatchProgress

VERSION = 1
DIRECTORY = Path(__file__).resolve().parent
LOCAL_BACKENDS = ("aer", "fake_kingston", "fake_miami")


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def runtime_value(result):
    if isinstance(result, Err):
        error = result.error
        message = "; ".join(error.problems) if hasattr(error, "problems") else f"{error.stage}: {error.message}"
        raise RuntimeError(message)
    return result.value


def runtime_target(backend, method, account_file):
    if backend == "aer":
        return Aer(method), RuntimeEnvironment()
    if backend in ("fake_fez", "fake_kingston", "fake_miami"):
        return FakeIBMBackend(backend, method=method), RuntimeEnvironment()
    if backend.startswith("fake_"):
        raise ValueError(f"unsupported fake backend {backend}")
    account = load_ibm_account(account_file)
    if isinstance(account, Err):
        raise RuntimeError(f"Could not load IBM account: {account.error.message}")
    return IBMHardware(backend), RuntimeEnvironment(ibm_account=account.value)


def configuration(options):
    """JSON-native physical/analysis signature. Plotting/batch/time-grid choices are separate."""
    fields = ("backend", "aer_method", "trajectories", "shots", "trotter_delta_t",
              "optimization_level", "seed_transpiler", "seed_simulator", "seed_trajectories",
              "fold_repetitions", "seed_folding", "seed_twirling", "zne_inference",
              "zne_polynomial_order", "no_zne", "no_pauli_twirling",
              "no_dynamical_decoupling", "no_readout_mitigation", "dd_sequence")
    return {"schema_version": VERSION, "physics": options.model.payload(),
            **{name: getattr(options, name) for name in fields},
            "zne_scale_factors": list(options.zne_scale_factors),
            "dd_spacing_fractions": options.dd_spacing_fractions,
            "dd_durations": options.dd_durations_data,
            "readout_calibration": options.readout_calibration_data,
            "measurement_bases": list(physics.BASES), "measurement_scheme": "three-basis U(1) reconstruction",
            "ancilla_strategy": "two boundary-local resettable ancillas",
            "exchange_synthesis": "explicit 14-operation RX/RZ/RZZ before native compilation"}


def output_paths(options, config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    model = options.model
    def number(value):
        return format(value, ".8g").replace(".", "p")
    flags = f"{'raw' if options.no_zne else 'zne'}_pt{int(not options.no_pauli_twirling)}"
    flags += f"_dd{'off' if options.no_dynamical_decoupling else options.dd_sequence}_ro{int(not options.no_readout_mitigation)}"
    stem = (f"randomized_{options.backend}_N{model.n_qubits}_J{number(model.J)}_h{number(model.h)}"
            f"_g{number(model.gamma)}_R{options.trajectories}_{flags}_{digest}")
    return {"result": options.output_directory / "results" / f"{stem}.json",
            "checkpoint": options.output_directory / "results" / f"{stem}_checkpoint.json",
            **{name: options.output_directory / "figures" / f"{stem}_{suffix}.png"
               for name, suffix in (("observables", "observables"), ("metrics", "transpilation_metrics"),
                                    ("scaling", "zne_scaling"), ("layout", "transpiled_layout"),
                                    ("left", "one_step_left"), ("right", "one_step_right"))}}


def canonical_time(value):
    value = float(value)
    rounded = round(value, 12)
    return rounded if abs(value - rounded) <= 8 * abs(np.spacing(value)) else value


def merge_records(existing, incoming):
    replacement_times = {canonical_time(row["time"]) for row in incoming}
    rows = [row for row in existing if canonical_time(row["time"]) not in replacement_times]
    for row in incoming:
        rows.append({**row, "time": canonical_time(row["time"])})
    return sorted(rows, key=lambda r: (r["time"], r["trajectory_index"], physics.BASES.index(r["basis"]),
                                       r["scale_index"], r["fold_repetition"]))


def validate_archive(payload, config):
    if payload.get("configuration") != config:
        raise ValueError("result/checkpoint physical or mitigation configuration differs; no circuits submitted")


def execute(options, config, paths):
    dts, counts, choices = physics.schedule(options.times, options.trajectories,
                                           options.seed_trajectories, options.trotter_delta_t)
    bases = [(ti, r, basis, counts[ti]) for ti in range(len(options.times))
             for r in range(options.trajectories) for basis in physics.BASES]
    variants_per_base = len(options.zne_scale_factors) * options.fold_repetitions
    total = len(bases) * variants_per_base
    checkpoint_path = paths["checkpoint"] if options.checkpoint_file is None else options.checkpoint_file
    invocation = {"times": options.times.tolist(), "step_dts": list(dts),
                  "step_counts": list(counts), "boundary_choices": [list(row) for row in choices]}
    batch_size = options.batch_size or (300 if options.backend in LOCAL_BACKENDS else 5)
    base_batch_size = batch_size // variants_per_base
    if base_batch_size < 1:
        raise ValueError(f"--batch-size must be >= {variants_per_base} (all variants of one base circuit)")
    records = []
    if checkpoint_path.exists() and not options.resume:
        previous = json.loads(checkpoint_path.read_text())
        validate_archive(previous, config)
        if not previous.get("simulation_complete", False):
            raise ValueError("incomplete checkpoint exists: use --resume with its original times, or a different --checkpoint-file; no circuits submitted")
    if options.resume:
        saved = json.loads(checkpoint_path.read_text())
        validate_archive(saved, config)
        if saved["invocation"] != invocation or saved["total_variant_count"] != total:
            raise ValueError("resume requires the same requested time grid, schedule and trajectory choices")
        records = saved["records"]
        expected = [(ti, r, basis, si, f) for ti, r, basis, _ in bases
                    for si in range(len(options.zne_scale_factors)) for f in range(options.fold_repetitions)]
        actual = [(row["time_index"], row["trajectory_index"], row["basis"], row["scale_index"],
                   row["fold_repetition"]) for row in records]
        if len(records) % variants_per_base or actual != expected[:len(records)]:
            raise ValueError("checkpoint is not a complete ordered base-circuit prefix")
        print(f"Resuming {len(records)}/{total} submitted variants", flush=True)

    def checkpoint(status):
        atomic_json(checkpoint_path, {"configuration": config, "invocation": invocation,
            "completed_variant_count": len(records), "total_variant_count": total,
            "simulation_complete": len(records) == total, "status": status, "records": records})

    checkpoint("sampling_complete" if len(records) == total else "sampling")
    print(f"Built {len(bases)} base circuits → {total} submitted variants; {options.shots // (options.trajectories * options.fold_repetitions)} shots each", flush=True)
    start = len(records) // variants_per_base
    # No full logical/compiled circuit batch survives between provider batches.
    for offset in range(start, len(bases), base_batch_size):
        batch = bases[offset:offset + base_batch_size]
        label = f"Batch {offset // base_batch_size + 1}/{int(np.ceil(len(bases)/base_batch_size))}"
        print(f"{label}: {len(batch)} base circuits → {len(batch) * variants_per_base} variants on {options.backend}", flush=True)
        diagnostics = []
        with BatchProgress(not options.no_progress, label, options.backend,
                           len(batch) * variants_per_base) as progress:
            progress.start("Preparing compiler inputs", len(batch))
            logical = []
            for ti, r, basis, count in batch:
                compact = physics.build_circuit(options.model, dts[:count], choices[r][:count], basis)
                logical.append(physics.prepare_for_compilation(compact))
                progress.advance()
            transformed_count = 0
            def transform(native, target):
                nonlocal transformed_count
                base_index = offset + transformed_count
                transformed_count += 1
                variants = []
                for si in range(len(options.zne_scale_factors)):
                    for f in range(options.fold_repetitions):
                        submitted, diagnostic = mitigation.transform(native, target, config,
                            base_index=base_index, scale_index=si, repetition=f)
                        variants.append(submitted)
                        diagnostics.append(diagnostic)
                        progress.advance()
                return tuple(variants)
            target, environment = runtime_target(options.backend, options.aer_method, options.account_file)
            result = runtime_value(run_sample_variants_batch_sync(tuple(logical), target,
                CompilerConfig(options.optimization_level, options.seed_transpiler),
                Sample(options.shots // (options.trajectories * options.fold_repetitions), options.seed_simulator),
                environment, transform, progress_callback=progress.stage))
        if len(result) != len(batch) * variants_per_base or len(diagnostics) != len(result):
            raise RuntimeError("provider returned an incomplete batch")
        index = 0
        for ti, r, basis, count in batch:
            for si, scale in enumerate(options.zne_scale_factors):
                for f in range(options.fold_repetitions):
                    record = result[index]
                    records.append({"time": float(options.times[ti]), "time_index": ti,
                        "trajectory_index": r, "basis": basis, "scale_index": si, "scale_factor": scale,
                        "fold_repetition": f, "step_dts": list(dts[:count]),
                        "boundary_choices": list(choices[r][:count]), "counts": dict(record.counts),
                        "shots": record.shots, "backend_name": record.backend_name, "job_id": record.job_id,
                        "original_operation_count": record.original_gate_count, "original_depth": record.original_depth,
                        "compiled_operation_count": record.compiled_gate_count, "compiled_depth": record.compiled_depth,
                        "diagnostic": diagnostics[index]})
                    index += 1
        checkpoint("sampling_complete" if len(records) == total else "sampling")
        print(f"Checkpointed {len(records)}/{total}: {checkpoint_path}", flush=True)
        del logical, compact, result, diagnostics
        gc.collect()
    return records, checkpoint_path, total - start * variants_per_base


def _embed(axis, circuit, title):
    import matplotlib.pyplot as plt
    figure = draw_circuit(circuit, fold=-1)
    try:
        with BytesIO() as stream:
            figure.savefig(stream, format="png", dpi=150, bbox_inches="tight")
            stream.seek(0)
            axis.imshow(plt.imread(stream, format="png"))
    finally:
        plt.close(figure)
    axis.axis("off")
    axis.set_title(title)


def draw_layout(options, circuit, path, *, one_steps=None):
    import matplotlib.pyplot as plt
    target, environment = runtime_target(options.backend, options.aer_method, options.account_file)
    figure = runtime_value(draw_transpiled_circuit_layout_sync(
        physics.prepare_for_compilation(circuit), target,
        CompilerConfig(options.optimization_level, options.seed_transpiler), environment,
        view="physical", logical_labels=physics.logical_roles(options.model)))
    try:
        if one_steps is not None:
            figure.set_size_inches(16, 14)
            figure.axes[0].set_position((0.025, 0.52, 0.63, 0.42))
            figure.axes[1].set_position((0.69, 0.55, 0.285, 0.36))
            for index, compact in enumerate(one_steps):
                axis = figure.add_axes((0.025 + index * 0.49, 0.035, 0.46, 0.40))
                _embed(axis, compact, f"One {'left' if index == 0 else 'right'} step before transpilation")
        figure.suptitle(f"Final-time Z-basis layout on {options.backend}")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_figures(payload, options, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    times = np.asarray(payload["times"])
    created = {}
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    reference = payload.get("exact_reference_observables")
    for axis, name, title in zip(axes.flat[:3], analysis.FAMILIES,
                                ("Site populations", "XY correlations (three-basis reconstruction)", "Excitation flow")):
        values = np.asarray(payload["observables"][name])
        errors = np.asarray(payload["standard_errors"]["total"][name])
        for index, (values_i, error_i) in enumerate(zip(values, errors)):
            line, = axis.plot(times, values_i, marker="o" if len(times) == 1 else None,
                              label=f"site {index}" if name == "populations" else f"bond {index}–{index+1}")
            axis.fill_between(times, values_i - error_i, values_i + error_i, color=line.get_color(), alpha=0.13)
            if reference is not None:
                axis.plot(times, reference[name][index], "--", color=line.get_color(), alpha=0.8)
        axis.set_title(title + ("; dashed = exact" if reference is not None else ""))
        axis.set_xlabel("Time")
        axis.set_ylabel(name.replace("_", " "))
        axis.grid(alpha=0.3)
        axis.legend(fontsize="small", ncol=2)
    diagnostic = axes[1, 1]
    for name in ("shot", "folding", "trajectory", "total"):
        maximum = np.maximum.reduce([np.max(payload["standard_errors"][name][family], axis=0)
                                     for family in analysis.FAMILIES])
        diagnostic.plot(times, maximum, label=f"max {name} s.e.")
    if reference is not None:
        maximum = np.maximum.reduce([np.max(np.abs(np.asarray(payload["observables"][name]) - reference[name]), axis=0)
                                     for name in analysis.FAMILIES])
        diagnostic.plot(times, maximum, label="max absolute error vs exact")
    diagnostic.set_title("Error and uncertainty diagnostics (raw units)")
    diagnostic.set_xlabel("Time")
    diagnostic.grid(alpha=0.3)
    diagnostic.legend(fontsize="small")
    model = options.model
    figure.suptitle(f"Experiment 11 on {options.backend}: N={model.n_qubits}, J={model.J}, h={model.h}, γ={model.gamma}")
    figure.tight_layout()
    paths["observables"].parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(paths["observables"], dpi=200, bbox_inches="tight")
    plt.close(figure)
    created["observables"] = paths["observables"].name
    print(f"Saved observable figure: {paths['observables']}", flush=True)

    summary = payload["transpilation_summary"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    positions = np.arange(len(summary))
    for axis, suffix, title in zip(axes, ("operation_count", "depth"), ("Mean operation count", "Mean depth")):
        for offset, prefix, label in ((-0.25, "mean_original_explicit_", "explicit before transpilation"),
                                     (0.0, "mean_base_transpiled_", "native transpiled"),
                                     (0.25, "mean_submitted_", "submitted after mitigation")):
            bars = axis.bar(positions + offset, [item[prefix + suffix] for item in summary], 0.25, label=label)
            axis.bar_label(bars, fmt="%.0f", padding=2)
        axis.set_xticks(positions, [str(item["requested_scale_factor"]) for item in summary])
        axis.set_xlabel("Requested noise scale")
        axis.set_title(title)
        axis.legend(fontsize="small")
        axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(paths["metrics"], dpi=200, bbox_inches="tight")
    plt.close(figure)
    created["metrics"] = paths["metrics"].name
    if not options.no_zne:
        figure, axes = plt.subplots(1, 3, figsize=(18, 5))
        for axis, name in zip(axes, analysis.FAMILIES):
            values = np.asarray(payload["scale_observables"][name])
            # Display all final-time site/bond scale data, without pretending one representative is universal.
            for index, row in enumerate(values[:, -1, :]):
                axis.plot(options.zne_scale_factors, row, "o-", label=f"{index}")
                axis.plot([0], [payload["observables"][name][index][-1]], "x", color=axis.lines[-1].get_color())
            axis.set_title(f"{name.replace('_', ' ')} at t={times[-1]:g}")
            axis.set_xlabel("Requested scale (x at 0 = extrapolation)")
            axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(paths["scaling"], dpi=200, bbox_inches="tight")
        plt.close(figure)
        created["scaling"] = paths["scaling"].name
    positive_steps = next((row["step_dts"] for row in payload["records"] if row["step_dts"]), ())
    steps = None
    if not options.no_circuit_plots and positive_steps:
        steps = tuple(physics.one_step(model, positive_steps[0], boundary) for boundary in (0, 1))
        for key, circuit in zip(("left", "right"), steps):
            figure = draw_circuit(circuit, fold=-1)
            try:
                figure.savefig(paths[key], dpi=200, bbox_inches="tight")
            finally:
                plt.close(figure)
            created[key] = paths[key].name
    if not options.no_layout:
        row = next(row for row in payload["records"] if row["time"] == max(payload["times"])
                   and row["trajectory_index"] == 0 and row["basis"] == "Z" and row["scale_index"] == 0
                   and row["fold_repetition"] == 0)
        circuit = physics.build_circuit(model, row["step_dts"], row["boundary_choices"], "Z")
        print("Generating transpiled layout (offline compilation, no sampling)...", flush=True)
        draw_layout(options, circuit, paths["layout"], one_steps=steps)
        created["layout"] = paths["layout"].name
    else:
        print("Skipped layout generation (--no-layout)", flush=True)
    return created


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--J", "--j", dest="J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--initial-site", type=int)
    parser.add_argument("--backend", default="aer", help="aer, fake_kingston (local noisy), fake_fez (layout-only), or IBM hardware name")
    parser.add_argument("--aer-method", choices=AER_METHODS, default="automatic")
    parser.add_argument("--account-file", type=Path, default=ROOT / "IBMRuntime" / "apikey.json")
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument("--times", type=float, nargs="+")
    grid.add_argument("--t-final", type=float)
    parser.add_argument("--time-points", type=int)
    parser.add_argument("--trotter-delta-t", type=float, default=0.1)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, help="maximum submitted variants per provider batch; no automatic retry/bisection")
    parser.add_argument("--optimization-level", type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument("--seed-transpiler", type=int, default=11)
    parser.add_argument("--seed-simulator", type=int, default=17)
    parser.add_argument("--seed-trajectories", type=int, default=29)
    parser.add_argument("--no-zne", action="store_true")
    parser.add_argument("--zne-scale-factors", type=float, nargs="+", default=(1.0, 1.5, 2.0))
    parser.add_argument("--zne-inference", choices=("linear", "polynomial", "richardson"), default="linear")
    parser.add_argument("--zne-polynomial-order", type=int, default=2)
    parser.add_argument("--fold-repetitions", type=int, default=1)
    parser.add_argument("--seed-folding", type=int, default=8675309)
    parser.add_argument("--no-pauli-twirling", action="store_true")
    parser.add_argument("--seed-twirling", type=int, default=314159)
    parser.add_argument("--no-dynamical-decoupling", action="store_true")
    parser.add_argument("--dd-sequence", choices=("xx", "yy", "xy4", "xy8", "xyxy"), default="xx")
    parser.add_argument("--dd-spacing-fractions", type=float, nargs="+")
    parser.add_argument("--dd-durations", type=Path, help="plain-Aer synthetic timing JSON; not a noise model")
    parser.add_argument("--no-readout-mitigation", action="store_true")
    parser.add_argument("--readout-calibration", type=Path)
    references = parser.add_mutually_exclusive_group()
    references.add_argument("--classical-reference", action="store_true", help="full many-qubit density-vector reference (expensive)")
    references.add_argument("--optimized-classical-reference", action="store_true", help="exact single-excitation amplitude reference, recommended at large N")
    parser.add_argument("--output-directory", type=Path, default=DIRECTORY)
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", type=Path, help="regenerate figures from a saved Experiment 11 JSON; never sample")
    parser.add_argument("--layout-only", action="store_true")
    parser.add_argument("--no-layout", action="store_true")
    parser.add_argument("--no-circuit-plots", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    options = parser.parse_args(arguments)
    options.backend = options.backend.lower()
    options.model = physics.Model(options.n_qubits, options.J, options.h, options.gamma, options.initial_site)
    if options.times is not None:
        if options.time_points is not None:
            raise ValueError("--time-points is not valid with --times")
        options.times = physics.validate_times(options.times)
    else:
        horizon = 2.0 if options.t_final is None else options.t_final
        count = 11 if options.time_points is None else options.time_points
        if not np.isfinite(horizon) or horizon <= 0 or count < 2:
            raise ValueError("t-final must be positive and finite, and time-points >= 2; use --times 0 for only t=0")
        options.times = np.linspace(0.0, horizon, count)
    if options.trajectories < 1 or options.fold_repetitions < 1 or options.shots < 1:
        raise ValueError("trajectory/fold/shot counts must be positive")
    if options.shots % (options.trajectories * options.fold_repetitions):
        raise ValueError("shots must be divisible by trajectories * fold-repetitions")
    if options.no_zne and options.fold_repetitions != 1:
        raise ValueError("--no-zne requires --fold-repetitions 1")
    if not np.isfinite(options.trotter_delta_t) or options.trotter_delta_t <= 0:
        raise ValueError("trotter-delta-t must be positive and finite")
    for name in ("seed_transpiler", "seed_simulator", "seed_trajectories", "seed_folding", "seed_twirling"):
        if getattr(options, name) < 0:
            raise ValueError("seeds must be nonnegative")
    options.zne_scale_factors = (1.0,) if options.no_zne else tuple(options.zne_scale_factors)
    if options.zne_scale_factors[0] != 1:
        raise ValueError("first ZNE scale must be 1")
    plan = qem.ZNEPlan(options.zne_scale_factors, "none" if options.no_zne else options.zne_inference,
                       degree=options.zne_polynomial_order)
    analysis.qem_value(qem.extrapolate(plan, [0.0] * len(options.zne_scale_factors)))
    minimum_batch = len(options.zne_scale_factors) * options.fold_repetitions
    if options.batch_size is not None and options.batch_size < minimum_batch:
        raise ValueError(f"batch-size must be >= {minimum_batch}")
    if options.dd_sequence == "xyxy":
        options.dd_sequence = "xy4"
    qem.DDPlan(sequence=options.dd_sequence,
               spacing=None if options.dd_spacing_fractions is None else tuple(options.dd_spacing_fractions))
    options.dd_durations_data = None
    if options.dd_durations is not None:
        from qiskit.transpiler import InstructionDurations
        if options.backend != "aer":
            raise ValueError("dd-durations is for plain Aer only; fake/hardware targets supply timings")
        data = json.loads(options.dd_durations.read_text())
        if not np.isfinite(data["dt"]) or data["dt"] <= 0:
            raise ValueError("DD dt must be positive and finite")
        InstructionDurations(data["instructions"], dt=data["dt"])
        options.dd_durations_data = data
    options.readout_calibration_data = None
    if options.readout_calibration is not None:
        data = json.loads(options.readout_calibration.read_text())
        if data["backend"].lower() != options.backend:
            raise ValueError("readout calibration backend differs")
        model = qem.ReadoutModel(tuple(qem.ReadoutBlock(tuple(block["physical_qubits"]),
                    tuple(tuple(row) for row in block["assignment"])) for block in data["blocks"]), source=data["source"])
        for block in model.blocks:
            analysis.qem_value(qem.mitigate_parity({"0": 1}, model, block.bits))
        options.readout_calibration_data = data
    if options.layout_only and (options.no_layout or options.no_plots or options.resume or options.plot_only is not None):
        raise ValueError("layout-only conflicts with no-layout/no-plots/resume/plot-only")
    return options


def _restore_plot_options(options, config):
    for name, value in config.items():
        if name not in ("physics", "schema_version", "dd_durations", "readout_calibration"):
            setattr(options, name, value)
    options.model = physics.Model(**config["physics"])
    return options


def main(options):
    if options.plot_only is not None:
        payload = json.loads(options.plot_only.read_text())
        if payload.get("experiment") != "Model1/Experiment11" or payload["configuration"]["schema_version"] != VERSION:
            raise ValueError("plot-only requires a supported Experiment 11 result")
        options = _restore_plot_options(options, payload["configuration"])
        options.output_directory = options.plot_only.parent.parent
        paths = output_paths(options, payload["configuration"])
        if not options.no_plots:
            payload["figures"] = plot_figures(payload, options, paths)
            payload["figure_generation_status"] = "complete"
            atomic_json(options.plot_only, payload)
        print("Sampler jobs submitted: 0 (plot-only)", flush=True)
        return payload
    config = configuration(options)
    paths = output_paths(options, config)
    if options.layout_only:
        dts, counts, choices = physics.schedule(options.times, options.trajectories,
            options.seed_trajectories, options.trotter_delta_t)
        circuit = physics.build_circuit(options.model, dts[:counts[-1]], choices[0][:counts[-1]], "Z")
        steps = None if options.no_circuit_plots or not dts else tuple(physics.one_step(options.model, dts[0], b) for b in (0, 1))
        draw_layout(options, circuit, paths["layout"], one_steps=steps)
        print(f"Saved layout: {paths['layout']}; sampler jobs submitted: 0", flush=True)
        return None
    if options.backend == "fake_fez":
        raise ValueError("fake_fez is layout-only; use fake_kingston for local noisy sampling")
    if not options.no_dynamical_decoupling and options.backend == "aer" and config["dd_durations"] is None:
        raise ValueError("plain Aer DD needs --dd-durations FILE or --no-dynamical-decoupling")
    if not options.no_readout_mitigation and options.backend != "aer" and config["readout_calibration"] is None:
        raise ValueError("noisy fake/hardware readout mitigation needs --readout-calibration FILE or --no-readout-mitigation")
    if not options.no_progress:
        from tqdm import tqdm  # Fail before execution if presentation dependency is unavailable.
    if 2 * options.model.gamma * options.trotter_delta_t >= 0.1:
        warnings.warn("2*gamma*trotter_delta_t is not very small; compare with the exact reference and consider smaller steps")
    existing = None
    if paths["result"].exists():
        existing = json.loads(paths["result"].read_text())
        validate_archive(existing, config)
    requested_reference = "full" if options.classical_reference else "optimized" if options.optimized_classical_reference else None
    reference_mode = requested_reference or (None if existing is None else existing.get("reference_mode"))
    if reference_mode == "full" and options.model.n_qubits >= 10:
        warnings.warn("full reference is memory/CPU intensive at N>=10; --optimized-classical-reference is recommended")
    print(f"Experiment 11 physics: {options.model.payload()}", flush=True)
    print(f"Configuration-specific result: {paths['result']}", flush=True)
    incoming, checkpoint_path, submitted_count = execute(options, config, paths)
    records = merge_records([] if existing is None else existing["records"], incoming)
    reconstructed = analysis.reconstruct(records, options.model, config)
    history = [] if existing is None else existing.get("run_history", [])
    history.append({"completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "requested_times": options.times.tolist(), "variant_count_in_invocation": len(incoming),
                    "submitted_variants_in_invocation": submitted_count,
                    "reference_mode": reference_mode})
    payload = {"experiment": "Model1/Experiment11", "configuration": config,
               **reconstructed, "records": records, "run_history": history,
               "shots_per_variant": options.shots // (options.trajectories * options.fold_repetitions),
               "transpilation_summary": analysis.resource_summary(records, options.zne_scale_factors),
               "reference_mode": reference_mode, "reference_generation_status": "pending" if reference_mode else "disabled",
               "figure_generation_status": "pending", "figures": {}}
    atomic_json(paths["result"], payload)
    print(f"Saved simulation data before reference/figure generation: {paths['result']}", flush=True)
    del incoming, records
    gc.collect()
    if reference_mode:
        print(f"Calculating {reference_mode} exact reference with the same J/h/gamma/initial site...", flush=True)
        reference = physics.exact_reference(options.model, payload["times"], full=reference_mode == "full")
        payload["exact_reference_observables"] = analysis.family_payload(reference)
        payload["aggregate_observable_error_vs_exact"] = np.maximum.reduce([
            np.max(np.abs(np.asarray(payload["observables"][name]) - values), axis=0)
            for name, values in zip(analysis.FAMILIES, reference)]).tolist()
        payload["reference_generation_status"] = "complete"
        atomic_json(paths["result"], payload)
    if options.no_plots:
        payload["figure_generation_status"] = "disabled"
    else:
        payload["figures"] = plot_figures(payload, options, paths)
        payload["figure_generation_status"] = "complete"
    atomic_json(paths["result"], payload)
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["status"] = "complete"
    atomic_json(checkpoint_path, checkpoint)
    print(f"Archive times: {payload['times']}", flush=True)
    print(f"Saved result: {paths['result']}", flush=True)
    return payload


if __name__ == "__main__":
    main(parse_arguments())
