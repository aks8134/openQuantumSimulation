"""Profile one saved Aer ZNE circuit without sampling or submitting jobs.

Run with the main environment. Reconstruct one trajectory from a checkpoint
and profile native-qem folding, with no conversion, lowering or execution.
Historical checkpoints can be inputs, but their old folded counts will differ.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter, process_time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_CHECKPOINT = Path(__file__).parent / "results" / (
    "mitiq_zne_randomized_lie_trotter_aer_N10_R4_F1_B3_"
    "pauli_off_dd_off_readout_off_linear_S1p0_1p5_2p0_checkpoint.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-index", type=int, default=21)
    parser.add_argument("--scale", type=float, default=2.0)
    args = parser.parse_args()
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    config = checkpoint["configuration"]
    if config["backend"].lower() != "aer":
        parser.error("this diagnostic is Aer-only; it never accesses IBM accounts")
    saved = next(
        (
            record for record in checkpoint["results"]
            if record["base_index"] == args.base_index
            and record["scale_factor"] == args.scale
        ),
        None,
    )
    if saved is None:
        parser.error("the selected base index and scale have no saved result")

    diagnostics = Path(__file__).parent / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="folding-base21-", dir=diagnostics))
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "base_index": args.base_index,
        "scale": args.scale,
        "fold_seed": saved["fold_seed"],
        "time": config["times"][saved["time_index"]],
        "trajectory_index": saved["trajectory_index"],
        "basis": saved["basis"],
        "python": sys.executable,
        "versions": {},
        "stages": [],
        "conversions": [],
        "jobs_submitted": 0,
        "shots_executed": 0,
    }

    def persist():
        temporary = output / "profile.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "profile.json")

    def snapshot(circuit):
        return {
            "representation": "qiskit",
            "operations": int(circuit.size()),
            "depth": int(circuit.depth()),
            "qubits": circuit.num_qubits,
            "counts": dict(sorted(circuit.count_ops().items())),
        }

    active_stage = "startup"

    def stage(name, action, *, circuit=False):
        nonlocal active_stage
        active_stage = name
        print(f"Starting: {name}", flush=True)
        wall, cpu = perf_counter(), process_time()
        value = action()
        record = {
            "name": name,
            "wall_seconds": perf_counter() - wall,
            "cpu_seconds": process_time() - cpu,
        }
        if circuit:
            record.update(snapshot(value))
        report["stages"].append(record)
        persist()
        print(json.dumps(record), flush=True)
        return value

    def imports():
        from Model1.Experiment5 import randomized_lie_trotter as experiment5
        from Model1.Experiment7 import randomized_lie_trotter as experiment7
        from IBMRuntime.interpreters.qiskit import _to_qiskit
        from qiskit import qpy
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_aer import AerSimulator
        from Model1.Experiment8 import zne_randomized_lie_trotter as experiment8
        return (
            experiment5, experiment7, _to_qiskit, qpy,
            generate_preset_pass_manager, AerSimulator, experiment8,
        )

    print(f"Diagnostic output: {output}", flush=True)
    (
        experiment5, experiment7, to_qiskit, qpy, pass_manager,
        aer_simulator, experiment8,
    ) = stage("imports (not folding time)", imports)
    report["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("qiskit", "qiskit-aer")
    }

    # Only this trajectory is constructed. Keep the original full schedule so
    # RNG prefixes agree with the saved run; --times 0.2 alone would differ.
    def reconstruct():
        schedule = experiment5.build_randomized_schedule(
            config["times"], config["number_of_trajectories"],
            config["seed_trajectories"], config["trotter_delta_t"],
        )
        evolution = experiment5._trajectory_evolution_circuits(
            config["number_of_system_qubits"], schedule,
            saved["trajectory_index"],
        )
        logical = experiment5._append_terminal_measurements(
            evolution[saved["time_index"]],
            config["number_of_system_qubits"], saved["basis"],
        )
        return to_qiskit(experiment7.prepare_circuit_for_compilation(logical))

    original = stage("explicit Experiment 7 circuit", reconstruct, circuit=True)
    target = aer_simulator(method=config["aer_method"]).target
    native = stage(
        "baseline optimizing compilation",
        lambda: pass_manager(
            target=target, optimization_level=config["optimization_level"],
            seed_transpiler=config["seed_transpiler"],
        ).run(original),
        circuit=True,
    )
    baseline = saved["fold_diagnostic"]
    report["checkpoint_baseline"] = {
        "operations": baseline["base_operation_count"],
        "depth": baseline["base_depth"],
        "eligible_gate_counts": baseline["eligible_gate_counts"],
    }
    report["baseline_matches_checkpoint"] = (
        int(native.size()) == baseline["base_operation_count"]
        and int(native.depth()) == baseline["base_depth"]
        and dict(Counter(
            item.operation.name for item in native.data
            if item.operation.name not in ("rz", "reset", "measure", "barrier")
        )) == dict(baseline["eligible_gate_counts"])
    )
    if not report["baseline_matches_checkpoint"]:
        warning = (
            "Recompiled baseline differs from the checkpoint "
            f"(saved: {baseline['base_operation_count']} operations, "
            f"depth {baseline['base_depth']}; current: {native.size()} "
            f"operations, depth {native.depth()}). Continuing to profile "
            "native folding using the current baseline; "
            "this is not an exact reproduction of the saved decomposition."
        )
        report["warnings"] = [warning]
        print(f"WARNING: {warning}", flush=True)
    persist()

    def fold(factor):
        return experiment8.fold_native_circuit(
            native, target, experiment8.FoldSpec(0, factor, 0, saved["fold_seed"]),
            args.base_index,
        )[0]

    roundtrip = stage("native scale-one control", lambda: fold(1.0), circuit=True)
    folded = stage("native local folding", lambda: fold(args.scale), circuit=True)
    report["folding_implementation"] = "qem; no conversion or post-fold lowering"

    report["folded_matches_checkpoint"] = (
        int(folded.size()) == saved["compiled_operation_count"]
        and int(folded.depth()) == saved["compiled_depth"]
    )
    report["unsupported_operation_names"] = sorted(
        set(folded.count_ops()) - set(target.operation_names)
    )
    report["operation_deltas"] = {
        "scale_one": int(roundtrip.size() - native.size()),
        "native_folding": int(folded.size() - roundtrip.size()),
    }
    for name, circuit in (
        ("baseline", native), ("scale_one", roundtrip), ("native_folded", folded),
    ):
        with (output / f"{name}.qpy").open("wb") as stream:
            qpy.dump(circuit, stream)
    persist()
    print("Operation deltas:", json.dumps(report["operation_deltas"]), flush=True)
    print("Reproduces saved folded counts:", report["folded_matches_checkpoint"], flush=True)
    print(f"Saved profile and single-circuit QPY examples: {output}", flush=True)


if __name__ == "__main__":
    main()
