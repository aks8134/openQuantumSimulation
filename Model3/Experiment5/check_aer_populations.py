"""Offline Aer validation of saved paths; never connects to IBM hardware.

Exactly trace each freshly prepared bath after its collision for a ten-qubit
density-matrix check. Independently run the full twelve-qubit regenerated
CZ circuit, including preparation measurements, resets and final readout.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import Kraus, Operator, Pauli
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

try:
    from . import circuit_depth as circuits
    from .analyze_hardware import analyze, moments, serializable
except ImportError:
    import circuit_depth as circuits
    from analyze_hardware import analyze, moments, serializable


def bath_channel(operation, occupied):
    """System is qarg 0, bath qarg 1, in Qiskit's little-endian convention."""
    unitary = Operator(operation).data
    # Aer 0.17.2 in this environment mishandles strided matrix views passed as
    # Kraus parameters. Explicit contiguous copies preserve the intended map;
    # the full-density tests below independently check trace and coherences.
    blocks = [np.array(unitary[2*a:2*a+2, 2*occupied:2*occupied+2], order="C", copy=True)
              for a in (0, 1)]
    np.testing.assert_allclose(sum(k.conj().T@k for k in blocks), np.eye(2), atol=1e-13)
    return Kraus(blocks).to_instruction()


def reduced_density_circuit(body, n):
    """Exact nonselective channel, derived from the actual logical operations.

    A bath is used once and immediately reset. Tracing it after that unitary
    therefore reproduces the system state exactly, with two fewer wires.
    No classically conditioned gates are supported by this transformation.
    """
    circuit = QuantumCircuit(n, name=body.name+"_bath_traced")
    occupied = {q: 0 for q in range(n, body.num_qubits)}
    awaiting_reset = None
    dephase = Kraus([np.diag([1, 0]), np.diag([0, 1])]).to_instruction()
    for instruction in body.data:
        op = instruction.operation
        q = [body.find_bit(bit).index for bit in instruction.qubits]
        if getattr(op, "condition", None) is not None or op.name in ("if_else", "while_loop", "switch_case"):
            raise ValueError("ignored-record transformation cannot handle classical control")
        if awaiting_reset is not None and (op.name != "reset" or q != [awaiting_reset]):
            raise ValueError("bath collision must be immediately followed by its reset")
        if op.name == "measure":
            if len(q) != 1 or q[0] >= n:
                raise ValueError("only initial system preparation measurements are supported")
            circuit.append(dephase, q)
        elif all(index < n for index in q):
            if op.name == "xx_plus_yy":
                circuit.unitary(Operator(op), q)
            else:
                circuit.append(op, q)
        elif op.name == "x" and len(q) == 1 and q[0] in occupied:
            occupied[q[0]] ^= 1
        elif op.name == "xx_plus_yy" and len(q) == 2 and q[0] < n and q[1] in occupied:
            circuit.append(bath_channel(op, occupied[q[1]]), [q[0]])
            awaiting_reset = q[1]
        elif op.name == "reset" and q == [awaiting_reset]:
            occupied[q[0]] = 0
            awaiting_reset = None
        else:
            raise ValueError(f"unsupported system/bath operation {op.name}{q}")
    if awaiting_reset is not None:
        raise ValueError("unreset bath at end of circuit")
    for site in range(n):
        circuit.save_expectation_value(Pauli("Z"), [site], label=f"z{site}")
    return circuit


def compile_native(body, n, seed, optimization):
    """Regenerate on a compact CZ line; no live target/calibrations are used."""
    native = transpile(body, basis_gates=["cz", "rz", "sx", "x", "reset", "measure"],
        coupling_map=CouplingMap.from_line(n+2), initial_layout=list(range(1, n+1))+[0, n+1],
        optimization_level=optimization, seed_transpiler=seed)
    mapping = native.layout.final_index_layout(filter_ancillas=True)
    return circuits.append_readout(native, mapping[:n], "z"), mapping


def sampled_circuit(native, n):
    """Insert conditional expectation saves before all final measurements.

    These are simulator diagnostics only and do not change the quantum state.
    The measurements and all original preparation/reset instructions remain.
    """
    register = next(reg for reg in native.cregs if reg.name == "outcome")
    indices = [native.find_bit(bit).index for bit in register]
    final = {}
    for position, instruction in enumerate(native.data):
        if instruction.operation.name == "measure" and instruction.clbits[0] in register:
            j = list(register).index(instruction.clbits[0])
            final[j] = (position, native.find_bit(instruction.qubits[0]).index)
    if set(final) != set(range(n)):
        raise ValueError("incomplete final site measurements")
    start = min(position for position, _ in final.values())
    if any(instruction.operation.name not in ("measure", "barrier") for instruction in native.data[start:]):
        raise ValueError("final outcome measurements must be terminal")
    sites = [final[j][1] for j in range(n)]
    if sites != native.metadata["physical_sites"]:
        raise ValueError("readout metadata and actual measurement mapping disagree")
    result = native.copy_empty_like()
    for position, instruction in enumerate(native.data):
        if position == start:
            for j, site in enumerate(sites):
                result.save_expectation_value(Pauli("Z"), [site], label=f"z{j}", pershot=True)
        result.append(instruction.operation, instruction.qubits, instruction.clbits)
    return result, indices


def outcome_counts(counts, indices):
    result = {}
    for label, count in counts.items():
        value = int(label.replace(" ", ""), 2)
        outcome = sum(((value >> index) & 1) << j for j, index in enumerate(indices))
        key = format(outcome, f"0{len(indices)}b")
        result[key] = result.get(key, 0)+count
    return result


def make_plots(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(1, len(report["physical_populations"])+1)
    colors = {"hardware": "#b71c1c", "ideal": "#00897b", "aer": "#1565c0"}
    fig, axis = plt.subplots(figsize=(10.5, 5), layout="constrained")
    axis.plot(x, report["physical_populations"], "k--", label="Physical steady-state target")
    axis.plot(x, report["ideal_populations"], "s-", color=colors["ideal"], markersize=7,
              label="Saved ideal: same four paths")
    axis.plot(x, report["density_populations"], "x", color="black", markersize=7,
              label="Aer density matrix (exact bath trace)")
    axis.errorbar(x, report["native_conditional_populations"], yerr=2*np.array(report["native_conditional_se"]),
                  fmt="o", color=colors["aer"], capsize=3, markersize=4,
                  label="Aer full CZ circuit, conditional estimator ±2 SE")
    axis.errorbar(x, report["hardware_populations"], yerr=2*np.array(report["hardware_shot_se"]),
                  fmt="d-", color=colors["hardware"], capsize=3, label="Saved Phoenix ±2 shot SE")
    axis.set(xlabel="Site", ylabel="Population", xticks=x,
             title="Same four trajectories: noiseless Aer versus saved ideal and hardware")
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"populations.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained", sharey=True)
    for axis, row in zip(axes.flat, report["paths"]):
        axis.plot(x, row["ideal"], "s-", color=colors["ideal"], label="Saved ideal")
        axis.plot(x, row["density"], "kx", markersize=7, label="Aer density matrix")
        axis.errorbar(x, row["native_conditional"], yerr=2*np.array(row["native_conditional_se"]),
                      fmt="o", color=colors["aer"], capsize=2, markersize=4, label="Full native Aer ±2 SE")
        axis.errorbar(x, row["hardware"], yerr=2*np.array(row["hardware_shot_se"]), fmt="d-",
                      color=colors["hardware"], capsize=2, label="Saved hardware ±2 shot SE")
        axis.set(title=f"Trajectory {row['trajectory']}", xlabel="Site", ylabel="Population", xticks=x)
        axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=4, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"per_path_populations.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    if args.shots < 2 or args.threads < 1:
        raise ValueError("at least two shots and one thread are required")
    original, measured, ideal_paths = analyze(args.run_directory)
    source = json.loads((args.run_directory/"source.json").read_bytes())
    manifest = json.loads((args.run_directory/"manifest.json").read_bytes())
    circuits.validate_source(source)
    hashes = original["verification"]["original_files_sha256"]
    output = args.output_directory or Path(__file__).resolve().parent/"analysis"/"hardware"/args.run_directory.name/"aer_population_check"
    if output.resolve() == args.run_directory.resolve() or args.run_directory.resolve() in output.resolve().parents:
        raise ValueError("output must be outside original run directory")
    output.mkdir(parents=True, exist_ok=True)
    n = original["physics"]["n_qubits"]
    density_simulator = AerSimulator(method="density_matrix", max_parallel_threads=args.threads,
        max_parallel_experiments=1, max_memory_mb=1200, fusion_threshold=1, fusion_max_qubit=2)
    sample_simulator = AerSimulator(method="statevector", max_parallel_threads=args.threads,
        max_parallel_shots=args.threads, max_parallel_experiments=1, fusion_threshold=1, fusion_max_qubit=2)
    rows, saved_circuits = [], []
    for index, trajectory in enumerate(measured["indices"]):
        print(f"Trajectory {trajectory}: reconstructing saved 72-step body", flush=True)
        body = circuits.build_body(source, trajectory, ancillas="boundary")
        reduced = reduced_density_circuit(body, n)
        start = time.monotonic()
        exact = density_simulator.run(reduced, shots=1, seed_simulator=args.seed).result()
        if not exact.success:
            raise RuntimeError(str(exact))
        density = np.array([(1-exact.data()[f"z{j}"])/2 for j in range(n)])
        density_seconds = time.monotonic()-start
        print(f"Trajectory {trajectory}: exact reduced density complete in {density_seconds:.1f}s; max saved-ideal difference {np.max(np.abs(density-ideal_paths['populations'][index])):.3g}", flush=True)
        native, mapping = compile_native(body, n, manifest["configuration"]["seed_transpiler"],
                                        manifest["configuration"]["optimization_level"])
        diagnostic, readout_indices = sampled_circuit(native, n)
        start = time.monotonic()
        sampled = sample_simulator.run(diagnostic, shots=args.shots, seed_simulator=args.seed+index).result()
        if not sampled.success:
            raise RuntimeError(str(sampled))
        counts = outcome_counts(sampled.get_counts(), readout_indices)
        populations, covariance, shots = moments(counts, n)
        conditional_samples = np.column_stack([(1-np.array(sampled.data()[f"z{j}"]))/2 for j in range(n)])
        if conditional_samples.shape != (args.shots, n) or shots != args.shots:
            raise ValueError("Aer diagnostic shot count mismatch")
        conditional = conditional_samples.mean(axis=0)
        conditional_cov = np.cov(conditional_samples, rowvar=False, ddof=1)/args.shots
        saved_circuits.extend([body, native])
        row = {"trajectory": trajectory, "ideal": ideal_paths["populations"][index], "density": density,
               "native_counts": populations, "native_counts_covariance": covariance,
               "native_conditional": conditional, "native_conditional_covariance": conditional_cov,
               "native_conditional_se": np.sqrt(np.maximum(0, np.diag(conditional_cov))),
               "hardware": measured["populations"][index],
               "hardware_shot_se": np.sqrt(np.diag(measured["per_path_population_shot_covariance"][index])),
               "native_final_mapping": mapping, "logical_metrics": circuits.metrics(body),
               "native_metrics": circuits.metrics(native), "density_seconds": density_seconds,
               "native_seconds": time.monotonic()-start, "outcome_counts": counts,
               "density_aer_metadata": exact.results[0].metadata,
               "native_aer_metadata": sampled.results[0].metadata}
        rows.append(row)
        (output/f"path_{trajectory}.json").write_text(json.dumps(serializable(row), indent=2)+"\n")
        print(f"Trajectory {trajectory}: full native simulation complete in {row['native_seconds']:.1f}s", flush=True)
    with gzip.open(output/"regenerated_circuits.qpy.gz", "wb") as stream:
        qpy.dump(saved_circuits, stream)
    r = len(rows)
    conditional_cov = sum(row["native_conditional_covariance"] for row in rows)/r**2
    counts_cov = sum(row["native_counts_covariance"] for row in rows)/r**2
    result = {"run_directory": str(args.run_directory.resolve()), "scope": "regenerated logical circuits and offline compact CZ line; original Phoenix QPY missing",
        "seed": args.seed, "shots_per_full_native_path": args.shots, "threads": args.threads,
        "versions": {name: importlib.metadata.version(name) for name in ("qiskit", "qiskit-aer", "numpy")},
        "paths": rows, "density_populations": np.mean([row["density"] for row in rows], axis=0),
        "ideal_populations": original["references"]["ideal_same_saved_paths"]["populations"],
        "physical_populations": original["references"]["physical_stationary"]["populations"],
        "hardware_populations": original["hardware"]["populations"],
        "hardware_shot_se": original["shot_standard_errors_fixed_paths"]["populations"],
        "native_counts_populations": np.mean([row["native_counts"] for row in rows], axis=0),
        "native_counts_se": np.sqrt(np.maximum(0, np.diag(counts_cov))),
        "native_conditional_populations": np.mean([row["native_conditional"] for row in rows], axis=0),
        "native_conditional_se": np.sqrt(np.maximum(0, np.diag(conditional_cov))), "input_sha256": hashes}
    result["max_density_discrepancy_all_paths_sites"] = float(max(np.max(np.abs(row["density"]-row["ideal"])) for row in rows))
    se = np.array([row["native_conditional_se"] for row in rows])
    difference = np.array([row["native_conditional"]-row["ideal"] for row in rows])
    result["max_native_conditional_discrepancy_all_paths_sites"] = float(np.abs(difference).max())
    result["max_native_conditional_standardized_discrepancy"] = float((np.abs(difference)/(se+1e-14)).max())
    result["density_agrees_to_1e-10"] = result["max_density_discrepancy_all_paths_sites"] < 1e-10
    result["native_agrees_within_4_nominal_se"] = bool(np.all(np.abs(difference) <= 4*se+1e-10))
    table = []
    for j in range(n):
        table.append({"site_1based": j+1, **{key: float(result[key][j]) for key in (
            "ideal_populations", "density_populations", "native_counts_populations", "native_counts_se",
            "native_conditional_populations", "native_conditional_se", "hardware_populations", "physical_populations")}})
    with (output/"populations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    make_plots(result, output)
    text = ["# Noiseless Aer population check", "",
        f"Saved run: `{args.run_directory.name}`; same {r} paths, all {n} sites, 72 cosine steps.", "",
        "## Result", "",
        f"Maximum exact density-matrix versus saved ideal difference over all paths/sites: **{result['max_density_discrepancy_all_paths_sites']:.6g}**.",
        f"Full native circuit conditional-estimator discrepancy: maximum **{result['max_native_conditional_discrepancy_all_paths_sites']:.6g}**, largest standardized discrepancy **{result['max_native_conditional_standardized_discrepancy']:.3f} SE**.",
        f"Exact check below 1e-10: **{result['density_agrees_to_1e-10']}**. Native sample check within four nominal SE at every site/path: **{result['native_agrees_within_4_nominal_se']}**.", "",
        "[Averaged populations](populations.png) ([PDF](populations.pdf)); [each trajectory](per_path_populations.png) ([PDF](per_path_populations.pdf)).", "",
        "| Site | Saved ideal | Aer exact density | Full native Aer conditional estimate | Conditional SE | Saved Phoenix | Physical target |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        text.append(f"| {row['site_1based']} | {row['ideal_populations']:.6f} | {row['density_populations']:.6f} | {row['native_conditional_populations']:.6f} | {row['native_conditional_se']:.6f} | {row['hardware_populations']:.6f} | {row['physical_populations']:.6f} |")
    text += ["", "## What was simulated", "",
        "The original twelve-qubit boundary-bath bodies were regenerated directly from the saved step sizes and jump choices. No new random boundary trajectories were chosen. No hardware account or live backend was accessed.", "",
        "For the deterministic check, every ignored preparation measurement was replaced by its exact nonselective dephasing channel. Each bath collision was replaced by the exact reduced system channel computed from the actual Qiskit two-qubit gate matrix and the bath's actual initial bit. The bath is immediately reset and never used jointly with the other bath, so this partial trace gives exactly the same system density matrix. Aer propagated the resulting ten-qubit density matrix; no sampled shots or initial-state approximation enter this check.", "",
        f"Independently, the full twelve-qubit bodies were compiled to a compact CZ-connected line using the saved optimization level/transpiler seed. Aer statevector sampling used {args.shots} shots/path, retaining all initial preparation measurements, 72 bath resets, and terminal Z measurements. Simulator-only expectation-value snapshots before final measurement give a lower-variance conditional population estimator; raw measured population counts and their separate shot SEs are also saved. Both estimators average over sampled preparation and reset outcomes. Snapshot values are not hardware-accessible observables and are not extra shots.", "",
        "## Scope and limits", "",
        "The original Phoenix ISA QPY files are absent. This validates regenerated logical operations and an offline CZ-line compilation; it does not certify the historical Phoenix compiled circuits or reproduce device calibrations, timings, routing decisions, or errors. The compressed regenerated QPY is saved separately and is not a replacement for the missing original artifact.", "",
        "The deterministic comparison is against the same four saved paths, not the infinite-trajectory mean or physical stationary target. The latter can differ because of trajectory selection, finite physical time, and timestep bias. Agreement here would support circuit/reference consistency, not establish a unique cause of hardware error.", "",
        "The bath-channel matrices are copied to contiguous storage before passing them to Aer. In this installed Aer 0.17.2 environment, passing the original strided submatrix views produced an incorrect non-trace-preserving result. Small-system tests against explicit system-plus-bath density matrices detected this; the contiguous-copy implementation passed those checks. This affected the new analysis helper, not the original hardware circuit or saved ideal calculation.", "",
        "The four-SE sampling criterion is a diagnostic using empirical marginal standard errors, not a certified simultaneous confidence bound. Raw counts and conditional estimators have different variances and must not have their SEs added. There is no device noise model, mitigation, or hardware execution in this analysis.", "",
        "Per-path JSON files retain counts, conditional covariances, mappings, resource metrics, Aer metadata and timing. Population CSV contains both estimators. All original run files were hash-verified unchanged.", ""]
    (output/"Analysis.md").write_text("\n".join(text))
    for name, digest in hashes.items():
        if hashlib.sha256((args.run_directory/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"original input changed: {name}")
    result["inputs_unchanged"] = True
    (output/"analysis.json").write_text(json.dumps(serializable(result), indent=2)+"\n")
    print(json.dumps({"output": str(output), "max_density_error": result["max_density_discrepancy_all_paths_sites"],
        "max_native_standardized_error": result["max_native_conditional_standardized_discrepancy"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
