"""Compile saved Model3 trajectories against the local FakeKingston snapshot.

No new trajectory simulation, backend execution, account, or hardware job.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qpy, transpile
from qiskit.circuit.library import XXPlusYYGate

DIRECTORY = Path(__file__).resolve().parent
DEFAULT_SOURCE = DIRECTORY.parent / "Experiment4/runs/budget_e8ec77f803de/replay_R8192.json"
BASES = ("z", "xy", "yx")


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--trajectory-indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--prefix-steps", type=int, nargs="+", help="default: full saved schedule")
    parser.add_argument("--ancillas", choices=("boundary", "shared"), default="boundary")
    parser.add_argument("--layout", choices=("chain", "automatic"), default="chain")
    parser.add_argument("--physical-path", type=int, nargs="+", help="optional ordered chain: left ancilla, sites, right ancilla")
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    parser.add_argument("--seed-transpiler", type=int, default=11)
    parser.add_argument("--output-directory", type=Path, default=DIRECTORY / "runs")
    parser.add_argument("--no-plots", action="store_true")
    options = parser.parse_args(arguments)
    if options.seed_transpiler < 0:
        raise ValueError("seed-transpiler must be nonnegative")
    for name in ("trajectory_indices", "prefix_steps"):
        values = getattr(options, name)
        if values is not None and (min(values) < 0 or len(set(values)) != len(values)):
            raise ValueError(f"{name.replace('_', '-')} must be distinct nonnegative integers")
    if options.physical_path is not None and options.layout != "chain":
        raise ValueError("physical-path requires chain layout")
    return options


def validate_source(data):
    if data.get("experiment") not in ("Model3/Experiment4", "Model3/Experiment6", "Model3/Experiment7"):
        raise ValueError("source must be an Experiment4, Experiment6 or Experiment7 replay JSON")
    config = data["configuration"]
    p = config["physics"]
    if config["system_formula"] != "even-odd" or p.get("delta", 0) != 0:
        raise ValueError("only the saved even-odd XX channel is supported")
    if config["order"] not in ("hamiltonian-first", "jump-first"):
        raise ValueError("unsupported factor order")
    n = p["n_qubits"]
    if not isinstance(n, int) or n < 2:
        raise ValueError("n-qubits must be an integer >= 2")
    if any(not np.isfinite(p[k]) or p[k] <= 0 for k in ("J", "gamma", "exchange_prefactor")):
        raise ValueError("positive finite J, gamma and exchange prefactor required")
    fractions = np.array([p["p_left"], 1-p["p_left"], p["p_right"], 1-p["p_right"]])
    if not np.all(np.isfinite(fractions)) or np.any(fractions < 0):
        raise ValueError("invalid reservoir populations")
    policy = config["sampling_policy"]
    if policy not in ("uniform", "rate-weighted"):
        raise ValueError("unsupported sampling policy")
    probabilities = fractions / 2 if policy == "rate-weighted" else np.full(4, .25)
    schedule = data["schedule"]
    if not np.allclose(probabilities, schedule["jump_probabilities"], atol=1e-14, rtol=1e-14):
        raise ValueError("saved sampling probabilities disagree with physics")
    dts = np.asarray(schedule["step_dts"], dtype=float)
    choices = np.asarray(schedule["jump_choices"])
    if choices.size == 0:
        choices = choices.astype(int)
    populations = np.asarray(data["preparation"]["populations"], dtype=float)
    if dts.ndim != 1 or not np.all(np.isfinite(dts)) or np.any(dts <= 0):
        raise ValueError("invalid step sizes")
    if (choices.ndim != 2 or choices.shape != (config["trajectories"], len(dts))
            or not np.issubdtype(choices.dtype, np.integer) or np.any((choices < 0) | (choices > 3))):
        raise ValueError("invalid saved trajectory choices")
    if np.any(probabilities[choices] <= 0):
        raise ValueError("trajectory selects a zero-probability jump")
    if populations.shape != (n,) or not np.all(np.isfinite(populations)) or np.any((populations < 0) | (populations > 1)):
        raise ValueError("invalid product-state populations")
    if not np.allclose(np.r_[0., np.cumsum(dts)], data["times"], rtol=1e-12, atol=1e-12):
        raise ValueError("saved time nodes disagree with timesteps")


def add_preparation(circuit, system, populations):
    """Prepare a diagonal product mixture by unconditioned Z measurements.

    Outcome records are unused by subsequent gates. Do not reset the system
    after measuring: each shot must retain its randomly prepared local bits.
    """
    fractional = any(0 < p < 1 for p in populations)
    register = ClassicalRegister(len(system), "preparation") if fractional else None
    if register is not None:
        circuit.add_register(register)
    for j, p in enumerate(populations):
        if p == 1:
            circuit.x(system[j])
        elif 0 < p < 1:
            if p == .5:
                circuit.h(system[j])
            else:
                circuit.ry(2*np.arcsin(np.sqrt(p)), system[j])
            circuit.measure(system[j], register[j])


def add_step(circuit, system, baths, dt, choice, physics, probabilities, order):
    def hamiltonian():
        angle = 4*physics["exchange_prefactor"]*physics["J"]*dt
        for parity in (0, 1):
            for j in range(parity, len(system)-1, 2):
                circuit.append(XXPlusYYGate(angle), [system[j], system[j+1]])

    def jump():
        side = int(choice >= 2)
        site = system[-1] if side else system[0]
        bath = baths[side] if len(baths) == 2 else baths[0]
        fraction = (physics["p_left"], 1-physics["p_left"],
                    physics["p_right"], 1-physics["p_right"])[choice]
        angle = 2*np.sqrt(physics["gamma"]*fraction*dt/probabilities[choice])
        if choice % 2 == 0:
            circuit.x(bath)  # Pump: exchange with an occupied bath qubit.
        circuit.append(XXPlusYYGate(angle), [site, bath])
        circuit.reset(bath)

    if order == "hamiltonian-first":
        hamiltonian()
        jump()
    else:
        jump()
        hamiltonian()


def build_body(data, trajectory, steps=None, ancillas="boundary"):
    p, config = data["configuration"]["physics"], data["configuration"]
    n = p["n_qubits"]
    count = len(data["schedule"]["step_dts"]) if steps is None else steps
    if not 0 <= trajectory < config["trajectories"] or not 0 <= count <= len(data["schedule"]["step_dts"]):
        raise ValueError("trajectory or prefix outside saved source")
    if ancillas not in ("boundary", "shared"):
        raise ValueError("unknown ancilla strategy")
    system, bath = QuantumRegister(n, "site"), QuantumRegister(2 if ancillas == "boundary" else 1, "bath")
    circuit = QuantumCircuit(system, bath, name=f"trajectory_{trajectory}_steps_{count}")
    add_preparation(circuit, system, data["preparation"]["populations"])
    for dt, choice in zip(data["schedule"]["step_dts"][:count], data["schedule"]["jump_choices"][trajectory][:count]):
        add_step(circuit, system, bath, dt, choice, p, data["schedule"]["jump_probabilities"], config["order"])
    circuit.metadata = {"experiment": "Model3/Experiment5", "trajectory": trajectory, "steps": count,
        "ancillas": ancillas, "preparation": "unconditioned product-mixture measurements; outcomes ignored"}
    return circuit


def chain_layout(backend, n, ancillas="boundary", supplied=None):
    """Find an actual CZ-connected path, then map sites between boundary baths."""
    size = n + (2 if ancillas == "boundary" else 1)
    adjacency = {j: set() for j in range(backend.num_qubits)}
    for a, b in backend.coupling_map.get_edges():
        if backend.target.instruction_supported(operation_name="cz", qargs=(a, b)):
            adjacency[a].add(b)
            adjacency[b].add(a)
    if supplied is not None:
        path = list(supplied)
        if (len(path) != size or len(set(path)) != size or any(q not in adjacency for q in path)
                or any(b not in adjacency[a] for a, b in zip(path, path[1:]))):
            raise ValueError("physical-path must be a simple CZ-connected path with one vertex per logical qubit")
    else:
        def extend(path):
            if len(path) == size:
                return path
            for q in sorted(adjacency[path[-1]]-set(path), key=lambda j: (len(adjacency[j]), j)):
                result = extend(path + [q])
                if result is not None:
                    return result
            return None
        path = None
        for first in sorted(adjacency, key=lambda j: (len(adjacency[j]), j)):
            path = extend([first])
            if path is not None:
                break
        if path is None:
            raise ValueError("no CZ-connected path large enough for this chain")
    layout = path[1:-1] + [path[0], path[-1]] if ancillas == "boundary" else path[1:] + [path[0]]
    return layout, path


def append_readout(body, physical_sites, basis):
    """Native end rotations, appended at the final layout without rerouting."""
    if basis not in BASES:
        raise ValueError("unknown readout basis")
    circuit = body.copy(name=f"{body.name}_{basis}")
    register = ClassicalRegister(len(physical_sites), "outcome")
    circuit.add_register(register)
    for j, q in enumerate(physical_sites):
        letter = "Z" if basis == "z" else basis[j % 2].upper()
        if letter == "X":
            circuit.rz(np.pi/2, q)
            circuit.sx(q)
            circuit.rz(np.pi/2, q)
        elif letter == "Y":
            circuit.sx(q)
            circuit.rz(np.pi/2, q)
        circuit.measure(q, register[j])
    circuit.metadata = {**(body.metadata or {}), "readout_basis": basis,
                        "physical_sites": list(physical_sites), "ignore_preparation_records": True}
    return circuit


def metrics(circuit):
    counts = dict(circuit.count_ops())
    active = {circuit.find_bit(q).index for instruction in circuit.data for q in instruction.qubits}
    return {"depth": circuit.depth(),
        "two_qubit_depth": circuit.depth(filter_function=lambda i: i.operation.num_qubits == 2 and i.operation.name != "barrier"),
        "two_qubit_gates": sum(i.operation.num_qubits == 2 and i.operation.name != "barrier" for i in circuit.data),
        "operations": circuit.size(), "operation_counts": counts,
        "active_qubits": len(active), "allocated_qubits": circuit.num_qubits,
        "classical_bits": circuit.num_clbits}


def assert_native(circuit, backend):
    for instruction in circuit.data:
        qargs = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        if not backend.target.instruction_supported(operation_name=instruction.operation.name, qargs=qargs):
            raise ValueError(f"Unsupported target operation: {instruction.operation.name}{qargs}")


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def make_plots(directory, result, source, options):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = result["records"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for trajectory in options.trajectory_indices:
        selected = [r for r in rows if r["trajectory"] == trajectory]
        axes[0].plot([r["steps"] for r in selected], [r["logical_body"]["depth"] for r in selected],
                     "o--", alpha=.5, label=f"Path {trajectory}, logical body")
        axes[0].plot([r["steps"] for r in selected], [r["readouts"]["z"]["depth"] for r in selected],
                     "o-", label=f"Path {trajectory}, native Z circuit")
        axes[1].plot([r["steps"] for r in selected], [r["native_body"]["two_qubit_gates"] for r in selected], "o-", label=f"Path {trajectory}")
    axes[0].set(xlabel="Trotter steps", ylabel="Circuit depth (layers)")
    axes[1].set(xlabel="Trotter steps", ylabel="Native two-qubit gates (CZ)")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    figure.suptitle("Model3 Experiment5: FakeKingston offline compilation; preparation and resets included")
    folder = directory / "figures"
    folder.mkdir(exist_ok=True)
    figure.savefig(folder / "depth.png", dpi=180)
    plt.close(figure)
    one = build_body(source, options.trajectory_indices[0], min(1, len(source["schedule"]["step_dts"])), options.ancillas)
    (directory / "one_step.txt").write_text(str(one.draw(output="text", fold=160)))
    figure = one.draw(output="mpl", fold=35, idle_wires=False)
    figure.savefig(folder / "one_step.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main(options, backend=None):
    raw = options.source.read_bytes()
    source = json.loads(raw)
    validate_source(source)
    steps = len(source["schedule"]["step_dts"])
    prefixes = sorted(options.prefix_steps if options.prefix_steps is not None else [steps])
    if any(p > steps for p in prefixes) or max(options.trajectory_indices) >= source["configuration"]["trajectories"]:
        raise ValueError("requested prefix or trajectory outside source")
    if backend is None:
        from qiskit_ibm_runtime.fake_provider import FakeKingston
        backend = FakeKingston()
    n = source["configuration"]["physics"]["n_qubits"]
    layout, path = (chain_layout(backend, n, options.ancillas, options.physical_path)
                    if options.layout == "chain" else (None, None))
    config = {"algorithm": "saved_trajectory_circuits_v1", "source_sha256": hashlib.sha256(raw).hexdigest(),
        "backend": backend.name, "versions": {k: importlib.metadata.version(k) for k in ("qiskit", "qiskit-ibm-runtime")},
        "trajectories": options.trajectory_indices, "prefix_steps": prefixes, "ancillas": options.ancillas,
        "layout": options.layout, "initial_layout": layout, "physical_path": path,
        "optimization_level": options.optimization_level, "seed_transpiler": options.seed_transpiler}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    directory = options.output_directory / f"depth_{digest}"
    circuits = directory / "circuits"
    circuits.mkdir(parents=True, exist_ok=True)
    result = {"experiment": "Model3/Experiment5", "configuration": config, "source": str(options.source.resolve()),
        "source_status": source["run_status"], "source_assessment": source["assessment"],
        "backend_qubits": backend.num_qubits, "backend_operations": sorted(backend.target.operation_names),
        "source_trajectories": source["configuration"]["trajectories"],
        "records": [], "status": "compiling", "figure_generation_status": "disabled" if options.no_plots else "pending",
        "scope": "Offline target compilation only. No backend execution or noise simulation. Selected saved trajectories only.",
        "depth_definition": "Qiskit unweighted dependency layers, including reset and measurement; not scheduled duration.",
        "readout": "Z for populations; alternating XY and YX cover every bond. I_j = prefactor*J*(<Y_j X_j+1>-<X_j Y_j+1>). Ignore preparation records.",
        "compilation": "Optimize body once, then append fixed native end rotations and measurement at final physical layout; no post-readout reoptimization."}
    save_json(directory / "depth.json", result)
    for trajectory in options.trajectory_indices:
        for prefix in prefixes:
            print(f"Compiling saved trajectory {trajectory}, {prefix}/{steps} steps...", flush=True)
            logical = build_body(source, trajectory, prefix, options.ancillas)
            started = time.perf_counter()
            native = transpile(logical, backend=backend, initial_layout=layout,
                optimization_level=options.optimization_level, seed_transpiler=options.seed_transpiler)
            elapsed = time.perf_counter() - started
            final_mapping = native.layout.final_index_layout(filter_ancillas=True)
            readouts = [append_readout(native, final_mapping[:n], basis) for basis in BASES]
            for circuit in [native, *readouts]:
                assert_native(circuit, backend)
            stem = f"trajectory_{trajectory}_steps_{prefix}"
            with (circuits / f"{stem}_logical.qpy").open("wb") as handle:
                qpy.dump(logical, handle)
            with (circuits / f"{stem}_native.qpy").open("wb") as handle:
                qpy.dump([native, *readouts], handle)
            row = {"trajectory": trajectory, "steps": prefix, "physical_time": float(sum(source["schedule"]["step_dts"][:prefix])),
                "jump_counts": np.bincount(source["schedule"]["jump_choices"][trajectory][:prefix], minlength=4).tolist(),
                "logical_body": metrics(logical), "native_body": metrics(native),
                "readouts": {basis: metrics(circuit) for basis, circuit in zip(BASES, readouts)},
                "final_logical_to_physical": final_mapping, "transpilation_seconds": elapsed,
                "logical_qpy": f"circuits/{stem}_logical.qpy", "native_qpy": f"circuits/{stem}_native.qpy"}
            result["records"].append(row)
            save_json(directory / "depth.json", result)
            print(f"  native Z depth={row['readouts']['z']['depth']}, 2q depth={row['native_body']['two_qubit_depth']}, CZ={row['native_body']['operation_counts'].get('cz', 0)}", flush=True)
    result["status"] = "complete"
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(directory / "depth.json", result)
    report = ["# Experiment5: FakeKingston circuit depth", "", result["scope"], "",
        f"Source: `{result['source']}` (status `{result['source_status']}`).",
        f"Ancillas: {options.ancillas}; physical chain: `{path}`; optimization level {options.optimization_level}; seed {options.seed_transpiler}.", "",
        "Preparation is a diagonal product mixture made by rotations and mid-circuit Z measurements; preparation results are ignored, with no system reset. All selected jump resets are included.", "",
        "| Trajectory | Steps | Logical body depth | Native body depth | Full Z depth | Full XY/YX depth | Two-qubit depth | CZ gates | Resets |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["records"]:
        b, reads = row["native_body"], row["readouts"]
        report.append(f"| {row['trajectory']} | {row['steps']} | {row['logical_body']['depth']} | {b['depth']} | {reads['z']['depth']} | {reads['xy']['depth']}/{reads['yx']['depth']} | {b['two_qubit_depth']} | {b['operation_counts'].get('cz', 0)} | {b['operation_counts'].get('reset', 0)} |")
    report += ["", result["depth_definition"], "", result["compilation"], "",
        "Depths describe the compiled subset, not all saved R trajectories or a minimum-depth proof. Three final measurement settings cover all populations and currents. R repetitions and measurement shots increase total work, not depth of one trajectory.", "",
        "Native QPY files contain four circuits in order: body, Z readout, XY readout, YX readout. Logical QPY files contain the body. depth.json records all gate counts and final layouts.", "",
        "The previous ideal covariance tolerance does not establish noisy hardware accuracy. No shots, Aer sampling, hardware jobs, timing schedule, or duration prediction were performed."]
    (directory / "Depth.md").write_text("\n".join(report) + "\n")
    if not options.no_plots:
        make_plots(directory, result, source, options)
        result["figure_generation_status"] = "complete"
        save_json(directory / "depth.json", result)
    if hashlib.sha256(options.source.read_bytes()).hexdigest() != config["source_sha256"]:
        raise ValueError("source changed during compilation")
    print(f"Saved depth report: {directory / 'Depth.md'}", flush=True)
    return result


if __name__ == "__main__":
    try:
        main(parse_arguments())
    except ValueError as error:
        raise SystemExit(str(error)) from error
