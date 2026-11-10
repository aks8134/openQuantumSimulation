"""Compile the saved FinalModel schedules against a complete Phoenix target.

Only capture-target contacts IBM, to read backend metadata. There is no QPU
submission path. compile uses whole circuits and a local frozen target.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
DEFAULT_STUDY = DIRECTORY / "runs/full_a029bea5f96d"
DEFAULT_INPUT = DIRECTORY / "inputs/phoenix_schedules.json"
DEFAULT_TARGET = DIRECTORY / "phoenix_targets/ibm_phoenix"
MODELS = ("model1", "model2", "model3")
FAMILIES = ("fixed", "constant", "warped_cosine")
BASES = ("z", "xy", "yx")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".partial", delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def sealed(value):
    return {**value, "sha256": digest(value)}


def read_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = value.pop("sha256")
    if digest(value) != expected:
        raise ValueError(f"Content hash mismatch: {path}")
    return {**value, "sha256": expected}


def versions():
    return {"python": list(sys.version_info[:3]), **{n: importlib.metadata.version(n)
            for n in ("qiskit", "qiskit-ibm-runtime", "numpy")}}


def steps(case):
    values = ([case["constant_dt"]] * case["steps"] if "constant_dt" in case else case["step_dts"])
    if (len(values) != case["steps"] or not values
            or any(not math.isfinite(v) or v <= 0 for v in values)):
        raise ValueError(f"Invalid saved schedule: {case['id']}")
    if digest(values) != case["schedule_sha256"]:
        raise ValueError(f"Schedule hash mismatch: {case['id']}")
    return values


def validate_case(case):
    n, spec = case["N"], case["physics"]
    if type(n) is not int or not 2 <= n <= 100 or type(case["steps"]) is not int:
        raise ValueError("Invalid system size or step count")
    if case["model"] not in MODELS or case["family"] not in FAMILIES:
        raise ValueError("Unknown model or schedule family")
    if case["id"] != f"{case['model']}_N{n}_{case['family']}":
        raise ValueError("Case identity mismatch")
    if any(not math.isfinite(spec[k]) or spec[k] <= 0 for k in ("J", "gamma", "prefactor")):
        raise ValueError("Invalid positive physical parameters")
    if any(not math.isfinite(spec[k]) or not 0 <= spec[k] <= 1 for k in ("p_left", "p_right")):
        raise ValueError("Invalid reservoir populations")
    if spec["preparation"] not in ("center_excitation", "half_mixed", "vacuum", "filled"):
        raise ValueError("Unsupported preparation")
    values = steps(case)
    if not math.isclose(sum(values), case["physical_time"], rel_tol=1e-11, abs_tol=1e-10):
        raise ValueError("Saved physical time differs from timestep sum")


def export_inputs(study, output):
    study = Path(study)
    provenance = json.loads((study / "configuration.json").read_text(encoding="utf-8"))
    config, cases = provenance["configuration"], []
    for model, spec in config["models"].items():
        for n in config["sizes"]:
            source = study / f"schedules_{model}_N{n}.json"
            saved = json.loads(source.read_text(encoding="utf-8"))
            if saved["parameters"] != spec or saved["N"] != n or saved["model"] != model:
                raise ValueError("Schedule checkpoint differs from its declared configuration")
            for family in FAMILIES:
                selection = saved["result"]["families"][family]
                validation = selection.get("validation", {})
                if (selection["status"] != "mean_forward_window_passed" or not validation.get("passed")
                        or validation["forward_duration"] < config["forward_window"] - 1e-10
                        or len(validation["forward_errors"]) != validation["forward_steps"] + 1
                        or max(validation["forward_errors"]) > config["mean_tolerance"]):
                    raise ValueError(f"Source schedule lacks complete mean validation: {source}, {family}")
                dts = selection["step_dts"]
                case = dict(id=f"{model}_N{n}_{family}", model=model, N=n, family=family,
                    physics=spec, steps=selection["selected_steps"], physical_time=selection["physical_time"],
                    schedule_sha256=digest(dts), source_checkpoint_sha256=file_digest(source),
                    mean_validation=dict(maximum_forward_error=validation["maximum_forward_error"],
                        forward_duration=validation["forward_duration"], tolerance=config["mean_tolerance"]))
                if all(v == dts[0] for v in dts):
                    case["constant_dt"] = dts[0]
                else:
                    case["step_dts"] = dts
                validate_case(case)
                cases.append(case)
    payload = sealed(dict(artifact_type="finalmodel_phoenix_schedules_v1", source_study=study.name,
        source_configuration_sha256=file_digest(study / "configuration.json"), source_configuration=config,
        channel=dict(order="hamiltonian-first", system_formula="even-odd", sampling_policy="rate-weighted"),
        scope="validated deterministic means; compilation paths do not establish sampling accuracy", cases=cases))
    atomic_json(output, payload)
    return payload


def read_inputs(path):
    payload = read_sealed(path)
    if payload["artifact_type"] != "finalmodel_phoenix_schedules_v1":
        raise ValueError("Wrong schedule input format")
    if payload["channel"] != dict(order="hamiltonian-first", system_formula="even-odd", sampling_policy="rate-weighted"):
        raise ValueError("Unsupported saved channel convention")
    identities = set()
    for case in payload["cases"]:
        validate_case(case)
        if case["id"] in identities:
            raise ValueError("Duplicate schedule identity")
        identities.add(case["id"])
    return payload


def make_service(account_file=None, account_name=None, instance=None):
    from qiskit_ibm_runtime import QiskitRuntimeService
    if account_file is None and account_name is None and (ROOT / "IBMRuntime/apikey.json").is_file():
        account_file = ROOT / "IBMRuntime/apikey.json"
    if account_file is None:
        return QiskitRuntimeService(**{k: v for k, v in {"name": account_name, "instance": instance}.items() if v is not None})
    account = json.loads(Path(account_file).read_text(encoding="utf-8"))
    token = account.get("apikey", account.get("api_key"))
    channel = account.get("channel", "ibm_quantum_platform")
    account_instance = instance or account.get("crn", account.get("instance"))
    if not isinstance(token, str) or not token or not isinstance(channel, str) or not channel:
        raise ValueError("Account file requires a nonempty apikey/api_key and channel")
    if account_instance is not None and not isinstance(account_instance, str):
        raise ValueError("Account instance must be a string")
    options = dict(channel=channel, token=token)
    if account_instance:
        options["instance"] = account_instance
    return QiskitRuntimeService(**options)


def capture_snapshot(backend, directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Target directory must be empty; use a new directory for another calibration")
    directory.mkdir(parents=True, exist_ok=True)
    target = backend.target
    if target.num_qubits != backend.num_qubits:
        raise ValueError("Backend target does not cover the full backend width")
    instructions = []
    for name in sorted(target.operation_names):
        for qargs, properties in target[name].items():
            instructions.append(dict(name=name, qubits=None if qargs is None else list(qargs),
                duration_seconds=getattr(properties, "duration", None), error=getattr(properties, "error", None)))
    compiler_methods = {}
    for stage in ("translation", "scheduling"):
        getter = getattr(backend, f"get_{stage}_stage_plugin", None)
        method = getter() if getter else None
        if method:
            compiler_methods[f"{stage}_method"] = method
    target_path = directory / "target.pkl"
    with target_path.open("wb") as stream:
        pickle.dump(target, stream, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = sealed(dict(artifact_type="finalmodel_backend_target_v1", backend=backend.name,
        backend_version=str(getattr(backend, "backend_version", "unknown")),
        captured_at_utc=datetime.now(timezone.utc).isoformat(), num_qubits=backend.num_qubits,
        use_fractional_gates=False, compiler_methods=compiler_methods, versions=versions(),
        target_sha256=file_digest(target_path), operation_names=sorted(target.operation_names),
        timing={k: getattr(target, k, None) for k in ("dt", "pulse_alignment", "acquire_alignment", "granularity", "min_length")},
        instructions=instructions))
    atomic_json(directory / "target.json", metadata)
    return metadata


def load_snapshot(directory):
    directory = Path(directory)
    metadata = read_sealed(directory / "target.json")
    if metadata["artifact_type"] != "finalmodel_backend_target_v1":
        raise ValueError("Wrong target snapshot format")
    if metadata["versions"] != versions():
        raise ValueError("Target cache requires the same Python/Qiskit/Runtime/NumPy versions; capture it in the compilation environment")
    if file_digest(directory / "target.pkl") != metadata["target_sha256"]:
        raise ValueError("Target cache hash mismatch")
    with (directory / "target.pkl").open("rb") as stream:
        target = pickle.load(stream)
    if target.num_qubits != metadata["num_qubits"] or sorted(target.operation_names) != metadata["operation_names"]:
        raise ValueError("Target metadata differs from its cached object")
    return target, metadata


def chain_layout(target, n, *, supplied=None, avoid=(), limit=500000, seconds=30.):
    width = n + 2
    eligible = {q for q in range(target.num_qubits) if q not in avoid and all(
        target.instruction_supported(operation_name=op, qargs=(q,)) for op in ("rz", "sx", "x", "reset", "measure"))}
    if width > len(eligible):
        raise ValueError(f"Only {len(eligible)} eligible physical qubits for {width} logical qubits")
    adjacency = {q: set() for q in eligible}
    if "cz" not in target.operation_names:
        raise ValueError("Chain layout requires the CZ connectivity used by the Phoenix hardware workflow")
    for pair in target["cz"]:
        if pair is not None and len(pair) == 2 and set(pair) <= eligible:
            a, b = pair
            adjacency[a].add(b)
            adjacency[b].add(a)
    if supplied is not None:
        path = list(supplied)
        if (len(path) != width or len(set(path)) != width or any(q not in eligible for q in path)
                or any(b not in adjacency[a] for a, b in zip(path, path[1:]))):
            raise ValueError(f"The supplied N={n} path is not a supported simple CZ path")
    else:
        expansions, deadline = 0, time.monotonic() + seconds
        def visit(path, used):
            nonlocal expansions
            expansions += 1
            if expansions > limit or time.monotonic() > deadline:
                raise ValueError(f"N={n} chain-search limit reached; supply a physical path or choose --layout automatic")
            if len(path) == width:
                return path.copy()
            for q in sorted(adjacency[path[-1]] - used, key=lambda v: (len(adjacency[v] - used), v)):
                result = visit(path + [q], used | {q})
                if result is not None:
                    return result
            return None
        path = None
        for start in sorted(eligible, key=lambda q: (len(adjacency[q]), q)):
            path = visit([start], {start})
            if path is not None:
                break
        if path is None:
            raise ValueError(f"No CZ chain of length {width} was found")
    return list(path[1:-1]) + [path[0], path[-1]], path


def build_body(case, seed):
    import numpy as np
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import XXPlusYYGate
    n, spec = case["N"], case["physics"]
    system, bath = QuantumRegister(n, "site"), QuantumRegister(2, "bath")
    circuit = QuantumCircuit(system, bath, name=case["id"])
    if spec["preparation"] == "center_excitation":
        circuit.x(system[n // 2])
    elif spec["preparation"] == "filled":
        circuit.x(system)
    elif spec["preparation"] == "half_mixed":
        register = ClassicalRegister(n, "preparation")
        circuit.add_register(register)
        circuit.h(system)
        circuit.measure(system, register)
    probabilities = np.array([spec["p_left"], 1 - spec["p_left"], spec["p_right"], 1 - spec["p_right"]]) / 2
    choices = np.random.default_rng(seed).choice(4, size=case["steps"], p=probabilities)
    for dt, choice in zip(steps(case), choices):
        for parity in (0, 1):
            for site in range(parity, n - 1, 2):
                circuit.append(XXPlusYYGate(4 * spec["prefactor"] * spec["J"] * dt), [system[site], system[site + 1]])
        side = int(choice >= 2)
        if choice % 2 == 0:
            circuit.x(bath[side])
        circuit.append(XXPlusYYGate(2 * np.sqrt(2 * spec["gamma"] * dt)), [system[-1] if side else system[0], bath[side]])
        circuit.reset(bath[side])
    circuit.metadata = dict(experiment="FinalModel/Experiment1/PhoenixResources", case=case["id"],
        schedule_sha256=case["schedule_sha256"], seed=seed, steps=case["steps"])
    return circuit, choices


def append_readout(native, physical_sites, basis):
    import numpy as np
    from qiskit import ClassicalRegister
    circuit = native.copy(name=f"{native.name}_{basis}")
    register = ClassicalRegister(len(physical_sites), "outcome")
    circuit.add_register(register)
    for i, q in enumerate(physical_sites):
        letter = "z" if basis == "z" else basis[i % 2]
        if letter == "x":
            circuit.rz(np.pi / 2, q)
            circuit.sx(q)
            circuit.rz(np.pi / 2, q)
        elif letter == "y":
            circuit.sx(q)
            circuit.rz(np.pi / 2, q)
        circuit.measure(q, register[i])
    return circuit


def native_metrics(circuit, target):
    counts, two, active = Counter(), Counter(), set()
    for instruction in circuit.data:
        name = instruction.operation.name
        qubits = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        if name != "barrier" and not target.instruction_supported(operation_name=name, qargs=qubits,
                parameters=instruction.operation.params):
            raise ValueError(f"Instruction unsupported by captured target: {name}{qubits}")
        if getattr(instruction.operation, "blocks", ()):
            raise ValueError("Unexpected control flow needs recursive resource accounting")
        counts[name] += 1
        if name != "barrier":
            active.update(qubits)
            if len(qubits) == 2:
                two[name] += 1
    return dict(operation_counts=dict(counts), operations=sum(counts.values()), two_qubit_counts=dict(two),
        two_qubit_gates=sum(two.values()), depth=circuit.depth(),
        two_qubit_depth=circuit.depth(filter_function=lambda i: i.operation.num_qubits == 2 and i.operation.name != "barrier"),
        allocated_qubits=circuit.num_qubits, active_qubits=len(active), classical_bits=circuit.num_clbits)


def timing_estimate(circuit, target):
    from qiskit.transpiler import PassManager
    from qiskit_ibm_runtime.transpiler.passes.scheduling import ALAPScheduleAnalysis
    if target.dt is None or target.dt <= 0:
        raise ValueError("Target has no positive dt")
    manager = PassManager([ALAPScheduleAnalysis(target=target)])
    manager.run(circuit)
    ends = {}
    for node, (block, start) in manager.property_set["node_start_time"].items():
        if node.name == "barrier":
            duration = 0
        elif node.name == "delay":
            if node.op.unit != "dt":
                raise ValueError("Delay is not expressed in dt")
            duration = int(node.op.duration)
        else:
            qubits = tuple(circuit.find_bit(q).index for q in node.qargs)
            value = target[node.name][qubits].duration
            if value is None or not math.isfinite(value) or value < 0:
                raise ValueError(f"Missing target duration: {node.name}{qubits}")
            duration = int(target.seconds_to_dt(value))
        ends[int(block)] = max(ends.get(int(block), 0), int(start) + duration)
    return dict(status="modeled", dynamic_blocks=len(ends), block_duration_dt=ends,
        modeled_block_duration_sum_seconds=sum(ends.values()) * target.dt,
        scope="ALAP block-duration sum from captured instruction durations; controller/reset block-transition overhead is not included; not measured wall-clock runtime")


def compile_one(case, target, metadata, settings, seed, *, initial_layout=None, physical_path=None, output=None):
    from qiskit import qpy, transpile
    body, choices = build_body(case, seed)
    routing = []
    def callback(**values):
        name = type(values["pass_"]).__name__
        if name in ("SabreSwap", "BasicSwap", "LookaheadSwap", "SabreLayout"):
            routing.append(dict(pass_name=name, swaps_in_dag=int(values["dag"].count_ops().get("swap", 0))))
    options = dict(target=target, optimization_level=settings["optimization_level"],
        seed_transpiler=settings["transpiler_seed"], initial_layout=initial_layout, num_processes=1,
        approximation_degree=1.0, callback=callback, **metadata["compiler_methods"])
    if initial_layout is not None:
        options["routing_method"] = "none"
    started = time.perf_counter()
    print(f"Transpiling full circuit: {case['id']}, seed={seed}, {case['N'] * case['steps']:,} exchange blocks", flush=True)
    native = transpile(body, **options)
    del body
    final_mapping = native.layout.final_index_layout(filter_ancillas=True)
    initial_mapping = native.layout.initial_index_layout(filter_ancillas=True)
    if initial_layout is not None and final_mapping != initial_layout:
        raise ValueError("Compiler changed the required fixed-chain mapping")
    rows = []
    for basis in settings["bases"]:
        circuit = append_readout(native, final_mapping[:case["N"]], basis)
        metrics = native_metrics(circuit, target)
        timing = dict(status="not_requested")
        if settings["timing"]:
            try:
                timing = timing_estimate(circuit, target)
            except Exception as error:
                timing = dict(status="unavailable", reason=f"{type(error).__name__}: {error}")
        row = dict(basis=basis, metrics=metrics, timing=timing)
        if settings["save_qpy"]:
            path = Path(output) / f"{case['id']}_seed{seed}_{basis}.qpy"
            temporary = path.with_suffix(".qpy.partial")
            with temporary.open("wb") as stream:
                qpy.dump(circuit, stream)
            temporary.replace(path)
            row["qpy"] = dict(file=path.name, sha256=file_digest(path))
        rows.append(row)
        del circuit
    return dict(status="compiled", case_id=case["id"], model=case["model"], N=case["N"], family=case["family"],
        steps=case["steps"], source_exchanges=case["N"] * case["steps"], source_logical_qubits=case["N"] + 2,
        seed=seed, schedule_sha256=case["schedule_sha256"], jump_choices_sha256=hashlib.sha256(choices.tobytes()).hexdigest(),
        initial_logical_to_physical=initial_mapping, final_logical_to_physical=final_mapping,
        physical_path=physical_path, routing_pass_records=routing, readouts=rows,
        wall_seconds=time.perf_counter() - started,
        scope="whole preparation circuit transpiled to frozen backend target; native readout appended at final system mapping; no execution")


def select_cases(payload, options):
    cases = [c for c in payload["cases"] if c["model"] in options.models and c["family"] in options.families
             and (options.sizes is None or c["N"] in options.sizes)]
    if not cases:
        raise ValueError("No saved schedules match the filters")
    return cases


def plan_rows(cases, seeds, bases):
    return [dict(model=c["model"], N=c["N"], family=c["family"], steps=c["steps"],
        source_exchanges=c["N"] * c["steps"], source_logical_qubits=c["N"] + 2,
        paths=len(seeds), readout_circuits=len(seeds) * len(bases)) for c in cases]


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(output, *, plot=True):
    output = Path(output)
    manifest = read_sealed(output / "manifest.json")
    rows = []
    for item in manifest["requested"]:
        path = output / "cases" / f"{item['id']}_seed{item['seed']}.json"
        saved = read_sealed(path) if path.exists() else {"status": "not_completed"}
        if saved["status"] == "compiled" and saved["context_sha256"] != digest(manifest["context"]):
            raise ValueError(f"Saved compilation context differs from the run manifest: {path}")
        base = {k: item[k] for k in ("model", "N", "family", "steps", "seed")}
        for basis in manifest["settings"]["bases"]:
            readout = next((r for r in saved.get("readouts", []) if r["basis"] == basis), {})
            metrics, timing = readout.get("metrics", {}), readout.get("timing", {})
            rows.append(dict(**base, basis=basis, status=saved["status"], CZ=metrics.get("operation_counts", {}).get("cz"),
                two_qubit_gates=metrics.get("two_qubit_gates"), native_depth=metrics.get("depth"),
                two_qubit_depth=metrics.get("two_qubit_depth"), active_qubits=metrics.get("active_qubits"),
                reset=metrics.get("operation_counts", {}).get("reset"), measure=metrics.get("operation_counts", {}).get("measure"),
                timing_status=timing.get("status"), modeled_block_sum_seconds=timing.get("modeled_block_duration_sum_seconds"),
                reason=saved.get("reason")))
    write_csv(output / "resources.csv", rows)
    good = [r for r in rows if r["status"] == "compiled"]
    completed = sum((output / "cases" / f"{r['id']}_seed{r['seed']}.json").is_file() and
        read_sealed(output / "cases" / f"{r['id']}_seed{r['seed']}.json")["status"] == "compiled"
        for r in manifest["requested"])
    lines = ["# Backend-specific compilation resources", "",
        f"Backend: `{manifest['target']['backend']}`; captured UTC: `{manifest['target']['captured_at_utc']}`.", "",
        f"Completed {completed}/{len(manifest['requested'])} requested preparation paths ({len(good)} readout circuits). Every row is a whole-circuit transpilation; failed, capped and unfinished rows remain explicit in resources.csv.", "",
        f"Layout policy: `{manifest['settings']['layout']}`; optimization level {manifest['settings']['optimization_level']}; transpiler seed {manifest['settings']['transpiler_seed']}. Source timesteps and physical parameters are preserved. Paths are generated with the recorded fixed seeds.", "",
        "Preparation measurements and bath resets are included. Native Z/XY/YX readout rotations are appended at the actual final system locations, following the existing hardware workflow. Counts are per path and basis, with no shot multiplier. A small compilation sample is not trajectory-accuracy validation.", "",
        "Depth is an unweighted instruction dependency depth. Optional timing is a local dynamic-block duration model from the frozen target, with unknown controller/block-transition overhead; it is not measured runtime. Native gate counts include actual routing/optimization effects; no 2NK formula or block-stitching extrapolation is substituted.", "",
        "The target identity, source input hash, compiler settings/versions, layouts, complete native operation counts and optional QPY hashes are recorded beside this report. No hardware jobs are submitted. See [the per-path CSV](resources.csv).", ""]
    if good and plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        basis = "z" if "z" in manifest["settings"]["bases"] else manifest["settings"]["bases"][0]
        models = sorted({r["model"] for r in rows})
        fig, axes = plt.subplots(2, len(models), figsize=(4.5 * len(models), 7.5), squeeze=False)
        for col, model in enumerate(models):
            for family in FAMILIES:
                group = [r for r in good if r["model"] == model and r["family"] == family and r["basis"] == basis]
                sizes = sorted({r["N"] for r in group})
                for index, key in enumerate(("two_qubit_gates", "native_depth")):
                    means = [sum(r[key] for r in group if r["N"] == n) / sum(r["N"] == n for r in group) for n in sizes]
                    if sizes:
                        axes[index, col].loglog(sizes, means, "o-", label=family)
                    axes[index, col].set(title=model, xlabel="System qubits N", ylabel="Native two-qubit gates" if index == 0 else "Compiled dependency depth")
                    axes[index, col].grid(alpha=.2, which="both")
        axes[0, 0].legend()
        fig.suptitle(f"{manifest['target']['backend']}: whole-circuit compilation, {basis.upper()} readout")
        fig.tight_layout()
        fig.savefig(output / "native_scaling.png", dpi=200)
        fig.savefig(output / "native_scaling.pdf")
        plt.close(fig)
        lines.append("![Backend-specific resources](native_scaling.png)")
    (output / "Results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def compile_command(options):
    payload = read_inputs(options.input)
    cases = select_cases(payload, options)
    target, metadata = load_snapshot(options.target_dir)
    if any(c["N"] + 2 > target.num_qubits for c in cases):
        raise ValueError("Some requested schedules require more qubits than the captured target")
    settings = dict(layout=options.layout, optimization_level=options.optimization_level,
        transpiler_seed=options.transpiler_seed, seeds=options.seeds, bases=options.bases,
        timing=options.timing, save_qpy=options.save_qpy, physical_path=options.physical_path,
        layout_map=None if options.layout_map is None else json.loads(options.layout_map.read_text(encoding="utf-8")),
        avoid_qubits=options.avoid_qubits, path_search_limit=options.path_search_limit,
        path_search_seconds=options.path_search_seconds, max_source_exchanges=options.max_source_exchanges,
        timeout_seconds=options.timeout_seconds)
    context = dict(input_sha256=payload["sha256"], target_sha256=metadata["sha256"], settings=settings,
        versions=versions(), script_sha256=file_digest(__file__))
    output = options.output / f"phoenix_{digest(context)[:12]}"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("cases", "tasks", "logs", "circuits"):
        (output / name).mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = read_sealed(manifest_path) if manifest_path.exists() else dict(context=context, target={k: metadata[k]
        for k in ("backend", "captured_at_utc", "num_qubits", "target_sha256", "compiler_methods")}, settings=settings, requested=[])
    if manifest["context"] != context:
        raise ValueError("Saved run context differs from requested compilation")
    requested = {f"{r['id']}_seed{r['seed']}": r for r in manifest["requested"]}
    for case in cases:
        for seed in options.seeds:
            requested[f"{case['id']}_seed{seed}"] = {k: case[k] for k in ("id", "model", "N", "family", "steps")} | {"seed": seed}
    manifest.pop("sha256", None)
    manifest["requested"] = list(requested.values())
    atomic_json(manifest_path, sealed(manifest))
    write_csv(output / "plan.csv", plan_rows(cases, options.seeds, options.bases))
    print(f"OUTPUT={output}", flush=True)
    layouts = {}
    layout_errors = {}
    failures = 0
    for case in cases:
        for seed in options.seeds:
            key = f"{case['id']}_seed{seed}"
            destination = output / "cases" / f"{key}.json"
            if destination.exists():
                saved = read_sealed(destination)
                if saved["status"] == "compiled":
                    if (saved["context_sha256"] != digest(context) or saved["schedule_sha256"] != case["schedule_sha256"]
                            or saved["case_id"] != case["id"] or saved["seed"] != seed
                            or {r["basis"] for r in saved["readouts"]} != set(settings["bases"])):
                        raise ValueError(f"Completed checkpoint provenance mismatch: {destination}")
                    qpy_valid = all(not settings["save_qpy"] or
                        ((output / "circuits" / r["qpy"]["file"]).is_file() and
                         file_digest(output / "circuits" / r["qpy"]["file"]) == r["qpy"]["sha256"])
                        for r in saved["readouts"])
                    if qpy_valid:
                        print(f"Reusing {key}", flush=True)
                        continue
            failure = None
            if options.max_source_exchanges is not None and case["N"] * case["steps"] > options.max_source_exchanges:
                failure = dict(status="source_size_cap", reason="Explicit --max-source-exchanges limit; no approximate resources substituted")
            try:
                if case["N"] in layout_errors:
                    raise ValueError(layout_errors[case["N"]])
                if options.layout == "chain" and case["N"] not in layouts:
                    supplied = options.physical_path
                    if settings["layout_map"] is not None:
                        supplied = settings["layout_map"].get(str(case["N"]))
                        if supplied is None:
                            raise ValueError(f"Layout map has no physical path for N={case['N']}")
                    layouts[case["N"]] = chain_layout(target, case["N"], supplied=supplied,
                        avoid=options.avoid_qubits, limit=options.path_search_limit, seconds=options.path_search_seconds)
            except ValueError as error:
                failure = dict(status="layout_unavailable", reason=str(error))
                layout_errors[case["N"]] = str(error)
            if failure:
                atomic_json(destination, sealed(failure))
                print(f"{key}: {failure['status']}: {failure['reason']}", flush=True)
                failures += 1
                continue
            initial, path = layouts.get(case["N"], (None, None))
            task = dict(case=case, seed=seed, settings=settings, target_dir=str(options.target_dir.resolve()),
                initial_layout=initial, physical_path=path, result=str(destination.resolve()),
                circuit_output=str((output / "circuits").resolve()), context_sha256=digest(context))
            task_file = output / "tasks" / f"{key}.json"
            atomic_json(task_file, task)
            atomic_json(destination, sealed(dict(status="compiling", context_sha256=digest(context))))
            print(f"Compiling {key}: {case['N'] * case['steps']:,} exchanges; log={output / 'logs' / (key + '.log')}", flush=True)
            with (output / "logs" / f"{key}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_compile-one", "--task", str(task_file.resolve())],
                    stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait(timeout=options.timeout_seconds or None)
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                    process.kill()
                    process.wait()
                    status = "timeout" if isinstance(error, subprocess.TimeoutExpired) else "interrupted"
                    atomic_json(destination, sealed(dict(status=status, reason="Worker stopped; no resource estimate substituted")))
                    report(output, plot=False)
                    if status == "interrupted":
                        raise
                    code = -1
            if not destination.exists() or read_sealed(destination)["status"] == "compiling":
                atomic_json(destination, sealed(dict(status="worker_failed", reason=f"Worker exit code {code}; inspect logs/{key}.log")))
            result = read_sealed(destination)
            if result["status"] != "compiled":
                failures += 1
            print(f"{key}: {result['status']}", flush=True)
            report(output, plot=False)
    report(output, plot=True)
    print(output / "Results.md", flush=True)
    return int(failures > 0)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True,
        metavar="{export-inputs,capture-target,plan,compile,report}")
    export = commands.add_parser("export-inputs", help="Package existing validated schedules; no IBM access")
    export.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY)
    export.add_argument("--output", type=Path, default=DEFAULT_INPUT)
    capture = commands.add_parser("capture-target", help="Read the full live backend target once; no execution")
    capture.add_argument("--backend", default="ibm_phoenix")
    capture.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET)
    credentials = capture.add_mutually_exclusive_group()
    credentials.add_argument("--account-file", type=Path)
    credentials.add_argument("--account-name")
    capture.add_argument("--instance")
    for name in ("plan", "compile"):
        sub = commands.add_parser(name, help="Offline size preview" if name == "plan" else "Offline whole-circuit compilation")
        sub.add_argument("--input", type=Path, default=DEFAULT_INPUT)
        sub.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
        sub.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
        sub.add_argument("--sizes", nargs="+", type=int)
        sub.add_argument("--seeds", nargs="+", type=int, default=[29, 30])
        sub.add_argument("--bases", nargs="+", choices=BASES, default=["z"])
        sub.add_argument("--output", type=Path, default=DIRECTORY / "phoenix_runs")
        if name == "plan":
            continue
        sub.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET)
        sub.add_argument("--layout", choices=("chain", "automatic"), default="chain")
        paths = sub.add_mutually_exclusive_group()
        paths.add_argument("--physical-path", nargs="+", type=int)
        paths.add_argument("--layout-map", type=Path, help="JSON mapping N to [left bath, sites..., right bath]")
        sub.add_argument("--avoid-qubits", type=int, nargs="*", default=[])
        sub.add_argument("--path-search-limit", type=int, default=500000)
        sub.add_argument("--path-search-seconds", type=float, default=30.)
        sub.add_argument("--optimization-level", type=int, choices=range(4), default=3)
        sub.add_argument("--transpiler-seed", type=int, default=11)
        sub.add_argument("--max-source-exchanges", type=int, help="Optional explicit memory/workload cap; otherwise no size cap")
        sub.add_argument("--timeout-seconds", type=float, default=0., help="Per-path worker timeout; zero means unlimited")
        sub.add_argument("--timing", action="store_true", help="Compute a labelled local dynamic-block duration model")
        sub.add_argument("--save-qpy", action="store_true", help="Save complete native circuits; can consume substantial disk space")
    summary = commands.add_parser("report", help="Regenerate tables/figures from saved metrics without target/account access")
    summary.add_argument("directory", type=Path)
    worker = commands.add_parser("_compile-one")
    worker.add_argument("--task", type=Path, required=True)
    return p


def main(arguments=None):
    options = parser().parse_args(arguments)
    if options.command == "export-inputs":
        payload = export_inputs(options.study_dir, options.output)
        print(f"Exported {len(payload['cases'])} schedules to {options.output}")
    elif options.command == "capture-target":
        service = make_service(options.account_file, options.account_name, options.instance)
        backend = service.backend(options.backend, use_fractional_gates=False)
        if backend.name != options.backend:
            raise ValueError("Resolved backend differs from requested backend")
        metadata = capture_snapshot(backend, options.target_dir)
        print(f"Captured {metadata['backend']}, {metadata['num_qubits']} qubits, {metadata['captured_at_utc']}; no jobs submitted")
    elif options.command == "report":
        report(options.directory)
    elif options.command == "_compile-one":
        task = json.loads(options.task.read_text(encoding="utf-8"))
        try:
            target, metadata = load_snapshot(task["target_dir"])
            result = compile_one(task["case"], target, metadata, task["settings"], task["seed"],
                initial_layout=task["initial_layout"], physical_path=task["physical_path"], output=task["circuit_output"])
            result["context_sha256"] = task["context_sha256"]
            atomic_json(task["result"], sealed(result))
        except Exception as error:
            atomic_json(task["result"], sealed(dict(status="compilation_failed", reason=f"{type(error).__name__}: {error}")))
            raise
    else:
        if (any(v < 0 for v in options.seeds) or len(set(options.seeds)) != len(options.seeds)
                or len(set(options.bases)) != len(options.bases)):
            raise ValueError("Seeds must be distinct/nonnegative and bases distinct")
        if options.command == "plan":
            payload = read_inputs(options.input)
            rows = plan_rows(select_cases(payload, options), options.seeds, options.bases)
            options.output.mkdir(parents=True, exist_ok=True)
            write_csv(options.output / "plan.csv", rows)
            print(f"{len(rows)} schedules, {sum(r['paths'] for r in rows)} preparation paths, {sum(r['readout_circuits'] for r in rows)} readout circuits")
            largest = max(rows, key=lambda r: r["source_exchanges"])
            print(f"Largest whole circuit: {largest['model']} N={largest['N']} {largest['family']}: {largest['source_exchanges']:,} exchange blocks")
            print(options.output / "plan.csv")
        else:
            if (options.path_search_limit < 1 or not math.isfinite(options.path_search_seconds) or options.path_search_seconds <= 0
                    or not math.isfinite(options.timeout_seconds) or options.timeout_seconds < 0
                    or options.transpiler_seed < 0 or any(q < 0 for q in options.avoid_qubits)
                    or options.max_source_exchanges is not None and options.max_source_exchanges < 1):
                raise ValueError("Invalid search, resource-cap, timeout or seed setting")
            if options.layout == "automatic" and (options.physical_path or options.layout_map or options.avoid_qubits):
                raise ValueError("Fixed paths and avoid-qubits require --layout chain")
            if options.physical_path is not None and (options.sizes is None or len(options.sizes) != 1):
                raise ValueError("--physical-path requires exactly one --sizes value")
            return compile_command(options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
