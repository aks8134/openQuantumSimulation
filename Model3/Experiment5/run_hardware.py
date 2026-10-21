"""Execute saved Model3 trajectories on IBM hardware, or recover a saved job.

Recompile logical circuits for the requested LIVE backend. FakeKingston QPY
circuits are never submitted to a different target. Only output observables
are measured; classical forward-validation steps are excluded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from qiskit import qpy, transpile

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from IBMRuntime.result import Err
from IBMRuntime.utils import load_ibm_account
from library.reservoir_scaling import atomic_json
from Model3.Experiment5 import circuit_depth as circuits

DEFAULT_SOURCE = DIRECTORY/"inputs/cosine72_seed29_paths0-3.json"


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--backend", default="ibm_phoenix")
    parser.add_argument("--account-file", type=Path, help="default IBMRuntime/apikey.json if present; otherwise the saved Qiskit account")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--trajectory-indices", type=int, nargs="+",
                           help="saved trajectory indices to execute (default: 0 1 2 3)")
    selection.add_argument("--trajectory-count", type=int, metavar="N",
                           help="execute the first N saved trajectories (indices 0 through N-1)")
    parser.add_argument("--shots", type=int, default=8192, help="shots per trajectory and measurement basis")
    parser.add_argument("--batch-size", type=int, default=12, help="circuits per Runtime job; batches are collected sequentially")
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    parser.add_argument("--seed-transpiler", type=int, default=11)
    parser.add_argument("--layout", choices=("chain", "automatic"), default="chain")
    parser.add_argument("--physical-path", type=int, nargs="+")
    parser.add_argument("--compile-only", action="store_true", help="resolve the live target and save circuits without submitting")
    parser.add_argument("--resume", type=Path, help="recover a manifest; collect submitted jobs and submit only batches with no recorded job ID")
    parser.add_argument("--output-directory", type=Path, default=DIRECTORY/"hardware_runs")
    options = parser.parse_args(arguments)
    if options.trajectory_count is not None:
        if options.trajectory_count < 1:
            raise ValueError("trajectory-count must be positive")
        options.trajectory_indices = list(range(options.trajectory_count))
    elif options.trajectory_indices is None:
        options.trajectory_indices = [0, 1, 2, 3]
    if not options.backend.strip():
        raise ValueError("backend name cannot be empty")
    if options.shots < 1 or options.batch_size < 1 or options.seed_transpiler < 0:
        raise ValueError("shots and batch-size must be positive; seed-transpiler must be nonnegative")
    if min(options.trajectory_indices) < 0 or len(set(options.trajectory_indices)) != len(options.trajectory_indices):
        raise ValueError("trajectory-indices must be distinct nonnegative integers")
    if options.physical_path is not None and options.layout != "chain":
        raise ValueError("physical-path requires chain layout")
    if options.resume is not None and options.compile_only:
        raise ValueError("resume and compile-only cannot be combined")
    return options


def make_service(options):
    from qiskit_ibm_runtime import QiskitRuntimeService
    path = options.account_file
    if path is None and (ROOT/"IBMRuntime/apikey.json").is_file():
        path = ROOT/"IBMRuntime/apikey.json"
    if path is None:
        return QiskitRuntimeService()
    loaded = load_ibm_account(path)
    if isinstance(loaded, Err):
        raise ValueError(f"cannot load IBM account: {loaded.error.message}")
    account = loaded.value
    kwargs = {"channel": account.channel, "token": account.api_key}
    if account.instance is not None:
        kwargs["instance"] = account.instance
    return QiskitRuntimeService(**kwargs)


def require_validated_source(source):
    circuits.validate_source(source)
    if source.get("run_status") == "passed":
        return
    provenance = source.get("provenance", {})
    if (source.get("artifact_type") == "model3_hardware_input_v1" and source.get("run_status") == "hardware_input"
            and provenance.get("full_replay_run_status") == "passed"
            and len(provenance.get("saved_trajectory_indices", [])) == source["configuration"]["trajectories"]):
        return
    raise ValueError("source must be a passing saved replay or an exported validated hardware input")


def prepare(source, backend, options):
    require_validated_source(source)
    if max(options.trajectory_indices) >= source["configuration"]["trajectories"]:
        raise ValueError("trajectory index outside saved replay")
    if "reset" not in backend.target.operation_names or "measure" not in backend.target.operation_names:
        raise ValueError("backend must support resets and measurements")
    n = source["configuration"]["physics"]["n_qubits"]
    layout, path = (circuits.chain_layout(backend, n, "boundary", options.physical_path)
                    if options.layout == "chain" else (None, None))
    publications, records = [], []
    for trajectory in options.trajectory_indices:
        logical = circuits.build_body(source, trajectory, ancillas="boundary")
        native = transpile(logical, backend=backend, initial_layout=layout,
            optimization_level=options.optimization_level, seed_transpiler=options.seed_transpiler)
        mapping = native.layout.final_index_layout(filter_ancillas=True)
        for basis in circuits.BASES:
            circuit = circuits.append_readout(native, mapping[:n], basis)
            circuits.assert_native(circuit, backend)
            records.append({"pub_index": len(publications), "trajectory": trajectory, "basis": basis,
                            "metrics": circuits.metrics(circuit), "final_logical_to_physical": mapping})
            publications.append(circuit)
        print(f"Compiled trajectory {trajectory}: {len(source['schedule']['step_dts'])} steps, Z depth={records[-3]['metrics']['depth']}", flush=True)
    return publications, records, path


def count_observables(counts, n):
    """Register-local bit strings are big endian; site j is bit j from the right."""
    shots = sum(counts.values())
    if shots < 1:
        raise ValueError("empty output counts")
    singles, pairs = np.zeros(n), np.zeros(n-1)
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        if len(bits) != n or set(bits)-{"0", "1"} or not isinstance(count, int) or count < 0:
            raise ValueError("invalid outcome-register counts")
        signs = 1-2*np.array([int(bit) for bit in bits[::-1]])
        singles += count*signs
        pairs += count*signs[:-1]*signs[1:]
    return singles/shots, pairs/shots, shots


def summarize(source, records, observations):
    if len(records) != len(observations):
        raise ValueError("observation count differs from saved circuit count")
    p = source["configuration"]["physics"]
    n, J, prefactor = p["n_qubits"], p["J"], p["exchange_prefactor"]
    by_path = {}
    for record, observation in zip(records, observations):
        singles, pairs, shots = count_observables(observation["outcome_counts"], n)
        by_path.setdefault(record["trajectory"], {})[record["basis"]] = (singles, pairs, shots)
    populations, currents = [], []
    for path in by_path.values():
        populations.append((1-path["z"][0])/2)
        # For even bonds xy measures XY and yx measures YX; odd bonds reverse.
        currents.append(prefactor*J*np.where(np.arange(n-1) % 2 == 0, 1., -1.)*
                        (path["yx"][1]-path["xy"][1]))
    populations, currents = np.asarray(populations), np.asarray(currents)
    means = {"populations": populations.mean(axis=0).tolist(), "excitation_flows": currents.mean(axis=0).tolist()}
    uncertainty = {"paths": len(by_path), "interpretation": "across-path standard errors include finite-shot fluctuations; no hardware accuracy guarantee",
        "populations": (populations.std(axis=0, ddof=1)/np.sqrt(len(by_path))).tolist() if len(by_path) > 1 else None,
        "excitation_flows": (currents.std(axis=0, ddof=1)/np.sqrt(len(by_path))).tolist() if len(by_path) > 1 else None}
    reference = {}
    if "final_covariances_real_gauge" in source:
        covariances = np.asarray(source["final_covariances_real_gauge"]).reshape(-1, n, n)
        average = covariances[list(by_path)].mean(axis=0)
        reference["ideal_same_saved_paths"] = {"populations": np.diag(average).tolist(),
            "excitation_flows": (4*prefactor*J*np.diag(average, 1)).tolist()}
    for name, values in (("physical_stationary", source.get("stationary_observables")),
                         ("ideal_mean_channel", source.get("mean_channel_observables"))):
        if values:
            reference[name] = {family: (np.asarray(values[family])[0, -1].tolist() if name == "ideal_mean_channel" else values[family])
                               for family in ("populations", "excitation_flows")}
    errors = {name: float(max(np.max(np.abs(np.asarray(means["populations"])-values["populations"])),
        np.max(np.abs(np.asarray(means["excitation_flows"])-values["excitation_flows"]))/J)) for name, values in reference.items()}
    return {"observables": means, "per_trajectory": {"indices": list(by_path), "populations": populations.tolist(), "excitation_flows": currents.tolist()},
            "across_path_standard_errors": uncertainty, "references": reference, "local_observable_errors": errors,
            "scope": "output only; hardware noise, preparation-bit noise and measurement shots are present; the classical R=8192 tolerance result is not a hardware tolerance claim"}


def collect_and_submit(manifest, directory, publications, service, backend, sampler_factory=None):
    if sampler_factory is None:
        from qiskit_ibm_runtime import SamplerV2
        sampler_factory = SamplerV2
    sampler = None
    for batch in manifest["batches"]:
        if batch.get("observations") is not None:
            continue
        if batch.get("job_id"):
            job = service.job(batch["job_id"])
        else:
            if batch.get("status") == "submitting":
                raise ValueError("a submission was interrupted before its job ID was saved; find that job on IBM Quantum and add its job_id to the manifest before resuming")
            if sampler is None:
                sampler = sampler_factory(mode=backend, options={"dynamical_decoupling": {"enable": False},
                    "twirling": {"enable_gates": False, "enable_measure": False}, "execution": {"init_qubits": True}})
            batch["status"] = "submitting"
            atomic_json(directory/"manifest.json", manifest)
            job = sampler.run(publications[batch["start"]:batch["stop"]], shots=manifest["configuration"]["shots"])
            batch.update(job_id=job.job_id(), status="submitted")
            atomic_json(directory/"manifest.json", manifest)
            print(f"Submitted job {batch['job_id']}: circuits {batch['start']}..{batch['stop']-1}", flush=True)
        result = job.result()
        if len(result) != batch["stop"]-batch["start"]:
            raise ValueError("Runtime returned an unexpected number of circuit results")
        observations = []
        for pub in result:
            # Ignore preparation measurements; use only the final system register.
            observations.append({"outcome_counts": {str(k): int(v) for k, v in pub.data.outcome.get_counts().items()}})
        if any(sum(row["outcome_counts"].values()) != manifest["configuration"]["shots"] for row in observations):
            raise ValueError("Runtime output shot count differs from requested shots")
        batch.update(observations=observations, status="collected")
        atomic_json(directory/"manifest.json", manifest)
    manifest["status"] = "collected"
    atomic_json(directory/"manifest.json", manifest)
    return [row for batch in manifest["batches"] for row in batch["observations"]]


def main(options, *, service=None, backend=None, sampler_factory=None):
    if options.resume is not None:
        directory = options.resume.resolve().parent
        manifest = json.loads(options.resume.read_text())
        if manifest.get("experiment") != "Model3/Experiment5/hardware":
            raise ValueError("resume needs a Model3 hardware manifest")
        backend_name = manifest["configuration"]["backend"]
        source = json.loads((directory/"source.json").read_text())
        raw = (directory/"source.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["configuration"]["source_sha256"]:
            raise ValueError("saved source changed after compilation")
        if hashlib.sha256((directory/"isa_circuits.qpy").read_bytes()).hexdigest() != manifest["configuration"]["isa_sha256"]:
            raise ValueError("saved ISA circuits changed after compilation")
        with (directory/"isa_circuits.qpy").open("rb") as handle:
            publications = qpy.load(handle)
        if len(publications) != len(manifest["circuits"]):
            raise ValueError("saved ISA circuit count disagrees with manifest")
        if service is None:
            service = make_service(options)
        if backend is None:
            backend = service.backend(backend_name, use_fractional_gates=False)
        for circuit in publications:
            circuits.assert_native(circuit, backend)
    else:
        raw = options.source.read_bytes()
        source = json.loads(raw)
        require_validated_source(source)
        if max(options.trajectory_indices) >= source["configuration"]["trajectories"]:
            raise ValueError("trajectory index outside saved replay")
        if service is None and backend is None:
            service = make_service(options)
        if backend is None:
            backend = service.backend(options.backend, use_fractional_gates=False)
        publications, records, path = prepare(source, backend, options)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        directory = options.output_directory/f"{options.backend}_{stamp}"
        directory.mkdir(parents=True, exist_ok=False)
        (directory/"source.json").write_bytes(raw)
        with (directory/"isa_circuits.qpy").open("wb") as handle:
            qpy.dump(publications, handle)
        config = {"backend": options.backend, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "isa_sha256": hashlib.sha256((directory/"isa_circuits.qpy").read_bytes()).hexdigest(),
            "trajectory_indices": options.trajectory_indices, "shots": options.shots, "batch_size": options.batch_size,
            "optimization_level": options.optimization_level, "seed_transpiler": options.seed_transpiler,
            "layout": options.layout, "physical_path": path, "ancillas": "boundary",
            "preparation_steps": len(source["schedule"]["step_dts"]),
            "versions": {name: importlib.metadata.version(name) for name in ("qiskit", "qiskit-ibm-runtime")}}
        manifest = {"experiment": "Model3/Experiment5/hardware", "status": "compiled",
            "configuration": config, "circuits": records, "batches": [
                {"start": start, "stop": min(start+options.batch_size, len(publications)), "job_id": None, "status": "not_submitted"}
                for start in range(0, len(publications), options.batch_size)]}
        atomic_json(directory/"manifest.json", manifest)
        print(f"Saved compilation: {directory/'manifest.json'}", flush=True)
        if options.compile_only:
            print("Compile only: no jobs submitted.", flush=True)
            return manifest
    if sampler_factory is None and service is None:
        raise ValueError("Runtime service required for hardware execution")
    observations = collect_and_submit(manifest, directory, publications, service, backend, sampler_factory)
    result = {"experiment": "Model3/Experiment5/hardware", "configuration": manifest["configuration"],
              "job_ids": [batch["job_id"] for batch in manifest["batches"]],
              **summarize(source, manifest["circuits"], observations)}
    atomic_json(directory/"hardware_results.json", result)
    print(f"Results: {directory/'hardware_results.json'}", flush=True)
    print(f"Output errors: {result['local_observable_errors']}", flush=True)
    return result


if __name__ == "__main__":
    try:
        main(parse_arguments())
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
