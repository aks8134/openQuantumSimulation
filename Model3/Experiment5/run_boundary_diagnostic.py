"""Isolated boundary collision/reset controls; no full-chain evolution.

Defaults: both endpoint pairs, pump/loss at 1/12/72 collisions, preparation
baselines and 72-reset controls: 20 circuits x 2048 shots = 40960 shots.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit.library import XXPlusYYGate
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library.reservoir_scaling import atomic_json
from Model3.Experiment5 import circuit_depth as circuits, run_hardware as hardware
from Model3.Experiment5.run_preparation_diagnostic import reference
from Model3.Experiment5.analyze_hardware import moments, serializable, source_integrity

EXPERIMENT = "Model3/Experiment5/boundary_diagnostic"
DEFAULT_REFERENCE = DIRECTORY/"hardware_runs/ibm_phoenix_20261025T150619_925833Z"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse_arguments(arguments=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--steps", nargs="+", type=int, default=[1, 12, 72])
    p.add_argument("--sides", nargs="+", choices=("left", "right"), default=["left", "right"])
    p.add_argument("--shots", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--account-file", type=Path)
    p.add_argument("--seed-transpiler", type=int, default=11)
    p.add_argument("--order-seed", type=int, default=29)
    p.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--compile-only", action="store_true")
    action.add_argument("--resume", type=Path)
    action.add_argument("--analyze-only", type=Path, metavar="RUN_DIRECTORY")
    p.add_argument("--output-directory", type=Path, default=DIRECTORY/"hardware_runs")
    p.add_argument("--analysis-output-directory", type=Path)
    a = p.parse_args(arguments)
    if (a.shots < 2 or a.batch_size < 1 or min(a.steps) < 1 or len(set(a.steps)) != len(a.steps)
            or len(set(a.sides)) != len(a.sides) or min(a.seed_transpiler, a.order_seed) < 0):
        raise ValueError("need shots >=2, positive distinct step counts, distinct sides, positive batch-size and nonnegative seeds")
    a.steps.sort()
    return a


def angle(source, side, direction, dt):
    index = (0 if side == "left" else 2)+(direction == "loss")
    p = source["configuration"]["physics"]
    fraction = [p["p_left"], 1-p["p_left"], p["p_right"], 1-p["p_right"]][index]
    probability = source["schedule"]["jump_probabilities"][index]
    if fraction <= 0 or probability <= 0:
        raise ValueError("selected side must have active pump and loss branches")
    return float(np.sqrt(p["gamma"]*fraction*dt/probability))


def plan(source, path, steps, sides):
    hardware.require_validated_source(source)
    if max(steps) > len(source["schedule"]["step_dts"]):
        raise ValueError("steps exceed the saved cosine schedule")
    cases = []
    for side in sides:
        site, bath = (path[1], path[0]) if side == "left" else (path[-2], path[-1])
        for direction in ("pump", "loss"):
            initial = int(direction == "loss")
            # Also validate branch availability before any backend access.
            angle(source, side, direction, source["schedule"]["step_dts"][0])
            for kind, count in [("baseline", 0), *[("collision", k) for k in steps], ("reset_only", max(steps))]:
                retention = float(np.prod([np.cos(angle(source, side, direction, dt))**2
                                          for dt in source["schedule"]["step_dts"][:count]]))
                target = (1-retention if direction == "pump" else retention) if kind == "collision" else float(initial)
                cases.append(dict(case_id=f"{side}_{kind}_{direction}_{count}", side=side, kind=kind, direction=direction,
                    steps=count, initial_system=initial, physical_site=site, physical_bath=bath,
                    ideal_system_population=target, ideal_bath_population=0.))
    return cases


def build(source, case):
    """Logical qubit 0 is system; qubit 1 is bath. Both are measured last.

    Barriers enforce waiting for bath resets even in reset-only circuits.
    The repeated conditional branch is a diagnostic, not the random mean model.
    """
    qc = QuantumCircuit(2, name=case["case_id"])
    if case["initial_system"]:
        qc.x(0)
    for dt in source["schedule"]["step_dts"][:case["steps"]]:
        qc.barrier(0, 1)
        if case["direction"] == "pump":
            qc.x(1)
        if case["kind"] == "collision":
            qc.append(XXPlusYYGate(2*angle(source, case["side"], case["direction"], dt), 0), [0, 1])
        qc.reset(1)
        qc.barrier(0, 1)
    qc.metadata = dict(case)
    return qc


def verify_plan(manifest, source):
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError("need a boundary diagnostic manifest")
    cfg = manifest["configuration"]
    if (cfg["shots"] < 2 or not cfg["steps"] or min(cfg["steps"]) < 1
            or len(set(cfg["steps"])) != len(cfg["steps"])
            or not cfg["sides"] or not set(cfg["sides"]) <= {"left", "right"}
            or len(set(cfg["sides"])) != len(cfg["sides"])):
        raise ValueError("invalid saved boundary settings")
    expected = {row["case_id"]: row for row in plan(source, cfg["physical_path"], cfg["steps"], cfg["sides"])}
    records = manifest["circuits"]
    if len(records) != len(expected) or {r["case_id"] for r in records} != expected.keys():
        raise ValueError("missing or duplicate boundary cases")
    for index, record in enumerate(records):
        if record["pub_index"] != index or record["shots"] != cfg["shots"]:
            raise ValueError("publication index or shot count mismatch")
        for key, value in expected[record["case_id"]].items():
            if isinstance(value, float):
                if not np.isclose(record[key], value, atol=1e-14, rtol=1e-14):
                    raise ValueError("ideal boundary reference mismatch")
            elif record[key] != value:
                raise ValueError("boundary metadata mismatch")
        if record["final_logical_to_physical"] != [record["physical_site"], record["physical_bath"]]:
            raise ValueError("boundary mapping mismatch")
    if cfg["total_circuits"] != len(records) or cfg["total_shots"] != cfg["shots"]*len(records):
        raise ValueError("boundary circuit/shot totals mismatch")


def analyze(folder):
    raw = {name: (folder/name).read_bytes() for name in ("source.json", "manifest.json")}
    source, manifest = [json.loads(raw[name]) for name in raw]
    verify_plan(manifest, source)
    cfg = manifest["configuration"]
    integrity = source_integrity(raw["source.json"], cfg["source_sha256"])
    if manifest["status"] != "collected":
        raise ValueError("need completely collected boundary results")
    stop, observations, jobs = 0, [], []
    for b in manifest["batches"]:
        if b["status"] != "collected" or not b.get("job_id") or b["start"] != stop or not stop < b["stop"] <= len(manifest["circuits"]):
            raise ValueError("invalid collected boundary batch")
        if len(b.get("observations", [])) != b["stop"]-b["start"]:
            raise ValueError("missing boundary observations")
        observations.extend(b["observations"]); stop = b["stop"]; jobs.append(b["job_id"])
    if stop != len(manifest["circuits"]):
        raise ValueError("incomplete boundary coverage")
    rows = []
    for record, obs in zip(manifest["circuits"], observations):
        value, cov, shots = moments(obs["outcome_counts"], 2)
        if shots != record["shots"]:
            raise ValueError("boundary shot count mismatch")
        rows.append(dict(record, system_population=float(value[0]), bath_population=float(value[1]),
            system_shot_se=float(np.sqrt(max(0., cov[0, 0]))), bath_shot_se=float(np.sqrt(max(0., cov[1, 1]))),
            system_error=float(value[0]-record["ideal_system_population"]), bath_error=float(value[1]),
            covariance_of_mean=cov))
    qpy_path = folder/"isa_circuits.qpy"
    if qpy_path.exists():
        raw_qpy = qpy_path.read_bytes()
    else:
        with zipfile.ZipFile(folder/"results_bundle.zip") as z:
            raw_qpy = z.read("isa_circuits.qpy")
    if sha(raw_qpy) != cfg["isa_sha256"]:
        raise ValueError("compiled boundary QPY hash mismatch")
    return dict(experiment=EXPERIMENT, run_directory=str(folder.resolve()), configuration=cfg, job_ids=jobs, cases=rows,
        verification=dict(source_integrity=integrity, isa_qpy_sha256_verified=True,
                          original_files_sha256={name: sha(value) for name, value in raw.items()}),
        scope="Isolated conditional pump/loss and reset controls; not the physical randomized chain or a steady-state validation. Raw measurements include preparation/readout errors.")


def save(report, output):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    if output.resolve().is_relative_to(Path(report["run_directory"]).resolve()):
        raise ValueError("analysis output must be outside original run")
    output.mkdir(parents=True, exist_ok=True)
    rows = report["cases"]; sides = report["configuration"]["sides"]
    fig, axes = plt.subplots(len(sides), 2, figsize=(12, 4.5*len(sides)), squeeze=False, layout="constrained")
    for line, side in enumerate(sides):
        chosen = [r for r in rows if r["side"] == side]
        for direction, color in [("pump", "#b71c1c"), ("loss", "#1565c0")]:
            data = sorted([r for r in chosen if r["direction"] == direction and r["kind"] != "reset_only"], key=lambda r: r["steps"])
            x = [r["steps"] for r in data]
            axes[line, 0].plot(x, [r["ideal_system_population"] for r in data], "--", color=color, label=f"{direction}: ideal")
            axes[line, 0].errorbar(x, [r["system_population"] for r in data], yerr=[2*r["system_shot_se"] for r in data], fmt="o-", color=color, capsize=3, label=f"{direction}: hardware ±2 SE")
            control = next(r for r in chosen if r["direction"] == direction and r["kind"] == "reset_only")
            axes[line, 0].errorbar([control["steps"]], [control["system_population"]], yerr=[2*control["system_shot_se"]], fmt="x", markersize=9, color=color, capsize=3, label=f"{direction}: reset-only")
        axes[line, 0].set(xlabel="Collision/reset count", ylabel="System population", title=f"{side}: system q{chosen[0]['physical_site']}, bath q{chosen[0]['physical_bath']}")
        ordered = sorted(chosen, key=lambda r: (r["kind"], r["direction"], r["steps"]))
        axes[line, 1].bar(range(len(ordered)), [r["bath_population"] for r in ordered], color="teal")
        axes[line, 1].set(xticks=range(len(ordered)), xticklabels=[f"{r['kind']}\n{r['direction']} {r['steps']}" for r in ordered], ylabel="Measured bath excitation", title="Bath after reset; ideal zero for every case")
        axes[line, 1].tick_params(axis="x", labelsize=6)
        for ax in axes[line]: ax.grid(axis="y", alpha=.2)
        axes[line, 0].legend(fontsize=7)
    for ext in ("png", "pdf"): fig.savefig(output/f"boundary_controls.{ext}", dpi=170)
    plt.close(fig)
    flat = [{k: v for k, v in row.items() if k not in ("metrics", "covariance_of_mean", "final_logical_to_physical")} for row in rows]
    with (output/"boundary_controls.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(flat)
    cfg = report["configuration"]
    text = ["# Isolated boundary collision/reset evaluation", "", f"{cfg['total_circuits']} circuits × {cfg['shots']} shots = {cfg['total_shots']} shots; jobs {', '.join(report['job_ids'])}.", "",
        "![Boundary controls](boundary_controls.png)", "", "| Case | System population | Ideal system | Bath population | Ideal bath |", "|---|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: r["case_id"]):
        text.append(f"| {r['case_id']} | {r['system_population']:.6f} | {r['ideal_system_population']:.6f} | {r['bath_population']:.6f} | 0 |")
    text += ["", "Baselines prepare system 0 or 1 and bath 0, then measure both. Collision cases repeatedly apply one conditional pump or loss branch with the original schedule's compensated bath angles, resetting the bath after every collision. These repetitions are not stochastic physical-reservoir evolution or a replacement steady-state schedule.", "",
        "Reset-only cases apply the same bath preparation (X for pump, none for loss) and reset count, without system–bath exchange. Synchronization barriers force the system to wait until the bath operations finish before readout. Collision and reset-only durations differ, so their difference does not isolate a pure CZ error rate.", "",
        "Compare the measured bath excitation after resets with the corresponding baseline bath excitation; baseline readout errors can produce apparent excitation even after a perfect reset. System survival in reset-only cases tests effects associated with waiting, bath pulses and resets together. Neither comparison by itself separates reset errors, relaxation, crosstalk and readout. These are isolated-pair circuits with different measurement context and synchronization from the full-chain experiment.", "",
        "Only the two active qubits are measured in each circuit. Error bars are nominal ±2 shot SE, assume independent stationary sampling, omit drift and are not simultaneous confidence guarantees. No inverse readout correction or mitigation is applied.", "",
        "Original source/compiled circuit hashes and raw counts were verified. No hardware execution is performed by the offline analyzer.", ""]
    (output/"Analysis.md").write_text("\n".join(text), encoding="utf-8")
    (output/"analysis.json").write_text(json.dumps(serializable(report), indent=2)+"\n")


def bundle(directory, output=None):
    with zipfile.ZipFile(directory/"results_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("manifest.json", "source.json", "isa_circuits.qpy", "backend_snapshot.json", "hardware_results.json"):
            if (directory/name).exists(): z.write(directory/name, name)
        if output:
            for path in output.iterdir():
                if path.is_file(): z.write(path, "analysis/"+path.name)


def main(options, *, backend=None, service=None, sampler_factory=None):
    if options.analyze_only:
        report = analyze(options.analyze_only)
        output = options.analysis_output_directory or DIRECTORY/"analysis"/"hardware"/options.analyze_only.name/"boundary"
        save(report, output); print(f"Analysis: {output/'Analysis.md'}")
        return report
    if options.resume:
        directory = options.resume.resolve().parent
        manifest = json.loads(options.resume.read_bytes()); cfg = manifest["configuration"]
        raw = (directory/"source.json").read_bytes(); source_integrity(raw, cfg["source_sha256"]); source = json.loads(raw)
        verify_plan(manifest, source)
        if sha((directory/"isa_circuits.qpy").read_bytes()) != cfg["isa_sha256"]:
            raise ValueError("saved boundary circuits changed")
        with (directory/"isa_circuits.qpy").open("rb") as f: pubs = qpy.load(f)
        if len(pubs) != len(manifest["circuits"]): raise ValueError("QPY count mismatch")
        if any(c.metadata.get("case_id") != r["case_id"] for c, r in zip(pubs, manifest["circuits"])):
            raise ValueError("compiled case order differs from manifest")
    else:
        source_file, backend_name, provenance = reference(options.reference_run, [])
        raw = source_file.read_bytes(); source = json.loads(raw)
        cases = plan(source, provenance["physical_path"], options.steps, options.sides)
        provenance.update(mode="isolated_boundary_reference", interpretation="Reference supplies physical endpoint pairs and the compensated collision-angle schedule. These isolated repetitions are not full-chain evolution.")
        cfg = dict(backend=backend_name, physical_path=provenance["physical_path"], reference=provenance,
            steps=options.steps, sides=options.sides, shots=options.shots, batch_size=options.batch_size,
            order_seed=options.order_seed, seed_transpiler=options.seed_transpiler, optimization_level=options.optimization_level,
            source_sha256=sha(raw), total_circuits=len(cases), total_shots=len(cases)*options.shots)
    pending = not options.resume or any(b.get("observations") is None for b in manifest["batches"])
    if pending:
        if backend is None:
            service = service or hardware.make_service(options)
            backend = service.backend(cfg["backend"], use_fractional_gates=False)
        if backend.name != cfg["backend"]: raise ValueError("backend must match saved reference")
        if not options.resume:
            refresh = getattr(backend, "refresh", None)
            if callable(refresh): refresh()
            order = np.random.default_rng(options.order_seed).permutation(len(cases))
            pubs, records = [], []
            for i in order:
                case = cases[i]
                native = transpile(build(source, case), backend=backend, initial_layout=[case["physical_site"], case["physical_bath"]],
                    optimization_level=options.optimization_level, seed_transpiler=options.seed_transpiler)
                mapping = native.layout.final_index_layout(filter_ancillas=True)
                if mapping != [case["physical_site"], case["physical_bath"]]: raise ValueError("boundary layout changed")
                measured = circuits.append_readout(native, mapping, "z")
                validate_isa_circuits([measured], backend.target)
                pubs.append(measured); records.append(dict(case, pub_index=len(records), shots=options.shots,
                    final_logical_to_physical=mapping, metrics=circuits.metrics(measured)))
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            directory = options.output_directory/f"{cfg['backend']}_boundary_{stamp}"
            directory.mkdir(parents=True, exist_ok=False); (directory/"source.json").write_bytes(raw)
            with (directory/"isa_circuits.qpy").open("wb") as f: qpy.dump(pubs, f)
            cfg["isa_sha256"] = sha((directory/"isa_circuits.qpy").read_bytes())
            hardware.save_backend_snapshot(backend, cfg["physical_path"], records, directory)
            manifest = dict(experiment=EXPERIMENT, status="compiled", configuration=cfg, circuits=records,
                batches=[dict(start=s, stop=min(s+options.batch_size,len(pubs)), job_id=None, status="not_submitted")
                         for s in range(0,len(pubs),options.batch_size)])
            verify_plan(manifest, source); atomic_json(directory/"manifest.json", manifest)
            print(f"Saved {len(pubs)} circuits, {cfg['total_shots']} shots, {len(manifest['batches'])} batch(es): {directory}", flush=True)
            if options.compile_only:
                bundle(directory); return manifest
        else:
            validate_isa_circuits(pubs, backend.target)
    if pending and sampler_factory is None and service is None: raise ValueError("Runtime service required for execution")
    hardware.collect_and_submit(manifest, directory, pubs, service, backend, sampler_factory)
    report = analyze(directory); atomic_json(directory/"hardware_results.json", serializable(report))
    output = options.analysis_output_directory or DIRECTORY/"analysis"/"hardware"/directory.name/"boundary"
    save(report, output); bundle(directory, output)
    print(f"Boundary analysis: {output/'Analysis.md'}\nBring back: {directory/'results_bundle.zip'}", flush=True)
    return report


if __name__ == "__main__":
    try:
        main(parse_arguments())
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
