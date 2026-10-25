"""Endpoint system survival with idle-duration sweeps and bath-reset controls.

No exchange gates or Trotter evolution. Hardware runs only on explicit CLI use.
Default: 64 circuits x 2048 shots, one batch. Offline analysis needs no account.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, qpy
from qiskit.transpiler import PassManager
from qiskit_ibm_runtime.transpiler.passes.scheduling import ASAPScheduleAnalysis
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

DIRECTORY = Path(__file__).resolve().parent
if str(DIRECTORY.parents[1]) not in sys.path:
    sys.path.insert(0, str(DIRECTORY.parents[1]))
from library.reservoir_scaling import atomic_json
from Model3.Experiment5 import circuit_depth, run_hardware as hardware
from Model3.Experiment5 import run_boundary_diagnostic as boundary
from Model3.Experiment5.run_preparation_diagnostic import reference as preparation_reference
from Model3.Experiment5.analyze_hardware import moments, serializable, source_integrity

EXPERIMENT = "Model3/Experiment5/reset_survival"
DEFAULT_REFERENCE = DIRECTORY/"hardware_runs/ibm_phoenix_boundary_20261025T215447_981524Z"
TIMING_SCOPE = ("Equal means equal summed target-model time before final measurement, not verified equal "
                "hardware elapsed time. Reset block-transition latency is unknown. Idle scales bracket "
                "the target model; no finite bracket is guaranteed to contain actual reset latency.")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse_arguments(arguments=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--steps", type=int, nargs="+", default=[12, 36, 72])
    p.add_argument("--idle-scales", type=float, nargs="+", default=[.5, 1., 2.])
    p.add_argument("--shots", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--order-seed", type=int, default=29)
    p.add_argument("--account-file", type=Path)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--compile-only", action="store_true")
    action.add_argument("--resume", type=Path)
    action.add_argument("--analyze-only", type=Path)
    p.add_argument("--output-directory", type=Path, default=DIRECTORY/"hardware_runs")
    p.add_argument("--analysis-output-directory", type=Path)
    a = p.parse_args(arguments)
    if (a.shots < 2 or a.batch_size < 1 or a.order_seed < 0 or min(a.steps) < 1
            or len(set(a.steps)) != len(a.steps) or len(set(a.idle_scales)) != len(a.idle_scales)
            or any(not np.isfinite(x) or x <= 0 for x in a.idle_scales) or 1. not in a.idle_scales):
        raise ValueError("need shots >=2, positive distinct steps/scales including scale 1, positive batch-size and nonnegative order seed")
    a.steps.sort(); a.idle_scales.sort()
    return a


def reference(folder):
    manifest = json.loads((folder/"manifest.json").read_bytes())
    if manifest.get("experiment") == boundary.EXPERIMENT:
        checked = boundary.analyze(folder)
        cfg = checked["configuration"]
        source = folder/"source.json"
        provenance = dict(reference_run=folder.name, reference_manifest_sha256=sha((folder/"manifest.json").read_bytes()),
                          reference_job_ids=checked["job_ids"], physical_path=cfg["physical_path"])
        return source, cfg["backend"], provenance
    source, name, provenance = preparation_reference(folder, [])
    return source, name, provenance


def aligned(value, alignment):
    return int(math.ceil(value/alignment)*alignment)


def timing_model(backend, path):
    """Read the live target, including dt constraints; do not infer reset latency."""
    target = backend.target
    if not target.dt or not np.isfinite(target.dt) or target.dt <= 0:
        raise ValueError("backend must supply a finite positive dt")
    alignment = math.lcm(*(int(getattr(target, name, 1) or 1)
                           for name in ["granularity", "pulse_alignment", "acquire_alignment"]))
    minimum = int(target.min_length or 1)
    durations = {}
    for side, system, bath in [("left", path[1], path[0]), ("right", path[-2], path[-1])]:
        d = {}
        for name, qubit in [("x", system), ("x", bath), ("reset", bath), ("measure", system), ("measure", bath)]:
            try:
                value = target[name][(qubit,)].duration
            except (KeyError, AttributeError) as error:
                raise ValueError(f"missing {name} duration on q{qubit}") from error
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {name} duration on q{qubit}")
            d[f"{name}_{qubit}"] = int(target.seconds_to_dt(value))
        if any(x <= 0 for x in d.values()):
            raise ValueError("instruction duration rounds to zero dt")
        for qubit in [system, bath]:
            if not target.instruction_supported(operation_name="delay", qargs=(qubit,)):
                raise ValueError(f"backend must support delay on q{qubit}")
        prepare = aligned(max(d[f"x_{system}"], minimum), alignment)
        bath_x = aligned(max(d[f"x_{bath}"], minimum), alignment)
        reset = aligned(max(d[f"reset_{bath}"], minimum), alignment)
        # Nonzero padding delays also need the target minimum length.
        for slot, key in [(prepare, f"x_{system}"), (bath_x, f"x_{bath}"), (reset, f"reset_{bath}")]:
            if 0 < slot-d[key] < minimum:
                raise ValueError("target padding smaller than minimum delay; unsupported timing constraints")
        durations[side] = dict(system=system, bath=bath, instruction_dt=d, prepare_dt=prepare,
                               bath_x_slot_dt=bath_x, reset_slot_dt=reset, cycle_dt=bath_x+reset)
    return dict(dt_seconds=float(target.dt), alignment_dt=alignment, min_length_dt=minimum,
                pairs=durations, scope=TIMING_SCOPE)


def plan(cfg):
    cases = []
    model = cfg["timing_model"]
    for side, timing in model["pairs"].items():
        for initial in [0, 1]:
            variants = [("baseline", 0, 1.)]
            for count in cfg["steps"]:
                variants += [("reset_zero", count, 1.), ("reset_one", count, 1.)]
                variants += [("idle", count, scale) for scale in cfg["idle_scales"]]
            for arm, count, scale in variants:
                ticks = aligned(max(timing["cycle_dt"]*scale, model["min_length_dt"]), model["alignment_dt"])
                cases.append(dict(case_id=f"{side}_s{initial}_{arm}_{count}_scale{scale:g}", side=side,
                    initial_system=initial, arm=arm, steps=count, idle_scale=scale,
                    physical_site=timing["system"], physical_bath=timing["bath"],
                    cycle_dt=ticks, modeled_wait_dt=count*ticks,
                    modeled_wait_us=count*ticks*model["dt_seconds"]*1e6,
                    ideal_system_population=float(initial), ideal_bath_population=0.))
    return cases


def build(cfg, case):
    """Direct native physical instructions: no optimizer can move preparation late."""
    qc = QuantumCircuit(cfg["num_qubits"])
    outcome = ClassicalRegister(2, "outcome"); qc.add_register(outcome)
    s, b = case["physical_site"], case["physical_bath"]
    t = cfg["timing_model"]["pairs"][case["side"]]
    def delay(ticks, *qubits):
        if ticks:
            for qubit in qubits:
                qc.delay(int(ticks), qubit, unit="dt")
    if case["initial_system"]:
        qc.x(s); delay(t["prepare_dt"]-t["instruction_dt"][f"x_{s}"], s)
    else:
        delay(t["prepare_dt"], s)
    delay(t["prepare_dt"], b); qc.barrier(s,b)
    for _ in range(case["steps"]):
        if case["arm"] == "idle":
            delay(case["cycle_dt"],s,b)
        else:
            delay(t["bath_x_slot_dt"],s)
            if case["arm"] == "reset_one":
                qc.x(b); delay(t["bath_x_slot_dt"]-t["instruction_dt"][f"x_{b}"],b)
            else:
                delay(t["bath_x_slot_dt"],b)
            qc.barrier(s,b)
            # Keep the system waiting explicitly during the modeled reset interval.
            delay(t["instruction_dt"][f"reset_{b}"],s); qc.reset(b)
            qc.barrier(s,b)
            delay(t["reset_slot_dt"]-t["instruction_dt"][f"reset_{b}"],s,b)
        qc.barrier(s,b)
    qc.measure(s,outcome[0]); qc.measure(b,outcome[1])
    qc.name = case["case_id"]; qc.metadata = dict(case)
    return qc


def schedule_summary(circuit, target):
    """Retain dynamic-block timing, without treating the blocks as a wall clock."""
    manager = PassManager([ASAPScheduleAnalysis(target=target)])
    manager.run(circuit)
    ends = {}
    measurement_starts = []
    for node, (block, start) in manager.property_set["node_start_time"].items():
        if node.name == "barrier": duration = 0
        elif node.name == "delay": duration = int(node.op.duration)
        else:
            qubits = tuple(circuit.find_bit(q).index for q in node.qargs)
            duration = int(target.seconds_to_dt(target[node.name][qubits].duration))
        ends[block] = max(ends.get(block,0),int(start)+duration)
        if node.name == "measure": measurement_starts.append((block,int(start)))
    if len(measurement_starts) != 2 or len(set(measurement_starts)) != 1:
        raise ValueError("final readout is not simultaneous in the timing model")
    block,start = measurement_starts[0]
    pre_measure = sum(value for key,value in ends.items() if key < block)+start
    return dict(dynamic_blocks=len(ends), block_durations_dt={str(k):v for k,v in ends.items()},
                summed_pre_measure_dt=pre_measure, summed_total_dt=sum(ends.values()),
                unknown_reset_block_latency=True)


def verify_plan(manifest):
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError("need a reset-survival manifest")
    cfg = manifest["configuration"]
    if (cfg["shots"] < 2 or cfg["batch_size"] < 1 or min(cfg["steps"]) < 1
            or len(set(cfg["steps"])) != len(cfg["steps"]) or 1. not in cfg["idle_scales"]
            or len(set(cfg["idle_scales"])) != len(cfg["idle_scales"])
            or any(not np.isfinite(x) or x <= 0 for x in cfg["idle_scales"])):
        raise ValueError("invalid saved control settings")
    expected = {r["case_id"]:r for r in plan(cfg)}
    records = manifest["circuits"]
    if len(records) != len(expected) or {r["case_id"] for r in records} != expected.keys():
        raise ValueError("incomplete or duplicate control plan")
    for i,r in enumerate(records):
        if r["pub_index"] != i or r["shots"] != cfg["shots"]:
            raise ValueError("publication/shot mismatch")
        if any(r[k] != value for k,value in expected[r["case_id"]].items()):
            raise ValueError("control metadata differs from plan")
        if r["final_logical_to_physical"] != [r["physical_site"],r["physical_bath"]]:
            raise ValueError("control measurement mapping mismatch")
        t=cfg["timing_model"]["pairs"][r["side"]]
        if r["schedule"]["summed_pre_measure_dt"] != t["prepare_dt"]+r["modeled_wait_dt"]:
            raise ValueError("modeled preparation-to-readout timing mismatch")
    if cfg["total_circuits"] != len(records) or cfg["total_shots"] != len(records)*cfg["shots"]:
        raise ValueError("control budget mismatch")


def load_circuits(folder, cfg):
    path=folder/"isa_circuits.qpy"
    if path.exists(): raw=path.read_bytes()
    else:
        with zipfile.ZipFile(folder/"results_bundle.zip") as archive:
            raw=archive.read("isa_circuits.qpy")
    if sha(raw) != cfg["isa_sha256"]: raise ValueError("compiled QPY hash mismatch")
    return qpy.load(io.BytesIO(raw))


def verify_circuits(pubs, manifest):
    if len(pubs) != len(manifest["circuits"]): raise ValueError("compiled circuit count mismatch")
    cfg=manifest["configuration"]; expected={r["case_id"]:r for r in plan(cfg)}
    for pub,record in zip(pubs,manifest["circuits"]):
        if pub != build(cfg,expected[record["case_id"]]):
            raise ValueError("compiled circuit differs from native control plan")
        if record["metrics"] != circuit_depth.metrics(pub):
            raise ValueError("saved circuit metrics differ from compiled circuit")


def wilson(value, shots):
    """Pointwise 95% binomial interval, including nonzero bounds at 0/n."""
    z=1.959963984540054
    center=(value+z*z/(2*shots))/(1+z*z/shots)
    half=z*np.sqrt(value*(1-value)/shots+z*z/(4*shots*shots))/(1+z*z/shots)
    return float(max(0.,center-half)),float(min(1.,center+half))


def analyze(folder):
    manifest=json.loads((folder/"manifest.json").read_bytes()); verify_plan(manifest)
    cfg=manifest["configuration"]
    integrity=source_integrity((folder/"source.json").read_bytes(),cfg["source_sha256"])
    pubs=load_circuits(folder,cfg); verify_circuits(pubs,manifest)
    if manifest["status"] != "collected": raise ValueError("need completely collected results")
    observations=[]; jobs=[]; stop=0
    for batch in manifest["batches"]:
        if (batch["status"] != "collected" or not batch.get("job_id") or batch["start"] != stop
                or not stop < batch["stop"] <= len(pubs)
                or len(batch.get("observations",[])) != batch["stop"]-stop):
            raise ValueError("incomplete control batch")
        observations.extend(batch["observations"]);jobs.append(batch["job_id"]);stop=batch["stop"]
    if stop != len(pubs): raise ValueError("missing observations")
    cases=[]
    for record,obs in zip(manifest["circuits"],observations):
        value,cov,shots=moments(obs["outcome_counts"],2)
        if shots != record["shots"]: raise ValueError("raw shot count mismatch")
        sl,su=wilson(value[0],shots);bl,bu=wilson(value[1],shots)
        cases.append(dict(record,system_population=float(value[0]),bath_population=float(value[1]),
                          system_shot_se=float(np.sqrt(max(0.,cov[0,0]))),bath_shot_se=float(np.sqrt(max(0.,cov[1,1]))),
                          system_ci95_low=sl,system_ci95_high=su,bath_ci95_low=bl,bath_ci95_high=bu))
    comparisons=[]
    for r in cases:
        if not r["arm"].startswith("reset_"): continue
        idle=next(x for x in cases if x["side"]==r["side"] and x["initial_system"]==r["initial_system"]
                  and x["steps"]==r["steps"] and x["arm"]=="idle" and x["idle_scale"]==1.)
        a,b=r["system_population"],idle["system_population"];difference=a-b
        low=difference-np.hypot(a-r["system_ci95_low"],idle["system_ci95_high"]-b)
        high=difference+np.hypot(r["system_ci95_high"]-a,b-idle["system_ci95_low"])
        comparisons.append(dict(reset_case=r["case_id"],idle_case=idle["case_id"],side=r["side"],
            initial_system=r["initial_system"],steps=r["steps"],arm=r["arm"],
            reset_minus_idle=difference,difference_ci95_low=float(low),difference_ci95_high=float(high),
            nominal_shot_se=float(np.hypot(r["system_shot_se"],idle["system_shot_se"])),
            interpretation="Model-time comparison; unknown reset-block latency may contribute."))
    return dict(experiment=EXPERIMENT,run_directory=str(folder.resolve()),configuration=cfg,job_ids=jobs,
        cases=cases,comparisons=comparisons,verification=dict(source_integrity=integrity,isa_qpy_sha256_verified=True,
        circuits_match_native_plan=True),scope=TIMING_SCOPE)


def save(report, output):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output)
    if output.resolve().is_relative_to(Path(report["run_directory"]).resolve()):
        raise ValueError("analysis output must be outside original run")
    output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    text=["# System survival: idle durations versus bath resets", "",
          f"Jobs: {', '.join(report['job_ids'])}. {report['configuration']['total_circuits']} circuits; "
          f"{report['configuration']['total_shots']} shots.", "", "![Populations](reset_survival.png)", "",
          TIMING_SCOPE, "", "| Case | Modeled wait, μs | System population | Bath population |",
          "|---|---:|---:|---:|"]
    rows=report["cases"]
    for row,side in enumerate(["left","right"]):
        for initial in [0,1]:
            ax=axes[row,initial]
            groups=[("reset_zero",1.),("reset_one",1.),*[("idle",x) for x in report["configuration"]["idle_scales"]]]
            base=next(r for r in rows if r["side"]==side and r["initial_system"]==initial and r["arm"]=="baseline")
            for arm,scale in groups:
                chosen=sorted([r for r in rows if r["side"]==side and r["initial_system"]==initial
                               and r["arm"]==arm and r["idle_scale"]==scale],key=lambda r:r["steps"])
                label=f"Idle ×{scale:g}" if arm=="idle" else arm.replace("_"," ")
                plotted=[base]+chosen
                errors=[[r["system_population"]-r["system_ci95_low"] for r in plotted],
                        [r["system_ci95_high"]-r["system_population"] for r in plotted]]
                ax.errorbar([0]+[r["steps"] for r in chosen],[base["system_population"]]+[r["system_population"] for r in chosen],
                    yerr=errors,fmt="o-",capsize=3,label=label)
            ax.axhline(initial,color="black",ls="--",alpha=.5)
            ax.set(title=f"{side}: system q{base['physical_site']}, initially {initial}",xlabel="Cycle count",ylabel="System population")
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Populations with pointwise 95% Wilson intervals; idle ×1 matches modeled time only")
    for ext in ["png","pdf"]: fig.savefig(output/f"reset_survival.{ext}",dpi=170)
    plt.close(fig)
    for r in sorted(rows,key=lambda r:r["case_id"]):
        text.append(f"| {r['case_id']} | {r['modeled_wait_us']:.4f} | {r['system_population']:.6f} | {r['bath_population']:.6f} |")
    text += ["", "Both initial system states are tested against both bath preparations. reset_zero resets a bath prepared in 0; "
             "reset_one applies X before each bath reset. No exchange gates or full-chain evolution are present. "
             "All arms have the same initial preparation slot and terminal two-qubit measurement. Reset-zero has a padded empty X slot, "
             "so both reset arms have the same target-model cycle duration. Explicit barriers prevent readout before the sequence finishes.", "",
             "Compare reset arms with the whole idle-duration sweep. A reset-minus-idle difference at scale 1 does not alone establish "
             "reset crosstalk: unknown elapsed-time differences can contribute. Curves are connected across separate circuits, not a "
             "single continuously monitored trajectory. Final bath measurement does not reveal every earlier bath reset.", "",
             "No inverse readout correction is applied. Population intervals are pointwise 95% Wilson; difference intervals use "
             "the Newcombe independent-proportions method. They assume independent stationary shots, omit device drift and are "
             "not simultaneous confidence bounds. Saved comparisons share idle controls and are therefore correlated. "
             "PUB order is randomized; actual QPU dispatch order is not verified.", "",
             "Source and compiled-circuit hashes, native circuit contents, mappings, complete batch coverage and shot counts were checked. "
             "Offline analysis performs no hardware queries or new jobs.", "",
             "Timing reference: [IBM dynamic scheduling documentation](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/transpiler-passes-scheduling-alap-schedule-analysis).", ""]
    (output/"Analysis.md").write_text("\n".join(text),encoding="utf-8")
    for name,data in [("populations",rows),("reset_idle_comparisons",report["comparisons"])]:
        flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in data]
        with (output/f"{name}.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    atomic_json(output/"analysis.json",serializable(report))


def main(options, *, backend=None, service=None, sampler_factory=None):
    if options.analyze_only:
        report=analyze(options.analyze_only)
        output=options.analysis_output_directory or DIRECTORY/"analysis"/"hardware"/options.analyze_only.name/"reset_survival"
        save(report,output);print(f"Analysis: {output/'Analysis.md'}");return report
    if options.resume:
        directory=options.resume.resolve().parent
        manifest=json.loads(options.resume.read_bytes());verify_plan(manifest);cfg=manifest["configuration"]
        source_integrity((directory/"source.json").read_bytes(),cfg["source_sha256"])
        pubs=load_circuits(directory,cfg);verify_circuits(pubs,manifest)
    else:
        source_file,name,provenance=reference(options.reference_run)
        raw=source_file.read_bytes()
        cfg=dict(backend=name,physical_path=provenance["physical_path"],reference=provenance,
                 shots=options.shots,batch_size=options.batch_size,order_seed=options.order_seed,
                 steps=options.steps,idle_scales=options.idle_scales,source_sha256=sha(raw))
    pending=not options.resume or any(b.get("observations") is None for b in manifest["batches"])
    if pending:
        if backend is None:
            service=service or hardware.make_service(options)
            backend=service.backend(cfg["backend"],use_fractional_gates=False)
        if backend.name != cfg["backend"]: raise ValueError("backend must match saved reference")
        if not options.resume:
            refresh=getattr(backend,"refresh",None)
            if callable(refresh): refresh()
            cfg.update(num_qubits=backend.num_qubits,timing_model=timing_model(backend,cfg["physical_path"]))
            cases=plan(cfg);pubs=[];records=[]
            for index in np.random.default_rng(options.order_seed).permutation(len(cases)):
                case=cases[index];native=build(cfg,case)
                validate_isa_circuits([native],backend.target)
                timing=schedule_summary(native,backend.target)
                record=dict(case,pub_index=len(records),shots=options.shots,
                    final_logical_to_physical=[case["physical_site"],case["physical_bath"]],
                    metrics=circuit_depth.metrics(native),schedule=timing)
                pubs.append(native);records.append(record)
            cfg.update(total_circuits=len(records),total_shots=len(records)*cfg["shots"])
            stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            directory=options.output_directory/f"{cfg['backend']}_reset_survival_{stamp}"
            directory.mkdir(parents=True,exist_ok=False);(directory/"source.json").write_bytes(raw)
            with (directory/"isa_circuits.qpy").open("wb") as f: qpy.dump(pubs,f)
            cfg["isa_sha256"]=sha((directory/"isa_circuits.qpy").read_bytes())
            hardware.save_backend_snapshot(backend,cfg["physical_path"],records,directory)
            manifest=dict(experiment=EXPERIMENT,status="compiled",configuration=cfg,circuits=records,
                batches=[dict(start=s,stop=min(s+cfg["batch_size"],len(pubs)),job_id=None,status="not_submitted")
                         for s in range(0,len(pubs),cfg["batch_size"])])
            verify_plan(manifest);verify_circuits(pubs,manifest);atomic_json(directory/"manifest.json",manifest)
            print(f"Saved {len(pubs)} circuits; {cfg['total_shots']} shots; {len(manifest['batches'])} batch(es): {directory}",flush=True)
            print(TIMING_SCOPE,flush=True)
            if options.compile_only:
                boundary.bundle(directory);return manifest
        else: validate_isa_circuits(pubs,backend.target)
    if pending and sampler_factory is None and service is None: raise ValueError("Runtime service required for execution")
    hardware.collect_and_submit(manifest,directory,pubs,service,backend,sampler_factory)
    report=analyze(directory);atomic_json(directory/"hardware_results.json",serializable(report))
    output=options.analysis_output_directory or DIRECTORY/"analysis"/"hardware"/directory.name/"reset_survival"
    save(report,output);boundary.bundle(directory,output)
    print(f"Analysis: {output/'Analysis.md'}\nBring back: {directory/'results_bundle.zip'}",flush=True)
    return report


if __name__ == "__main__":
    try: main(parse_arguments())
    except (ValueError,OSError) as error: raise SystemExit(str(error)) from error
