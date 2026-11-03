"""Offline IBM instruction-memory audit of saved Model2 ISA circuits.

Counts the published validator heuristic, not measured control-system memory.
Never submits, retranspiles, rewrites QPY, or changes a hardware manifest.
--manifest audits the actual local hardware archive. --fake-run evaluates
saved FakeKingston circuits and explicitly labelled footprint projections.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from library.reservoir_scaling import atomic_json
from Model2.Experiment5 import input_data, qpy_storage, run_hardware

INSTRUCTION_LIMIT = 26_800_000
EXECUTION_LIMIT = 10_000_000
WEIGHTS = {"rz": 1, "delay": 1, "sx": 2, "x": 2, "cx": 5, "cz": 5,
           "ecr": 5, "measure": 10, "reset": 17, "init": 50}
LIMIT_SOURCE = "https://quantum.cloud.ibm.com/docs/en/guides/job-limits"


def footprint(circuit, init_qubits=True):
    """Charge two-qubit gates to both operands, and implicit init to all qubits."""
    result = np.full(circuit.num_qubits, 50 if init_qubits else 0, dtype=np.int64)
    indices = {qubit: j for j, qubit in enumerate(circuit.qubits)}
    for instruction in circuit.data:
        name = instruction.operation.name
        if name == "barrier":
            continue
        if name not in WEIGHTS:
            raise ValueError(f"no published weight for {name!r}; refusing an incomplete count")
        for qubit in instruction.qubits:
            result[indices[qubit]] += WEIGHTS[name]
    operations = circuit.count_ops()
    two_qubit = sum(operations.get(name, 0) for name in ("cx", "cz", "ecr"))
    if operations.get("rz", 0) > 30_000_000 or operations.get("sx", 0) > 20_000_000 or two_qubit > 5_000_000:
        raise ValueError("a circuit exceeds a published per-circuit gate limit")
    return result


def partition(vectors, executions, controls, control_executions,
              instruction_limit=INSTRUCTION_LIMIT, execution_limit=EXECUTION_LIMIT):
    """Contiguous whole-path partition; lower-bound equality certifies this model."""
    vectors = np.asarray(vectors, dtype=np.int64)
    controls = np.asarray(controls, dtype=np.int64)
    executions = np.asarray(executions, dtype=np.int64)
    if (vectors.ndim != 2 or not len(vectors) or controls.shape != vectors.shape[1:]
            or executions.shape != (len(vectors),) or np.any(vectors < 0) or np.any(controls < 0)
            or np.any(executions < 1) or not 0 <= control_executions < execution_limit
            or np.any(controls >= instruction_limit)):
        raise ValueError("invalid footprints, controls or limits")
    if (np.any(vectors+controls > instruction_limit)
            or np.any(executions+control_executions > execution_limit)):
        raise ValueError("a complete path plus its controls cannot fit in one job")
    total = vectors.sum(axis=0)
    lower = max(1, max((int(a)+int(b)-1)//int(b) for a, b in zip(total, instruction_limit-controls)),
                (int(executions.sum())+execution_limit-control_executions-1)//(execution_limit-control_executions))
    jobs = []
    begin = 0
    load = controls.copy()
    shots = int(control_executions)
    for index, (vector, count) in enumerate(zip(vectors, executions)):
        if np.any(load+vector > instruction_limit) or shots+count > execution_limit:
            jobs.append(dict(path_start=begin, path_stop=index, trajectories=index-begin,
                             maximum_instructions=int(load.max()), bottleneck_position=int(load.argmax()), executions=shots))
            begin, load, shots = index, controls.copy(), int(control_executions)
        load += vector
        shots += int(count)
    jobs.append(dict(path_start=begin, path_stop=len(vectors), trajectories=len(vectors)-begin,
                     maximum_instructions=int(load.max()), bottleneck_position=int(load.argmax()), executions=shots))
    one_job = total+controls
    return dict(trajectories=len(vectors), lower_bound_jobs=int(lower), feasible_jobs=len(jobs),
                minimum_certified_for_counting_model=len(jobs) == lower, jobs=jobs,
                single_job_instructions=one_job.tolist(), single_job_maximum=int(one_job.max()),
                bottleneck_position=int(one_job.argmax()),
                single_job_executions=int(executions.sum())+int(control_executions),
                total_executions_after_repartition=int(executions.sum())+len(jobs)*int(control_executions),
                instruction_limit=instruction_limit, execution_limit=execution_limit)


def repeat_rows(values, count):
    values = np.asarray(values)
    return np.tile(values, (math.ceil(count/len(values)), *([1]*(values.ndim-1))))[:count]


def summarize_matrix(vectors, controls, executions, control_executions):
    return dict(per_path_minimum=np.min(vectors, axis=0).tolist(),
                per_path_mean=np.mean(vectors, axis=0).tolist(),
                per_path_maximum=np.max(vectors, axis=0).tolist(),
                controls_instructions=np.asarray(controls).tolist(), control_executions=int(control_executions),
                at_limit=partition(vectors, executions, controls, control_executions),
                at_80_percent=partition(vectors, executions, controls, control_executions,
                                        instruction_limit=int(.8*INSTRUCTION_LIMIT)))


def hardware_archive(path):
    print("Verifying saved source and native-archive hashes with bounded reads...", flush=True)
    folder, manifest, source = run_hardware.read_saved(path)
    del source
    config = manifest["configuration"]
    if config["isa_sha256"] is None:
        raise ValueError("compilation must be complete; use --recover-compiled first for a completed legacy QPY")
    if config.get("isa_storage") != "batch-qpy-v1" and not config.get("isa_index_sha256"):
        legacy = qpy_storage.Archive(folder/"isa_circuits.qpy", len(manifest["circuits"]))
        program_batches = (programs for programs, _ in legacy.iter_batches(manifest["batches"]))
    else:
        store = run_hardware.Publications(manifest, folder)
        program_batches = (store[b["start"]:b["stop"]] for b in manifest["batches"])
    paths, path_shots, bases = {}, {}, {}
    control_profiles, control_shots = [], []
    original_jobs = []
    complete = None
    width = None
    for (index, batch), programs in zip(enumerate(manifest["batches"], 1), program_batches):
        records = manifest["circuits"][batch["start"]:batch["stop"]]
        run_hardware.validate_publications(manifest, programs, records)
        control = None
        controls_executions = 0
        batch_count = None
        batch_shots = 0
        for program, row in zip(programs, records):
            value = footprint(program)
            if width is None:
                width = len(value)
                complete = np.zeros(width, dtype=np.int64)
            if len(value) != width:
                raise ValueError("compiled backend widths differ; cannot combine physical-qubit vectors")
            if batch_count is None:
                batch_count = np.zeros(width, dtype=np.int64)
                control = np.zeros(width, dtype=np.int64)
            complete += value
            batch_count += value
            batch_shots += row["shots"]
            if row["kind"] == "control":
                control += value
                controls_executions += row["shots"]
            else:
                trajectory = row["trajectory"]
                paths.setdefault(trajectory, np.zeros(width, dtype=np.int64))[:] += value
                path_shots[trajectory] = path_shots.get(trajectory, 0)+row["shots"]
                if row["basis"] in bases.setdefault(trajectory, set()):
                    raise ValueError("duplicate path/basis")
                bases[trajectory].add(row["basis"])
        control_profiles.append(control)
        control_shots.append(controls_executions)
        original_jobs.append(dict(batch=index-1, circuits=len(records), maximum_instructions=int(batch_count.max()),
                                  bottleneck_position=int(batch_count.argmax()), executions=batch_shots))
        del programs, program
        print(f"Audited job {index}/{len(manifest['batches'])}: peak {batch_count.max():,} instructions/qubit", flush=True)
    if (set(paths) != set(range(config["trajectories"]))
            or any(value != set(config["measurement_bases"]) for value in bases.values())):
        raise ValueError("missing path/basis groups")
    vectors = np.array(list(paths.values()))
    # All jobs normally share identical controls. A componentwise maximum is a
    # conservative bound if compilation produced small control-count differences.
    controls = np.max(control_profiles, axis=0)
    result = summarize_matrix(vectors, controls, list(path_shots.values()), max(control_shots))
    result.update(kind="exact_saved_raw_archive", physical_qubit_indices=list(range(width)),
                  original_jobs=original_jobs, original_single_job_maximum=int(complete.max()),
                  original_single_job_instructions=complete.tolist(), path_order=list(paths),
                  original_total_executions=sum(j["executions"] for j in original_jobs),
                  controls_identical=all(np.array_equal(p, controls) for p in control_profiles),
                  compilation_backend=config["backend"], physical_path=config["physical_path"],
                  manifest=str(path.resolve()), archive_hash=config["isa_sha256"])
    return {"raw": result}


def fake_projection(folder, target_paths, shots):
    config = json.loads((folder/"configuration.json").read_text())
    count = config["trajectory_count"]
    if shots % config["randomizations"] or shots < 2*config["randomizations"]:
        raise ValueError("projection shots must split into at least two shots per twirl variant")
    raw, mitigated = [], []
    controls = {"raw": None, "all": None}
    masks_seen = set()
    for name in ["control_all_zero", "control_all_one", "control_initial_mixture"]+[f"trajectory_{j}" for j in range(count)]:
        unit = json.loads((folder/f"{name}.json").read_text())
        if qpy_storage.file_sha256(folder/unit["qpy_file"]) != unit["qpy_sha256"]:
            raise ValueError("FakeKingston QPY hash mismatch")
        records = unit["records"]
        archive = qpy_storage.Archive(folder/unit["qpy_file"], len(records))
        programs, _ = next(archive.iter_batches([dict(start=0, stop=len(records))]))
        total = np.zeros(12, dtype=np.int64)
        raw_z = None
        for program, row in zip(programs, records):
            if program.num_qubits != 12:
                raise ValueError("expected the saved twelve-qubit compact simulation target")
            value = footprint(program)
            total += value
            if row["arm"] == "raw":
                raw_z = value
            elif row["kind"] != "control":
                masks_seen.add((row["factor_index"], row["randomization"]))
        if raw_z is None:
            raise ValueError("missing paired raw circuit")
        if name.startswith("control"):
            for arm, value in [("raw", raw_z), ("all", total)]:
                controls[arm] = value.copy() if controls[arm] is None else controls[arm]+value
        else:
            # Z has no terminal basis rotations. X adds SX+2RZ (weight 4),
            # Y adds SX+RZ (weight 3). XY and YX together add 7 at each site.
            all_bases = 3*raw_z
            all_bases[:10] += 7
            raw.append(all_bases)
            mitigated.append(total)
        del programs, program
        print(f"Counted saved FakeKingston {name}", flush=True)
    if len(masks_seen) != 3*config["randomizations"]:
        raise ValueError("incomplete mitigation variants")
    output = {}
    for name, values, control, factor, control_executions in [
            ("raw", raw, controls["raw"], 3, 3*config["control_shots"]),
            ("mitigation", mitigated, controls["all"], 4, 6*config["control_shots"])]:
        projection = repeat_rows(values, target_paths)
        result = summarize_matrix(projection, control, np.full(target_paths, factor*shots), control_executions)
        result.update(kind="replicated_saved_FakeKingston_footprint_projection", observed_paths=count,
                      target_paths=target_paths, shots_per_arm=shots, measured_per_path_minimum=np.min(values, axis=0).tolist(),
                      measured_per_path_maximum=np.max(values, axis=0).tolist(),
                      system_positions=list(range(10)), bath_positions=[10, 11],
                      bases=["z", "xy", "yx"] if name == "raw" else ["z"],
                      physical_compilation_backend="FakeKingston, compact logical layout",
                      source_run=str(folder.resolve()), source_noise_model_sha256=config["noise_model_sha256"],
                      scope="Projection repeats observed native footprint rows, not quantum trajectories. It is not an audit of all 16,384 paths or of the Windows Phoenix archive.")
        output[name] = result
    return output


def save_report(data, folder):
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder/"instruction_audit.json", data)
    projections = [r for r in data["workloads"].values() if r["kind"].startswith("replicated_")]
    classification = (f"**Footprint projection:** {projections[0]['observed_paths']} saved FakeKingston paths are repeated to model "
                      f"{projections[0]['target_paths']:,} native path groups. These are not counts from the complete Phoenix archive."
                      if projections else "**Actual archive audit:** every native program in the supplied manifest was counted; "
                      "the compilation backend is identified below.")
    lines = ["# Model2 control-instruction audit", "", classification, "",
             "This is an offline application of [IBM's published validation heuristic]("+LIMIT_SOURCE+"). "
             "It is not a measurement of controller memory or a guarantee that the IBM compiler accepts a payload.", "",
             "The limit is 26,800,000 weighted instructions per qubit per job, plus 10,000,000 shot executions per job. "
             "Weights: RZ/delay 1, SX/X 2, CZ/CX/ECR 5, measure 10, reset 17, init 50. "
             "Two-qubit operations are charged to both operands. We conservatively include implicit init on every allocated qubit for every circuit. "
             "Memory counts are program-storage counts, not multiplied by shots; executions are counted separately. "
             "Manually inserted twirls/folds are already present in the QPY and are counted once. "
             "Unknown operations/control flow fail rather than being omitted.", "",
             "A path's matched bases/mitigation variants stay together. Proposed new jobs each receive a complete control set. "
             "These proposed partitions are calculations only; the saved run, job IDs and QPY are unchanged.", "",
             "| Workload | Paths | Single-job peak instructions/qubit | Lower-bound jobs | Feasible jobs | Jobs at 80% memory |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in data["workloads"].items():
        at = result["at_limit"]
        lines.append(f"| {name} | {at['trajectories']:,} | {at['single_job_maximum']:,} | {at['lower_bound_jobs']} | {at['feasible_jobs']} | {result['at_80_percent']['feasible_jobs']} |")
    for name, result in data["workloads"].items():
        at = result["at_limit"]
        lines += ["", f"## {name}", "", f"Input classification: **{result['kind']}**.", "",
                  f"Compilation backend: **{result.get('compilation_backend', result.get('physical_compilation_backend'))}**.", "",
                  result.get("scope", "Counts come from every native program in the supplied hardware manifest's archive."), "",
                  f"Bottleneck physical index/logical position: **{at['bottleneck_position']}**. "
                  f"A one-job configuration would use **{at['single_job_executions']:,} executions** with one control set. "
                  f"The calculated partition uses **{at['total_executions_after_repartition']:,} total executions** including repeated controls.", "",
                  f"The lower bound and constructed partition agree: **{at['minimum_certified_for_counting_model']}**. "
                  "When they agree, this is the minimum for the supplied footprint vectors and counting model, not an unconditional hardware limit.", ""]
        with (folder/f"{name}_partitions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(at["jobs"][0]))
            writer.writeheader()
            writer.writerows(at["jobs"])
        with (folder/f"{name}_per_qubit.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["position", "one_job_instructions", "per_path_minimum",
                                                       "per_path_mean", "per_path_maximum", "controls_instructions"])
            writer.writeheader()
            for position, total in enumerate(at["single_job_instructions"]):
                writer.writerow(dict(position=position, one_job_instructions=total,
                    **{key: result[key][position] for key in ("per_path_minimum", "per_path_mean", "per_path_maximum", "controls_instructions")}))
    lines += ["", "## Exact Phoenix archive check on Windows", "", "From the repository root, after syncing the audit script:", "", "```powershell",
              "python Model2/Experiment5/instruction_audit.py --manifest $runManifest", "```", "",
              "This can take time because it hashes and reads every compiled program. It uses the bounded QPY loader, "
              "does not recompile or access an IBM account, and does not submit or merge jobs. "
              "It writes an audit report beside the saved run. Parameter or layout changes, missing bases, invalid QPY, "
              "and unsupported instructions are not silently estimated.", "",
              "The current 6056 failure is not diagnosed by this count. Payload serialization size, controller details not captured "
              "by the public heuristic, execution duration, and provider compiler faults can still restrict a job.", ""]
    (folder/"InstructionAudit.md").write_text("\n".join(lines))


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--fake-run", type=Path)
    parser.add_argument("--project-trajectories", type=int, default=16384)
    parser.add_argument("--project-shots", type=int, default=32)
    parser.add_argument("--output-directory", type=Path)
    options = parser.parse_args(arguments)
    if min(options.project_trajectories, options.project_shots) < 1:
        raise ValueError("positive projection sizes required")
    base = options.manifest.parent if options.manifest else options.fake_run
    workloads = (hardware_archive(options.manifest) if options.manifest else
                 fake_projection(options.fake_run, options.project_trajectories, options.project_shots))
    data = {"limit_source": LIMIT_SOURCE, "weights": WEIGHTS, "implicit_init": True,
            "instruction_limit": INSTRUCTION_LIMIT, "execution_limit": EXECUTION_LIMIT, "workloads": workloads}
    output = options.output_directory or base/"instruction_audit"
    save_report(data, output)
    print(json.dumps({name: {k: result["at_limit"][k] for k in
          ("single_job_maximum", "lower_bound_jobs", "feasible_jobs", "minimum_certified_for_counting_model")}
          for name, result in workloads.items()}, indent=2), flush=True)
    print(output/"InstructionAudit.md", flush=True)
    return data


if __name__ == "__main__":
    main()
