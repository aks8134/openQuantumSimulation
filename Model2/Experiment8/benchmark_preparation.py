"""Offline 1/2/4-process benchmark of the real Experiment8 preparation kernel.

Uses a synthetic 120-qubit backend with the saved Phoenix path durations.
No account lookup, Sampler, simulation or hardware manifest is used or changed.
All modes compile the same distinct 44-step paths and all 15 local controls.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import platform
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONTEXT = ROOT/"Model2/Experiment8/job_estimates/ibm_phoenix_16834/context.json"
_WORK = None
THREAD_LIMITS = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1", RAYON_NUM_THREADS="1", QISKIT_NUM_PROCS="1", QISKIT_PARALLEL="FALSE")


def cpu_allocation():
    """Record inherited Linux quotas as well as visible CPU affinity."""
    affinity = list(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    constraints = []
    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        memberships = []
    for row in memberships:
        _, controllers, relative = row.split(":", 2)
        if not controllers:
            root, names = Path("/sys/fs/cgroup"), ("cpu.max",)
        elif "cpu" in controllers.split(","):
            root, names = Path("/sys/fs/cgroup/cpu"), ("cpu.cfs_quota_us", "cpu.cfs_period_us")
        else:
            continue
        directory = root/relative.lstrip("/")
        while directory == root or root in directory.parents:
            try:
                if len(names) == 1:
                    quota, period = (directory/names[0]).read_text().split()
                    cores = None if quota == "max" else int(quota)/int(period)
                else:
                    quota, period = (int((directory/name).read_text()) for name in names)
                    cores = None if quota < 0 else quota/period
                if cores is not None:
                    constraints.append(dict(path=str(directory), cpu_cores=cores))
            except (OSError, ValueError):
                pass
            if directory == root:
                break
            directory = directory.parent
    visible = len(affinity) if affinity is not None else os.cpu_count()
    effective = min([visible]+[c["cpu_cores"] for c in constraints]) if visible else None
    return dict(logical_cpus=os.cpu_count(), affinity=affinity, quota_constraints=constraints,
                effective_cpu_capacity=effective)


def initialize_worker(cache):
    global _WORK
    os.environ.update(THREAD_LIMITS)
    with (Path(cache)/"backend.pkl").open("rb") as handle:
        backend = pickle.load(handle)
    with (Path(cache)/"inputs.json").open() as handle:
        inputs = json.load(handle)
    _WORK = backend, inputs["source"], inputs["configuration"]


def compile_task(task, destination):
    from Model2.Experiment8 import run_hardware as run
    from Model2.Experiment8.preparation import compile_group
    backend, source, configuration = _WORK
    started, cpu_started = time.perf_counter(), time.process_time()
    name, rows = task
    name += ".qpy"
    programs, _, vector = compile_group(source, configuration, rows, backend,
        job_index=0, shard_name=name, verbose=False)
    destination = Path(destination)/name
    temporary = destination.with_suffix(".qpy.partial")
    with temporary.open("wb") as handle:
        run.qpy.dump(programs, handle)
    if temporary.stat().st_size > run.MAX_SHARD_BYTES or len(programs) > run.MAX_SHARD_CIRCUITS:
        raise ValueError("benchmark exceeded the production shard bound")
    temporary.replace(destination)
    file_hash = run.qpy_storage.file_sha256(destination)
    result = dict(name=name, rows=rows, instructions=vector.tolist(), file_sha256=file_hash,
        size_bytes=destination.stat().st_size, seconds=time.perf_counter()-started,
        process_cpu_seconds=time.process_time()-cpu_started, pid=os.getpid())
    return result


class MemoryMonitor:
    """Sample parent plus recursive children; summed RSS includes shared pages."""
    def __init__(self):
        try:
            import psutil
        except ImportError:
            self.psutil = None
        else:
            self.psutil = psutil
            self.root = psutil.Process()
        self.peak_rss = self.peak_uss = 0
        self.samples = 0
        self.stopped = threading.Event()

    def sample(self):
        if self.psutil is None:
            return
        rss = uss = 0
        for process in [self.root]+self.root.children(recursive=True):
            try:
                rss += process.memory_info().rss
                uss += getattr(process.memory_full_info(), "uss", 0)
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                pass
        self.peak_rss, self.peak_uss = max(self.peak_rss, rss), max(self.peak_uss, uss)
        self.samples += 1

    def __enter__(self):
        self.sample()
        def poll():
            while not self.stopped.wait(.2):
                self.sample()
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stopped.set()
        self.thread.join()
        self.sample()

    def report(self):
        return dict(available=self.psutil is not None, samples=self.samples,
            peak_aggregate_rss_mib=self.peak_rss/1024**2 if self.psutil else None,
            peak_aggregate_uss_mib=self.peak_uss/1024**2 if self.psutil else None,
            scope="Sampled parent plus recursive children. RSS counts shared pages in each process; "
                  "USS counts private pages. Neither is total system memory.")


def benchmark_mode(cache, destination, tasks, workers):
    destination.mkdir()
    results = []
    started = time.perf_counter()
    with MemoryMonitor() as memory:
        def record(result):
            results.append(result)
            print(f"workers={workers}: {len(results)}/{len(tasks)} groups saved; "
                  f"{result['name']} {result['seconds']:.2f}s; elapsed {time.perf_counter()-started:.1f}s", flush=True)
        if workers == 1:
            initialize_worker(cache)
            for task in tasks:
                record(compile_task(task, destination))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize_worker, initargs=(str(cache),)) as pool:
                iterator, pending = iter(tasks), {}
                for _ in range(min(workers, len(tasks))):
                    task = next(iterator)
                    pending[pool.submit(compile_task, task, str(destination))] = task[0]
                while pending:
                    done, _ = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
                    if not done:
                        print(f"workers={workers}: {len(results)}/{len(tasks)} saved, {len(pending)} active; "
                              f"elapsed {time.perf_counter()-started:.1f}s", flush=True)
                    for future in done:
                        pending.pop(future)
                        record(future.result())
                        task = next(iterator, None)
                        if task is not None:
                            pending[pool.submit(compile_task, task, str(destination))] = task[0]
    return dict(workers=workers, wall_seconds=time.perf_counter()-started,
        task_cpu_seconds=sum(r["process_cpu_seconds"] for r in results),
        memory=memory.report(), groups=sorted(results, key=lambda r:r["name"]),
        qpy_bytes=sum(r["size_bytes"] for r in results))


def verify_outputs(destination, result, configuration):
    """Check public QPY round trips and compare semantics, ignoring circuit names."""
    from Model2.Experiment8 import run_hardware as run
    signatures = {}
    for group in result["groups"]:
        info = destination/group["name"]
        if run.qpy_storage.file_sha256(info) != group["file_sha256"]:
            raise ValueError("benchmark QPY changed after writing")
        with info.open("rb") as handle:
            programs = run.qpy.load(handle)
        run.base.validate_circuits({"configuration":configuration}, programs, group["rows"],
                                   expected_experiment=run.EXPERIMENT)
        for program, row in zip(programs, group["rows"]):
            semantic = dict(width=program.num_qubits, classical_bits=program.num_clbits,
                registers=[(r.name, len(r)) for r in program.cregs], global_phase=float(program.global_phase),
                metadata=program.metadata, instructions=[(i.operation.name, i.operation.label,
                    [program.find_bit(q).index for q in i.qubits], [program.find_bit(b).index for b in i.clbits],
                    [float(p) for p in i.operation.params]) for i in program.data])
            signatures[row["case_id"]] = run.input_data.digest(semantic)
    return signatures


def make_inputs(cache, context_file, count):
    import numpy as np
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import InstructionProperties
    from Model2.Experiment8 import estimate_jobs as estimate
    context = estimate.verify_seal(json.loads(context_file.read_text()))
    path, width = context["physical_path"], context["num_qubits"]
    # Include the actual active path in a connected synthetic graph. This is
    # explicitly not a reconstruction of Phoenix's target or calibration.
    ordered = path+[q for q in range(width) if q not in path]
    model = context["timing_model"]
    edges = {(a,b) for a,b in zip(ordered, ordered[1:])} | {(b,a) for a,b in zip(ordered, ordered[1:])}
    edges |= {tuple(row["qubits"]) for row in model["instructions"] if row["name"] == "cz"}
    backend = GenericBackendV2(width, basis_gates=["cz", "rz", "sx", "x"], coupling_map=edges,
        seed=17, dt=model["dt_seconds"])
    for constraint, value in model["constraints"].items():
        setattr(backend.target, constraint, value)
    for row in model["instructions"]:
        backend.target.update_instruction_properties(row["name"], tuple(row["qubits"]),
            InstructionProperties(duration=row["duration_seconds"], error=.001))
    if estimate.run.timing.timing_model(backend, path) != model:
        raise ValueError("synthetic path timing does not match the saved timing model")
    if count > context["trajectory_count"]:
        raise ValueError("benchmark count exceeds the source cohort")
    source = estimate.compilation_source(context["trajectory_count"])
    configuration, prototype, controls = estimate.case_rows()
    configuration.update(physical_path=path, trajectory_count=context["trajectory_count"], capture_timing=False,
        dd_policy=context["dd_policy"])
    identities = [int(i) for i in np.random.default_rng(29).permutation(context["trajectory_count"])[:count]]
    tasks, index = [], 0
    for identity in identities:
        rows = [dict(row, trajectory=identity, dd_condition="on", pub_index=index+i, batch=0,
                     case_id=row["case_id"].replace("_0", f"_{identity}", 1)) for i,row in enumerate(prototype)]
        tasks.append((f"trajectory_{identity:05d}", rows))
        index += len(rows)
    for control in ("all_zero", "all_one", "initial_mixture"):
        rows = [dict(row, pub_index=index+i, batch=0, dd_condition="on")
                for i,row in enumerate(r for r in controls if r["control"] == control)]
        tasks.append((f"control_{control}", rows))
        index += len(rows)
    with (cache/"backend.pkl").open("wb") as handle:
        pickle.dump(backend, handle, protocol=pickle.HIGHEST_PROTOCOL)
    estimate.atomic_json(cache/"inputs.json", dict(source=source, configuration=configuration))
    return context, configuration, identities, tasks


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--workers", nargs="+", type=int, default=[1,2,4])
    parser.add_argument("--timing-context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--require-cpus", type=int, help="refuse a known smaller affinity/quota before compiling")
    options = parser.parse_args(arguments)
    if (options.trajectories < 4 or options.trajectories % 4 or not options.workers
            or min(options.workers) < 1 or options.workers[0] != 1 or len(set(options.workers)) != len(options.workers)):
        parser.error("need a positive multiple of four trajectories and worker modes including baseline 1")
    os.environ.update(THREAD_LIMITS)
    allocation = cpu_allocation()
    if options.require_cpus and (allocation["effective_cpu_capacity"] or 0) < options.require_cpus:
        parser.error(f"need {options.require_cpus} CPUs; allocation: {allocation}")
    if (allocation["effective_cpu_capacity"] or 0) < max(options.workers):
        print(f"CPU capacity is limited: {allocation}. Worker scaling will be constrained.", flush=True)
    from Model2.Experiment8 import estimate_jobs as estimate
    folder = options.output_directory.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    cache = folder/"inputs"
    cache.mkdir()
    context, configuration, identities, tasks = make_inputs(cache, options.timing_context, options.trajectories)
    report = dict(artifact_type="model2_dd_preparation_parallel_benchmark_v1", complete=False,
        created_at_utc=datetime.now(timezone.utc).isoformat(), versions=estimate.versions(),
        host=dict(platform=platform.platform(), python=platform.python_version(), **allocation,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"), slurm_cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK")),
        thread_limits=THREAD_LIMITS, source_context_sha256=context["sha256"],
        physical_path=context["physical_path"], synthetic_backend_qubits=context["num_qubits"],
        trajectories=identities, circuits=13*len(identities)+15, control_circuits=15,
        source_and_compiler_code_sha256=estimate.code_hashes(),
        benchmark_code_sha256={str(p.relative_to(ROOT)):estimate.run.qpy_storage.file_sha256(p)
            for p in (Path(__file__), Path(__file__).with_name("preparation.py"))},
        scope="Local synthetic backend with saved Phoenix path durations, not the Phoenix target/calibration. "
              "Identical trajectories, seeds, native transformations, checks and QPY writes for each mode. "
              "Wall time includes process startup, compilation, validation, QPY writes/hashes and shutdown. "
              "Output comparison is outside timed sections. No IBM access, hardware execution or simulation. "
              "These workers are a benchmark prototype, not enabled in the production CLI.", results=[])
    estimate.atomic_json(folder/"benchmark.json", report)
    baseline = None
    for workers in options.workers:
        # Fresh tasks each time: the production kernel annotates its row copies.
        fresh = json.loads(json.dumps(tasks))
        result = benchmark_mode(cache, folder/f"workers_{workers}", fresh, workers)
        signatures = verify_outputs(folder/f"workers_{workers}", result, configuration)
        if baseline is None:
            baseline = signatures
        elif signatures != baseline:
            raise ValueError("parallel worker output differs from the serial baseline")
        result.update(semantic_sha256=estimate.run.input_data.digest(signatures),
            semantic_comparison_passed=True, output_signatures=signatures,
            speedup_vs_serial=(report["results"][0]["wall_seconds"]/result["wall_seconds"] if report["results"] else 1.))
        report["results"].append(result)
        estimate.atomic_json(folder/"benchmark.json", report)
        print(f"FINISHED workers={workers}: {result['wall_seconds']:.2f}s, "
              f"speedup={result['speedup_vs_serial']:.2f}x, peak aggregate RSS "
              f"{result['memory']['peak_aggregate_rss_mib']} MiB; outputs match", flush=True)
    report["complete"] = True
    estimate.atomic_json(folder/"benchmark.json", report)
    print(f"Benchmark: {folder/'benchmark.json'}", flush=True)
    return report


if __name__ == "__main__":
    main()
