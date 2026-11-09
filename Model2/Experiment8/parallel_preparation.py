"""Bounded preparation workers and atomic, per-group recovery receipts.

Only the parent changes the hardware manifest. Workers receive a Target and
compile-only source, never a backend service, account or hardware submission API.
"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import signal
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from Model2.Experiment8.preparation import compile_group, compiler_methods

_WORK = None
COMPUTED = {"isa_file", "isa_index", "metrics", "final_logical_to_physical", "dd_check"}
THREAD_LIMITS = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1", RAYON_NUM_THREADS="1", QISKIT_NUM_PROCS="1", QISKIT_PARALLEL="FALSE")


def row_identity(rows):
    return [{k:v for k,v in row.items() if k not in COMPUTED} for row in rows]


def sealed(value):
    from Model2.Experiment8 import run_hardware as run
    return dict(value, sha256=run.input_data.digest({k:v for k,v in value.items() if k != "sha256"}))


def read_sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != sealed(value)["sha256"]:
        raise ValueError(f"preparation receipt hash mismatch: {path}")
    return value


def receipt_name(name):
    return name.removesuffix(".qpy")+".prepare.json"


def write_group(folder, source, configuration, context, backend, rows, job_index, name,
                *, target_only=False, templates=None, verbose=False):
    from Model2.Experiment8 import run_hardware as run
    started = time.monotonic()
    folder = Path(folder)
    identity = run.input_data.digest(row_identity(rows))
    destination = folder/name
    legacy = destination.exists()
    if legacy:
        if not 0 < destination.stat().st_size <= run.MAX_SHARD_BYTES:
            raise ValueError(f"legacy QPY exceeds shard bounds: {destination}")
        with destination.open("rb") as stream:
            programs = run.qpy.load(stream)
        from Model2.Experiment8.preparation_recovery import recover_group
        windows, vector = recover_group(source, configuration, rows, programs, backend,
                                       shard_name=name, target_only=target_only)
    else:
        programs, windows, vector = compile_group(source, configuration, rows, backend,
            job_index=job_index, shard_name=name, templates=templates, verbose=verbose, target_only=target_only)
        # A crash leaves only a .partial file. The final .qpy is never overwritten.
        with tempfile.NamedTemporaryFile(dir=folder, prefix=name+".", suffix=".partial", delete=False) as stream:
            partial = Path(stream.name)
            run.qpy.dump(programs, stream)
        if not 0 < partial.stat().st_size <= run.MAX_SHARD_BYTES or len(programs) > run.MAX_SHARD_CIRCUITS:
            raise ValueError("compiled shard exceeds its bounded storage limit")
        if destination.exists():
            raise ValueError("another preparation process wrote this shard; stop concurrent preparations")
        partial.replace(destination)
    windows_file = None
    if windows:
        windows_file = name.removesuffix(".qpy")+("_recovered" if legacy else "")+"_dd_windows.json"
        run.atomic_json(folder/windows_file, windows)
    result = sealed(dict(artifact_type="model2_dd_preparation_receipt_v1", context_sha256=context,
        captured_at_utc=datetime.now(timezone.utc).isoformat(), row_identity_sha256=identity,
        job_index=job_index, isa_file=name, isa_sha256=run.qpy_storage.file_sha256(destination),
        size_bytes=destination.stat().st_size, rows=rows, instructions_per_qubit=vector.tolist(),
        compiler_versions=run.dd_runtime.versions(), origin="verified_legacy_qpy" if legacy else "new_compilation",
        windows_file=windows_file, windows_sha256=run.qpy_storage.file_sha256(folder/windows_file) if windows_file else None,
        seconds=time.monotonic()-started))
    run.atomic_json(folder/receipt_name(name), result)
    return receipt_name(name)


def initialize_worker(cache):
    global _WORK
    from Model2.Experiment8 import run_hardware as run
    os.environ.update(THREAD_LIMITS)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cache = Path(cache)
    header = read_sealed(cache/"cache.json")
    for name, expected in header["files"].items():
        if run.qpy_storage.file_sha256(cache/name) != expected:
            raise ValueError("preparation worker input changed")
    with (cache/"target.pkl").open("rb") as stream:
        target = pickle.load(stream)
    with (cache/"inputs.json").open() as stream:
        inputs = json.load(stream)
    backend = SimpleNamespace(target=target, num_qubits=target.num_qubits,
        compiler_methods=inputs["compiler_methods"])
    _WORK = backend, inputs, {}


def worker_task(folder, rows, job_index, name):
    backend, inputs, templates = _WORK
    return write_group(folder, inputs["source"], inputs["configuration"], inputs["context_sha256"],
        backend, rows, job_index, name, target_only=True, templates=templates)


class Preparation:
    def __init__(self, runner, folder, manifest, source, backend, workers):
        self.run, self.folder, self.manifest = runner, folder, manifest
        self.source, self.backend, self.workers = source, backend, workers
        self.c = manifest["configuration"]
        self.methods = compiler_methods(backend)
        from Model2.Experiment8 import preparation, preparation_recovery
        files = [Path(preparation.__file__), Path(preparation_recovery.__file__), Path(runner.timing.__file__),
            Path(runner.flat_block_order.__code__.co_filename), Path(runner.protocol.__file__),
            Path(runner.protocol.mit.__file__), Path(runner.raw.__file__), Path(runner.raw.circuits.__file__)]
        self.context = runner.input_data.digest(dict(configuration=self.c, compiler_methods=self.methods,
            code_sha256={str(p.relative_to(runner.ROOT)):runner.qpy_storage.file_sha256(p) for p in files}))
        self.pool = self.cache = None
        self.templates = {}
        self.old_environment = {}
        self.started = time.monotonic()
        self.done = {r["trajectory"] for r in manifest["circuits"] if r["kind"] == "trajectory"
                     and r.get("isa_file") in manifest["files"]}
        self.initial_done = len(self.done)
        self.recovered = self.compiled = 0

    def __enter__(self):
        print(f"Preparation: workers={self.workers}; {len(self.done):,}/{self.c['trajectory_count']:,} "
              "trajectories registered. Scanning saved groups and receipts.", flush=True)
        return self

    def start_pool(self):
        if self.pool is not None:
            return
        self.cache = tempfile.TemporaryDirectory(prefix="preparation_workers_", dir=self.folder)
        cache = Path(self.cache.name)
        with (cache/"target.pkl").open("wb") as stream:
            pickle.dump(self.backend.target, stream, protocol=pickle.HIGHEST_PROTOCOL)
        # The exact compilation input is small; ideal/covariance analysis arrays
        # are unnecessary in workers and remain solely in the original source.
        source = {key:self.source[key] for key in ("configuration", "preparation", "schedule")}
        self.run.atomic_json(cache/"inputs.json", dict(source=source, configuration=self.c,
            compiler_methods=self.methods, context_sha256=self.context))
        self.run.atomic_json(cache/"cache.json", sealed(dict(files={name:self.run.qpy_storage.file_sha256(cache/name)
            for name in ("target.pkl", "inputs.json")})))
        self.old_environment = {key:os.environ.get(key) for key in THREAD_LIMITS}
        os.environ.update(THREAD_LIMITS)
        self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker, initargs=(str(cache),))

    def report(self, job_index, active=0, *, name=None):
        elapsed = time.monotonic()-self.started
        advanced = len(self.done)-self.initial_done
        remaining = self.c["trajectory_count"]-len(self.done)
        eta = f"{remaining*elapsed/advanced/3600:.2f} h" if advanced else "estimating"
        print(f"Preparation: {len(self.done):,}/{self.c['trajectory_count']:,} trajectories saved | "
              f"job {job_index+1}/{len(self.manifest['batches'])} | recovered {self.recovered:,} | "
              f"compiled {self.compiled:,} | active {active} | elapsed {elapsed/60:.1f} min | "
              f"ETA {eta}"+(f" | {name}" if name else ""), flush=True)

    def accept(self, name, rows, job_index, *, reused=False):
        run = self.run
        receipt = read_sealed(self.folder/receipt_name(name))
        if (receipt.get("artifact_type") != "model2_dd_preparation_receipt_v1"
                or receipt.get("context_sha256") != self.context or receipt.get("isa_file") != name
                or receipt.get("job_index") != job_index or receipt.get("compiler_versions") != run.dd_runtime.versions()
                or receipt.get("row_identity_sha256") != run.input_data.digest(row_identity(rows))
                or row_identity(receipt["rows"]) != row_identity(rows)):
            raise ValueError(f"preparation receipt context/row mismatch: {name}")
        path = self.folder/name
        if (not 0 < path.stat().st_size <= run.MAX_SHARD_BYTES or len(receipt["rows"]) > run.MAX_SHARD_CIRCUITS
                or path.stat().st_size != receipt["size_bytes"]
                or run.qpy_storage.file_sha256(path) != receipt["isa_sha256"]):
            raise ValueError(f"preparation QPY hash/size mismatch: {name}")
        vector = np.asarray(receipt["instructions_per_qubit"])
        if vector.shape != (self.backend.num_qubits,) or vector.dtype.kind not in "iu" or np.any(vector < 0):
            raise ValueError("invalid preparation instruction vector")
        windows_file = receipt.get("windows_file")
        if windows_file:
            if Path(windows_file).name != windows_file or run.qpy_storage.file_sha256(self.folder/windows_file) != receipt["windows_sha256"]:
                raise ValueError("preparation timing diagnostic hash mismatch")
        for i, row in enumerate(receipt["rows"]):
            if (row.get("isa_file") != name or row.get("isa_index") != i
                    or not row.get("dd_check", {}).get("original_operations_and_local_times_preserved")):
                raise ValueError("incomplete preparation row proof")
        # No global manifest write here: small receipts are the durable journal.
        # The parent remains its only writer, checkpointing at job boundaries.
        for original, saved in zip(rows, receipt["rows"]):
            original.update(saved)
        self.manifest["files"][name] = dict(sha256=receipt["isa_sha256"], circuit_count=len(rows),
            pub_indices=[r["pub_index"] for r in rows])
        run.register(self.folder, self.manifest, receipt_name(name))
        if windows_file:
            run.register(self.folder, self.manifest, windows_file)
        if rows[0]["kind"] == "trajectory":
            self.done.add(rows[0]["trajectory"])
            if reused or receipt["origin"] == "verified_legacy_qpy":
                self.recovered += 1
            else:
                self.compiled += 1
        return vector

    def batch(self, groups, job_index):
        pending = {}
        total = np.zeros(self.backend.num_qubits, dtype=np.int64)
        names = []

        def receive():
            nonlocal total
            done, _ = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
            if not done:
                self.report(job_index, len(pending))
            for future in done:
                name, rows = pending.pop(future)
                if future.result() != receipt_name(name):
                    raise ValueError("worker returned another preparation receipt")
                total += self.accept(name, rows, job_index)
                self.report(job_index, len(pending), name=name)

        for key, rows in groups.items():
            name = f"batch_{job_index:04d}_{key[0]}_{key[1]}.qpy"
            names.append(name)
            if name in self.manifest["files"]:
                programs = self.run.load_shard(self.folder, self.manifest, name, self.backend)
                total += sum((self.run.instruction_audit.footprint(p) for p in programs), np.zeros_like(total))
                del programs
                continue
            if (self.folder/receipt_name(name)).exists():
                total += self.accept(name, rows, job_index, reused=True)
                self.report(job_index, len(pending), name=f"reused receipt {name}")
                continue
            copied_rows = json.loads(json.dumps(rows))
            if self.workers == 1:
                write_group(self.folder, self.source, self.c, self.context, self.backend, copied_rows,
                    job_index, name, templates=self.templates, verbose=True)
                total += self.accept(name, rows, job_index)
                self.report(job_index, name=name)
            else:
                self.start_pool()
                pending[self.pool.submit(worker_task, str(self.folder), copied_rows, job_index, name)] = name, rows
                while len(pending) >= self.workers:
                    receive()
        while pending:
            receive()
        return names, total

    def __exit__(self, kind, error, traceback):
        try:
            if self.pool is not None:
                if kind is not None:
                    print("Stopping preparation; letting active groups finish their atomic receipts. "
                          "Resume will reuse completed groups.", flush=True)
                self.pool.shutdown(wait=True, cancel_futures=True)
        finally:
            for key, value in self.old_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if self.cache is not None:
                self.cache.cleanup()
