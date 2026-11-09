"""Real spawned preparation, legacy recovery, durable interruption and tamper refusal."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from qiskit.converters import circuit_to_dag

from Model2.Experiment8 import run_hardware as run
from Model2.Experiment8 import parallel_preparation as parallel
from Model2.Experiment8.preparation import compiler_methods, native_template
from Model2.Experiment8.preparation_recovery import wire_signatures
from test_model2_combined_dd import compiled


def interrupted_copy(compiled, root, *, keep_qpy, keep_receipts=False):
    backend, original, _ = compiled
    folder = root/"run"
    shutil.copytree(original, folder)
    _, manifest, _ = run.read_saved(folder/"manifest.json")
    removed = [name for name in manifest["files"] if name.endswith((".qpy", ".prepare.json"))
               or "_dd_windows.json" in name or name.endswith("_compilation.json")]
    for name in removed:
        manifest["files"].pop(name)
        if (name.endswith(".qpy") and not keep_qpy) or (name.endswith(".prepare.json") and not keep_receipts):
            (folder/name).unlink()
    for row in manifest["circuits"]:
        for key in parallel.COMPUTED:
            row.pop(key, None)
    for batch in manifest["batches"]:
        batch["compiled"] = False
        for key in ("isa_file", "isa_sha256", "resources", "compiler_versions"):
            batch.pop(key, None)
    manifest["status"] = "compiling"
    run.checkpoint(folder, manifest)
    return backend, folder, manifest


def resume(folder, backend, workers):
    opts = run.parse_arguments(["prepare", "--resume", str(folder/"manifest.json"), "--workers", str(workers)])
    with patch.object(run.raw.runtime, "make_service", side_effect=AssertionError("no IBM account")):
        return run.prepare(opts, backend=backend)


def signatures(folder, manifest, backend):
    answer = {}
    for name in manifest["files"]:
        if name.endswith(".qpy"):
            for circuit in run.load_shard(folder, manifest, name, backend):
                answer[circuit.metadata["mitigation_case_id"]] = (circuit_to_dag(circuit), wire_signatures(circuit))
    return answer


def test_four_workers_interrupt_then_resume_from_receipts_without_compilation(compiled, tmp_path, capsys):
    backend, folder, before = interrupted_copy(compiled, tmp_path, keep_qpy=False)
    partial = folder/"batch_0000_trajectory_unused.qpy.partial"
    partial.write_bytes(b"unfinished legacy write")
    old_manifest = (folder/"manifest.json").read_bytes()
    accept = parallel.Preparation.accept
    def stop_after_path(self, name, rows, job_index, **kwargs):
        vector = accept(self, name, rows, job_index, **kwargs)
        if rows[0]["kind"] == "trajectory":
            raise KeyboardInterrupt
        return vector
    with patch.object(parallel.Preparation, "accept", stop_after_path), pytest.raises(KeyboardInterrupt):
        resume(folder, backend, 4)
    assert (folder/"manifest.json").read_bytes() == old_manifest
    assert list(folder.glob("*.prepare.json"))
    hashes = {p.name:run.qpy_storage.file_sha256(p) for p in folder.glob("*.qpy")}
    # Active groups finish receipts before shutdown, including any result not
    # returned to the parent when Ctrl+C arrived. Reuse must need no compiler.
    with patch.object(run, "transpile", side_effect=AssertionError("receipt should avoid recompilation")), \
         patch.object(run.timing, "schedule", side_effect=AssertionError("receipt should avoid rescheduling")):
        _, after = resume(folder, backend, 2)
    assert after["status"] == "compiled" and all(b["compiled"] for b in after["batches"])
    assert hashes == {name:run.qpy_storage.file_sha256(folder/name) for name in hashes}
    assert partial.read_bytes() == b"unfinished legacy write"
    old_backend, old_folder, old_manifest = compiled
    assert signatures(folder, after, backend) == signatures(old_folder, old_manifest, old_backend)
    assert after["configuration"]["total_shots"] == before["configuration"]["total_shots"]
    assert "ETA" in capsys.readouterr().out


def test_legacy_unregistered_qpy_is_recovered_and_never_rewritten(compiled, tmp_path):
    backend, folder, _ = interrupted_copy(compiled, tmp_path, keep_qpy=True)
    old_hashes = {p.name:run.qpy_storage.file_sha256(p) for p in folder.glob("*.qpy")}
    _, manifest = resume(folder, backend, 4)
    assert old_hashes == {name:run.qpy_storage.file_sha256(folder/name) for name in old_hashes}
    for name in old_hashes:
        proof = parallel.read_sealed(folder/parallel.receipt_name(name))
        assert proof["origin"] == "verified_legacy_qpy"
    _, old_folder, old_manifest = compiled
    assert signatures(folder, manifest, backend) == signatures(old_folder, old_manifest, backend)
    # A completed run also permits changing worker count without any compiler.
    plan_hash = manifest["plan_sha256"]
    with patch.object(run, "transpile", side_effect=AssertionError("completed jobs must be reused")):
        _, again = resume(folder, backend, 2)
    assert again["plan_sha256"] == plan_hash


@pytest.mark.parametrize("tamper", ["qpy", "receipt", "context"])
def test_modified_receipts_and_qpy_are_refused_without_overwriting(compiled, tmp_path, tamper):
    backend, folder, _ = interrupted_copy(compiled, tmp_path, keep_qpy=True, keep_receipts=True)
    proof_file = sorted(folder.glob("*.prepare.json"))[0]
    proof = json.loads(proof_file.read_text())
    qpy_file = folder/proof["isa_file"]
    if tamper == "qpy":
        with qpy_file.open("ab") as handle:
            handle.write(b"corrupt")
    else:
        proof["context_sha256"] = "other compiler context"
        run.atomic_json(proof_file, parallel.sealed(proof) if tamper == "context" else proof)
    before = qpy_file.read_bytes()
    with pytest.raises(ValueError, match="preparation .*mismatch"):
        resume(folder, backend, 4)
    assert qpy_file.read_bytes() == before


def test_legacy_changed_dd_pulse_is_refused(compiled, tmp_path):
    backend, folder, _ = interrupted_copy(compiled, tmp_path, keep_qpy=True, keep_receipts=True)
    name = sorted(p.name for p in folder.glob("*trajectory*.qpy"))[0]
    (folder/parallel.receipt_name(name)).unlink()
    with (folder/name).open("rb") as handle:
        programs = run.qpy.load(handle)
    circuit = programs[0]
    index = next(i for i,item in enumerate(circuit.data) if item.operation.label == run.timing.DD_LABEL)
    circuit.data.pop(index)
    with (folder/name).open("wb") as handle:
        run.qpy.dump(programs, handle)
    changed_hash = run.qpy_storage.file_sha256(folder/name)
    with pytest.raises(ValueError, match="legacy QPY differs"):
        resume(folder, backend, 2)
    assert run.qpy_storage.file_sha256(folder/name) == changed_hash


def test_target_only_compilation_retains_ibm_translation_plugin(compiled, monkeypatch):
    backend, folder, manifest = compiled
    monkeypatch.setattr(backend, "get_translation_stage_plugin", lambda:"ibm_dynamic_circuits", raising=False)
    _, _, source = run.read_saved(folder/"manifest.json")
    row = next(r for r in manifest["circuits"] if r["kind"] == "control")
    expected = native_template(source, manifest["configuration"], row, backend)
    target_backend = SimpleNamespace(target=backend.target, compiler_methods=compiler_methods(backend))
    with patch.object(run, "transpile", wraps=run.transpile) as compiler:
        actual = native_template(source, manifest["configuration"], row, target_backend, target_only=True)
    assert compiler.call_args.kwargs["translation_method"] == "ibm_dynamic_circuits"
    assert compiler.call_args.kwargs["num_processes"] == 1
    assert "backend" not in compiler.call_args.kwargs
    assert circuit_to_dag(actual) == circuit_to_dag(expected)


def test_workers_are_invocation_only_and_scientific_resume_options_stay_frozen():
    assert run.parse_arguments(["prepare", "--resume", "manifest.json", "--workers", "4"]).workers == 4
    with pytest.raises(SystemExit):
        run.parse_arguments(["prepare", "--resume", "manifest.json", "--workers", "0"])
    with pytest.raises(SystemExit):
        run.parse_arguments(["prepare", "--resume", "manifest.json", "--workers", "4", "--shots", "64"])
