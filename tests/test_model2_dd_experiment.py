"""Schedule identity, IBM metadata rendering, and no-duplicate job recovery."""
from __future__ import annotations

import copy
from collections import Counter
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SuperOp

from library.reservoir_scaling import atomic_json
from Model2.Experiment7 import run_hardware as run, schedule_tools as timing
from Model2.Experiment7.analyze_hardware import analyze


def backend(size=12):
    return GenericBackendV2(size, basis_gates=["cz", "rz", "sx", "x"],
        coupling_map=[[i, i+1] for i in range(size-1)]+[[i+1, i] for i in range(size-1)], seed=17)


def compiler_text():
    # Synthetic metadata for service-bookkeeping tests, never a hardware result.
    rows = ["main,broadcast,Hub,10,40,broadcast", "main,receive,Receive,50,5,receive"]
    for q in range(12):
        rows += [f"main,sx_{q},Qubit {q},0,10,play", f"main,cz_{q},Qubit {q},100,20,play",
                 f"main,measure_{q},Qubit {q},140,60,play", f"main,measure_{q},AWGR0_{q},150,80,capture"]
    return "\n".join(rows)+"\n"


def fake_draw(value, destination, *, channels=None):
    Path(destination).write_text("<!doctype html><p>Test timing view</p>", encoding="utf-8")


def toy_body(source, trajectory, steps=None, populations=None):
    """Short channel for runner tests; full frozen-circuit smoke is separate."""
    q = QuantumCircuit(12)
    run.raw.circuits.add_preparation(q, list(range(10)), populations or [.5]*10)
    if steps is None:
        q.sx(1)
        for _ in range(44):
            q.cz(0, 10)
            q.reset(10)
    q.metadata = dict(trajectory=trajectory, steps=0 if steps == 0 else 44)
    return q


@pytest.fixture(scope="module")
def scout(tmp_path_factory):
    root = tmp_path_factory.mktemp("model2_dd")
    b = backend()
    options = run.parse_arguments(["prepare", "--backend", b.name, "--trajectory-count", "2",
        "--physical-path", *map(str, range(12)), "--calibration-policy", "record-only",
        "--optimization-level", "1", "--output-directory", str(root)])
    with patch.object(run.raw, "build_body", side_effect=toy_body):
        folder, m = run.prepare(options, backend=b)
    return root, b, folder, m


def copy_scout(scout, tmp_path):
    import shutil
    _, b, original, _ = scout
    folder = tmp_path/"scout"
    shutil.copytree(original, folder)
    return b, folder


def fake_collect(folder, *, dry_run=True):
    path = folder/"manifest.json"
    _, m, _ = run.read_saved(path)
    m["submission_mode"] = "dry_run" if dry_run else "hardware"
    for i, batch in enumerate(m["batches"]):
        pubs = []
        for row in m["circuits"][batch["start"]:batch["stop"]]:
            shots = row["shots"]
            if row.get("control") == "all_zero":
                counts = {"0"*10: shots}
            elif row.get("control") == "all_one":
                counts = {"1"*10: shots}
            else:
                counts = {"0"*10: shots//2, "1"*10: shots-shots//2}
            text = compiler_text()
            if row.get("arm") != "baseline" and row["kind"] == "evolution":
                for q, count in row["dd_check"]["added_x_by_qubit"].items():
                    text += "".join(f"main,x_{q},Qubit {q},{30+i},1,play\n" for i in range(count))
            pubs.append(dict(pub_index=row["pub_index"], timing=text, outcome_counts=counts))
        name = f"job_{i:04d}_results.json"
        job_id = f"test_job_{i}"
        atomic_json(folder/name, dict(job_id=job_id, submission_mode=m["submission_mode"], pubs=pubs))
        batch.update(status="collected", job_id=job_id, result_file=name,
                     result_sha256=run.qpy_storage.file_sha256(folder/name))
    m["status"] = "collected"; run.checkpoint(folder, m)
    return m


def select_policy(folder, output):
    return run.create_policy(run.parse_arguments(["policy", "--manifest", str(folder/"manifest.json"),
        "--system-sites", "1", "2", "3", "--note", "Inspect synthetic service timing in the test fixture",
        "--output", str(output)]))


def paired_options(folder, policy, output):
    return run.parse_arguments(["apply-policy", "--manifest", str(folder/"manifest.json"),
        "--policy", str(policy), "--batch-size", "12", "--shots", "16", "--output-directory", str(output)])


def test_dd_preserves_open_system_channel_and_operation_times():
    b = backend(3)
    q = QuantumCircuit(3)
    q.sx(0); q.sx(1); q.cz(1, 2); q.reset(2); q.cz(0, 1)
    policy = dict(physical_qubits=[0, 1], min_idle_ratio=2.)
    baseline, e = timing.schedule(q, b.target)
    dd, f = timing.schedule(q, b.target, policy=policy)
    check = timing.verify_pair(baseline, dd, e, f, policy)
    assert check["added_x_pulses"] >= 2
    assert check["original_operations_and_local_times_preserved"]
    assert baseline.count_ops()["reset"] == dd.count_ops()["reset"] == 1
    def channel(circuit):
        result = QuantumCircuit(3)
        for item in circuit.data:
            if item.operation.name not in ("delay", "barrier"):
                result.append(item.operation, [circuit.find_bit(q).index for q in item.qubits])
        return SuperOp(result).data
    np.testing.assert_allclose(channel(baseline), channel(dd), atol=2e-13)
    assert all(e["qubits"][0] in [0, 1] for e in f if e["dd"])
    changed = copy.deepcopy(f)
    next(e for e in changed if e["name"] == "cz")["start_dt"] += 1
    with pytest.raises(ValueError, match="moved or changed"):
        timing.verify_pair(baseline, dd, e, changed, policy)


def test_no_protection_after_final_readout_and_no_short_window_pulses():
    b = backend(3)
    q = QuantumCircuit(3)
    from qiskit import ClassicalRegister
    outcome = ClassicalRegister(2, "outcome"); q.add_register(outcome)
    q.sx(0); q.sx(1); q.measure(0, outcome[0]); q.measure(1, outcome[1]); q.reset(2)
    policy = dict(physical_qubits=[0, 1], min_idle_ratio=2.)
    _, events = timing.schedule(q, b.target, policy=policy)
    for qubit in (0, 1):
        final = next(e for e in events if e["name"] == "measure" and e["qubits"] == [qubit])
        assert not any(e["dd"] and e["qubits"] == [qubit]
                       and (e["block"], e["start_dt"]) >= (final["block"], final["start_dt"])
                       for e in events)
    _, events = timing.schedule(q, b.target, policy={**policy, "min_idle_ratio": 1e10})
    assert not any(e["dd"] for e in events)


def test_compiler_rows_keep_branches_and_controller_overlap():
    text = compiler_text()+"then,sx_1,Qubit 1,300,10,play\nthen,sx_1,Qubit 1,400,10,play\n"
    rows = timing.parse_timing(text.splitlines())
    gaps = timing.gap_table(rows, list(range(12)), 1e-9)
    first = next(r for r in gaps if r["branch"] == "main" and r["physical_qubit"] == 1)
    assert first["duration_dt"] == 90 and first["role"] == "system"
    assert "Hub" in first["controller_or_readout_overlap"]
    assert any(r["branch"] == "then" and r["start_dt"] == 310 for r in gaps)
    assert not any(r["start_dt"] == 200 and r["stop_dt"] == 300 for r in gaps)
    with pytest.raises(ValueError, match="no local timing fallback"):
        timing.parse_timing(None)
    assert timing.server_comparison(rows, rows)["non_x_anchor_timing_equal"]
    changed = copy.deepcopy(rows); changed[0]["duration_dt"] += 1
    assert not timing.server_comparison(rows, changed)["non_x_anchor_timing_equal"]
    assert not timing.server_comparison(rows, rows, expected_added_x_by_qubit={"1": 2})["added_x_counts_match"]


@pytest.mark.skipif(importlib.util.find_spec("plotly") is None, reason="optional visualization dependency")
def test_real_ibm_timing_drawer_produces_portable_html(tmp_path):
    destination = tmp_path/"timing.html"
    timing.draw(compiler_text(), destination)
    content = destination.read_text()
    assert "Plotly.newPlot" in content and "Qubit 1" in content and "Hub" in content
    assert "cdn.plot.ly" not in content.split("Plotly.newPlot")[-1]


def test_scout_packing_and_frozen_source(scout):
    _, _, folder, original = scout
    _, m, source = run.read_saved(folder/"manifest.json")
    assert m == original
    assert len(m["batches"]) == 1 and m["configuration"]["total_circuits"] == 9
    assert m["configuration"]["total_shots"] == 6*32+3*2048
    assert source["configuration"]["physics"]["gamma"] == 2.
    assert source["preparation"]["populations"] == [.5]*10
    for row in m["circuits"]:
        if row["kind"] == "evolution":
            assert row["metrics"]["operation_counts"]["reset"] == 44
    with pytest.raises(ValueError, match="collect every scout"):
        run.require_timing(folder, m)
    with patch.object(run, "MAX_QPY_SHARD_BYTES", 1):
        with pytest.raises(ValueError, match="bounded per-path"):
            run.load_qpy(folder, m, "native_00000.qpy")


def test_policy_requires_collected_timing_and_rejects_bath_selection(scout, tmp_path):
    _, folder = copy_scout(scout, tmp_path)
    with pytest.raises(ValueError, match="collect every scout"):
        select_policy(folder, tmp_path/"not_ready.json")
    fake_collect(folder)
    p = select_policy(folder, tmp_path/"policy.json")
    assert p["physical_qubits"] == [1, 2, 3]
    with pytest.raises(SystemExit):
        run.parse_arguments(["policy", "--manifest", str(folder/"manifest.json"), "--system-sites", "0",
                             "--note", "invalid bath", "--output", str(tmp_path/"bad.json")])
    with pytest.raises(ValueError, match="randomized mock"):
        analyze(folder/"manifest.json")


def test_matched_compile_hardware_analysis_and_cloud_clone(scout, tmp_path):
    b, folder = copy_scout(scout, tmp_path)
    before = (folder/"source.json").read_bytes()
    fake_collect(folder)
    select_policy(folder, tmp_path/"policy.json")
    paired, m = run.apply_policy(paired_options(folder, tmp_path/"policy.json", tmp_path), backend=b)
    assert (folder/"source.json").read_bytes() == before
    assert len(m["batches"]) == 2  # one complete matched path + six controls per job
    assert len(m["circuits"]) == 24
    assert sum(r.get("dd_check", {}).get("added_x_pulses", 0) for r in m["circuits"]) > 0
    for batch in m["batches"]:
        rows = m["circuits"][batch["start"]:batch["stop"]]
        assert len([r for r in rows if r["kind"] == "control"]) == 6
        assert batch["resources"]["peak_instructions_per_qubit"] <= batch["resources"]["instruction_allowance"]
    fake_collect(paired)
    options = run.parse_arguments(["clone-for-hardware", "--manifest", str(paired/"manifest.json"),
                                   "--output-directory", str(tmp_path)])
    clone, cloned = run.clone_for_hardware(options)
    assert "submission_mode" not in cloned and all(b["job_id"] is None for b in cloned["batches"])
    assert cloned["files"] == m["files"]
    with pytest.raises(ValueError, match="real hardware"):
        analyze(paired/"manifest.json")
    fake_collect(clone, dry_run=False)
    report = analyze(clone/"manifest.json")
    assert report["all_non_x_server_anchors_equal"]
    assert report["all_added_x_counts_match"]
    np.testing.assert_allclose(report["observables"]["populations"]["dd_minus_baseline"], 0)
    assert report["observables"]["populations"]["arms"]["baseline"]["maximum_error_same_paths"] > 0
    assert (clone/"analysis"/"Analysis.md").exists()
    with patch.object(timing, "draw", side_effect=fake_draw) as draw:
        checks = run.render(SimpleNamespace(manifest=clone/"manifest.json"))
    assert len(checks) == 6 and draw.call_count == 24
    assert all(call.args[0].startswith(compiler_text()) for call in draw.call_args_list)
    # The conservative preflight must catch a compiler schedule with no DD X.
    _, changed, _ = run.read_saved(paired/"manifest.json")
    first = changed["batches"][0]
    file = paired/first["result_file"]
    saved = json.loads(file.read_text())
    for pub in saved["pubs"]:
        pub["timing"] = compiler_text()
    atomic_json(file, saved)
    first["result_sha256"] = run.qpy_storage.file_sha256(file)
    run.checkpoint(paired, changed)
    with pytest.raises(ValueError, match="service timing changed"):
        run.clone_for_hardware(options)


def test_policy_timing_drift_and_result_tampering(scout, tmp_path):
    b, folder = copy_scout(scout, tmp_path)
    fake_collect(folder)
    select_policy(folder, tmp_path/"policy.json")
    options = paired_options(folder, tmp_path/"policy.json", tmp_path)
    with patch.object(timing, "timing_model", return_value={}):
        with pytest.raises(ValueError, match="durations/constraints changed"):
            run.apply_policy(options, backend=b)
    p = json.loads(options.policy.read_text()); p["physical_qubits"] = [0]
    atomic_json(options.policy, p)
    with pytest.raises(ValueError, match="system-only"):
        run.apply_policy(options, backend=b)
    result = folder/"job_0000_results.json"
    result.write_text(result.read_text()+" ")
    _, m, _ = run.read_saved(folder/"manifest.json")
    with pytest.raises(ValueError, match="archive changed"):
        run.require_timing(folder, m)


def test_dry_run_never_falls_back_and_submission_resumes_known_ids(scout, tmp_path):
    b, folder = copy_scout(scout, tmp_path)
    manifest = folder/"manifest.json"
    options = run.parse_arguments(["submit", "--manifest", str(manifest), "--dry-run"])
    with patch.object(run.importlib.metadata, "version", return_value="0.49.0"):
        with patch.object(run.raw.runtime, "make_service", side_effect=AssertionError("no account access")):
            with pytest.raises(ValueError, match="No hardware fallback"):
                run.submit(options)
    sampler = Mock(); sampler.run.return_value.job_id.return_value = "test_saved_id"
    factory = Mock(return_value=sampler)
    run.submit(options, backend=b, sampler_factory=factory)
    assert sampler.run.call_args.kwargs == {"dry_run": True}
    assert factory.call_args.kwargs["options"]["experimental"] == {"execution": {"scheduler_timing": True}}
    run.submit(options, backend=b, sampler_factory=factory)
    assert sampler.run.call_count == 1
    hardware = run.parse_arguments(["submit", "--manifest", str(manifest), "--hardware"])
    with pytest.raises(ValueError, match="cannot switch"):
        run.submit(hardware, backend=b, sampler_factory=factory)
    _, m, _ = run.read_saved(manifest)
    m["batches"][0].update(job_id=None, status="submitting"); run.checkpoint(folder, m)
    with pytest.raises(ValueError, match="ambiguous submission"):
        run.submit(options, backend=b, sampler_factory=factory)
    assert sampler.run.call_count == 1


def test_collect_only_known_done_jobs_and_preserve_missing_timing_results(scout, tmp_path):
    b, folder = copy_scout(scout, tmp_path)
    manifest = folder/"manifest.json"
    sampler = Mock(); sampler.run.return_value.job_id.return_value = "known_id"
    run.submit(run.parse_arguments(["submit", "--manifest", str(manifest), "--dry-run"]),
               backend=b, sampler_factory=Mock(return_value=sampler))
    service = Mock(); job = service.job.return_value; job.status.return_value = "RUNNING"
    options = run.parse_arguments(["collect", "--manifest", str(manifest)])
    with patch.object(run, "render"):
        run.collect(options, service=service)
    job.result.assert_not_called(); service.job.assert_called_once_with("known_id")
    _, m, _ = run.read_saved(manifest)
    pubs = []
    for row in m["circuits"]:
        counts = {"0"*10: row["shots"]}
        pubs.append(SimpleNamespace(metadata={}, data=SimpleNamespace(outcome=SimpleNamespace(get_counts=lambda v=counts: v))))
    job.status.return_value = "DONE"; job.result.return_value = pubs
    job.metrics.return_value = {}; job.usage.return_value = 1.25
    with pytest.raises(ValueError, match="no local timing fallback"):
        run.collect(options, service=service)
    _, m, _ = run.read_saved(manifest)
    assert m["status"] == "collected" and (folder/"job_0000_results.json").exists()
    assert sampler.run.call_count == 1
    with pytest.raises(ValueError, match="no local timing fallback"):
        run.require_timing(folder, m)


def test_hardware_maintenance_does_not_create_submitting_entry(scout, tmp_path):
    b, folder = copy_scout(scout, tmp_path)
    options = run.parse_arguments(["submit", "--manifest", str(folder/"manifest.json"), "--hardware"])
    factory = Mock()
    with patch.object(run.raw, "backend_readiness", side_effect=ValueError("backend maintenance")):
        with pytest.raises(ValueError, match="maintenance"):
            run.submit(options, backend=b, sampler_factory=factory)
    factory.assert_not_called()
    _, m, _ = run.read_saved(folder/"manifest.json")
    assert m["batches"][0]["status"] == "not_submitted" and m["batches"][0]["job_id"] is None


def test_download_only_skips_optional_requests_and_preserves_policy_evidence(scout, tmp_path):
    b, folder = copy_scout(scout, tmp_path)
    manifest = folder/"manifest.json"
    sampler = Mock(); sampler.run.return_value.job_id.return_value = "download_existing_id"
    run.submit(run.parse_arguments(["submit", "--manifest", str(manifest), "--dry-run"]),
               backend=b, sampler_factory=Mock(return_value=sampler))
    _, m, _ = run.read_saved(manifest)
    pubs = []
    for row in m["circuits"]:
        counts = {"0"*10: row["shots"]}
        metadata = {"compilation": {"scheduler_timing": {"timing": compiler_text()}}}
        pubs.append(SimpleNamespace(metadata=metadata,
            data=SimpleNamespace(outcome=SimpleNamespace(get_counts=lambda c=counts: c))))
    service = Mock(); job = service.job.return_value
    job.status.return_value = "DONE"; job.result.return_value = pubs
    options = run.parse_arguments(["collect", "--manifest", str(manifest), "--download-only"])
    with patch.object(run, "render", side_effect=AssertionError("no plotting during download")):
        checked = run.collect(options, service=service)
    assert checked["status"] == "collected"
    job.metrics.assert_not_called(); job.usage.assert_not_called()
    assert sampler.run.call_count == job.result.call_count == 1
    with patch.object(run.raw.runtime, "make_service", side_effect=AssertionError("already saved")):
        run.collect(options)
    p = select_policy(folder, tmp_path/"downloaded_policy.json")
    original = (folder/"job_0000_results.json").read_bytes()
    job.metrics.return_value = {}; job.usage.return_value = 1.25
    with patch.object(run, "render"):
        run.collect(run.parse_arguments(["collect", "--manifest", str(manifest)]), service=service)
    _, after, _ = run.read_saved(manifest)
    assert (folder/"job_0000_results.json").read_bytes() == original
    assert after["plan_sha256"] == checked["plan_sha256"]
    assert run.require_timing(folder, after) == p["timing_evidence"]
    assert (folder/"job_0000_details.json").exists()
    assert sampler.run.call_count == job.result.call_count == 1


def test_partial_render_resumes_completed_views_and_repairs_changed_html(scout, tmp_path):
    _, folder = copy_scout(scout, tmp_path)
    fake_collect(folder)
    options = run.parse_arguments(["render", "--manifest", str(folder/"manifest.json"),
                                   "--trajectory", "0", "--basis", "z"])
    def interrupted(value, destination, *, channels=None):
        if channels is not None:
            raise KeyboardInterrupt
        fake_draw(value, destination, channels=channels)
    with patch.object(timing, "draw", side_effect=interrupted):
        with pytest.raises(KeyboardInterrupt):
            run.render(options)
    out = folder/"timing"
    full = out/"path_00000_z_baseline.html"
    active = out/"path_00000_z_baseline_active.html"
    assert full.exists() and not active.exists()
    assert full.name in (out/"index.html").read_text()
    assert active.name not in (out/"index.html").read_text()
    with patch.object(timing, "draw", side_effect=fake_draw) as draw:
        run.render(options)
    assert draw.call_count == 1 and draw.call_args.kwargs["channels"] == list(map(lambda q: f"Qubit {q}", range(12)))
    assert active.exists() and not (out/"path_00001_z_baseline.html").exists()
    assert not (out/"path_00000_xy_baseline.html").exists()
    with patch.object(timing, "draw", side_effect=AssertionError("reuse completed views")):
        run.render(options)
    full.write_text("truncated HTML")
    with patch.object(timing, "draw", side_effect=fake_draw) as draw:
        run.render(options)
    assert draw.call_count == 1 and draw.call_args.kwargs["channels"] is None
    assert (out/"server_comparison_path_0_z.json").exists()
    assert not (out/"server_comparison.json").exists()


def test_three_arm_threshold_comparison_end_to_end(scout, tmp_path):
    b, folder = copy_scout(scout, tmp_path)
    fake_collect(folder)
    policy_path = tmp_path/"two_cutoffs.json"
    policy = run.create_policy(run.parse_arguments(["policy", "--manifest", str(folder/"manifest.json"),
        "--system-sites", *map(str, range(1, 11)), "--min-idle-ratios", "2", "10",
        "--note", "Compare broad and selective XX using identical inspected paths", "--output", str(policy_path)]))
    assert policy["artifact_type"] == "model2_dd_policy_comparison_v1"
    assert policy["min_idle_ratios"] == [2., 10.]
    options = run.parse_arguments(["apply-policy", "--manifest", str(folder/"manifest.json"),
        "--policy", str(policy_path), "--shots", "16", "--control-shots", "32",
        "--batch-size", "27", "--output-directory", str(tmp_path)])
    with patch.object(timing, "schedule", wraps=timing.schedule) as schedules:
        paired, m = run.apply_policy(options, backend=b)
    ratios = Counter(call.kwargs["policy"]["min_idle_ratio"] for call in schedules.call_args_list if call.kwargs.get("policy"))
    assert ratios == {2.: 6, 10.: 6}
    arms = ["baseline", "dd_ratio_2", "dd_ratio_10"]
    assert m["configuration"]["arms"] == arms
    assert len(m["batches"]) == 1 and len(m["circuits"]) == 27
    assert m["configuration"]["total_shots"] == 2*3*3*16+3*3*32
    assert m["files"]["scheduled_00000.qpy"]["circuit_count"] == 9
    assert m["files"]["controls.qpy"]["circuit_count"] == 9
    assert len(run.load_qpy(paired, m, "scheduled_00000.qpy")) == 9
    permutations = set()
    for trajectory in range(2):
        for basis in ("z", "xy", "yx"):
            rows = [r for r in m["circuits"] if r["kind"] == "evolution"
                    and r["trajectory"] == trajectory and r["basis"] == basis]
            assert len(rows) == 3 and {r["arm"] for r in rows} == set(arms)
            permutations.add(tuple(r["arm"] for r in rows))
            counts = {r["arm"]: r["dd_check"]["added_x_pulses"] for r in rows if r["arm"] != "baseline"}
            assert counts["dd_ratio_2"] >= counts["dd_ratio_10"]
    assert len(permutations) == 6
    for arm in arms:
        assert {r["control"] for r in m["circuits"] if r["kind"] == "control" and r["arm"] == arm} == set(run.CONTROLS)
    fake_collect(paired)
    with patch.object(timing, "draw", side_effect=fake_draw) as draw:
        checks = run.render(run.parse_arguments(["render", "--manifest", str(paired/"manifest.json"),
            "--trajectory", "0", "--basis", "z", "--view", "full"]))
    assert len(checks) == 2 and draw.call_count == 3
    assert {v["arm"] for v in checks} == set(arms[1:])
    clone_options = run.parse_arguments(["clone-for-hardware", "--manifest", str(paired/"manifest.json"),
                                        "--output-directory", str(tmp_path)])
    clone, cloned = run.clone_for_hardware(clone_options)
    assert cloned["files"] == m["files"] and all(b["job_id"] is None for b in cloned["batches"])
    fake_collect(clone, dry_run=False)
    report = analyze(clone/"manifest.json")
    assert len(report["server_timing_checks"]) == 12
    assert set(report["method_errors"]) == set(arms)
    assert len(report["error_comparisons"]) == 3
    direct = report["observables"]["populations"]["comparisons"]["dd_ratio_10_minus_dd_ratio_2"]
    assert direct["reference"] == "dd_ratio_2" and direct["treatment"] == "dd_ratio_10"
    np.testing.assert_allclose(direct["difference"], 0)
    # Two independent 16-shot PUBs per path, averaged over two paths. Baseline
    # variance must not be added to the direct DD-policy contrast.
    np.testing.assert_allclose(direct["shot_standard_errors_fixed_paths"], np.sqrt(2*.25/(16-1)/2))
    assert (clone/"analysis"/"method_errors.csv").exists()
    # A failure unique to the second policy must block preflight cloning too.
    _, changed, _ = run.read_saved(paired/"manifest.json")
    batch = changed["batches"][0]; result_path = paired/batch["result_file"]
    result = json.loads(result_path.read_text())
    for row, pub in zip(changed["circuits"], result["pubs"]):
        if row["arm"] == "dd_ratio_10" and row.get("dd_check", {}).get("added_x_pulses", 0):
            pub["timing"] = compiler_text(); break
    else:
        raise AssertionError("test needs a nonempty second policy")
    atomic_json(result_path, result); batch["result_sha256"] = run.qpy_storage.file_sha256(result_path)
    run.checkpoint(paired, changed)
    with pytest.raises(ValueError, match="service timing changed"):
        run.clone_for_hardware(clone_options)


@pytest.mark.parametrize("ratios", [(10, 2), (2, 2), (1, 10), (2, float("inf"))])
def test_comparison_policy_rejects_invalid_ratios(tmp_path, ratios):
    with pytest.raises(SystemExit):
        run.parse_arguments(["policy", "--manifest", str(tmp_path/"manifest.json"),
            "--system-sites", "1", "--min-idle-ratios", *map(str, ratios),
            "--note", "invalid ratio test", "--output", str(tmp_path/"policy.json")])
