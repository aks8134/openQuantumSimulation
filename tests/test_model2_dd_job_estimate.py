"""Native DD footprints, model lower bounds, and offline resumable job estimates."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from qiskit.providers.fake_provider import GenericBackendV2

from Model2.Experiment8 import estimate_jobs as estimate


def numeric_unit(context, kind, identity, value=300):
    width = context["num_qubits"]
    return estimate.seal(dict(artifact_type="model2_dd_footprint_unit_v1",
        context_sha256=context["sha256"], kind=kind, identity=identity,
        circuits=15 if kind == "controls" else 13,
        instructions=dict(dd=[value]*width, scheduled_without_dd=[value-20]*width,
                          before_scheduling=[value-40]*width),
        added_x_by_qubit=[2]*width, cases=[], seconds=1.))


def options(root, count=4):
    return estimate.parse_arguments(["--backend", "test", "--trajectory-count", str(count),
        "--physical-path", *map(str,range(12)), "--output-directory", str(root)])


def model_context():
    return estimate.seal(dict(backend="test", num_qubits=12, physical_path=list(range(12)),
        captured_at_utc="test fixture", dd_policy=dict(sequence="XX", min_idle_ratio=10.)))


def test_full_count_proves_model_minimum_but_sample_does_not(tmp_path):
    context = model_context()
    cfg = options(tmp_path)
    units = [numeric_unit(context, "trajectory", i) for i in range(4)]
    controls = numeric_unit(context, "controls", "all", 100)
    with patch.object(estimate.audit, "INSTRUCTION_LIMIT", 1000):
        full = estimate.estimate(units, controls, context, cfg)
        sample = estimate.estimate(units[:2], controls, context, cfg)
    p = full["scenarios"]["selected_fraction"]
    assert (p["jobs"], p["lower_bound_jobs"]) == (2, 2)
    assert p["minimum_certified_for_full_cohort_counting_model"]
    assert full["minimum_without_circuit_cap"]["selected_fraction"]["full_cohort_minimum_jobs"] == 2
    assert p["circuits"] == 13*4+15*2
    assert p["shots"] == 4*32*4+6*2048*2
    assert not full["scenarios"]["current_circuit_cap"]["within_counted_limits"]
    balanced = full["scenarios"]["balanced_runner_selected_fraction"]
    assert balanced["runner_batch_size_for_this_grouping"] == 41
    assert balanced["minimum_certified_for_full_cohort_counting_model"]
    assert sample["configuration"]["footprint_projection"]
    assert not sample["scenarios"]["selected_fraction"]["minimum_certified_for_full_cohort_counting_model"]
    assert sample["minimum_without_circuit_cap"]["selected_fraction"]["full_cohort_minimum_jobs"] is None


def test_compile_source_and_variants_match_frozen_protocol():
    source = estimate.compilation_source(4)
    from Model2.Experiment5 import input_data
    expected = input_data.make_source(input_data.load_spec(), 4, 29)
    assert source["schedule"] == expected["schedule"]
    assert source["preparation"] == expected["preparation"]
    cfg, paths, controls = estimate.case_rows()
    assert len(paths) == 13 and len(controls) == 15
    assert sorted(r["folds"] for r in paths) == [0]*5+[110]*4+[220]*4


@pytest.fixture(scope="module")
def cached(tmp_path_factory):
    folder = tmp_path_factory.mktemp("dd_job_estimate")
    backend = GenericBackendV2(12, basis_gates=["cz", "rz", "sx", "x"],
        coupling_map=[[q,q+1] for q in range(11)]+[[q+1,q] for q in range(11)], seed=17)
    a = estimate.parse_arguments(["--backend", backend.name, "--trajectory-count", "2",
        "--physical-path", *map(str,range(12)), "--output-directory", str(folder)])
    with patch.object(estimate.run.raw.runtime, "make_service", side_effect=AssertionError("no IBM account")):
        context = estimate.open_context(a, folder, backend=backend)
    return folder, backend, a, context


def test_real_native_folded_twirled_dd_footprints(cached):
    folder, backend, a, context = cached
    estimate.initialize_worker(folder)
    with patch.object(estimate.run.raw.runtime, "make_service", side_effect=AssertionError("no IBM account")):
        control = estimate.count_unit("controls", "all")
        path = estimate.count_unit("trajectory", 0)
    assert (control["circuits"], path["circuits"]) == (15, 13)
    assert sorted(c["cz"] for c in path["cases"]) == [880]*5+[1100]*4+[1320]*4
    assert all(c["resets"] == 44 and c["original_operations_and_local_times_preserved"] for c in path["cases"])
    assert sum(path["added_x_by_qubit"]) > 0
    # Every protected XX window replaces one delay with XX plus three delays:
    # two X (weight 2 each), two extra delays (weight 1 each): +6 per pair.
    np.testing.assert_array_equal(np.asarray(path["instructions"]["dd"]),
        np.asarray(path["instructions"]["scheduled_without_dd"])+3*np.asarray(path["added_x_by_qubit"]))


def test_numeric_resume_report_only_and_shot_changes_are_offline(cached, tmp_path):
    import shutil
    folder, backend, a, context = cached
    copy = tmp_path/"cache"
    shutil.copytree(folder, copy)
    for kind, identity in [("controls", "all"), ("trajectory", 0), ("trajectory", 1)]:
        estimate.atomic_json(estimate.unit_path(copy, kind, identity), numeric_unit(context, kind, identity))
    args = ["--backend", backend.name, "--trajectory-count", "2", "--output-directory", str(copy), "--report-only"]
    with patch.object(estimate.run.raw.runtime, "make_service", side_effect=AssertionError("no backend query")), \
         patch.object(estimate, "count_unit", side_effect=AssertionError("no recompilation")):
        report = estimate.main(estimate.parse_arguments(args))
        changed = estimate.main(estimate.parse_arguments(args+["--shots", "64"]))
    assert report["configuration"]["counted_paths"] == 2
    assert changed["configuration"]["shots_per_arm_factor"] == 64
    assert (copy/"JobEstimate.md").exists()
    file = copy/"path_00001.json"
    file.write_text(file.read_text().replace('"circuits": 13', '"circuits": 12'))
    with pytest.raises(ValueError, match="hash mismatch"):
        estimate.main(estimate.parse_arguments(args))


def test_missing_counts_cannot_be_called_a_full_report(cached, tmp_path):
    import shutil
    folder, backend, a, context = cached
    copy = tmp_path/"cache"
    shutil.copytree(folder, copy)
    args = ["--backend", backend.name, "--trajectory-count", "2", "--output-directory", str(copy), "--report-only"]
    with pytest.raises(ValueError, match="incomplete"):
        estimate.main(estimate.parse_arguments(args))


def test_accounting_accepts_literal_16834_without_changing_hardware_limit(tmp_path):
    a = options(tmp_path, 16834)
    assert a.trajectory_count == 16834
    assert a.sample_trajectories is None  # exhaustive by default


def test_counting_resumes_only_missing_units(cached, tmp_path):
    import shutil
    folder, backend, a, context = cached
    copy = tmp_path/"cache"
    shutil.copytree(folder, copy)
    first = numeric_unit(context, "trajectory", 0)
    estimate.atomic_json(estimate.unit_path(copy, "trajectory", 0), first)
    calls = []
    def counted(kind, identity):
        calls.append((kind, identity))
        return numeric_unit(context, kind, identity)
    with patch.object(estimate, "count_unit", side_effect=counted):
        units, control = estimate.count_missing(copy, context, [0,1], 1)
    assert calls == [("controls", "all"), ("trajectory", 1)]
    assert units[0]["sha256"] == first["sha256"]
    with patch.object(estimate, "count_unit", side_effect=AssertionError("all checkpoints must be reused")):
        resumed, _ = estimate.count_missing(copy, context, [0,1], 1)
    assert resumed == units


def test_spawned_worker_uses_only_cached_target_and_source(cached):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    folder, backend, a, context = cached
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
            initializer=estimate.initialize_worker, initargs=(str(folder),)) as pool:
        result = pool.submit(estimate.count_unit, "controls", "all").result(timeout=120)
    assert result["context_sha256"] == context["sha256"]
    assert result["circuits"] == 15


def test_report_only_preserves_completed_cache_after_transport_code_updates(cached, tmp_path):
    import json
    import shutil
    folder, backend, a, context = cached
    copy = tmp_path/"cache"
    shutil.copytree(folder, copy)
    old = dict(context, versions={"old": "compiler"}, code_hashes={"old": "production transport code"})
    old = estimate.seal(old)
    estimate.atomic_json(copy/"context.json", old)
    for kind, identity in [("controls", "all"), ("trajectory", 0), ("trajectory", 1)]:
        estimate.atomic_json(estimate.unit_path(copy, kind, identity), numeric_unit(old, kind, identity))
    (copy/"target.pkl").unlink()
    (copy/"source.json").unlink()
    args = ["--backend", backend.name, "--trajectory-count", "2", "--output-directory", str(copy)]
    with patch.object(estimate, "initialize_worker", side_effect=AssertionError("no old target deserialization")), \
         patch.object(estimate.run.raw.runtime, "make_service", side_effect=AssertionError("no IBM account")):
        report = estimate.main(estimate.parse_arguments(args+["--report-only"]))
    assert report["context_sha256"] == old["sha256"]
    with pytest.raises(ValueError, match="completed caches remain usable"):
        estimate.main(estimate.parse_arguments(args))
