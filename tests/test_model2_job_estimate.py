"""Full mitigation budget accounting, sample provenance and read-only archives."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import numpy as np
from qiskit.providers.fake_provider import GenericBackendV2

from Model2.Experiment6 import estimate_jobs as estimate


class JobEstimateTests(unittest.TestCase):
    def profile(self):
        return estimate.make_profile(dict(spec_sha256=estimate.input_data.SPEC_SHA256, steps=44, seed=29,
            randomizations=4, fold_factors=[1., 1.25, 1.5]), [[100, 200], [110, 190]], [10, 20], [3, 5],
            dict(kind="saved_fake_kingston", backend="fake_kingston"))

    def test_bundled_full_scope_budget_and_current_runner_overhead(self):
        profile = json.loads(estimate.DEFAULT_PROFILE.read_text())
        result = estimate.estimate(profile)
        self.assertEqual(result["configuration"]["trajectory_count"], 16384)
        self.assertEqual(result["configuration"]["observed_paths"], 32)
        self.assertTrue(result["configuration"]["footprint_expansion"])
        self.assertEqual(result["source"]["backend"], "fake_kingston")
        cases = result["scenarios"]
        self.assertEqual(cases["at_published_limit"]["jobs"], 16)
        self.assertEqual(cases["at_selected_fraction"]["jobs"], 20)
        self.assertTrue(cases["at_selected_fraction"]["minimum_for_supplied_counting_model"])
        self.assertEqual(cases["at_selected_fraction"]["shots"], 2342912)
        self.assertEqual(cases["current_runner_circuit_cap"]["jobs"], 432)
        self.assertEqual(cases["current_runner_circuit_cap"]["circuits"], 219472)
        self.assertEqual(cases["current_runner_circuit_cap"]["shots"], 7405568)
        for case in cases.values():
            self.assertTrue(case["within_counted_limits"])
            self.assertEqual(sum(j["trajectories"] for j in case["partitions"]), 16384)
            self.assertEqual(case["evolution_shots"], 16384*32*4)
            self.assertEqual(case["control_shots"], case["jobs"]*6*2048)
            self.assertGreaterEqual(case["minimum_paths_per_job"], 2)
            self.assertEqual(case["circuits"], 16384*13+case["jobs"]*15)

    def test_shots_change_execution_budget_but_not_instruction_storage(self):
        vectors, controls = np.array([[10]]*6), np.array([2])
        with patch.object(estimate.audit, "EXECUTION_LIMIT", 100):
            low = estimate.pack(vectors, controls, 1, 5, 1000)
            high = estimate.pack(vectors, controls, 20, 5, 1000)
        self.assertEqual(low["jobs"], 1)
        self.assertEqual(high["jobs"], 2)
        self.assertEqual(high["shots"], 130)
        self.assertEqual(high["partitions"][0]["maximum_instructions"], 42)

    def test_control_reservation_and_singleton_rebalance_preserve_whole_paths(self):
        packed = estimate.pack(np.array([[3]]*7), np.array([1]), 4, 6, 10)
        self.assertEqual([j["trajectories"] for j in packed["partitions"]], [3, 2, 2])
        self.assertEqual(packed["shots"], 7*4+3*6)
        self.assertEqual(packed["maximum_instructions"], 10)
        self.assertTrue(packed["minimum_for_supplied_counting_model"])
        with self.assertRaisesRegex(ValueError, "at least two"):
            estimate.pack(np.array([[4]]*3), np.array([1]), 4, 6, 10)
        with self.assertRaisesRegex(ValueError, "one whole"):
            estimate.pack(np.array([[10]]*2), np.array([1]), 4, 6, 10)

    def test_greedy_only_claims_minimum_if_it_matches_lower_bound(self):
        vectors = np.array([[4, 1]]*4+[[1, 4]]*4)
        result = estimate.pack(vectors, np.array([0, 0]), 1, 1, 10)
        self.assertEqual(result["jobs"], 3)
        self.assertEqual(result["lower_bound_jobs"], 2)
        self.assertFalse(result["minimum_for_supplied_counting_model"])

    def test_current_runner_cap_reports_memory_failure_without_claiming_feasibility(self):
        profile = json.loads(estimate.DEFAULT_PROFILE.read_text())
        result = estimate.estimate(profile, circuit_cap=10**9)
        capped = result["scenarios"]["current_runner_circuit_cap"]
        self.assertEqual(capped["jobs"], 1)
        self.assertFalse(capped["within_counted_limits"])
        self.assertFalse(capped["minimum_for_supplied_counting_model"])
        self.assertGreater(capped["lower_bound_jobs"], 1)

    def test_observed_maximum_envelope_and_protocol_hash_validation(self):
        profile = self.profile()
        output = estimate.estimate(profile, trajectory_count=8)
        self.assertEqual(output["scenarios"]["observed_maximum_envelope"]["maximum_instructions"], 8*200+20)
        profile["path_instructions"][0][0] += 1
        with self.assertRaisesRegex(ValueError, "hash"):
            estimate.estimate(profile)
        profile = self.profile()
        profile["protocol"]["randomizations"] = 8
        estimate.seal(profile)
        with self.assertRaisesRegex(ValueError, "four randomizations"):
            estimate.estimate(profile)
        profile = self.profile()
        profile["path_ids"] = [0, 0]
        estimate.seal(profile)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            estimate.estimate(profile)

    def test_invalid_settings_fail_instead_of_silently_changing_full_scope(self):
        for setting in (dict(trajectory_count=1), dict(trajectory_count=16385), dict(shots=30),
                        dict(shots=4), dict(control_shots=3), dict(instruction_fraction=0),
                        dict(instruction_fraction=float("nan")), dict(circuit_cap=40)):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                estimate.estimate(self.profile(), **setting)

    def test_default_cli_is_offline_and_writes_report_and_partitions(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(estimate.audit.run_hardware.runtime, "make_service", side_effect=AssertionError("no account")), \
             patch.object(estimate, "profile_from_manifest", side_effect=AssertionError("no archive needed")), \
             patch.object(estimate, "profile_from_fake", side_effect=AssertionError("no archive needed")):
            before = estimate.DEFAULT_PROFILE.read_bytes()
            report = estimate.main(["--output-directory", temporary])
            self.assertEqual(before, estimate.DEFAULT_PROFILE.read_bytes())
            self.assertEqual(report["configuration"]["trajectory_count"], 16384)
            self.assertEqual(len(list(Path(temporary).glob("*_jobs.csv"))), 4)
            text = (Path(temporary)/"JobEstimate.md").read_text()
            self.assertIn("not installed in the hardware runner", text)
            self.assertIn("FakeKingston", text)
            self.assertTrue((Path(temporary)/"job_estimate.json").exists())

    def test_outputs_cannot_overwrite_source_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/"input"/"sample.json"
            source.parent.mkdir()
            source.write_text(json.dumps(self.profile()))
            with self.assertRaisesRegex(ValueError, "outside"):
                estimate.main(["--footprints", str(source), "--output-directory", str(source.parent)])
            with self.assertRaisesRegex(ValueError, "outside"):
                estimate.main(["--footprints", str(source), "--output-directory", str(root/"output"),
                               "--export-footprints", str(source)])

    def test_raw_experiment_manifest_is_not_mistaken_for_mitigation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"manifest.json"
            path.write_text(json.dumps({"experiment": "Model2/Experiment5/hardware"}))
            with self.assertRaisesRegex(ValueError, "lacks twirled/folded"):
                estimate.profile_from_manifest(path)

    def test_compiled_mitigation_archive_is_counted_without_query_or_modification(self):
        from Model2.Experiment6 import run_hardware as runner
        backend = GenericBackendV2(12, basis_gates=["cz", "rz", "sx", "x"],
            coupling_map=[[j, j+1] for j in range(11)]+[[j+1, j] for j in range(11)], seed=17)
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(runner.raw.runtime, "make_service", side_effect=AssertionError("no account")):
            root = Path(temporary)
            runner.main(runner.parse_arguments(["--compile-only", "--backend", backend.name,
                "--trajectory-count", "2", "--shots", "8", "--control-shots", "128", "--bootstrap", "100",
                "--physical-path", *map(str, range(12)), "--min-t1-us", "1", "--min-t2-us", "1",
                "--max-cz-error", ".1", "--max-readout-error", ".1", "--output-directory", str(root/"input")]),
                backend=backend)
            folder = next((root/"input").glob("mitigation44_*"))
            path = folder/"manifest.json"
            before = {p.name: estimate.qpy_storage.file_sha256(p) for p in folder.iterdir() if p.is_file()}
            with patch.object(runner, "transpile", side_effect=AssertionError("no compilation")):
                report = estimate.main(["--manifest", str(path), "--trajectory-count", "2", "--shots", "8",
                    "--output-directory", str(root/"report"), "--export-footprints", str(root/"profile.json")])
            self.assertEqual(before, {p.name: estimate.qpy_storage.file_sha256(p) for p in folder.iterdir() if p.is_file()})
            self.assertFalse(report["configuration"]["footprint_expansion"])
            self.assertEqual(report["source"]["backend"], backend.name)
            self.assertEqual(report["source"]["kind"], "compiled_mitigation_manifest")
            profile = json.loads((root/"profile.json").read_text())
            vectors, controls = estimate.validate_profile(profile)
            _, manifest, _ = runner.read_saved(path)
            np.testing.assert_array_equal(vectors.sum(axis=0)+controls,
                manifest["batches"][0]["resources"]["instructions_per_qubit"])
            service = Mock()
            service.backend.return_value = backend
            live_output = root/"live_estimate"
            with patch("qiskit_ibm_runtime.SamplerV2", side_effect=AssertionError("no Sampler")):
                live = estimate.main(["--backend", backend.name, "--trajectory-count", "2", "--shots", "8",
                    "--control-shots", "128", "--physical-path", *map(str, range(12)),
                    "--calibration-policy", "record-only", "--output-directory", str(live_output),
                    "--export-footprints", str(root/"live_profile.json")], service=service)
            service.backend.assert_called_once_with(backend.name, use_fractional_gates=False)
            service.job.assert_not_called()
            self.assertEqual(live["source"]["kind"], "backend_compilation")
            self.assertTrue(live["source"]["compilation_complete"])
            self.assertEqual(live["source"]["physical_path"], list(range(12)))
            self.assertEqual(live["source"]["requested_sample_paths"], 2)
            self.assertFalse(live["configuration"]["footprint_expansion"])
            live_vectors, live_controls = estimate.validate_profile(json.loads((root/"live_profile.json").read_text()))
            np.testing.assert_array_equal(vectors, live_vectors)
            np.testing.assert_array_equal(controls, live_controls)
            snapshot = Path(live["source"]["snapshot_directory"])
            self.assertEqual(estimate.qpy_storage.file_sha256(snapshot/"backend_snapshot.json"), live["source"]["backend_snapshot_sha256"])
            self.assertEqual(estimate.qpy_storage.file_sha256(snapshot/"source.json"), live["source"]["source_sha256"])
            self.assertTrue((snapshot/"footprints.json").exists())
            self.assertFalse(list(snapshot.glob("*.qpy")))
            self.assertFalse((snapshot/"manifest.json").exists())
            self.assertEqual(before, {p.name: estimate.qpy_storage.file_sha256(p) for p in folder.iterdir() if p.is_file()})

    def test_backend_settings_are_explicit_and_invalid_inputs_never_query_service(self):
        from Model2.Experiment6 import run_hardware as runner
        cases = (["--sample-trajectories", "2"], ["--physical-path", *map(str, range(12))],
                 ["--backend", "ibm_phoenix", "--sample-trajectories", "0"],
                 ["--backend", "ibm_phoenix", "--trajectory-count", "2", "--sample-trajectories", "3"],
                 ["--backend", "ibm_phoenix", "--shots", "30"],
                 ["--backend", "ibm_phoenix", "--calibration-policy", "record-only"],
                 ["--backend", "ibm_phoenix", "--physical-path", "1", "2"])
        with patch.object(runner.raw.runtime, "make_service", side_effect=AssertionError("no query")):
            for arguments in cases:
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    estimate.main(arguments)

    def test_backend_identity_and_native_cz_support_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            backend = GenericBackendV2(12, basis_gates=["cx", "rz", "sx", "x"], seed=5)
            with self.assertRaisesRegex(ValueError, "differs"):
                estimate.main(["--backend", "ibm_phoenix", "--trajectory-count", "2",
                    "--output-directory", temporary], backend=backend)
            with self.assertRaisesRegex(ValueError, "native CZ"):
                estimate.main(["--backend", backend.name, "--trajectory-count", "2",
                    "--output-directory", temporary], backend=backend)


if __name__ == "__main__":
    unittest.main()
