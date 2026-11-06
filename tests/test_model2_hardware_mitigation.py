"""Frozen Model2 hardware mitigation, storage, calibration and job recovery."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
from qiskit.providers.fake_provider import GenericBackendV2

from Model2.Experiment6 import run_hardware as runner, analyze_hardware as analysis


class HardwareMitigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.backend = GenericBackendV2(12, basis_gates=["cz", "rz", "sx", "x"],
            coupling_map=[[j, j+1] for j in range(11)]+[[j+1, j] for j in range(11)], seed=17)
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(runner.raw.runtime, "make_service", side_effect=AssertionError("no account")):
            runner.main(cls.options(cls.temporary.name), backend=cls.backend)
        cls.template = next(Path(cls.temporary.name).glob("mitigation44_*"))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def options(cls, folder, *extra):
        return runner.parse_arguments(["--compile-only", "--backend", cls.backend.name,
            "--trajectory-count", "4", "--shots", "8", "--control-shots", "128", "--bootstrap", "100",
            "--batch-size", "41", "--physical-path", *map(str, range(12)), "--min-t1-us", "1",
            "--min-t2-us", "1", "--max-cz-error", ".1", "--max-readout-error", ".1",
            "--output-directory", str(folder), *extra])

    def saved(self, folder):
        destination = Path(folder)/"run"
        shutil.copytree(self.template, destination)
        return destination

    def resume(self, folder, *extra):
        return runner.parse_arguments(["--resume", str(folder/"manifest.json"),
            "--analysis-output-directory", str(folder.parent/"analysis"), *extra])

    def sampler(self, folder):
        sampler, jobs = Mock(), []
        def submit(items, shots):
            saved = json.loads((folder/"manifest.json").read_text())
            self.assertTrue(all(b["compiled"] for b in saved["batches"]))
            results = []
            for circuit, _, requested in items:
                row = saved["circuits"][circuit.metadata["pub_index"]]
                self.assertEqual(row["shots"], requested)
                self.assertEqual(circuit.metadata["mitigation_case_id"], row["case_id"])
                zero, one = [format(value, "010b") for value in (row["measurement_mask"], row["measurement_mask"] ^ 1023)]
                counts = ({zero: requested} if row.get("control") == "all_zero" else
                          {one: requested} if row.get("control") == "all_one" else
                          {zero: requested//2, one: requested-requested//2})
                results.append(SimpleNamespace(data=SimpleNamespace(
                    outcome=SimpleNamespace(get_counts=lambda counts=counts: counts),
                    preparation=SimpleNamespace(get_counts=Mock(side_effect=AssertionError("ignore preparation bits"))))))
            job = Mock()
            job.job_id.return_value = f"model2_mit_job_{len(jobs)}"
            job.result.return_value = results
            job.metrics.return_value = {"usage": {"qpu_charge_time_seconds": 1}}
            jobs.append(job)
            return job
        sampler.run.side_effect = submit
        return sampler, jobs

    def test_default_is_offline_frozen_and_exact_pilot_budget(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(runner.raw.runtime, "make_service", side_effect=AssertionError("offline")), \
             patch.object(runner, "transpile", side_effect=AssertionError("no compilation")):
            manifest = runner.main(runner.parse_arguments(["--output-directory", temporary]))
            config = manifest["configuration"]
            self.assertEqual((config["total_jobs"], config["total_circuits"], config["total_shots"]), (2, 862, 90112))
            self.assertEqual(config["physics"]["gamma"], 2.)
            self.assertEqual(config["steps"], 44)
            self.assertEqual(config["physical_path"], None)
            self.assertEqual(config["fold_factors"], [1., 1.25, 1.5])
            self.assertEqual(config["sampler_options"]["twirling"], {"enable_gates": False, "enable_measure": False})
            for batch in manifest["batches"]:
                rows = manifest["circuits"][batch["start"]:batch["stop"]]
                self.assertEqual(sum(r["kind"] == "control" for r in rows), 15)
                self.assertEqual(sum(r["kind"] == "trajectory" and r["arm"] == "raw" for r in rows), 32)
            folder = next(Path(temporary).glob("mitigation44_*"))
            _, saved, source = runner.read_saved(folder/"manifest.json")
            self.assertEqual(source["preparation"]["populations"], [.5]*10)
            self.assertEqual(source["schedule"]["jump_probabilities"], [.125, .375, .125, .375])
            self.assertFalse(source["finite_path_assessment"]["empirical_classical_checks_passed"])
            runner.main(self.resume(folder))  # Plain resume also stays offline.

    def test_native_cz_resets_masks_layout_and_resources_survive_streaming_qpy(self):
        folder, manifest, _ = runner.read_saved(self.template/"manifest.json", require_circuits=True)
        store = runner.Publications(manifest, folder, self.backend)
        cz_counts = set()
        for batch in manifest["batches"]:
            programs = store[batch["start"]:batch["stop"]]
            rows = manifest["circuits"][batch["start"]:batch["stop"]]
            self.assertLess(batch["resources"]["maximum_instructions"], batch["resources"]["instruction_budget"])
            for program, row in zip(programs, rows):
                if row["kind"] == "trajectory":
                    cz_counts.add(program.count_ops()["cz"])
                    self.assertEqual(program.count_ops()["reset"], 44)
                    self.assertEqual(program.count_ops()["measure"], 20)
            with self.assertRaisesRegex(ValueError, "identity"):
                runner.validate_circuits(manifest, programs[::-1], rows)
        self.assertEqual(cz_counts, {880, 1100, 1320})

    def test_mock_execution_decodes_masks_and_returns_all_six_profiles(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(runner.raw.runtime, "make_service", side_effect=AssertionError("no live account")):
            folder = self.saved(temporary)
            sampler, jobs = self.sampler(folder)
            report = runner.main(self.resume(folder, "--execute"), backend=self.backend,
                                 service=Mock(), sampler_factory=Mock(return_value=sampler))
            self.assertEqual(len(jobs), 2)
            self.assertEqual(len(report["statistics"]["profiles"]), 6)
            for profile in report["statistics"]["profiles"].values():
                np.testing.assert_allclose(profile["populations"], .5, atol=1e-14)
            self.assertEqual(len(report["statistics"]["job_local_controls"]), 2)
            self.assertEqual(report["per_path_ideal_populations"].shape, (4, 10))
            self.assertEqual(report["verification"]["compiled_batches_not_available"], [])
            with zipfile.ZipFile(folder/"results_bundle.zip") as archive:
                self.assertEqual(sum(name.endswith(".qpy") for name in archive.namelist()), 2)
                self.assertIn("analysis/Analysis.md", archive.namelist())
            forbidden = Mock(side_effect=AssertionError("no duplicate submission"))
            runner.main(self.resume(folder, "--execute"), sampler_factory=forbidden)
            forbidden.assert_not_called()
            for path in folder.glob("*.qpy"):
                path.unlink()
            portable = analysis.analyze(folder)
            self.assertEqual(len(portable["verification"]["compiled_batches_not_available"]), 2)

    def test_job_local_spam_precedes_averaging_instead_of_pooling_responses(self):
        _, manifest, source = runner.read_saved(self.template/"manifest.json")
        manifest = copy.deepcopy(manifest)
        config = manifest["configuration"]
        config.update(shots=80, control_shots=4000)
        manifest["status"] = "collected"
        for index, batch in enumerate(manifest["batches"]):
            a, b, m = (.1, .9, .3) if index == 0 else (.2, .8, .65)
            rows = manifest["circuits"][batch["start"]:batch["stop"]]
            observations = []
            for row in rows:
                budget = config["control_shots"] if row["kind"] == "control" else config["shots"]
                row["shots"] = budget if row["arm"] == "raw" else budget//config["randomizations"]
                probability = ({"all_zero": a, "all_one": b, "initial_mixture": (a+b)/2}[row["control"]]
                               if row["kind"] == "control" else m)
                ones = round(probability*row["shots"])
                mask = row["measurement_mask"]
                observations.append({"outcome_counts": {format(mask, "010b"): row["shots"]-ones,
                                                        format(mask ^ 1023, "010b"): ones}})
            batch.update(status="collected", job_id=f"synthetic_{index}", observations=observations)
            batch["resources"]["executions"] = sum(r["shots"] for r in rows)
        config["total_shots"] = sum(r["shots"] for r in manifest["circuits"])
        report = analysis.analyze_counts(manifest, source)
        np.testing.assert_allclose(report["statistics"]["profiles"]["raw"]["populations"], .475)
        for name in ("raw_spam_corrected", "twirled_spam_corrected", "zne_spam_corrected"):
            np.testing.assert_allclose(report["statistics"]["profiles"][name]["populations"], .5, atol=1e-13)
        self.assertIn("job_local", report["statistics"]["calibration_scope"])

    def test_known_job_resume_retrieves_saved_id_without_resubmitting(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            folder = self.saved(temporary)
            sampler, jobs = self.sampler(folder)
            original = sampler.run.side_effect
            def interrupted(items, shots):
                job = original(items, shots)
                job.result.side_effect = RuntimeError("interrupted collection")
                return job
            sampler.run.side_effect = interrupted
            with self.assertRaisesRegex(RuntimeError, "interrupted collection"):
                runner.main(self.resume(folder, "--execute"), backend=self.backend, service=Mock(), sampler_factory=Mock(return_value=sampler))
            saved = json.loads((folder/"manifest.json").read_text())
            self.assertEqual(saved["batches"][0]["job_id"], "model2_mit_job_0")
            jobs[0].result.side_effect = None
            sampler.run.side_effect = original
            service = Mock()
            service.job.return_value = jobs[0]
            report = runner.main(self.resume(folder, "--execute"), backend=self.backend, service=service, sampler_factory=Mock(return_value=sampler))
            self.assertEqual(len(jobs), 2)
            service.job.assert_called_once_with("model2_mit_job_0")
            self.assertEqual(len(report["job_ids"]), 2)

    def test_ambiguous_submission_or_missing_returned_id_never_auto_retries(self):
        for mode in ("timeout", "missing_id"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
                folder = self.saved(temporary)
                sampler = Mock()
                if mode == "timeout":
                    sampler.run.side_effect = TimeoutError("unknown outcome")
                    expected = TimeoutError
                else:
                    sampler.run.return_value.job_id.return_value = None
                    expected = ValueError
                with self.assertRaises(expected):
                    runner.main(self.resume(folder, "--execute"), backend=self.backend, service=Mock(), sampler_factory=Mock(return_value=sampler))
                with patch.object(runner.raw.runtime, "make_service", side_effect=AssertionError("no connection")), \
                     self.assertRaisesRegex(ValueError, "job ID"):
                    runner.main(self.resume(folder, "--execute"), sampler_factory=Mock(side_effect=AssertionError("no retry")))

    def test_hashes_protect_source_native_programs_and_immutable_masks(self):
        for kind in ("source", "qpy", "plan"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                folder = self.saved(temporary)
                if kind == "plan":
                    path = folder/"manifest.json"
                    changed = json.loads(path.read_text())
                    changed["circuits"][0]["measurement_mask"] ^= 1
                    path.write_text(json.dumps(changed))
                else:
                    path = folder/"source.json" if kind == "source" else next(folder.glob("*.qpy"))
                    with path.open("ab") as handle:
                        handle.write(b" ")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    runner.read_saved(folder/"manifest.json")

    def test_partial_compilation_reuses_completed_batches(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            original = runner.qpy.dump
            calls = 0
            def interrupted(programs, handle):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("interrupted QPY write")
                return original(programs, handle)
            with patch.object(runner.qpy, "dump", side_effect=interrupted), self.assertRaisesRegex(RuntimeError, "QPY write"):
                runner.main(self.options(temporary), backend=self.backend)
            folder = next(Path(temporary).glob("mitigation44_*"))
            _, saved, _ = runner.read_saved(folder/"manifest.json")
            first = saved["batches"][0]
            file_hash = runner.qpy_storage.file_sha256(folder/first["isa_file"])
            done = {r["trajectory"] for r in saved["circuits"][first["start"]:first["stop"]] if r["kind"] == "trajectory"}
            transpile = runner.transpile
            def forbid_done(body, **kwargs):
                if body.metadata["steps"] == 44 and body.metadata["trajectory"] in done:
                    raise AssertionError("completed path recompiled")
                return transpile(body, **kwargs)
            with patch.object(runner, "transpile", side_effect=forbid_done):
                manifest = runner.main(self.resume(folder, "--compile-only"), backend=self.backend)
            self.assertTrue(all(b["compiled"] for b in manifest["batches"]))
            self.assertEqual(runner.qpy_storage.file_sha256(folder/first["isa_file"]), file_hash)

    def test_memory_budget_and_execution_limit_fail_before_submission(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            options = self.options(temporary, "--instruction-fraction", ".000001")
            options.compile_only, options.execute = False, True
            factory = Mock(side_effect=AssertionError("no hardware"))
            with self.assertRaisesRegex(ValueError, "instruction budget"):
                runner.main(options, backend=self.backend, service=Mock(), sampler_factory=factory)
            factory.assert_not_called()
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(runner.raw.runtime, "make_service", side_effect=AssertionError("offline budget")), \
             self.assertRaisesRegex(ValueError, "execution limit"):
            runner.main(runner.parse_arguments(["--trajectory-count", "2", "--shots", "2000000", "--output-directory", temporary]))

    def test_maintenance_blocks_retry_and_explicit_6056_retry_retains_history(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            folder = self.saved(temporary)
            path = folder/"manifest.json"
            manifest = json.loads(path.read_text())
            manifest["batches"][0].update(job_id="failed_6056", status="submitted")
            path.write_text(json.dumps(manifest))
            before = path.read_bytes()
            service, failed = Mock(), Mock()
            failed.status.return_value = "ERROR"
            failed.error_message.return_value = "Error code 6056; Internal Compiler Error"
            failed.metrics.return_value = {"usage": {"qpu_charge_time_seconds": 0}}
            service.job.return_value = failed
            forbidden = Mock(side_effect=AssertionError("no submit during maintenance"))
            options = self.resume(folder, "--execute", "--retry-failed-job", "failed_6056")
            with patch.object(self.backend, "status", create=True,
                              return_value=SimpleNamespace(operational=True, status_msg="maintenance")), \
                 self.assertRaisesRegex(ValueError, "not ready"):
                runner.main(options, backend=self.backend, service=service, sampler_factory=forbidden)
            self.assertEqual(before, path.read_bytes())
            sampler, jobs = self.sampler(folder)
            with patch.object(self.backend, "status", create=True,
                              return_value=SimpleNamespace(operational=True, status_msg="active")):
                runner.main(options, backend=self.backend, service=service, sampler_factory=Mock(return_value=sampler))
            final = json.loads(path.read_text())
            self.assertEqual(len(jobs), 2)
            self.assertEqual(final["batches"][0]["failed_attempts"][0]["job_id"], "failed_6056")
            self.assertEqual(final["batches"][0]["failed_attempts"][0]["metrics"]["usage"]["qpu_charge_time_seconds"], 0)

    def test_inspection_needs_no_circuits_and_does_not_reset_a_job(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            folder = self.saved(temporary)
            path = folder/"manifest.json"
            manifest = json.loads(path.read_text())
            manifest["batches"][0].update(job_id="failed_6056", status="submitted")
            path.write_text(json.dumps(manifest))
            before = path.read_bytes()
            for native in folder.glob("*.qpy"):
                native.unlink()
            service = Mock()
            service.job.return_value.status.return_value = "ERROR"
            service.job.return_value.error_message.return_value = "6056"
            service.job.return_value.metrics.return_value = {}
            with patch.object(self.backend, "status", create=True,
                              return_value=SimpleNamespace(operational=False, status_msg="maintenance")):
                report = runner.main(self.resume(folder, "--inspect-job", "failed_6056"), service=service, backend=self.backend)
            self.assertFalse(report["backend_status"]["operational"])
            self.assertEqual(before, path.read_bytes())

    def test_reference_manifest_reuses_only_matching_model2_mapping(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(runner.raw.runtime, "make_service", side_effect=AssertionError("offline reference")):
            _, _, source = runner.read_saved(self.template/"manifest.json")
            spec = runner.input_data.load_spec()
            opts = runner.raw.parse_arguments(["--backend", self.backend.name, "--trajectory-count", "4", "--seed", "29"])
            reference = runner.raw.plan(source, spec, opts)
            reference["configuration"]["physical_path"] = list(range(12))
            reference["plan_sha256"] = runner.raw.plan_hash(reference)
            path = Path(temporary)/"reference.json"
            path.write_text(json.dumps(reference))
            manifest = runner.main(runner.parse_arguments(["--backend", self.backend.name, "--reference-manifest", str(path),
                "--output-directory", str(Path(temporary)/"plans")]))
            self.assertEqual(manifest["configuration"]["requested_physical_path"], list(range(12)))
            self.assertEqual(manifest["configuration"]["reference"]["plan_sha256"], reference["plan_sha256"])
            self.assertTrue(all(b["job_id"] is None for b in manifest["batches"]))
            with self.assertRaisesRegex(ValueError, "reference backend"):
                runner.main(runner.parse_arguments(["--backend", "different_backend", "--reference-manifest", str(path),
                    "--output-directory", str(Path(temporary)/"bad")]))

    def test_invalid_cli_is_rejected_before_account_access(self):
        for argv in (["--randomizations", "3"], ["--shots", "7"], ["--batch-size", "40"],
                     ["--fold-factors", "1", "1.00001", "1.00002"], ["--physical-path", "0", "1"],
                     ["--calibration-policy", "record-only"], ["--analyze-only"], ["--inspect-job", "job"],
                     ["--retry-failed-job", "job"], ["--resume", "manifest", "--shots", "32"],
                     ["--allow-calibration-drift"], ["--resume", "manifest", "--allow-calibration-drift"],
                     ["--resume", "manifest", "--analyze-only", "--allow-calibration-drift"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                options = runner.parse_arguments(argv)
                # Fold rounding is validated by planning, before any service.
                if argv[0] == "--fold-factors":
                    runner.new_plan(options)

    def test_resume_accepts_recorded_drift_without_changing_plan_or_repeating_collected_job(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            folder = self.saved(temporary)
            path = folder/"manifest.json"
            _, manifest, _ = runner.read_saved(path)
            batch = manifest["batches"][0]
            observations = []
            for row in manifest["circuits"][batch["start"]:batch["stop"]]:
                mask, shots = row["measurement_mask"], row["shots"]
                zero, one = format(mask, "010b"), format(mask ^ 1023, "010b")
                counts = ({zero: shots} if row.get("control") == "all_zero" else
                          {one: shots} if row.get("control") == "all_one" else {zero: shots//2, one: shots//2})
                observations.append(dict(outcome_counts=counts))
            batch.update(status="collected", job_id="already_collected", observations=observations)
            runner.checkpoint(manifest, folder)
            runner.atomic_json(folder/"execution_backend_snapshot.json", {"earlier_execution": True})
            previous_execution_sha = runner.qpy_storage.file_sha256(folder/"execution_backend_snapshot.json")
            before = path.read_bytes()
            config, plan_hash = copy.deepcopy(manifest["configuration"]), manifest["plan_sha256"]
            originals = {name: runner.qpy_storage.file_sha256(folder/name) for name in
                ["source.json", "selected_path.json", "backend_snapshot.json"]+[b["isa_file"] for b in manifest["batches"]]}
            native_target = self.backend.target
            changed_properties = copy.deepcopy(native_target.qubit_properties)
            changed_properties[5].t1 = .5e-6  # Saved threshold is 1 microsecond.
            class TargetView:
                qubit_properties = changed_properties
                def __getattr__(self, name): return getattr(native_target, name)
                def __getitem__(self, name): return native_target[name]
            changed_backend = SimpleNamespace(name=self.backend.name, target=TargetView(),
                num_qubits=self.backend.num_qubits, coupling_map=self.backend.coupling_map)
            forbidden = Mock(side_effect=AssertionError("no job submitted before drift is accepted"))
            with self.assertRaisesRegex(ValueError, "allow-calibration-drift"):
                runner.main(self.resume(folder, "--execute"), backend=changed_backend, service=Mock(), sampler_factory=forbidden)
            self.assertEqual(before, path.read_bytes())
            forbidden.assert_not_called()
            sampler, jobs = self.sampler(folder)
            service = Mock()
            with patch.object(runner, "transpile", side_effect=AssertionError("no recompilation")), \
                 patch.object(runner.raw.quality_path, "select", side_effect=AssertionError("keep fixed path")), \
                 patch.object(runner, "finish_analysis", return_value={"complete": True}):
                runner.main(self.resume(folder, "--execute", "--allow-calibration-drift"),
                    backend=changed_backend, service=service, sampler_factory=Mock(return_value=sampler))
            _, final, _ = runner.read_saved(path)
            self.assertEqual(len(jobs), 1)
            service.job.assert_not_called()
            self.assertEqual(final["configuration"], config)
            self.assertEqual(final["plan_sha256"], plan_hash)
            self.assertEqual(final["batches"][0]["job_id"], "already_collected")
            self.assertEqual(final["batches"][0]["observations"], observations)
            self.assertEqual(final["status"], "collected")
            for name, digest in originals.items():
                self.assertEqual(runner.qpy_storage.file_sha256(folder/name), digest)
            drift = final["resume_calibration_checks"][0]
            self.assertEqual(drift["physical_path"], list(range(12)))
            self.assertEqual(drift["threshold_failures"], [dict(qubit=5, quantity="t1_seconds", value=.5e-6,
                threshold=1e-6, condition="minimum")])
            self.assertEqual(runner.qpy_storage.file_sha256(folder/drift["file"]), drift["sha256"])
            previous = drift["previous_execution_snapshot"]
            self.assertEqual(previous["sha256"], previous_execution_sha)
            self.assertEqual(runner.qpy_storage.file_sha256(folder/previous["file"]), previous_execution_sha)
            runner.bundle(folder, final)
            with zipfile.ZipFile(folder/"results_bundle.zip") as archive:
                self.assertIn(drift["file"], archive.namelist())
                self.assertIn(previous["file"], archive.namelist())

    def test_drift_override_retains_compilation_readiness_and_native_checks(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            folder = self.saved(temporary)
            path = folder/"manifest.json"
            forbidden = Mock(side_effect=AssertionError("no submission"))
            options = self.resume(folder, "--execute", "--allow-calibration-drift")
            before = path.read_bytes()
            with patch.object(self.backend, "status", create=True,
                    return_value=SimpleNamespace(operational=True, status_msg="maintenance")), \
                 self.assertRaisesRegex(ValueError, "not ready"):
                runner.main(options, backend=self.backend, service=Mock(), sampler_factory=forbidden)
            self.assertEqual(before, path.read_bytes())
            with patch.object(runner.protocol, "assert_supported", side_effect=ValueError("unsupported native operation")), \
                 self.assertRaisesRegex(ValueError, "unsupported native"):
                runner.main(options, backend=self.backend, service=Mock(), sampler_factory=forbidden)
            forbidden.assert_not_called()
            _, manifest, _ = runner.read_saved(path)
            self.assertTrue(all(b["job_id"] is None for b in manifest["batches"]))
            manifest["batches"][0]["compiled"] = False
            runner.checkpoint(manifest, folder)
            with patch.object(runner.raw.runtime, "make_service", side_effect=AssertionError("no account")), \
                 self.assertRaisesRegex(ValueError, "all jobs already compiled"):
                runner.main(options, sampler_factory=forbidden)


if __name__ == "__main__":
    unittest.main()
