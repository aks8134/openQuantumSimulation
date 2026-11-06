"""Recover completed jobs without submissions; analyze original partial paths."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from Model2 import audit_hardware_runs as audit
from Model2.Experiment6 import run_hardware as run, analyze_hardware as analyze


class HardwareAuditTests(unittest.TestCase):
    def fixture(self, temporary, *, collected=False):
        root = Path(temporary)
        options = run.parse_arguments(["--trajectory-count", "4", "--shots", "8", "--control-shots", "128",
            "--batch-size", "41", "--bootstrap", "100", "--output-directory", str(root/"input")])
        folder, manifest, source = run.new_plan(options)
        manifest["configuration"]["physical_path"] = list(range(12))
        for index, batch in enumerate(manifest["batches"]):
            batch.update(compiled=True, isa_file=f"test_{index}.qpy", isa_sha256="test_archive",
                resources=dict(test=True), status="submitted", job_id=f"existing_job_{index}")
        if collected:
            manifest["batches"][0].update(status="collected", observations=self.observations(manifest, 0))
        run.checkpoint(manifest, folder)
        return root, folder, manifest, source

    def observations(self, manifest, index):
        batch = manifest["batches"][index]
        values = []
        for row in manifest["circuits"][batch["start"]:batch["stop"]]:
            zero = format(row["measurement_mask"], "010b")
            one = format(row["measurement_mask"] ^ 1023, "010b")
            shots = row["shots"]
            counts = ({zero: shots} if row.get("control") == "all_zero" else
                      {one: shots} if row.get("control") == "all_one" else {zero: shots//2, one: shots//2})
            values.append(dict(outcome_counts=counts))
        return values

    def job(self, manifest, index, status):
        job = Mock()
        job.status.return_value = status
        job.metrics.return_value = dict(usage=dict(qpu_charge_time_seconds=1))
        job.error_message.return_value = "Error code 6056; Internal Compiler Error"
        job.result.return_value = [SimpleNamespace(data=SimpleNamespace(outcome=SimpleNamespace(
            get_counts=lambda value=value: value["outcome_counts"]))) for value in self.observations(manifest, index)]
        return job

    def test_partial_analysis_preserves_original_paths_and_does_not_claim_full_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, manifest, source = self.fixture(temporary, collected=True)
            with self.assertRaisesRegex(ValueError, "all mitigation jobs"):
                analyze.analyze_counts(manifest, source)
            report = analyze.analyze_counts(manifest, source, allow_partial=True)
            selected = sorted(r["trajectory"] for r in manifest["circuits"][:41]
                              if r["kind"] == "trajectory" and r["arm"] == "raw")
            self.assertFalse(report["analysis"]["complete"])
            self.assertEqual(report["analysis"]["trajectory_indices"], selected)
            self.assertEqual(report["analysis"]["collected_jobs"], 1)
            expected = np.asarray(source["final_covariances_real_gauge"])[selected].diagonal(axis1=-2, axis2=-1)
            np.testing.assert_array_equal(report["per_path_ideal_populations"], expected)
            self.assertEqual(len(report["statistics"]["profiles"]), 6)
            for value in report["statistics"]["profiles"].values():
                np.testing.assert_allclose(value["populations"], .5, atol=1e-14)

    def test_only_done_jobs_are_collected_and_backup_preserves_original_manifest(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), \
             patch("qiskit_ibm_runtime.SamplerV2", side_effect=AssertionError("no submissions")):
            root, folder, manifest, _ = self.fixture(temporary)
            before = (folder/"manifest.json").read_bytes()
            done, running = self.job(manifest, 0, "DONE"), self.job(manifest, 1, "RUNNING")
            running.result.side_effect = AssertionError("never wait on an active job")
            service = Mock(); service.job.side_effect = [done, running]
            report = audit.main(["--manifest", str(folder/"manifest.json"), "--collect-existing", "--analyze",
                "--output-directory", str(root/"report")], service=service)
            item = report["runs"][0]
            self.assertEqual(item["collected_jobs"], 1)
            self.assertEqual(item["state"], "incomplete")
            self.assertIn("--resume", item["resume_command"])
            self.assertEqual(item["analysis_summary"]["trajectories"], 2)
            self.assertFalse(item["analysis_summary"]["complete"])
            self.assertEqual(next(folder.glob("manifest.before_audit_*.json")).read_bytes(), before)
            _, saved = audit.load_manifest(folder/"manifest.json")
            self.assertEqual(saved["plan_sha256"], manifest["plan_sha256"])
            self.assertEqual(saved["batches"][0]["status"], "collected")
            self.assertEqual(saved["batches"][1]["job_id"], "existing_job_1")
            done.result.assert_called_once(); running.result.assert_not_called()
            # A second audit skips the already collected ID and recovers just the remainder.
            done_second = self.job(manifest, 1, "DONE")
            service2 = Mock(); service2.job.return_value = done_second
            report2 = audit.main(["--manifest", str(folder/"manifest.json"), "--collect-existing", "--analyze",
                "--output-directory", str(root/"report")], service=service2)
            service2.job.assert_called_once_with("existing_job_1")
            self.assertTrue(report2["runs"][0]["complete"])
            self.assertEqual(audit.load_manifest(folder/"manifest.json")[1]["status"], "collected")
            self.assertNotIn("resume_command", report2["runs"][0])

    def test_failed_and_ambiguous_submissions_are_not_reset_for_retry(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root, folder, manifest, _ = self.fixture(temporary)
            before = (folder/"manifest.json").read_bytes()
            service = Mock(); service.job.side_effect = [self.job(manifest, 0, "ERROR"), self.job(manifest, 1, "RUNNING")]
            report = audit.main(["--manifest", str(folder/"manifest.json"), "--check-ibm",
                "--output-directory", str(root/"report")], service=service)
            row = report["runs"][0]
            self.assertEqual(row["state"], "failed_saved_job")
            self.assertIn("--retry-failed-job existing_job_0", row["retry_commands"][0])
            self.assertNotIn("resume_command", row)
            self.assertEqual((folder/"manifest.json").read_bytes(), before)
            manifest["batches"][0].update(status="submitting", job_id=None)
            run.checkpoint(manifest, folder)
            report = audit.main(["--manifest", str(folder/"manifest.json"), "--output-directory", str(root/"report")])
            self.assertEqual(report["runs"][0]["state"], "ambiguous_submission")
            self.assertNotIn("resume_command", report["runs"][0])

    def test_bad_source_or_wrong_result_shots_do_not_overwrite_saved_counts(self):
        for failure in ("source", "shots"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
                root, folder, manifest, _ = self.fixture(temporary)
                before = (folder/"manifest.json").read_bytes()
                job = self.job(manifest, 0, "DONE")
                if failure == "source":
                    with (folder/"source.json").open("ab") as handle: handle.write(b" ")
                else:
                    job.result.return_value[0].data.outcome.get_counts = lambda: {"0"*10: 1}
                service = Mock(); service.job.side_effect = [job, self.job(manifest, 1, "RUNNING")]
                report = audit.main(["--manifest", str(folder/"manifest.json"), "--collect-existing",
                    "--output-directory", str(root/"report")], service=service)
                self.assertEqual(report["runs"][0]["collected_jobs"], 0)
                self.assertIn("lookup_or_collection_error", report["runs"][0]["jobs"][0])
                self.assertEqual((folder/"manifest.json").read_bytes(), before)
                self.assertFalse(list(folder.glob("manifest.before_audit_*")))

    def test_local_inventory_never_constructs_an_ibm_service(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(audit.raw.runtime, "make_service", side_effect=AssertionError("offline")):
            root, folder, _, _ = self.fixture(temporary, collected=True)
            before = (folder/"manifest.json").read_bytes()
            report = audit.main(["--manifest", str(folder/"manifest.json"), "--analyze", "--output-directory", str(root/"report")])
            self.assertEqual(report["runs"][0]["collected_jobs"], 1)
            self.assertEqual((folder/"manifest.json").read_bytes(), before)

    def test_raw_partial_cohort_preserves_nonprefix_ideal_paths_and_current_bases(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(audit.raw.runtime, "make_service", side_effect=AssertionError("offline")):
            root, input_folder, _, source = self.fixture(temporary)
            source_root = root/"raw"
            source_root.mkdir()
            options = audit.raw.parse_arguments(["--trajectory-count", "4", "--shots", "8", "--control-shots", "128",
                "--batch-size", "9", "--order-seed", "29"])
            manifest = audit.raw.plan(source, audit.input_data.load_spec(), options)
            audit.atomic_json(source_root/"source.json", source)
            audit.atomic_json(source_root/"spec.json", audit.input_data.load_spec())
            manifest["configuration"]["source_sha256"] = audit.qpy_storage.file_sha256(source_root/"source.json")
            first = manifest["batches"][0]
            selected = sorted({r["trajectory"] for r in manifest["circuits"][first["start"]:first["stop"]]
                               if r["kind"] == "evolution"})
            self.assertNotEqual(selected, [0, 1])
            observations = []
            for row in manifest["circuits"][first["start"]:first["stop"]]:
                shots = row["shots"]
                counts = ({"0"*10: shots} if row.get("control") == "all_zero" else
                          {"1"*10: shots} if row.get("control") == "all_one" else
                          {"0"*10: shots//2, "1"*10: shots//2})
                observations.append(dict(outcome_counts=counts))
            first.update(status="collected", job_id="raw_job_0", observations=observations)
            manifest["status"] = "compiled"
            manifest["plan_sha256"] = audit.raw.plan_hash(manifest)
            audit.atomic_json(source_root/"manifest.json", manifest)
            result = audit.main(["--manifest", str(source_root/"manifest.json"), "--analyze",
                                 "--output-directory", str(root/"report")])
            self.assertNotIn("analysis_error", result["runs"][0])
            report = json.loads((root/"report"/"raw"/"analysis.json").read_text())
            self.assertEqual(report["trajectory_indices"], selected)
            ideal = np.asarray(source["final_covariances_real_gauge"])[selected].diagonal(axis1=-2, axis2=-1).mean(axis=0)
            np.testing.assert_allclose(report["ideal_same_paths"]["populations"], ideal)
            np.testing.assert_allclose(report["observables"]["excitation_flows"], 0.)
            self.assertEqual(report["planned_jobs"], 2)
            self.assertFalse(report["complete"])


if __name__ == "__main__":
    unittest.main()
