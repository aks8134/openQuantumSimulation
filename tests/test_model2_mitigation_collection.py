"""Interrupted collection recovery without QPY deserialization or submission."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from Model2.Experiment6 import run_hardware as run, analyze_hardware as analysis


class CollectionTests(unittest.TestCase):
    def fixture(self, root, *, second_collected=False):
        options = run.parse_arguments(["--trajectory-count", "4", "--shots", "8", "--control-shots", "128",
            "--batch-size", "41", "--bootstrap", "100", "--output-directory", str(root)])
        folder, m, source = run.new_plan(options)
        m["configuration"]["physical_path"] = list(range(12))
        for i, batch in enumerate(m["batches"]):
            batch.update(compiled=True, isa_file=f"batch_{i}.qpy", isa_sha256="not_needed_for_collection",
                resources={"test": True}, status="submitted", job_id=f"saved_job_{i}")
            # Present but invalid QPY proves that collection never reads it.
            (folder/batch["isa_file"]).write_bytes(b"do not deserialize or hash for count recovery")
        if second_collected:
            m["batches"][1].update(status="collected", observations=self.counts(m, 1))
        run.checkpoint(m, folder)
        return folder, m, source

    def counts(self, m, index):
        batch = m["batches"][index]
        observations = []
        for row in m["circuits"][batch["start"]:batch["stop"]]:
            z, o = format(row["measurement_mask"], "010b"), format(row["measurement_mask"] ^ 1023, "010b")
            shots = row["shots"]
            counts = ({z: shots} if row.get("control") == "all_zero" else
                      {o: shots} if row.get("control") == "all_one" else {z: shots//2, o: shots//2})
            observations.append(dict(outcome_counts=counts))
        return observations

    def job(self, m, index, status="DONE"):
        job = Mock()
        job.status.return_value = status
        job.result.return_value = [SimpleNamespace(data=SimpleNamespace(outcome=SimpleNamespace(
            get_counts=lambda counts=row["outcome_counts"]: counts))) for row in self.counts(m, index)]
        return job

    def options(self, folder, *extra):
        return run.parse_arguments(["--resume", str(folder/"manifest.json"), "--collect-only", *extra])

    def test_collect_only_recovers_known_done_jobs_without_circuits_calibration_metrics_or_sampler(self):
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(run, "Publications", side_effect=AssertionError("no QPY")), \
             patch.object(run, "select_path", side_effect=AssertionError("no calibration")), \
             patch.object(run, "finish_analysis", side_effect=AssertionError("collection only")):
            folder, m, _ = self.fixture(Path(root))
            before = (folder/"manifest.json").read_bytes()
            service = Mock()
            jobs = [self.job(m, i) for i in range(2)]
            service.job.side_effect = jobs
            forbidden = Mock(side_effect=AssertionError("no submission"))
            result = run.main(self.options(folder), service=service, sampler_factory=forbidden)
            self.assertEqual(result["status"], "collected")
            self.assertEqual(result["plan_sha256"], m["plan_sha256"])
            self.assertEqual(result["configuration"], m["configuration"])
            self.assertEqual([b["job_id"] for b in result["batches"]], ["saved_job_0", "saved_job_1"])
            self.assertEqual(next(folder.glob("manifest.before_collect_*.json")).read_bytes(), before)
            self.assertEqual(len(list(folder.glob("job_*_counts.json"))), 2)
            forbidden.assert_not_called()
            service.backend.assert_not_called()
            for job in jobs:
                job.metrics.assert_not_called()
            with patch.object(run.raw.runtime, "make_service", side_effect=AssertionError("already local")):
                run.main(self.options(folder))
            self.assertFalse((folder/"results_bundle.zip").exists())

    def test_active_failed_and_ambiguous_jobs_are_never_waited_on_or_reset(self):
        for status in ("RUNNING", "ERROR", "CANCELLED"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
                folder, m, _ = self.fixture(Path(root))
                m["batches"][1].update(status="submitting", job_id=None)
                run.checkpoint(m, folder)
                before = (folder/"manifest.json").read_bytes()
                job = self.job(m, 0, status)
                job.result.side_effect = AssertionError("only DONE results")
                service = Mock(); service.job.return_value = job
                run.main(self.options(folder), service=service)
                job.result.assert_not_called()
                service.job.assert_called_once_with("saved_job_0")
                self.assertEqual((folder/"manifest.json").read_bytes(), before)

    def test_interrupted_manifest_checkpoint_recovers_journal_without_network(self):
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            folder, m, _ = self.fixture(Path(root), second_collected=True)
            before = (folder/"manifest.json").read_bytes()
            service = Mock(); service.job.return_value = self.job(m, 0)
            with patch.object(run, "checkpoint", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                run.main(self.options(folder), service=service)
            self.assertEqual((folder/"manifest.json").read_bytes(), before)
            self.assertTrue((folder/"job_0000_counts.json").exists())
            with patch.object(run.raw.runtime, "make_service", side_effect=AssertionError("local journal exists")):
                final = run.main(self.options(folder))
            self.assertEqual(final["status"], "collected")
            self.assertEqual(final["batches"][0]["observations"], self.counts(m, 0))

    def test_optional_metrics_are_requested_after_counts_are_durable(self):
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            folder, m, _ = self.fixture(Path(root), second_collected=True)
            job = self.job(m, 0)
            def metrics():
                saved = json.loads((folder/"manifest.json").read_text())
                self.assertEqual(saved["batches"][0]["status"], "collected")
                self.assertTrue((folder/"job_0000_counts.json").exists())
                raise KeyboardInterrupt
            job.metrics.side_effect = metrics
            service = Mock(); service.job.return_value = job
            with self.assertRaises(KeyboardInterrupt):
                run.main(self.options(folder, "--include-job-metrics"), service=service)
            with patch.object(run.raw.runtime, "make_service", side_effect=AssertionError("counts saved")):
                final = run.main(self.options(folder))
            self.assertEqual(final["status"], "collected")
            self.assertEqual(job.result.call_count, 1)

    def test_invalid_shots_or_orphan_identity_never_replace_manifest(self):
        for failure in ("shots", "job_id"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
                folder, m, _ = self.fixture(Path(root), second_collected=True)
                before = (folder/"manifest.json").read_bytes()
                service = Mock(); job = self.job(m, 0); service.job.return_value = job
                if failure == "shots":
                    job.result.return_value[0].data.outcome.get_counts = lambda: {"0"*10: 1}
                else:
                    run.atomic_json(folder/"job_0000_counts.json", dict(job_id="different_job"))
                with self.assertRaises(ValueError):
                    run.main(self.options(folder), service=service)
                self.assertEqual((folder/"manifest.json").read_bytes(), before)

    def test_count_analysis_is_offline_skips_existing_qpy_and_preserves_the_manifest(self):
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            folder, m, source = self.fixture(Path(root))
            m["status"] = "collected"
            for i, batch in enumerate(m["batches"]):
                batch.update(status="collected", observations=self.counts(m, i))
            run.checkpoint(m, folder)
            before = (folder/"manifest.json").read_bytes()
            with patch.object(run, "Publications", side_effect=AssertionError("no QPY")), \
                 patch.object(run.raw.runtime, "make_service", side_effect=AssertionError("offline")):
                report = analysis.analyze(folder, manifest=m, source=source)
            self.assertEqual(len(report["verification"]["compiled_batches_not_deserialized"]), 2)
            self.assertFalse(report["verification"]["compiled_circuit_verification_requested"])
            self.assertEqual(len(report["statistics"]["profiles"]), 6)
            for profile in report["statistics"]["profiles"].values():
                np.testing.assert_allclose(profile["populations"], .5, atol=1e-14)
            self.assertEqual((folder/"manifest.json").read_bytes(), before)
            original = zipfile.ZipFile.write
            def refuse_qpy(archive, path, *args, **kwargs):
                self.assertNotEqual(Path(path).suffix, ".qpy")
                return original(archive, path, *args, **kwargs)
            with patch.object(zipfile.ZipFile, "write", new=refuse_qpy):
                run.bundle(folder, m)
            old_zip = (folder/"results_bundle.zip").read_bytes()
            with patch.object(zipfile.ZipFile, "write", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                run.bundle(folder, m)
            self.assertEqual((folder/"results_bundle.zip").read_bytes(), old_zip)

    def test_submitted_circuit_lists_are_released_before_result_collection(self):
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            folder, m, _ = self.fixture(Path(root))
            for batch in m["batches"]:
                batch.update(status="not_submitted", job_id=None)
            run.checkpoint(m, folder)
            references = []
            test = self

            class Program:
                pass

            class Publications:
                def __getitem__(self, selected):
                    programs = [Program() for _ in range(selected.stop-selected.start)]
                    references.extend(weakref.ref(program) for program in programs)
                    return programs

            class Sampler:
                submitted = 0

                def run(self, programs, shots):
                    index = self.submitted
                    self.submitted += 1
                    job = test.job(m, index)
                    job.job_id.return_value = f"new_job_{index}"
                    response = job.result.return_value

                    def result():
                        test.assertTrue(all(reference() is None for reference in references))
                        return response

                    job.result.side_effect = result
                    return job

            sampler = Sampler()
            with patch.object(run.raw, "backend_readiness"):
                run.collect_and_submit(m, folder, Publications(), None, None,
                    sampler_factory=lambda **options: sampler)
            self.assertEqual(sampler.submitted, 2)
            self.assertEqual(m["status"], "collected")


if __name__ == "__main__":
    unittest.main()
