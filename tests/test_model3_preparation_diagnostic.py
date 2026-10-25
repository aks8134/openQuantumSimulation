"""Fixed-chain preparation followup with local targets and mocked Runtime."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
from qiskit import qpy

from Model3.Experiment5 import analyze_preparation_diagnostic as analysis
from Model3.Experiment5 import run_preparation_diagnostic as runner
from Model3.Experiment5 import run_hardware as hardware, quality_path
from tests import test_model3_experiment5_hardware as fixtures


class PreparationTests(unittest.TestCase):
    def setup(self, folder, *extra):
        backend = fixtures.HardwareRunnerTests().backend()
        reference = Path(folder)/"reference"; reference.mkdir()
        raw = json.dumps(fixtures.source()).encode()
        (reference/"source.json").write_bytes(raw)
        (reference/"manifest.json").write_text(json.dumps(dict(experiment="Model3/Experiment5/prefix_diagnostic", status="collected",
            configuration=dict(backend=backend.name, source_sha256=hashlib.sha256(raw).hexdigest(), physical_path=[0, 1, 2, 3, 4]),
            circuits=[dict(final_logical_to_physical=[1, 2, 3, 0, 4])], batches=[dict(job_id="prior_prefix_job")])))
        options = runner.parse_arguments(["--reference-run", str(reference), "--focus-qubits", "1", "3",
            "--output-directory", str(Path(folder)/"runs"), "--analysis-output-directory", str(Path(folder)/"analysis"), *extra])
        return options, backend

    def sampler(self, failed=False):
        sampler = Mock(); job = Mock(); job.job_id.return_value = "offline_controls_job"
        job.result.return_value = [fixtures.pub({"000": 4096}), fixtures.pub({"111": 4096}),
                                  fixtures.pub({"000": 2304, "111": 1792} if failed else {"000": 2048, "111": 2048})]
        job.metrics.return_value = {}
        sampler.run.return_value = job
        return sampler, job

    def test_defaults_and_invalid_counts(self):
        options = runner.parse_arguments([])
        self.assertEqual(options.shots, 4096)
        self.assertEqual(options.focus_qubits, [56, 76])
        self.assertEqual(options.batch_size, 256)
        for args in (["--shots", "1"], ["--batch-size", "0"], ["--focus-qubits", "56", "56"],
                     ["--focus-qubits", "-1"], ["--resume", "x.json", "--compile-only"]):
            with self.subTest(args=args), self.assertRaises(ValueError):
                runner.parse_arguments(args)

    def test_compile_only_keeps_chain_and_builds_only_three_zero_step_controls(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware, "make_service", side_effect=AssertionError("no IBM account")), \
             patch.object(quality_path, "select", side_effect=AssertionError("must not rerank")):
            options, backend = self.setup(folder, "--compile-only")
            original = hardware.circuits.build_body
            with patch.object(hardware.circuits, "build_body", wraps=original) as build:
                manifest = runner.main(options, backend=backend, sampler_factory=Mock(side_effect=AssertionError("no job")))
            self.assertEqual(len(build.call_args_list), 3)
            self.assertTrue(all(c.kwargs.get("steps") == 0 for c in build.call_args_list))
            cfg = manifest["configuration"]
            self.assertEqual(cfg["total_circuits"], 3)
            self.assertEqual(cfg["total_shots"], 12288)
            self.assertEqual(cfg["trajectory_indices"], [])
            self.assertEqual(cfg["physical_path"], [0, 1, 2, 3, 4])
            self.assertEqual(len(manifest["batches"]), 1)
            path = next((Path(folder)/"runs").glob("*/isa_circuits.qpy"))
            with path.open("rb") as stream: compiled = qpy.load(stream)
            self.assertEqual(len(compiled), 3)
            self.assertEqual([c.count_ops()["measure"] for c in compiled], [3, 3, 6])
            self.assertTrue(all(not c.count_ops().get("cz", 0) and not c.count_ops().get("reset", 0) for c in compiled))
            self.assertFalse((path.parent/"hardware_results.json").exists())

    def test_collected_counts_and_focus_analysis_and_archive_are_verified(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware, "make_service", side_effect=AssertionError("no IBM account")):
            options, backend = self.setup(folder)
            sampler, _ = self.sampler()
            result = runner.main(options, backend=backend, sampler_factory=Mock(return_value=sampler))
            sampler.run.assert_called_once()
            self.assertEqual(len(sampler.run.call_args.args[0]), 3)
            self.assertEqual(sampler.run.call_args.kwargs["shots"], 4096)
            self.assertTrue(result["control_assessment"]["passed"])
            path = Path(result["run_directory"])
            raw = {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}
            report = analysis.analyze(path)
            self.assertEqual(report["total_shots"], 12288)
            self.assertEqual([s["site_1based"] for s in report["sites"] if s["focus"]], [1, 3])
            np.testing.assert_allclose([s["mixture_residual"] for s in report["sites"]], 0.)
            with zipfile.ZipFile(path/"results_bundle.zip") as archive:
                self.assertIn("isa_circuits.qpy", archive.namelist())
                self.assertIn("ControlReport.md", archive.namelist())
                self.assertIn("analysis/preparation_controls.png", archive.namelist())
                self.assertIn("analysis/Analysis.md", archive.namelist())
            analysis.save(report, Path(folder)/"second_analysis")
            for name, value in raw.items(): self.assertEqual((path/name).read_bytes(), value)

    def test_failed_mixture_produces_report_without_any_evolution_run(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware, "make_service", side_effect=AssertionError("no account")):
            options, backend = self.setup(folder)
            sampler, _ = self.sampler(failed=True)
            result = runner.main(options, backend=backend, sampler_factory=Mock(return_value=sampler))
            self.assertFalse(result["control_assessment"]["passed"])
            self.assertEqual(result["configuration"]["total_circuits"], 3)
            sampler.run.assert_called_once()
            report = (Path(folder)/"analysis/Analysis.md").read_text()
            self.assertIn("Diagnostic passed: **False**", report)
            self.assertIn("initial mixture inconsistent", report)

    def test_invalid_reference_rejected_before_account_access(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(hardware, "make_service", side_effect=AssertionError("no account")):
            options, _ = self.setup(folder)
            options.focus_qubits = [5]
            with self.assertRaisesRegex(ValueError, "system sites"):
                runner.main(options)
            options.focus_qubits = [1]
            (options.reference_run/"source.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                runner.main(options)

    def test_interrupted_job_is_recovered_without_resubmission(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware, "make_service", side_effect=AssertionError("no account")):
            options, backend = self.setup(folder)
            sampler, job = self.sampler()
            counts = job.result.return_value
            job.result.side_effect = RuntimeError("interrupted collection")
            sampler_factory = Mock(return_value=sampler)
            with self.assertRaisesRegex(RuntimeError, "interrupted collection"):
                runner.main(options, backend=backend, sampler_factory=sampler_factory)
            manifest = next((Path(folder)/"runs").glob("*/manifest.json"))
            job.result.side_effect = None; job.result.return_value = counts
            service = Mock(); service.job.return_value = job
            resumed = runner.parse_arguments(["--resume", str(manifest), "--analysis-output-directory", str(Path(folder)/"analysis")])
            forbidden = Mock(side_effect=AssertionError("must not resubmit"))
            result = runner.main(resumed, backend=backend, service=service, sampler_factory=forbidden)
            forbidden.assert_not_called()
            service.job.assert_called_once_with("offline_controls_job")
            self.assertTrue(result["control_assessment"]["passed"])


if __name__ == "__main__":
    unittest.main()
