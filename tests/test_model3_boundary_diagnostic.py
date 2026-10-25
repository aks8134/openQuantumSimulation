"""Isolated boundary controls, tested with local targets and mocked Runtime."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
from qiskit import qpy
from qiskit.quantum_info import DensityMatrix

from Model3.Experiment5 import run_boundary_diagnostic as runner, run_hardware as hardware
from tests import test_model3_experiment5_hardware as fixtures
from tests import test_model3_preparation_diagnostic as prep


class BoundaryTests(unittest.TestCase):
    def setup(self, folder, *extra):
        prepared, backend = prep.PreparationTests().setup(folder)
        options = runner.parse_arguments(["--reference-run", str(prepared.reference_run), "--steps", "1", "2",
            "--shots", "8", "--output-directory", str(Path(folder)/"boundary_runs"),
            "--analysis-output-directory", str(Path(folder)/"boundary_analysis"), *extra])
        return options, backend

    def sampler(self):
        sampler = Mock(); job = Mock(); job.job_id.return_value = "offline_boundary_job"
        def results():
            return [fixtures.pub({"00": 4, "01": 4}) for _ in sampler.run.call_args.args[0]]
        job.result.side_effect = results; job.metrics.return_value = {}
        sampler.run.return_value = job
        return sampler, job

    def test_default_budget_and_bad_settings(self):
        options = runner.parse_arguments([])
        raw, _, metadata = runner.reference(options.reference_run, [])
        source = json.loads(raw.read_bytes())
        cases = runner.plan(source, metadata["physical_path"], options.steps, options.sides)
        self.assertEqual(len(cases), 20)
        self.assertEqual(options.shots*len(cases), 40960)
        for args in (["--shots", "1"], ["--steps", "0"], ["--steps", "1", "1"], ["--sides", "right", "right"]):
            with self.subTest(args=args), self.assertRaises(ValueError): runner.parse_arguments(args)

    def test_ideal_predictions_match_explicit_joint_density_and_reset(self):
        source = fixtures.source()
        for case in runner.plan(source, [0, 1, 2, 3, 4], [1, 2], ["left", "right"]):
            state = DensityMatrix.from_label("00")  # build() includes the system preparation.
            final = state.evolve(runner.build(source, case))
            probabilities = final.probabilities()
            np.testing.assert_allclose(probabilities[1]+probabilities[3], case["ideal_system_population"], atol=1e-14)
            np.testing.assert_allclose(probabilities[2]+probabilities[3], 0., atol=1e-14)

    def test_native_compilation_has_fixed_pairs_and_synchronized_resets(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware, "make_service", side_effect=AssertionError("no IBM account")):
            options, backend = self.setup(folder, "--compile-only")
            manifest = runner.main(options, backend=backend, sampler_factory=Mock(side_effect=AssertionError("no submission")))
            self.assertEqual(manifest["configuration"]["total_circuits"], 16)
            file = next((Path(folder)/"boundary_runs").glob("*/isa_circuits.qpy"))
            with file.open("rb") as f: compiled = qpy.load(f)
            for circuit, case in zip(compiled, manifest["circuits"]):
                self.assertEqual(case["final_logical_to_physical"], [1, 0] if case["side"] == "left" else [3, 4])
                ops = circuit.count_ops()
                self.assertEqual(ops.get("reset", 0), case["steps"])
                self.assertEqual(ops.get("cz", 0), 2*case["steps"] if case["kind"] == "collision" else 0)
                self.assertEqual(ops["measure"], 2)
                self.assertEqual(ops.get("barrier", 0), 2*case["steps"])
            self.assertTrue((file.parent/"results_bundle.zip").is_file())

    def test_collect_and_offline_analysis_preserve_original_data(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware, "make_service", side_effect=AssertionError("no account")):
            options, backend = self.setup(folder); sampler, _ = self.sampler()
            report = runner.main(options, backend=backend, sampler_factory=Mock(return_value=sampler))
            self.assertEqual(len(report["cases"]), 16)
            self.assertTrue(all(r["system_population"] == .5 and r["bath_population"] == 0. for r in report["cases"]))
            directory = Path(report["run_directory"])
            with zipfile.ZipFile(directory/"results_bundle.zip") as z:
                self.assertIn("isa_circuits.qpy", z.namelist())
                self.assertIn("analysis/boundary_controls.png", z.namelist())
            raw = {p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
            offline = runner.parse_arguments(["--analyze-only", str(directory), "--analysis-output-directory", str(Path(folder)/"offline")])
            runner.main(offline, sampler_factory=Mock(side_effect=AssertionError("no submission")))
            for name,data in raw.items(): self.assertEqual((directory/name).read_bytes(), data)

    def test_interrupted_collection_resumes_original_job(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware, "make_service", side_effect=AssertionError("no account")):
            options, backend = self.setup(folder); sampler, job = self.sampler()
            job.result.side_effect = RuntimeError("collection interrupted")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                runner.main(options, backend=backend, sampler_factory=Mock(return_value=sampler))
            path = next((Path(folder)/"boundary_runs").glob("*/manifest.json"))
            job.result.side_effect = None; job.result.return_value = [fixtures.pub({"00": 8}) for _ in range(16)]
            service = Mock(); service.job.return_value = job
            opts = runner.parse_arguments(["--resume", str(path), "--analysis-output-directory", str(Path(folder)/"boundary_analysis")])
            forbidden = Mock(side_effect=AssertionError("no resubmission"))
            report = runner.main(opts, backend=backend, service=service, sampler_factory=forbidden)
            self.assertEqual(report["job_ids"], ["offline_boundary_job"])
            service.job.assert_called_once_with("offline_boundary_job"); forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
