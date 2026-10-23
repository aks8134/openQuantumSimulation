"""Population-only live runner with per-PUB shots; all jobs are mocked."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from qiskit.primitives.containers import SamplerPub

from Model3.Experiment5 import run_hardware as hardware
from tests import test_model3_experiment5_hardware as fixtures
from tests.test_model3_experiment5_hardware import pub, source


class PopulationHardwareTests(unittest.TestCase):
    def options(self, folder, *extra):
        path = Path(folder)/"source.json"
        path.write_text(json.dumps(source()))
        return hardware.parse_arguments(["--source", str(path), "--trajectory-count", "2",
            "--bases", "z", "--population-controls", "--shots", "64", "--control-shots", "512",
            "--batch-size", "256", "--output-directory", str(Path(folder)/"runs"), *extra])

    def backend(self):
        return fixtures.HardwareRunnerTests().backend()

    def results(self, pubs):
        results = []
        for value in pubs:
            value = SamplerPub.coerce(value, shots=64)
            if value.shots == 512:
                name = value.circuit.metadata.get("control")
                counts = ({"000": 512} if name == "all_zero" else {"111": 512}
                          if name == "all_one" else {"000": 256, "111": 256})
            else:
                counts = {"001": 64}
            results.append(pub(counts))
        return results

    def test_cli_restricts_controls_to_population_mode(self):
        opts = hardware.parse_arguments(["--bases", "z", "--population-controls", "--trajectory-count", "128"])
        self.assertEqual(opts.bases, ["z"])
        self.assertEqual(opts.trajectory_indices, list(range(128)))
        for arguments in (["--bases", "xy"], ["--bases", "z", "z"], ["--population-controls"],
                          ["--bases", "z", "--population-controls", "--layout", "automatic"],
                          ["--control-shots", "0"]):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                hardware.parse_arguments(arguments)

    def test_controls_are_interleaved_and_measure_same_sites(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            options = self.options(folder)
            data = source(); before = json.dumps(data, sort_keys=True)
            publications, records, path = hardware.prepare(data, self.backend(), options)
            self.assertEqual(json.dumps(data, sort_keys=True), before)
            self.assertEqual(len(publications), 5)
            self.assertEqual([row["pub_index"] for row in records], list(range(5)))
            self.assertEqual([row["shots"] for row in records], [512, 64, 512, 64, 512])
            self.assertEqual([records[k]["control"] for k in (0, 2, 4)], ["all_zero", "all_one", "initial_mixture"])
            self.assertEqual([records[k]["trajectory"] for k in (1, 3)], [0, 1])
            for k in (0, 2, 4):
                self.assertEqual(records[k]["final_logical_to_physical"][:3], records[1]["final_logical_to_physical"][:3])
                self.assertNotIn("reset", publications[k].count_ops())
            self.assertEqual(publications[4].count_ops()["measure"], 6)

    def test_mixed_shots_summary_and_snapshot_without_control_contamination(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            options = self.options(folder)
            sampler = Mock()
            job = Mock(); job.job_id.return_value = "population_job"
            job.result.side_effect = lambda: self.results(sampler.run.call_args.args[0])
            job.metrics.return_value = {"timestamps": {"running": "2026-10-23T00:00:00Z"}}
            sampler.run.return_value = job
            result = hardware.main(options, backend=self.backend(), service=Mock(), sampler_factory=Mock(return_value=sampler))
            sent = [SamplerPub.coerce(value) for value in sampler.run.call_args.args[0]]
            self.assertEqual([value.shots for value in sent], [512, 64, 512, 64, 512])
            self.assertEqual(result["configuration"]["total_shots"], 1664)
            self.assertEqual(result["configuration"]["total_circuits"], 5)
            self.assertEqual(result["error_scope"], "populations_only")
            self.assertEqual(result["per_trajectory"]["indices"], [0, 1])
            np.testing.assert_allclose(result["observables"]["populations"], [1., 0., 0.])
            self.assertNotIn("excitation_flows", result["observables"])
            self.assertTrue(all("excitation_flows" not in value for value in result["references"].values()))
            self.assertEqual(len(result["population_controls"]), 3)
            self.assertTrue(all(row["max_absolute_error"] == 0 for row in result["population_controls"]))
            manifest_path = next(Path(folder).glob("runs/*/manifest.json"))
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["batches"][0]["job_metrics"], job.metrics.return_value)
            self.assertTrue((manifest_path.parent/"backend_snapshot.json").exists())
            snapshot = json.loads((manifest_path.parent/"backend_snapshot.json").read_text())
            self.assertEqual(len(snapshot["physical_path"]), 5)
            self.assertTrue(snapshot["instructions"])

    def test_wrong_control_shot_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            sampler = Mock(); job = Mock(); job.job_id.return_value = "wrong_control_shots"
            job.result.return_value = [pub({"000": 64})]*5
            sampler.run.return_value = job
            with self.assertRaisesRegex(ValueError, "shot count"):
                hardware.main(self.options(folder), backend=self.backend(), service=Mock(), sampler_factory=Mock(return_value=sampler))

    def test_resume_mixed_shots_recovers_without_resubmitting(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            sampler = Mock(); job = Mock(); job.job_id.return_value = "resume_population"
            job.result.side_effect = RuntimeError("interrupted collection")
            sampler.run.return_value = job
            backend = self.backend()
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                hardware.main(self.options(folder), backend=backend, service=Mock(), sampler_factory=Mock(return_value=sampler))
            path = next(Path(folder).glob("runs/*/manifest.json"))
            job.result.side_effect = lambda: self.results(sampler.run.call_args.args[0])
            service = Mock(); service.job.return_value = job
            forbidden = Mock(side_effect=AssertionError("must not resubmit"))
            result = hardware.main(hardware.parse_arguments(["--resume", str(path)]), backend=backend,
                                   service=service, sampler_factory=forbidden)
            forbidden.assert_not_called()
            self.assertEqual(result["configuration"]["total_shots"], 1664)
            self.assertEqual(len(result["population_controls"]), 3)

    def test_portable_128_path_input_preserves_original_four(self):
        path = hardware.DIRECTORY/"inputs/cosine72_seed29_paths0-127.json"
        data = json.loads(path.read_text())
        original = json.loads(hardware.DEFAULT_SOURCE.read_text())
        hardware.require_validated_source(data)
        self.assertLess(path.stat().st_size, 1024*1024)
        self.assertEqual(data["configuration"]["trajectories"], 128)
        self.assertEqual(data["schedule"]["step_dts"], original["schedule"]["step_dts"])
        self.assertEqual(data["schedule"]["jump_choices"][:4], original["schedule"]["jump_choices"])
        self.assertEqual(data["final_covariances_real_gauge"][0][:4], original["final_covariances_real_gauge"][0])


if __name__ == "__main__":
    unittest.main()
