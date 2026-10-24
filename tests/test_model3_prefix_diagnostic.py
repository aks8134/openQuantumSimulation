"""Prefix reference, compilation, count analysis and recovery without IBM access."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
from qiskit import qpy
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import DensityMatrix, partial_trace

from Model3.Experiment5 import analyze_prefix_diagnostic as analysis
from Model3.Experiment5 import quality_path, run_hardware as hardware
from Model3.Experiment5 import run_prefix_diagnostic as runner
from tests import test_model3_experiment5_hardware as fixtures


class PrefixTests(unittest.TestCase):
    def setup(self, folder, *extra):
        backend = fixtures.HardwareRunnerTests().backend()
        reference = Path(folder)/"reference"; reference.mkdir()
        source = fixtures.source()
        raw = json.dumps(source).encode()
        (reference/"source.json").write_bytes(raw)
        settings = dict(avoid_qubits=[], min_t1_seconds=80e-6, min_t2_seconds=60e-6,
                        max_cz_error=.005, max_readout_error=.02, max_expansions=100)
        selected = quality_path.validate(backend, [0, 1, 2, 3, 4], source, [0, 1], settings)
        manifest = dict(experiment="Model3/Experiment5/hardware", status="collected",
                        configuration=dict(backend=backend.name, source_sha256=runner.digest(raw),
                            trajectory_indices=[0, 1], layout="chain", physical_path=selected["physical_path"],
                            path_selection=selected, controls_only=False),
                        circuits=[dict(trajectory=i, final_logical_to_physical=[1, 2, 3, 0, 4]) for i in (0, 1)],
                        batches=[dict(job_id="saved_reference", status="collected")])
        (reference/"manifest.json").write_text(json.dumps(manifest))
        options = runner.parse_arguments(["--reference-run", str(reference), "--trajectory-count", "2",
            "--prefix-steps", "0", "1", "2", "--shots", "8", "--batch-size", "4",
            "--output-directory", str(Path(folder)/"runs"),
            "--analysis-output-directory", str(Path(folder)/"analysis"), *extra])
        return options, backend

    def sampler(self):
        sampler = Mock()
        def submit(circuits, shots):
            job = Mock()
            job.job_id.return_value = f"offline_{sampler.run.call_count}"
            job.result.return_value = [fixtures.pub({"001": shots//2, "110": shots-shots//2}) for _ in circuits]
            job.metrics.return_value = {}
            return job
        sampler.run.side_effect = submit
        return sampler

    def test_defaults_and_invalid_selections(self):
        options = runner.parse_arguments([])
        self.assertEqual(options.prefix_steps, [0, 12, 24, 48, 72])
        self.assertEqual(len(options.trajectory_indices), 32)
        self.assertEqual(options.shots, 128)
        self.assertEqual(len(options.prefix_steps)*len(options.trajectory_indices)*options.shots, 20480)
        for args in (["--shots", "1"], ["--trajectory-count", "0"], ["--trajectory-indices", "0", "0"],
                     ["--prefix-steps", "0", "0"], ["--prefix-steps", "-1"], ["--order-seed", "-1"],
                     ["--resume", "manifest.json", "--compile-only"]):
            with self.subTest(args=args), self.assertRaises(ValueError):
                runner.parse_arguments(args)

    def test_ideal_matches_explicit_system_bath_dilation_for_both_orders(self):
        # Independent small Hilbert-space calculation with one fresh bath at
        # each step. No Aer or measurement sampling is involved.
        for order in ("hamiltonian-first", "jump-first"):
            source = fixtures.source()
            source["configuration"]["order"] = order
            source["preparation"]["populations"] = [.2, .6, .8]
            reference = runner.ideal_references(source, [1, 0], [0, 1, 2])
            p = source["configuration"]["physics"]; n = p["n_qubits"]
            for column, trajectory in enumerate([1, 0]):
                diagonal = [np.prod([source["preparation"]["populations"][j] if bits >> j & 1
                                    else 1-source["preparation"]["populations"][j] for j in range(n)])
                            for bits in range(2**n)]
                state = DensityMatrix(np.diag(diagonal))
                for step in range(3):
                    probabilities = state.probabilities()
                    actual = [sum(probabilities[k] for k in range(2**n) if k >> j & 1) for j in range(n)]
                    np.testing.assert_allclose(actual, reference["prefixes"][step]["per_trajectory_populations"][column], atol=2e-13)
                    if step == 2:
                        break
                    dt = source["schedule"]["step_dts"][step]
                    choice = source["schedule"]["jump_choices"][trajectory][step]
                    def system(state):
                        for parity in (0, 1):
                            for j in range(parity, n-1, 2):
                                state = state.evolve(XXPlusYYGate(4*p["exchange_prefactor"]*p["J"]*dt), [j, j+1])
                        return state
                    def jump(state):
                        fraction = [p["p_left"], 1-p["p_left"], p["p_right"], 1-p["p_right"]][choice]
                        angle = 2*np.sqrt(p["gamma"]*fraction*dt/source["schedule"]["jump_probabilities"][choice])
                        bath = np.diag([0., 1.]) if choice % 2 == 0 else np.diag([1., 0.])
                        joint = DensityMatrix(np.kron(bath, state.data))
                        joint = joint.evolve(XXPlusYYGate(angle), [0 if choice < 2 else n-1, n])
                        return partial_trace(joint, [n])
                    state = jump(system(state)) if order == "hamiltonian-first" else system(jump(state))

    def test_real_cosine_prefixes_reproduce_saved_final_covariances(self):
        source = json.loads((runner.DIRECTORY/"inputs/cosine72_seed29_paths0-127.json").read_bytes())
        reference = runner.ideal_references(source, list(range(32)), [0, 12, 24, 48, 72])
        self.assertEqual([r["prefix_steps"] for r in reference["prefixes"]], [0, 12, 24, 48, 72])
        np.testing.assert_allclose(reference["prefixes"][0]["per_trajectory_populations"], .5)
        saved = np.asarray(source["final_covariances_real_gauge"]).reshape(-1, 10, 10)[:32]
        np.testing.assert_allclose(reference["prefixes"][-1]["per_trajectory_populations"],
                                   np.diagonal(saved, axis1=-2, axis2=-1), atol=2e-12)
        self.assertAlmostEqual(reference["prefixes"][-1]["physical_time"], 22.809542863156878)

    def test_compile_only_keeps_fixed_path_preparation_and_prefix_resets(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware, "make_service", side_effect=AssertionError("no IBM account")):
            options, backend = self.setup(folder, "--compile-only")
            forbidden = Mock(side_effect=AssertionError("no submission"))
            manifest = runner.main(options, backend=backend, sampler_factory=forbidden)
            forbidden.assert_not_called()
            path = next((Path(folder)/"runs").glob("*/manifest.json"))
            with (path.parent/"isa_circuits.qpy").open("rb") as stream:
                saved = qpy.load(stream)
            self.assertEqual(len(saved), 6)
            pairs = [(r["prefix_steps"], r["trajectory"]) for r in manifest["circuits"]]
            self.assertEqual(set(pairs), {(k, i) for k in range(3) for i in (0, 1)})
            self.assertNotEqual(pairs, sorted(pairs))
            for row, circuit in zip(manifest["circuits"], saved):
                self.assertEqual(row["final_logical_to_physical"], [1, 2, 3, 0, 4])
                self.assertEqual(circuit.count_ops().get("reset", 0), row["prefix_steps"])
                self.assertEqual(circuit.count_ops()["measure"], 6)
                self.assertEqual(circuit.count_ops().get("cz", 0), 6*row["prefix_steps"])
                self.assertEqual([r.name for r in circuit.cregs], ["preparation", "outcome"])
            self.assertEqual(manifest["configuration"]["total_shots"], 48)
            self.assertEqual(len(manifest["batches"]), 2)
            self.assertTrue((path.parent/"results_bundle.zip").is_file())

    def test_invalid_reference_or_prefix_fails_before_account_access(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(hardware, "make_service", side_effect=AssertionError("no account")):
            options, backend = self.setup(folder)
            options.prefix_steps = [3]
            with self.assertRaisesRegex(ValueError, "prefix exceeds"):
                runner.main(options)
            options.prefix_steps = [0, 1]
            options.trajectory_indices = [2]
            with self.assertRaisesRegex(ValueError, "subset"):
                runner.main(options)
            options.trajectory_indices = [0]
            (options.reference_run/"source.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                runner.main(options)

    def test_collect_analyze_reject_corruption_and_completed_resume_is_offline(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware, "make_service", side_effect=AssertionError("no IBM account")):
            options, backend = self.setup(folder)
            sampler = self.sampler()
            report = runner.main(options, backend=backend, sampler_factory=Mock(return_value=sampler))
            self.assertEqual(sampler.run.call_count, 2)
            self.assertEqual([row["prefix_steps"] for row in report["prefixes"]], [0, 1, 2])
            self.assertTrue(all(np.allclose(row["hardware_populations"], .5) for row in report["prefixes"]))
            run = Path(report["run_directory"])
            with zipfile.ZipFile(run/"results_bundle.zip") as archive:
                self.assertIn("isa_circuits.qpy", archive.namelist())
                self.assertIn("analysis/populations_by_prefix.png", archive.namelist())
            resume = runner.parse_arguments(["--resume", str(run/"manifest.json"),
                "--analysis-output-directory", str(Path(folder)/"analysis2")])
            runner.main(resume, sampler_factory=Mock(side_effect=AssertionError("no resubmission")))
            qpy_path = run/"isa_circuits.qpy"
            original_qpy = qpy_path.read_bytes(); qpy_path.unlink()
            self.assertIn("inside bundle", analysis.analyze(run)["verification"]["isa_qpy"])
            qpy_path.write_bytes(original_qpy)
            original = json.loads((run/"manifest.json").read_text())
            for change in (lambda m: m["circuits"][0].update(prefix_steps=99),
                           lambda m: m["batches"][0]["observations"][0].update(outcome_counts={"000": 7}),
                           lambda m: m["circuits"][0].update(final_logical_to_physical=[2, 1, 3, 0, 4])):
                modified = copy.deepcopy(original); change(modified)
                (run/"manifest.json").write_text(json.dumps(modified))
                with self.assertRaises(ValueError):
                    analysis.analyze(run)
            (run/"manifest.json").write_text(json.dumps(original))

    def test_interrupted_job_is_collected_without_resubmission(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware, "make_service", side_effect=AssertionError("no IBM account")):
            options, backend = self.setup(folder, "--batch-size", "256")
            sampler = Mock(); job = Mock(); job.job_id.return_value = "saved_pending_job"
            job.result.side_effect = RuntimeError("interrupted collection")
            sampler.run.return_value = job
            with self.assertRaisesRegex(RuntimeError, "interrupted collection"):
                runner.main(options, backend=backend, sampler_factory=Mock(return_value=sampler))
            path = next((Path(folder)/"runs").glob("*/manifest.json"))
            job.result.side_effect = None
            job.result.return_value = [fixtures.pub({"000": 4, "111": 4}) for _ in range(6)]
            job.metrics.return_value = {}
            service = Mock(); service.job.return_value = job
            resume = runner.parse_arguments(["--resume", str(path), "--analysis-output-directory", str(Path(folder)/"analysis")])
            forbidden = Mock(side_effect=AssertionError("must not resubmit"))
            report = runner.main(resume, backend=backend, service=service, sampler_factory=forbidden)
            self.assertEqual(report["job_ids"], ["saved_pending_job"])
            service.job.assert_called_once_with("saved_pending_job")
            forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
