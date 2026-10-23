"""Calibration selection and staged execution using only injected local backends."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from qiskit import qpy
from qiskit.circuit import Measure
from qiskit.circuit.library import CZGate
from qiskit.transpiler import InstructionProperties, QubitProperties, Target

from Model3.Experiment5 import control_quality, quality_path
from Model3.Experiment5 import run_hardware as hardware
from Model3.Experiment5 import run_population_diagnostic as workflow
from tests import test_model3_experiment5_hardware as fixtures


def calibrated_backend():
    target = Target(num_qubits=10, qubit_properties=[QubitProperties(t1=150e-6, t2=100e-6)]*10)
    edges = {}
    for path, error in (([0, 1, 2, 3, 4], .004), ([5, 6, 7, 8, 9], .001)):
        for a, b in zip(path, path[1:]):
            for pair in ((a, b), (b, a)):
                edges[pair] = InstructionProperties(error=error, duration=68e-9)
    target.add_instruction(CZGate(), edges)
    target.add_instruction(Measure(), {(q,): InstructionProperties(error=.01, duration=1e-6) for q in range(10)})
    return SimpleNamespace(name="offline_calibration", target=target)


class QualityPathTests(unittest.TestCase):
    def test_selection_filters_and_ranks_entire_paths(self):
        backend = calibrated_backend(); source = fixtures.source()
        chosen = quality_path.select(backend, source, [0, 1], avoid=())
        self.assertEqual(set(chosen["physical_path"]), {5, 6, 7, 8, 9})
        self.assertEqual(chosen["metrics"]["worst_cz_error"], .001)
        self.assertEqual(sum(chosen["edge_gate_weights"]), 12.)  # 2 steps × (2 system bonds + bath) × 2 CZ.
        avoided = quality_path.select(backend, source, [0, 1], avoid=(7,))
        self.assertEqual(set(avoided["physical_path"]), {0, 1, 2, 3, 4})
        self.assertNotIn(7, avoided["physical_path"])
        self.assertTrue(avoided["search_complete"])
        with self.assertRaisesRegex(ValueError, "thresholds were not relaxed"):
            quality_path.select(backend, source, [0, 1], avoid=(7,), max_cz_error=.002)

    def test_coherence_and_fresh_edge_checks(self):
        backend = calibrated_backend(); source = fixtures.source()
        choice = quality_path.select(backend, source, [0, 1], avoid=())
        for pair in ((6, 7), (7, 6)):
            backend.target.update_instruction_properties("cz", pair, InstructionProperties(error=.02, duration=68e-9))
        with self.assertRaisesRegex(ValueError, "CZ criteria"):
            quality_path.validate(backend, choice["physical_path"], source, [0, 1], choice["settings"])
        with self.assertRaisesRegex(ValueError, "no qualifying"):
            quality_path.select(backend, source, [0, 1], min_t1=200e-6)

    def test_search_budget_is_reported_and_does_not_claim_optimality(self):
        backend = calibrated_backend()
        choice = quality_path.select(backend, fixtures.source(), [0, 1], avoid=(), max_expansions=5)
        self.assertFalse(choice["search_complete"])
        self.assertEqual(choice["candidate_paths_examined"], 1)


class ControlQualityTests(unittest.TestCase):
    def controls(self):
        return [{"control": name, "shots": 2048, "populations": values,
                 "physical_sites": [4, 5], "expected_populations": expected}
                for name, values, expected in (("all_zero", [0., 0.], [0., 0.]),
                    ("all_one", [1., 1.], [1., 1.]), ("initial_mixture", [.5, .5], [.5, .5]))]

    def test_perfect_controls_pass_and_prior_anomaly_fails(self):
        controls = self.controls()
        self.assertTrue(control_quality.assess_controls(controls)["passed"])
        controls[-1]["populations"][1] = .38
        assessment = control_quality.assess_controls(controls)
        self.assertFalse(assessment["passed"])
        self.assertEqual(assessment["failed_sites_1based"], [2])
        self.assertIn("initial mixture", assessment["sites"][1]["reasons"][0])
        controls = self.controls(); controls[1]["populations"][0] = .94
        self.assertFalse(control_quality.assess_controls(controls)["passed"])


class WorkflowTests(unittest.TestCase):
    def setup_options(self, folder):
        path = Path(folder)/"source.json"; path.write_text(json.dumps(fixtures.source()))
        return workflow.parse_arguments(["controls", "--source", str(path), "--trajectory-count", "2",
            "--selection-file", str(Path(folder)/"latest.json"), "--output-directory", str(Path(folder)/"runs")])

    def backend(self):
        return fixtures.HardwareRunnerTests().backend()

    def sampler(self, control=True):
        sampler = Mock(); job = Mock(); job.job_id.return_value = "controls_job" if control else "full_job"
        def result():
            pubs = sampler.run.call_args.args[0]
            if control:
                return [fixtures.pub({"000": 2048}), fixtures.pub({"111": 2048}),
                        fixtures.pub({"000": 1024, "111": 1024})]
            return [fixtures.pub({"001": 64}) for _ in pubs]
        job.result.side_effect = result; job.metrics.return_value = {}
        sampler.run.return_value = job
        return sampler

    def test_controls_only_builds_no_evolution_and_full_reuses_exact_path(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware, "make_service", side_effect=AssertionError("no real account")):
            opts = self.setup_options(folder); backend = self.backend()
            sampler = self.sampler()
            original = hardware.circuits.build_body
            with patch.object(hardware.circuits, "build_body", wraps=original) as build:
                controls = workflow.main(opts, backend=backend, sampler_factory=Mock(return_value=sampler))
                self.assertEqual(len(build.call_args_list), 3)
                self.assertTrue(all(call.kwargs.get("steps") == 0 for call in build.call_args_list))
            self.assertEqual(controls["configuration"]["total_shots"], 6144)
            self.assertEqual(controls["configuration"]["total_circuits"], 3)
            self.assertEqual(controls["configuration"]["trajectory_indices"], [])
            self.assertEqual(controls["configuration"]["preparation_steps"], 0)
            self.assertTrue(controls["control_assessment"]["passed"])
            control_path = controls["configuration"]["physical_path"]
            folder_path = Path(controls["run_directory"])
            with (folder_path/"isa_circuits.qpy").open("rb") as stream:
                compiled = qpy.load(stream)
            self.assertEqual(len(compiled), 3)
            self.assertTrue(all("cz" not in circuit.count_ops() for circuit in compiled))
            with zipfile.ZipFile(folder_path/"results_bundle.zip") as archive:
                self.assertIn("isa_circuits.qpy", archive.namelist())
                self.assertIn("ControlReport.md", archive.namelist())
            full_options = workflow.parse_arguments(["full", "--trajectory-count", "2",
                "--selection-file", str(opts.selection_file), "--output-directory", str(Path(folder)/"runs")])
            with patch.object(quality_path, "select", side_effect=AssertionError("must reuse exact path")):
                full = workflow.main(full_options, backend=backend, sampler_factory=Mock(return_value=self.sampler(False)))
            self.assertEqual(full["configuration"]["physical_path"], control_path)
            self.assertEqual(full["configuration"]["total_shots"], 128)
            self.assertEqual(full["configuration"]["total_circuits"], 2)
            self.assertEqual(full["configuration"]["path_selection"]["mode"], "reuse_control_path")

    def test_failed_saved_controls_block_full_before_account_or_submission(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware, "make_service", side_effect=AssertionError("no account")):
            opts = self.setup_options(folder)
            result = workflow.main(opts, backend=self.backend(), sampler_factory=Mock(return_value=self.sampler()))
            path = Path(result["run_directory"])/"manifest.json"
            manifest = json.loads(path.read_text())
            manifest["batches"][0]["observations"][2]["outcome_counts"] = {"000": 2048}
            path.write_text(json.dumps(manifest))
            # The cached pointer and hardware_results still say pass. Recompute raw counts.
            full = workflow.parse_arguments(["full", "--trajectory-count", "2", "--selection-file", str(opts.selection_file)])
            forbidden = Mock(side_effect=AssertionError("no submission"))
            with self.assertRaisesRegex(ValueError, "control screening failed"):
                workflow.main(full, sampler_factory=forbidden)
            forbidden.assert_not_called()

    def test_deteriorated_saved_path_blocks_full_without_reranking(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware, "make_service", side_effect=AssertionError("no account")):
            opts = self.setup_options(folder); backend = self.backend()
            result = workflow.main(opts, backend=backend, sampler_factory=Mock(return_value=self.sampler()))
            a, b = result["configuration"]["physical_path"][:2]
            for pair in ((a, b), (b, a)):
                backend.target.update_instruction_properties("cz", pair, InstructionProperties(error=.04, duration=68e-9))
            full = workflow.parse_arguments(["full", "--trajectory-count", "2", "--selection-file", str(opts.selection_file)])
            forbidden = Mock(side_effect=AssertionError("no submission"))
            with self.assertRaisesRegex(ValueError, "CZ criteria"):
                workflow.main(full, backend=backend, sampler_factory=forbidden)
            forbidden.assert_not_called()

    def test_interrupted_controls_resume_through_saved_pointer(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware, "make_service", side_effect=AssertionError("no account")):
            opts = self.setup_options(folder); backend = self.backend()
            sampler = self.sampler(); job = sampler.run.return_value
            job.result.side_effect = RuntimeError("collection interrupted")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                workflow.main(opts, backend=backend, sampler_factory=Mock(return_value=sampler))
            self.assertTrue(opts.selection_file.exists())
            pointer = json.loads(opts.selection_file.read_text())
            manifest = opts.selection_file.parent/pointer["manifest"]
            full_opts = workflow.parse_arguments(["full", "--trajectory-count", "2", "--selection-file", str(opts.selection_file)])
            with self.assertRaisesRegex(ValueError, "completely collected"):
                workflow.main(full_opts)
            job.result.side_effect = None
            job.result.return_value = [fixtures.pub({"000": 2048}), fixtures.pub({"111": 2048}),
                                       fixtures.pub({"000": 1024, "111": 1024})]
            service = Mock(); service.job.return_value = job
            forbidden = Mock(side_effect=AssertionError("must not resubmit controls"))
            recovered = hardware.main(hardware.parse_arguments(["--resume", str(manifest)]), backend=backend,
                                      service=service, sampler_factory=forbidden)
            forbidden.assert_not_called()
            self.assertTrue(recovered["control_assessment"]["passed"])
            # The pointer's cached pass is still null; raw-count revalidation is authoritative.
            self.assertIsNone(json.loads(opts.selection_file.read_text())["controls_passed"])
            loaded = workflow.load_controls(opts.selection_file)
            self.assertTrue(loaded[-1]["control_assessment"]["passed"])


if __name__ == "__main__":
    unittest.main()
