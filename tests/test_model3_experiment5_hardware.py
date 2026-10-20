"""Live-target compilation and mocked Runtime submission; no IBM account/job."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from qiskit import qpy
from qiskit.providers.fake_provider import GenericBackendV2

from Model3.Experiment5 import run_hardware as hardware
from Model3.Experiment5.export_hardware_input import export
from Model3.Experiment5.circuit_depth import assert_native


def source(n=3):
    return {"experiment": "Model3/Experiment7", "run_status": "passed",
        "configuration": {"physics": {"n_qubits": n, "J": 1.7, "gamma": .8, "exchange_prefactor": .5,
            "p_left": .9, "p_right": .1, "delta": 0.}, "trajectories": 2,
            "order": "hamiltonian-first", "sampling_policy": "rate-weighted", "system_formula": "even-odd"},
        "preparation": {"name": "half_mixed", "populations": [.5]*n},
        "times": [0., .5, .75],
        "schedule": {"step_dts": [.5, .25], "jump_choices": [[0, 3], [2, 1]],
                     "jump_probabilities": [.45, .05, .05, .45]},
        "stationary_observables": {"populations": [.5]*n, "excitation_flows": [0.]*(n-1)},
        "mean_channel_observables": {"populations": [[[.5]*n]], "excitation_flows": [[[0.]*(n-1)]]}}


def pub(counts):
    return SimpleNamespace(data=SimpleNamespace(outcome=SimpleNamespace(get_counts=lambda: counts),
        preparation=SimpleNamespace(get_counts=Mock(side_effect=AssertionError("must ignore preparation bits")))))


class HardwareRunnerTests(unittest.TestCase):
    def backend(self):
        return GenericBackendV2(6, basis_gates=["cz", "rz", "sx", "x"],
            coupling_map=[[i, i+1] for i in range(5)]+[[i+1, i] for i in range(5)], seed=7)

    def options(self, folder, *extra):
        p = Path(folder)/"input.json"
        if not p.exists():
            p.write_text(json.dumps(source()))
        return hardware.parse_arguments(["--source", str(p), "--trajectory-indices", "0", "1",
            "--shots", "8", "--output-directory", str(Path(folder)/"output"), *extra])

    def test_defaults_and_bad_requests(self):
        options = hardware.parse_arguments([])
        self.assertEqual(options.backend, "ibm_phoenix")
        self.assertEqual(options.shots, 8192)
        self.assertEqual(options.trajectory_indices, [0, 1, 2, 3])
        self.assertFalse(options.compile_only)
        for args in (("--shots", "0"), ("--batch-size", "0"), ("--backend", " "),
                     ("--trajectory-indices", "0", "0"), ("--seed-transpiler", "-1"),
                     ("--resume", "x.json", "--compile-only")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                hardware.parse_arguments(args)

    def test_portable_input_preserves_exact_paths_and_does_not_claim_subset_accuracy(self):
        original = source()
        original["assessment"] = {"sampling_budget_met": True}
        before = json.dumps(original, sort_keys=True)
        compact = export(original, [1, 0], "a"*64)
        self.assertEqual(json.dumps(original, sort_keys=True), before)
        hardware.require_validated_source(compact)
        self.assertEqual(compact["configuration"]["trajectories"], 2)
        self.assertEqual(compact["run_status"], "hardware_input")
        self.assertNotIn("sampling_budget_met", compact["assessment"])
        self.assertEqual(compact["provenance"]["saved_trajectory_indices"], [1, 0])
        self.assertEqual(compact["schedule"]["jump_choices"], original["schedule"]["jump_choices"][::-1])
        self.assertEqual(compact["mean_channel_observables"], original["mean_channel_observables"])
        for compact_index, original_index in enumerate([1, 0]):
            old = hardware.circuits.build_body(original, original_index)
            new = hardware.circuits.build_body(compact, compact_index)
            old.metadata = new.metadata = {}
            self.assertEqual(old, new)
        for indices in ([], [-1], [2], [0, 0]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                export(original, indices, "a"*64)
        original["run_status"] = "failed"
        with self.assertRaisesRegex(ValueError, "passing full replay"):
            export(original, [0], "a"*64)

    def test_default_input_is_small_self_contained_and_retains_8192_path_provenance(self):
        options = hardware.parse_arguments([])
        self.assertLess(options.source.stat().st_size, 256*1024)
        compact = json.loads(options.source.read_text())
        hardware.require_validated_source(compact)
        self.assertEqual(len(compact["schedule"]["step_dts"]), 72)
        self.assertEqual(compact["configuration"]["trajectories"], 4)
        self.assertEqual(compact["provenance"]["full_replay_trajectories"], 8192)
        self.assertEqual(len(compact["schedule"]["jump_choices"]), 4)
        self.assertEqual(len(compact["final_covariances_real_gauge"][0]), 4)

    def test_live_compilation_is_native_and_keeps_boundary_baths_and_preparation(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            backend = self.backend()
            publications, records, path = hardware.prepare(source(), backend, self.options(folder))
            self.assertEqual(len(publications), 6)
            self.assertEqual(len(path), 5)
            self.assertEqual([r["basis"] for r in records], ["z", "xy", "yx"]*2)
            for circuit in publications:
                assert_native(circuit, backend)
                self.assertEqual(circuit.count_ops()["reset"], 2)
                self.assertEqual(circuit.count_ops()["measure"], 6)
                self.assertEqual([r.name for r in circuit.cregs], ["preparation", "outcome"])
                self.assertEqual(len(circuit.metadata["physical_sites"]), 3)

    def test_compile_only_never_loads_account_or_submits_and_saves_reloadable_qpy(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware, "make_service", side_effect=AssertionError("no account")):
            backend = self.backend()
            result = hardware.main(self.options(folder, "--compile-only"), backend=backend,
                                   sampler_factory=Mock(side_effect=AssertionError("no sampler")))
            self.assertEqual(result["status"], "compiled")
            manifest = next(Path(folder).glob("output/*/manifest.json"))
            with (manifest.parent/"isa_circuits.qpy").open("rb") as handle:
                saved = qpy.load(handle)
            self.assertEqual(len(saved), 6)
            self.assertTrue(all(b["job_id"] is None for b in result["batches"]))
            self.assertFalse((manifest.parent/"hardware_results.json").exists())

    def test_register_bit_order_and_both_current_bond_parities(self):
        singles, pairs, shots = hardware.count_observables({"01": 2, "10": 6}, 2)
        np.testing.assert_allclose(singles, [.5, -.5])
        np.testing.assert_allclose(pairs, [-1.])
        self.assertEqual(shots, 8)
        records = [{"trajectory": 0, "basis": b} for b in ("z", "xy", "yx")]
        result = hardware.summarize(source(), records, [
            {"outcome_counts": {"000": 8}}, {"outcome_counts": {"001": 8}}, {"outcome_counts": {"100": 8}}])
        np.testing.assert_allclose(result["observables"]["populations"], [0., 0., 0.])
        np.testing.assert_allclose(result["observables"]["excitation_flows"], [1.7, 1.7])
        self.assertIsNone(result["across_path_standard_errors"]["populations"])

    def test_submission_saves_job_before_collection_and_ignores_preparation_counts(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            options = self.options(folder)
            service = Mock()
            backend = self.backend()
            def collect():
                manifest = json.loads(next(Path(folder).glob("output/*/manifest.json")).read_text())
                self.assertEqual(manifest["batches"][0]["job_id"], "test_job")
                return [pub({"000": 8}) for _ in range(6)]
            job = Mock(); job.job_id.return_value = "test_job"; job.result.side_effect = collect
            sampler = Mock(); sampler.run.return_value = job
            factory = Mock(return_value=sampler)
            result = hardware.main(options, service=service, backend=backend, sampler_factory=factory)
            self.assertEqual(result["job_ids"], ["test_job"])
            self.assertEqual(len(sampler.run.call_args.args[0]), 6)
            self.assertEqual(sampler.run.call_args.kwargs["shots"], 8)
            np.testing.assert_allclose(result["observables"]["populations"], [0., 0., 0.])
            manifest = next(Path(folder).glob("output/*/manifest.json"))
            self.assertTrue((manifest.parent/"hardware_results.json").exists())
            self.assertEqual(factory.call_args.kwargs["options"]["execution"], {"init_qubits": True})

    def test_resume_recovers_existing_job_without_resubmitting(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            backend = self.backend(); service = Mock()
            job = Mock(); job.job_id.return_value = "existing_job"
            job.result.side_effect = [RuntimeError("collection interrupted"), [pub({"000": 8}) for _ in range(6)]]
            sampler = Mock(); sampler.run.return_value = job
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                hardware.main(self.options(folder), service=service, backend=backend, sampler_factory=Mock(return_value=sampler))
            manifest = next(Path(folder).glob("output/*/manifest.json"))
            service.job.return_value = job
            factory = Mock(side_effect=AssertionError("must not submit another job"))
            result = hardware.main(hardware.parse_arguments(["--resume", str(manifest)]),
                service=service, backend=backend, sampler_factory=factory)
            service.job.assert_called_once_with("existing_job")
            self.assertEqual(result["job_ids"], ["existing_job"])
            factory.assert_not_called()
            # A fully collected manifest can be replayed without touching a job.
            service.reset_mock()
            hardware.main(hardware.parse_arguments(["--resume", str(manifest)]),
                service=service, backend=backend, sampler_factory=factory)
            service.job.assert_not_called()

    def test_ambiguous_submission_is_not_automatically_duplicated(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            backend = self.backend(); service = Mock(); sampler = Mock()
            sampler.run.side_effect = RuntimeError("network interrupted during submission")
            with self.assertRaises(RuntimeError):
                hardware.main(self.options(folder), service=service, backend=backend, sampler_factory=Mock(return_value=sampler))
            manifest = next(Path(folder).glob("output/*/manifest.json"))
            factory = Mock(side_effect=AssertionError("must not duplicate uncertain submission"))
            with self.assertRaisesRegex(ValueError, "interrupted before its job ID"):
                hardware.main(hardware.parse_arguments(["--resume", str(manifest)]),
                    service=service, backend=backend, sampler_factory=factory)
            factory.assert_not_called()

    def test_batches_are_collected_in_order_and_bad_shot_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            jobs = []
            for index in range(2):
                job = Mock(); job.job_id.return_value = f"job{index}"
                job.result.return_value = [pub({"000": 8}) for _ in range(3)]
                jobs.append(job)
            sampler = Mock(); sampler.run.side_effect = jobs
            result = hardware.main(self.options(folder, "--batch-size", "3"), service=Mock(),
                backend=self.backend(), sampler_factory=Mock(return_value=sampler))
            self.assertEqual(result["job_ids"], ["job0", "job1"])
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            job = Mock(); job.job_id.return_value = "short_shots"
            job.result.return_value = [pub({"000": 7}) for _ in range(6)]
            sampler = Mock(); sampler.run.return_value = job
            with self.assertRaisesRegex(ValueError, "shot count"):
                hardware.main(self.options(folder), service=Mock(), backend=self.backend(),
                    sampler_factory=Mock(return_value=sampler))

    def test_invalid_source_fails_before_account_access(self):
        with tempfile.TemporaryDirectory() as folder:
            options = self.options(folder)
            data = source(); data["run_status"] = "failed"
            options.source.write_text(json.dumps(data))
            with patch.object(hardware, "make_service", side_effect=AssertionError("no account")), \
                 self.assertRaisesRegex(ValueError, "passing saved replay"):
                hardware.main(options)

    def test_live_service_resolves_requested_backend_without_fractional_gates(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            service = Mock(); service.backend.return_value = self.backend()
            job = Mock(); job.job_id.return_value = "resolved_backend_job"
            job.result.return_value = [pub({"000": 8}) for _ in range(6)]
            sampler = Mock(); sampler.run.return_value = job
            hardware.main(self.options(folder, "--backend", "ibm_phoenix"),
                service=service, sampler_factory=Mock(return_value=sampler))
            service.backend.assert_called_once_with("ibm_phoenix", use_fractional_gates=False)

    def test_resume_rejects_changed_isa_circuits_before_service_access(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            hardware.main(self.options(folder, "--compile-only"), backend=self.backend())
            manifest = next(Path(folder).glob("output/*/manifest.json"))
            path = manifest.parent/"isa_circuits.qpy"
            path.write_bytes(path.read_bytes()+b"changed")
            with patch.object(hardware, "make_service", side_effect=AssertionError("no account")), \
                 self.assertRaisesRegex(ValueError, "ISA circuits changed"):
                hardware.main(hardware.parse_arguments(["--resume", str(manifest)]))


if __name__ == "__main__":
    unittest.main()
