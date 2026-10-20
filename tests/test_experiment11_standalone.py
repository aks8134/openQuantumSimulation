import ast
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from qiskit.quantum_info import Operator, SuperOp
from qiskit_aer import AerSimulator
from qiskit.circuit.library import XXPlusYYGate

from IBMRuntime import Err, ExecutionFailure, Ok, SampleResult, XXPlusYY, Reset, RZ
from IBMRuntime.interpreters.qiskit import _to_qiskit
from Model1.Experiment11 import physics, mitigation, analysis
from Model1.Experiment11 import mitigated_randomized_lie_trotter as experiment


OFF = ("--no-zne", "--no-pauli-twirling", "--no-dynamical-decoupling", "--no-readout-mitigation", "--no-progress")


class PhysicsTests(unittest.TestCase):
    def test_source_imports_no_other_experiments(self):
        directory = Path(experiment.__file__).parent
        for path in directory.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
                for name in names:
                    if name.startswith("Model1.Experiment"):
                        self.assertTrue(name.startswith("Model1.Experiment11"), (path, name))

    def test_default_compact_and_explicit_circuits_match_previous_strategy(self):
        # Old experiments are imported ONLY by this compatibility test.
        from Model1.Experiment5 import randomized_lie_trotter as old
        from Model1.Experiment7 import randomized_lie_trotter as explicit
        times = (0, 0.2, 0.4)
        dts, counts, choices = physics.schedule(times, 2, 29, 0.1)
        previous, metadata, _ = old.build_sample_circuits(times, 4, trajectories=2,
            seed_trajectories=29, trotter_delta_t=0.1, measurement_bases=physics.BASES)
        for prior, (ti, r, basis) in zip(previous, metadata):
            current = physics.build_circuit(physics.Model(), dts[:counts[ti]], choices[r][:counts[ti]], basis)
            self.assertEqual(current.operations, prior.operations)
            self.assertEqual(physics.prepare_for_compilation(current).operations,
                             explicit.prepare_circuit_for_compilation(prior).operations)

    def test_remainder_and_no_extra_baseline(self):
        dts, counts, choices = physics.schedule((0.7,), 4, 29, 0.2)
        np.testing.assert_allclose(dts, (0.2, 0.2, 0.2, 0.1), atol=1e-15)
        self.assertEqual(counts, (4,))
        self.assertEqual(len(choices), 4)
        self.assertEqual(len(choices[0]), 4)

    def test_parameters_and_boundary_local_ancillas(self):
        model = physics.Model(4, J=2.0, h=-0.3, gamma=0.8, initial_site=0)
        for boundary in (0, 1):
            circuit = physics.one_step(model, 0.1, boundary)
            gates = [op for op in circuit.operations if isinstance(op, XXPlusYY)]
            self.assertEqual(len(gates), 4)
            self.assertTrue(all(np.isclose(g.angle, 0.4) for g in gates[:-1]))
            self.assertAlmostEqual(gates[-1].angle, 2 * np.sqrt(2 * 0.8 * 0.1))
            self.assertEqual(gates[-1].second, 5 if boundary == 0 else 0)
            self.assertEqual(circuit.operations[-1], Reset(gates[-1].second))
            fields = [op for op in circuit.operations if isinstance(op, RZ)]
            self.assertEqual(len(fields), 4)
            self.assertTrue(all(np.isclose(op.angle, -0.03) for op in fields))

    def test_explicit_exchange_exact_up_to_known_global_phase(self):
        from IBMRuntime import Circuit
        for angle in (-0.7, 0.0, 0.4, 1.8):
            circuit = Circuit(2, 0, physics.explicit_exchange(angle, 0, 1))
            np.testing.assert_allclose(Operator(_to_qiskit(circuit)).data,
                                       -Operator(XXPlusYYGate(angle, 0)).data, atol=1e-12)

    def test_optimized_reference_matches_full_for_tuned_parameters(self):
        for n in (2, 3, 4):
            model = physics.Model(n, J=0.8, h=0.6, gamma=1.7, initial_site=0)
            reduced = physics.exact_reference(model, (0, 0.13, 0.7))
            full = physics.exact_reference(model, (0, 0.13, 0.7), full=True)
            for first, second in zip(reduced, full):
                np.testing.assert_allclose(first, second, atol=1e-12)


class ConfigurationTests(unittest.TestCase):
    def test_physical_parameters_have_distinct_artifacts_and_guard_resume(self):
        initial = experiment.parse_arguments((*OFF,))
        first = experiment.configuration(initial)
        for key, value in (("--J", "2"), ("--h", "0.3"), ("--gamma", "5"), ("--initial-site", "0")):
            other = experiment.parse_arguments((*OFF, key, value))
            cfg = experiment.configuration(other)
            self.assertNotEqual(first, cfg)
            self.assertNotEqual(experiment.output_paths(initial, first)["result"],
                                experiment.output_paths(other, cfg)["result"])
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                experiment.validate_archive({"configuration": first}, cfg)

    def test_plot_and_batch_choices_do_not_change_physics_signature(self):
        first = experiment.parse_arguments((*OFF,))
        second = experiment.parse_arguments((*OFF, "--no-layout", "--no-circuit-plots", "--batch-size", "6"))
        self.assertEqual(experiment.configuration(first), experiment.configuration(second))

    def test_fake_backends_never_read_account(self):
        with patch.object(experiment, "load_ibm_account", side_effect=AssertionError("must not authenticate")):
            for name in ("aer", "fake_fez", "fake_kingston", "fake_miami"):
                _, environment = experiment.runtime_target(name, "automatic", Path("unused"))
                self.assertIsNone(environment.ibm_account)

    def test_invalid_inputs(self):
        for extra in (("--gamma", "-1"), ("--J", "nan"), ("--times", "0.2", "0.1"),
                      ("--trotter-delta-t", "0"), ("--trajectories", "0"),
                      ("--shots", "3"), ("--layout-only", "--no-layout")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                experiment.parse_arguments((*OFF, *extra))


class MitigationAndAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.options = experiment.parse_arguments((*OFF, "--n-qubits", "3", "--J", "2"))
        self.config = experiment.configuration(self.options)

    def test_no_mitigation_native_identity_and_fold_channel_preservation(self):
        from qiskit import transpile
        circuit = physics.build_circuit(self.options.model, (0.1,), (0,), "Z")
        target = AerSimulator().target
        native = transpile(_to_qiskit(physics.prepare_for_compilation(circuit)), target=target, optimization_level=1)
        same, _ = mitigation.transform(native, target, self.config, base_index=0, scale_index=0, repetition=0)
        self.assertIs(same, native)
        cfg = {**self.config, "no_zne": False, "zne_scale_factors": [1.0, 3.0]}
        folded, diagnostic = mitigation.transform(native, target, cfg, base_index=0, scale_index=1, repetition=0)
        np.testing.assert_allclose(SuperOp(native.remove_final_measurements(inplace=False)).data,
                                   SuperOp(folded.remove_final_measurements(inplace=False)).data, atol=1e-11)
        self.assertEqual(folded.count_ops()["reset"], native.count_ops()["reset"])
        self.assertGreater(diagnostic["inserted_physical_gate_count"], 0)

    def test_three_basis_flow_uses_tuned_J_and_alternating_signs(self):
        cfg = {**self.config, "trajectories": 1}
        rows = [dict(time=0.1, trajectory_index=0, basis=basis, scale_index=0,
                     fold_repetition=0, counts={"000": 100}, diagnostic={"readout_model": None})
                for basis in physics.BASES]
        output = analysis.reconstruct(rows, self.options.model, cfg)
        np.testing.assert_allclose(output["observables"]["populations"], 0)
        np.testing.assert_allclose(output["observables"]["xy_correlations"], 2)
        np.testing.assert_allclose(output["observables"]["excitation_flows"], [[-2], [2]])

    def test_readout_mapping_uses_actual_terminal_measurement_wires(self):
        from qiskit import QuantumCircuit
        circuit = QuantumCircuit(4, 2)
        circuit.measure(3, 0)
        circuit.measure(1, 1)
        cfg = {**self.config, "no_readout_mitigation": False, "backend": "test",
            "readout_calibration": {"source": "synthetic test", "blocks": [
                {"physical_qubits": [1], "assignment": [[0.9, 0.1], [0.1, 0.9]]},
                {"physical_qubits": [3], "assignment": [[0.8, 0.2], [0.2, 0.8]]}]}}
        model = mitigation._readout_model(circuit, cfg)
        self.assertEqual(model.blocks[0].bits, (1,))
        self.assertEqual(model.blocks[1].bits, (0,))
        with self.assertRaises(ValueError):
            mitigation._readout_model(circuit, {**cfg, "readout_calibration": None})

    def test_combined_folding_twirling_dd_and_readout_preserve_ideal_channel(self):
        from qiskit import QuantumCircuit
        circuit = QuantumCircuit(2, 1)
        circuit.x(0)
        circuit.rzz(0.3, 0, 1)
        circuit.rx(0.2, 1)
        circuit.reset(1)
        circuit.rx(-0.1, 0)
        circuit.measure(0, 0)
        durations = json.loads((Path(experiment.__file__).parent / "aer_timings.json").read_text())
        cfg = {**self.config, "no_zne": False, "zne_scale_factors": [1.0, 3.0],
            "no_pauli_twirling": False, "no_dynamical_decoupling": False,
            "no_readout_mitigation": False, "dd_sequence": "xy4", "dd_durations": durations}
        transformed, diagnostic = mitigation.transform(circuit, AerSimulator().target, cfg,
            base_index=0, scale_index=1, repetition=0)
        first = circuit.remove_final_measurements(inplace=False)
        second = transformed.remove_final_measurements(inplace=False)
        np.testing.assert_allclose(SuperOp(first).data, SuperOp(second).data, atol=1e-11)
        self.assertGreater(diagnostic["inserted_physical_gate_count"], 0)
        self.assertIsNotNone(diagnostic["twirling"])
        self.assertEqual(diagnostic["dd"]["sequence"], "xy4")
        self.assertEqual(diagnostic["readout_model"]["blocks"][0]["bits"], (0,))

    def test_append_and_replace_times(self):
        def rows(t, value):
            return [dict(time=t, trajectory_index=0, basis=b, scale_index=0, fold_repetition=0,
                         counts={"0": value}) for b in physics.BASES]
        old = rows(0.0, 10) + rows(0.2, 20)
        merged = experiment.merge_records(old, rows(0.2, 30) + rows(0.4, 40))
        self.assertEqual(sorted({row["time"] for row in merged}), [0.0, 0.2, 0.4])
        self.assertEqual(len(merged), 9)
        self.assertTrue(all(row["counts"] == {"0": 30} for row in merged if row["time"] == 0.2))


class CheckpointTests(unittest.TestCase):
    def options(self, directory, *extra):
        return experiment.parse_arguments((*OFF, "--n-qubits", "2", "--trajectories", "1",
            "--shots", "32", "--times", "0.1", "--batch-size", "1", "--no-plots",
            "--output-directory", directory, *extra))

    @staticmethod
    def fake_provider(circuits, target, compiler, workload, environment, transform, **_kwargs):
        # Compile/transform locally, but never execute a simulator/hardware job.
        from qiskit import transpile
        backend = AerSimulator()
        results = []
        for compact in circuits:
            native = transpile(_to_qiskit(compact), target=backend.target,
                               optimization_level=compiler.optimization_level, seed_transpiler=compiler.seed_transpiler)
            for submitted in transform(native, backend.target):
                results.append(SampleResult(counts=(("00", workload.shots),), shots=workload.shots,
                    backend_name="test", job_id="mock-job", original_gate_count=len(compact.operations),
                    original_depth=native.depth(), compiled_gate_count=submitted.size(), compiled_depth=submitted.depth()))
        return Ok(tuple(results))

    def test_completed_batches_survive_failure_and_resume_missing_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            options = self.options(directory)
            cfg = experiment.configuration(options)
            paths = experiment.output_paths(options, cfg)
            calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    return Err(ExecutionFailure("collect", "test interruption"))
                return self.fake_provider(*args, **kwargs)
            with patch.object(experiment, "run_sample_variants_batch_sync", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "test interruption"):
                    experiment.execute(options, cfg, paths)
            saved = json.loads(paths["checkpoint"].read_text())
            self.assertEqual(saved["completed_variant_count"], 1)
            with self.assertRaisesRegex(ValueError, "incomplete checkpoint"):
                experiment.execute(options, cfg, paths)
            options.resume = True
            with patch.object(experiment, "run_sample_variants_batch_sync", side_effect=self.fake_provider) as provider:
                records, _, submitted = experiment.execute(options, cfg, paths)
            self.assertEqual(provider.call_count, 2)
            self.assertEqual(len(records), 3)
            self.assertEqual(submitted, 2)
            self.assertTrue(json.loads(paths["checkpoint"].read_text())["simulation_complete"])
            with patch.object(experiment, "run_sample_variants_batch_sync", side_effect=AssertionError("must not resample")):
                _, _, submitted = experiment.execute(options, cfg, paths)
            self.assertEqual(submitted, 0)

    def test_data_is_saved_before_figures_and_can_be_replotted_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            options = self.options(directory)
            options.no_plots = False
            options.no_layout = options.no_circuit_plots = True
            with patch.object(experiment, "run_sample_variants_batch_sync", side_effect=self.fake_provider), \
                 patch.object(experiment, "plot_figures", side_effect=RuntimeError("plot failed")):
                with self.assertRaisesRegex(RuntimeError, "plot failed"):
                    experiment.main(options)
            path = experiment.output_paths(options, experiment.configuration(options))["result"]
            payload = json.loads(path.read_text())
            self.assertEqual(len(payload["records"]), 3)
            self.assertEqual(payload["figure_generation_status"], "pending")
            recovery = experiment.parse_arguments(("--plot-only", str(path), "--no-layout", "--no-circuit-plots"))
            with patch.object(experiment, "run_sample_variants_batch_sync", side_effect=AssertionError("must not sample")):
                recovered = experiment.main(recovery)
            self.assertEqual(recovered["figure_generation_status"], "complete")
            self.assertTrue((Path(directory) / "figures").is_dir())


if __name__ == "__main__":
    unittest.main()
