import unittest
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_ibm_runtime.fake_provider import FakeFez

from Model1.Experiment9 import mitigated_randomized_lie_trotter as experiment9
import qem


class Experiment9ConfigurationTests(unittest.TestCase):
    def test_no_layout_is_available_without_changing_physics_or_mitigation(self):
        enabled = experiment9.parse_arguments(())
        disabled = experiment9.parse_arguments(("--no-layout",))
        self.assertFalse(enabled.no_layout)
        self.assertTrue(disabled.no_layout)
        self.assertFalse(disabled.no_circuit_plots)
        self.assertEqual(experiment9.output_paths(enabled), experiment9.output_paths(disabled))
        self.assertEqual(experiment9._prospective_signature(enabled),
                         experiment9._prospective_signature(disabled))

    def test_fake_backends_no_mitigation_resolve_locally_without_reading_credentials(self):
        for name in ("fake_kingston", "fake_miami"):
            with self.subTest(backend=name):
                options = experiment9.parse_arguments(("--backend", name, "--n-qubits", "10",
                    "--aer-method", "statevector", "--no-zne", "--fold-repetitions", "1",
                    "--no-pauli-twirling", "--no-dynamical-decoupling", "--no-readout-mitigation"))
                with patch.object(experiment9.experiment5.hardware_tools, "load_ibm_account",
                                  side_effect=AssertionError("must not read credentials")):
                    target, environment = experiment9.experiment5.hardware_tools._runtime_target(
                        options.backend, options.aer_method, options.account_file)
                self.assertEqual(target.backend_name, name)
                self.assertEqual(target.method, "statevector")
                self.assertIsNone(environment.ibm_account)
                self.assertEqual(options.zne_scale_factors, (1.0,))
                self.assertIn(name, experiment9.output_paths(options)["result"].name)
                with patch.object(experiment9.experiment8, "main") as driver:
                    experiment9.main(options)
                driver.assert_called_once_with(options)

    def test_progress_can_be_disabled_without_changing_execution_settings(self):
        enabled = experiment9.parse_arguments(())
        disabled = experiment9.parse_arguments(("--no-progress",))
        self.assertFalse(enabled.no_progress)
        self.assertTrue(disabled.no_progress)
        self.assertEqual(experiment9._settings_payload(enabled), experiment9._settings_payload(disabled))
        self.assertEqual(experiment9.output_paths(enabled), experiment9.output_paths(disabled))

    def test_all_mitigation_layers_are_enabled_by_default(self):
        options = experiment9.parse_arguments(())

        self.assertFalse(options.no_zne)
        self.assertFalse(options.no_pauli_twirling)
        self.assertFalse(options.no_dynamical_decoupling)
        self.assertFalse(options.no_readout_mitigation)

    def test_every_layer_can_be_disabled_independently(self):
        options = experiment9.parse_arguments(
            (
                "--no-zne",
                "--no-pauli-twirling",
                "--no-dynamical-decoupling",
                "--no-readout-mitigation",
            )
        )

        self.assertEqual(options.zne_scale_factors, (1.0,))
        self.assertTrue(options.no_pauli_twirling)
        self.assertTrue(options.no_dynamical_decoupling)
        self.assertTrue(options.no_readout_mitigation)
        self.assertIn(
            "_pauli_off_dd_off_readout_off_",
            experiment9.output_paths(options)["result"].name,
        )

    def test_timed_dd_rule_and_spacing_are_configurable(self):
        options = experiment9.parse_arguments(
            ("--dd-sequence", "xyxy", "--dd-spacing-fractions", "0.125", "0.25", "0.25", "0.25", "0.125")
        )

        self.assertEqual(options.dd_sequence, "xy4")
        self.assertEqual(options.dd_spacing_fractions, (0.125, 0.25, 0.25, 0.25, 0.125))
        self.assertIn(
            "_pauli_on_dd_xy4_readout_on_",
            experiment9.output_paths(options)["result"].name,
        )

    def test_obsolete_dd_spacing_is_not_silently_reinterpreted(self):
        with self.assertRaisesRegex(ValueError, "circuit-column"):
            experiment9.parse_arguments(("--dd-spacing", "2"))

    def test_main_rejects_aer_dd_without_timing_before_execution(self):
        options = experiment9.parse_arguments(("--backend", "aer"))
        with patch.object(experiment9.experiment8, "main", side_effect=AssertionError("submitted")):
            with self.assertRaisesRegex(ValueError, "no calibrated durations"):
                experiment9.main(options)

    def test_main_rejects_hardware_readout_without_calibration(self):
        options = experiment9.parse_arguments(("--backend", "ibm_kingston", "--no-dynamical-decoupling"))
        with patch.object(experiment9.experiment8, "main", side_effect=AssertionError("submitted")):
            with self.assertRaisesRegex(ValueError, "readout-calibration"):
                experiment9.main(options)

    def test_new_filenames_and_settings_separate_old_results(self):
        options = experiment9.parse_arguments(("--no-dynamical-decoupling",))
        self.assertTrue(experiment9.output_paths(options)["result"].name.startswith("qem_"))
        settings = experiment9._settings_payload(options)
        self.assertEqual(settings, json.loads(json.dumps(settings)))

    def test_calibration_file_validation_and_compatibility_metadata(self):
        payload = {"backend": "aer", "source": "synthetic calibration test", "blocks": [
            {"physical_qubits": [0], "assignment": [[0.9, 0.2], [0.1, 0.8]]},
        ]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            options = experiment9.parse_arguments(("--backend", "aer", "--readout-calibration", str(path)))
            self.assertEqual(experiment9._settings_payload(options)["readout_calibration"], payload)
            payload["backend"] = "different_backend"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                experiment9.parse_arguments(("--backend", "aer", "--readout-calibration", str(path)))

    def test_hooks_are_restored_after_driver_error(self):
        options = experiment9.parse_arguments(("--no-dynamical-decoupling",))
        original = experiment9.experiment8.output_paths
        with patch.object(experiment9.experiment8, "main", side_effect=RuntimeError("driver test")):
            with self.assertRaisesRegex(RuntimeError, "driver test"):
                experiment9.main(options)
        self.assertIs(experiment9.experiment8.output_paths, original)
        self.assertEqual(experiment9._ACTIVE_DIAGNOSTICS, ())


class PauliTwirlingTests(unittest.TestCase):
    def test_cz_and_cx_twirling_preserve_the_ideal_unitary(self):
        for gate_name in ("cz", "cx"):
            with self.subTest(gate=gate_name):
                circuit = QuantumCircuit(2)
                circuit.h(0)
                getattr(circuit, gate_name)(0, 1)
                circuit.rz(0.31, 1)
                getattr(circuit, gate_name)(0, 1)

                twirled, diagnostic = experiment9.pauli_twirl_native_circuit(
                    circuit,
                    seed=123,
                )

                self.assertTrue(Operator(circuit).equiv(Operator(twirled)))
                self.assertEqual(diagnostic["twirled"], ((gate_name, 2),))


class DynamicalDecouplingTests(unittest.TestCase):
    def test_fake_backends_n10_baseline_validate_reset_support_without_running(self):
        from IBMRuntime import FakeIBMBackend, RuntimeEnvironment
        from IBMRuntime.interpreters import qiskit as interpreter
        circuits, _, _ = experiment9.experiment7.build_sample_circuits((0.2,), 10,
            trajectories=1, trotter_delta_t=0.1, measurement_bases=("XY",))
        for name in ("fake_kingston", "fake_miami"):
            with self.subTest(backend=name):
                options = experiment9.parse_arguments(("--backend", name, "--n-qubits", "10",
                    "--no-zne", "--fold-repetitions", "1", "--no-pauli-twirling",
                    "--no-dynamical-decoupling", "--no-readout-mitigation"))
                resolved = interpreter._resolve_backend(FakeIBMBackend(name), RuntimeEnvironment())
                self.assertIsInstance(resolved, interpreter.Ok, resolved)
                backend = resolved.value
                compiled = interpreter._compile_circuits(backend, tuple(
                    experiment9.experiment7.prepare_circuit_for_compilation(circuit) for circuit in circuits),
                    experiment9.CompilerConfig(optimization_level=3, seed_transpiler=options.seed_transpiler))
                if name == "fake_miami":
                    # The bundled Miami snapshot lacks reset. Do not silently
                    # fabricate a target instruction or reset calibration.
                    self.assertNotIn("reset", backend.native.target.operation_names)
                    self.assertIsInstance(compiled, interpreter.Err, compiled)
                    self.assertEqual(compiled.error.stage, "compile")
                    self.assertIn("reset", compiled.error.message)
                    continue
                self.assertIsInstance(compiled, interpreter.Ok, compiled)
                native = compiled.value[0].native
                submitted, diagnostic = experiment9.transform_native_variant(native, backend.native.target,
                    experiment9.experiment8.FoldSpec(0, 1.0, 0, 17), 0, options)
                self.assertIs(submitted, native)
                self.assertFalse(diagnostic.dd_applied)
                self.assertEqual(native.count_ops()["reset"], 2)
                self.assertEqual(native.count_ops()["measure"], 10)
                self.assertEqual(native.num_qubits, backend.native.num_qubits)
                self.assertIsNotNone(native.layout)
                for item in native.data:
                    qargs = tuple(native.find_bit(bit).index for bit in item.qubits)
                    self.assertTrue(backend.native.target.instruction_supported(
                        operation_name=item.operation.name, qargs=qargs, parameters=item.operation.params))

    def test_synthetic_aer_timing_file_loads_and_is_labelled_as_assumed(self):
        path = Path(experiment9.__file__).with_name("aer_timings.json")
        options = experiment9.parse_arguments(("--backend", "aer", "--dd-durations", str(path)))
        data = options.dd_durations_data
        self.assertIn("Synthetic", data["source"])
        self.assertIn("NOT IBM", data["source"])
        names = [entry[0] for entry in data["instructions"]]
        self.assertEqual(len(names), len(set(names)))
        durations = experiment9._dd_durations(options)
        self.assertEqual(durations.get("rz", [0], unit="dt"), 0)
        self.assertEqual(durations.get("rx", [0], unit="dt"), 40)
        self.assertEqual(durations.get("rzz", [0, 1], unit="dt"), 300)
        self.assertEqual(durations.get("unitary", [0, 1], unit="dt"), 500)
        self.assertEqual(durations.get("reset", [0], unit="dt"), 1000)
        self.assertEqual(data, experiment9._settings_payload(options)["dd_durations"])

    def test_n10_native_mitigations_schedule_with_synthetic_timings_without_jobs(self):
        from IBMRuntime.interpreters.qiskit import _to_qiskit
        from qiskit_aer import AerSimulator
        path = Path(experiment9.__file__).with_name("aer_timings.json")
        options = experiment9.parse_arguments(("--n-qubits", "10", "--backend", "aer",
            "--dd-durations", str(path), "--no-progress"))
        backend = AerSimulator()
        cases = []
        for boundary in (experiment9.experiment5.LEFT_BOUNDARY, experiment9.experiment5.RIGHT_BOUNDARY):
            logical = experiment9.experiment7.build_one_step_circuit(10, 0.1, boundary)
            measured = experiment9.experiment5._append_terminal_measurements(logical, 10, "XY")
            cases.append((f"boundary_{boundary}", measured, 1))
        full, _, _ = experiment9.experiment7.build_sample_circuits((1.0,), 10,
            trajectories=1, trotter_delta_t=0.1, measurement_bases=("XY",))
        cases.append(("ten_substeps", full[0], 10))
        for name, measured, reset_count in cases:
            native = transpile(_to_qiskit(experiment9.experiment7.prepare_circuit_for_compilation(measured)),
                target=backend.target, optimization_level=3, seed_transpiler=options.seed_transpiler)
            for scale in (1.0, 1.5, 2.0):
                with self.subTest(circuit=name, scale=scale):
                    submitted, diagnostic = experiment9.transform_native_variant(native, backend.target,
                        experiment9.experiment8.FoldSpec(0, scale, 0, 17), 0, options)
                    self.assertTrue(diagnostic.dd_applied)
                    self.assertEqual(submitted.num_qubits, native.num_qubits)
                    self.assertEqual(submitted.count_ops()["measure"], 10)
                    self.assertEqual(submitted.count_ops()["reset"], reset_count)
                    self.assertGreater(submitted.count_ops().get("delay", 0), 0)

    def test_timed_xx_rule_is_inserted_on_active_qubits(self):
        from qiskit.transpiler import InstructionDurations
        circuit = QuantumCircuit(1, 1)
        circuit.x(0)
        circuit.delay(128, 0, "dt")
        circuit.measure(0, 0)
        durations = InstructionDurations([("x", None, 16), ("measure", None, 32)], dt=1e-9)

        padded, diagnostic = experiment9.apply_dynamical_decoupling(
            circuit,
            durations=durations,
        )

        self.assertTrue(diagnostic["applied"])
        self.assertEqual(diagnostic["inserted_x"], 2)
        self.assertGreater(diagnostic["inserted_delay"], 0)
        self.assertEqual(padded.count_ops().get("x", 0), 3)


class NoMitigationFastPathTests(unittest.TestCase):
    def test_native_fold_and_twirl_preserve_reset_channel_and_physical_width(self):
        from qiskit.quantum_info import SuperOp
        from qiskit_aer import AerSimulator
        options = experiment9.parse_arguments(("--backend", "aer", "--n-qubits", "2", "--no-dynamical-decoupling"))
        native = QuantumCircuit(2, 2)
        native.rx(0.2, 0)
        native.rzz(0.3, 0, 1)
        native.reset(1)
        native.rx(0.4, 1)
        native.measure((0, 1), (0, 1))
        transformed, diagnostic = experiment9.transform_native_variant(native, AerSimulator().target,
            experiment9.experiment8.FoldSpec(2, 3.0, 0, 17), 0, options)
        self.assertEqual(transformed.num_qubits, native.num_qubits)
        self.assertEqual(transformed.count_ops()["reset"], 1)
        self.assertEqual(transformed.count_ops()["measure"], 2)
        self.assertEqual(diagnostic.folded_gate_occurrences, 3)
        np.testing.assert_allclose(SuperOp(native.remove_final_measurements(inplace=False)).data,
            SuperOp(transformed.remove_final_measurements(inplace=False)).data, atol=1e-12)

    def test_all_disabled_submits_the_once_transpiled_native_circuit(self):
        backend = FakeFez()
        circuit = QuantumCircuit(2, 2)
        circuit.sx(0)
        circuit.cz(0, 1)
        circuit.measure((0, 1), (0, 1))
        native = transpile(
            circuit,
            backend,
            optimization_level=1,
            seed_transpiler=0,
        )
        options = SimpleNamespace(
            no_pauli_twirling=True,
            no_dynamical_decoupling=True,
            dd_sequence="xx",
            dd_spacing=-1,
            seed_twirling=314159,
            seed_transpiler=0,
            zne_scale_factors=(1.0,),
            fold_repetitions=1,
        )
        spec = experiment9.experiment8.FoldSpec(0, 1.0, 0, 17)

        with patch.object(experiment9.qem, "fold", side_effect=AssertionError("baseline transformed")):
            submitted, diagnostic = experiment9.transform_native_variant(
                native,
                backend.target,
                spec,
                base_index=0,
                options=options,
            )

        self.assertIs(submitted, native)
        self.assertEqual(diagnostic.inserted_operation_count, 0)
        self.assertEqual(diagnostic.actual_physical_scale_factor, 1.0)
        self.assertFalse(diagnostic.twirling_enabled)
        self.assertFalse(diagnostic.dd_enabled)


class ReadoutMitigationTests(unittest.TestCase):
    def test_local_symmetric_attenuation_is_inverted(self):
        # Raw <ZZ>=0.64 with p0=p1=0.1 has attenuation 0.8^2=0.64,
        # so the corrected expectation is one.
        counts = {"00": 82, "01": 9, "10": 9}

        value, variance = experiment9._corrected_pauli(
            counts,
            (0, 1),
            qem.ReadoutModel(tuple(qem.ReadoutBlock((bit,), ((0.9, 0.1), (0.1, 0.9))) for bit in range(2))),
            enabled=True,
        )

        self.assertAlmostEqual(value, 1.0)
        self.assertGreaterEqual(variance, 0.0)

    def test_actual_measurement_routing_not_initial_layout_selects_calibration(self):
        circuit = QuantumCircuit(5, 2)
        circuit.measure(4, 0)
        circuit.measure(1, 1)
        options = SimpleNamespace(no_readout_mitigation=False, backend="ibm_kingston",
            readout_calibration_data={"source": "measured", "blocks": [
                {"physical_qubits": [1], "assignment": [[0.8, 0.3], [0.2, 0.7]]},
                {"physical_qubits": [4], "assignment": [[0.9, 0.1], [0.1, 0.9]]},
            ]})
        model = experiment9._readout_model(circuit, options)
        self.assertEqual(model.blocks[0].bits, (1,))
        self.assertEqual(model.blocks[1].bits, (0,))

    def test_missing_physical_qubit_calibration_is_an_error(self):
        circuit = QuantumCircuit(3, 1)
        circuit.measure(2, 0)
        options = SimpleNamespace(no_readout_mitigation=False, backend="ibm_kingston",
            readout_calibration_data={"source": "measured", "blocks": [
                {"physical_qubits": [1], "assignment": [[0.9, 0.1], [0.1, 0.9]]},
            ]})
        with self.assertRaises(ValueError):
            experiment9._readout_model(circuit, options)

    def test_identity_aer_readout_and_diagnostic_roundtrip(self):
        options = experiment9.parse_arguments(("--no-zne", "--no-pauli-twirling", "--no-dynamical-decoupling"))
        circuit = QuantumCircuit(2, 2)
        circuit.measure((0, 1), (0, 1))
        _, diagnostic = experiment9.transform_native_variant(circuit, None,
            experiment9.experiment8.FoldSpec(0, 1.0, 0, 0), 0, options)
        from dataclasses import asdict
        rebuilt = experiment9._diagnostic_from_record(json.loads(json.dumps({"fold_diagnostic": asdict(diagnostic)})))
        self.assertEqual(rebuilt, diagnostic)
        model = qem.ReadoutModel(tuple(qem.ReadoutBlock(bits, assignment) for bits, assignment in rebuilt.readout_blocks))
        value, variance = experiment9._corrected_pauli({"00": 80, "01": 20}, (0,), model, enabled=True)
        self.assertAlmostEqual(value, 0.6)
        self.assertGreater(variance, 0)

    def test_identity_readout_observable_reconstruction_matches_experiment8(self):
        options = experiment9.parse_arguments(("--n-qubits", "2", "--trajectories", "1",
            "--no-zne", "--no-pauli-twirling", "--no-dynamical-decoupling"))
        circuit = QuantumCircuit(2, 2)
        circuit.measure((0, 1), (0, 1))
        diagnostics = tuple(experiment9.transform_native_variant(circuit, None,
            experiment9.experiment8.FoldSpec(0, 1.0, 0, 0), base, options)[1] for base in range(3))
        grouped = [[[[{"Z": {"00": 80, "01": 20}, "X": {"00": 50, "01": 50},
                         "XY": {"00": 75, "01": 25}}]]]]
        previous = experiment9._ACTIVE_DIAGNOSTICS
        try:
            experiment9._ACTIVE_DIAGNOSTICS = diagnostics
            measured, _ = experiment9._raw_observable_arrays(grouped, options, 1)
        finally:
            experiment9._ACTIVE_DIAGNOSTICS = previous
        expected, _ = experiment9.experiment8._raw_observable_arrays(grouped, options, 1)
        for actual, reference in zip(measured, expected):
            np.testing.assert_allclose(actual, reference)


if __name__ == "__main__":
    unittest.main()
