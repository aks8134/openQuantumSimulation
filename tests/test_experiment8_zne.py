import json
import unittest
from types import SimpleNamespace

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez

from IBMRuntime import SampleResult
from Model1.Experiment7 import randomized_lie_trotter as experiment7
from Model1.Experiment8 import zne_randomized_lie_trotter as experiment8


def sample_result(counts, shots=100):
    return SampleResult(
        counts=tuple(counts.items()),
        shots=shots,
        backend_name="test",
        job_id="test-job",
        original_gate_count=3,
        original_depth=2,
        compiled_gate_count=7,
        compiled_depth=5,
    )


class Experiment8ConfigurationTests(unittest.TestCase):
    def test_defaults_extend_experiment7_working_point(self):
        options = experiment8.parse_arguments(())

        self.assertEqual(options.trajectories, 4)
        self.assertEqual(options.trotter_delta_t, 0.1)
        self.assertEqual(options.zne_scale_factors, (1.0, 1.5, 2.0))
        self.assertEqual(options.zne_inference, "linear")
        self.assertEqual(options.fold_repetitions, 1)
        self.assertEqual(experiment8.MEASUREMENT_BASES, ("Z", "X", "XY"))

    def test_only_three_measurement_circuits_are_built_per_trajectory_time(self):
        circuits, metadata, _ = experiment7.build_sample_circuits(
            np.asarray((0.0, 0.1)),
            number_of_qubits=3,
            trajectories=2,
            seed_trajectories=29,
            trotter_delta_t=0.1,
            measurement_bases=experiment8.MEASUREMENT_BASES,
        )

        self.assertEqual(len(circuits), 2 * 2 * 3)
        self.assertEqual(len(metadata), len(circuits))
        self.assertEqual(
            {basis for _, _, basis in metadata},
            set(experiment8.MEASUREMENT_BASES),
        )

    def test_invalid_scales_and_incompatible_references_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "first ZNE scale"):
            experiment8.parse_arguments(
                ("--zne-scale-factors", "1.5", "2.0")
            )
        with self.assertRaisesRegex(ValueError, "choose either"):
            experiment8.parse_arguments(
                (
                    "--classical-reference",
                    "--optimized-classical-reference",
                )
            )

    def test_shots_are_split_over_trajectories_and_fold_repetitions(self):
        options = experiment8.parse_arguments(
            (
                "--trajectories",
                "4",
                "--fold-repetitions",
                "2",
                "--shots",
                "80",
            )
        )
        self.assertEqual(experiment8.shots_per_variant(options), 10)

        options.shots = 81
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            experiment8.shots_per_variant(options)

    def test_no_zne_uses_one_unfurled_variant_and_all_scale_one_shots(self):
        options = experiment8.parse_arguments(
            (
                "--no-zne",
                "--trajectories",
                "4",
                "--shots",
                "80",
            )
        )
        base_metadata = tuple(
            (0, 0, basis) for basis in experiment8.MEASUREMENT_BASES
        )
        metadata = experiment8.expand_metadata(base_metadata, options)

        self.assertTrue(options.no_zne)
        self.assertEqual(options.zne_scale_factors, (1.0,))
        self.assertEqual(options.fold_repetitions, 1)
        self.assertEqual(experiment8._variant_count_per_base(options), 1)
        self.assertEqual(experiment8.shots_per_variant(options), 20)
        self.assertEqual(len(metadata), len(base_metadata))
        self.assertTrue(all(item.scale_factor == 1.0 for item in metadata))
        self.assertIn("no_zne_", experiment8.output_paths(options)["result"].name)
        configuration = experiment8._checkpoint_configuration(
            options,
            np.asarray((0.0,)),
            metadata,
        )
        self.assertFalse(configuration["zne"]["enabled"])
        self.assertIsNone(configuration["zne"]["inference"])
        self.assertFalse(configuration["folding"]["enabled"])

    def test_no_zne_rejects_multiple_fold_repetitions(self):
        with self.assertRaisesRegex(ValueError, "requires --fold-repetitions 1"):
            experiment8.parse_arguments(
                ("--no-zne", "--fold-repetitions", "2")
            )

    def test_checkpoint_configuration_survives_json_round_trip(self):
        options = experiment8.parse_arguments(
            (
                "--n-qubits",
                "2",
                "--trajectories",
                "1",
                "--fold-repetitions",
                "1",
                "--shots",
                "100",
                "--times",
                "0.1",
                "--trotter-delta-t",
                "0.1",
                "--zne-scale-factors",
                "1",
                "3",
            )
        )
        base_metadata = tuple(
            (0, 0, basis) for basis in experiment8.MEASUREMENT_BASES
        )
        metadata = experiment8.expand_metadata(base_metadata, options)
        configuration = experiment8._checkpoint_configuration(
            options,
            np.asarray((0.1,)),
            metadata,
        )

        self.assertEqual(json.loads(json.dumps(configuration)), configuration)

    def test_fold_specs_can_select_only_one_missing_scale(self):
        options = experiment8.parse_arguments(
            (
                "--zne-scale-factors",
                "1",
                "1.25",
                "1.5",
                "--fold-repetitions",
                "2",
            )
        )

        specs = experiment8.fold_specs(options, base_index=3, scale_indices=(1,))

        self.assertEqual(len(specs), 2)
        self.assertEqual({item.scale_index for item in specs}, {1})
        self.assertEqual({item.scale_factor for item in specs}, {1.25})
        self.assertEqual({item.repetition for item in specs}, {0, 1})

    def test_reused_and_executed_variants_are_restored_to_target_order(self):
        metadata = (
            experiment8.VariantMetadata(0, 0, 0, "Z", 0, 1.0, 0, 11),
            experiment8.VariantMetadata(0, 0, 0, "Z", 1, 1.25, 0, 12),
            experiment8.VariantMetadata(0, 0, 0, "Z", 2, 1.5, 0, 13),
        )
        reused = {
            0: ("result-1", "diagnostic-1"),
            2: ("result-1.5", "diagnostic-1.5"),
        }

        results, diagnostics = experiment8.combine_reused_and_executed_variants(
            metadata,
            reused,
            (metadata[1],),
            ("result-1.25",),
            ("diagnostic-1.25",),
        )

        self.assertEqual(
            results,
            ("result-1", "result-1.25", "result-1.5"),
        )
        self.assertEqual(
            diagnostics,
            ("diagnostic-1", "diagnostic-1.25", "diagnostic-1.5"),
        )


class PhysicalLocalFoldingTests(unittest.TestCase):
    def setUp(self):
        self.backend = AerSimulator()

    def test_scale_three_folds_rx_and_rzz_but_not_virtual_rz(self):
        circuit = QuantumCircuit(2)
        circuit.rz(0.2, 0)
        circuit.rx(0.3, 0)
        circuit.rzz(0.4, 0, 1)

        folded, diagnostic = experiment8.fold_native_circuit(
            circuit,
            self.backend.target,
            experiment8.FoldSpec(1, 3.0, 0, 17),
        )

        self.assertEqual(folded.count_ops(), {"rz": 1, "rx": 3, "rzz": 3})
        self.assertEqual(diagnostic.eligible_gate_counts, (("rx", 1), ("rzz", 1)))
        self.assertEqual(diagnostic.virtual_rz_count, 1)
        self.assertEqual(diagnostic.actual_physical_scale_factor, 3.0)
        self.assertTrue(Operator(folded).equiv(Operator(circuit)))

    def test_reset_and_measurement_are_retained_once(self):
        circuit = QuantumCircuit(2, 2)
        circuit.rx(0.3, 0)
        circuit.reset(1)
        circuit.measure((0, 1), (0, 1))

        folded, diagnostic = experiment8.fold_native_circuit(
            circuit,
            self.backend.target,
            experiment8.FoldSpec(1, 3.0, 0, 23),
        )

        self.assertEqual(folded.count_ops()["rx"], 3)
        self.assertEqual(folded.count_ops()["reset"], 1)
        self.assertEqual(folded.count_ops()["measure"], 2)
        self.assertEqual(diagnostic.reset_count, 1)
        self.assertEqual(diagnostic.measurement_count, 2)

    def test_ibm_sx_inverse_uses_one_physical_sx_and_virtual_rz(self):
        circuit = QuantumCircuit(1)
        circuit.sx(0)

        folded, diagnostic = experiment8.fold_native_circuit(
            circuit,
            FakeFez().target,
            experiment8.FoldSpec(1, 3.0, 0, 41),
        )

        self.assertEqual(folded.count_ops(), {"sx": 3, "rz": 2})
        self.assertEqual(diagnostic.inserted_operation_count, 4)
        self.assertEqual(diagnostic.inserted_physical_gate_count, 2)
        self.assertEqual(diagnostic.inserted_virtual_gate_count, 2)
        self.assertEqual(diagnostic.actual_physical_scale_factor, 3.0)
        self.assertTrue(Operator(folded).equiv(Operator(circuit)))


class ZNEObservableTests(unittest.TestCase):
    def test_constant_scale_data_extrapolates_to_the_same_observables(self):
        options = SimpleNamespace(
            n_qubits=2,
            trajectories=1,
            fold_repetitions=1,
            zne_scale_factors=(1.0, 3.0),
            zne_inference="linear",
            zne_polynomial_order=2,
            seed_folding=31,
        )
        base_metadata = tuple(
            (0, 0, basis) for basis in experiment8.MEASUREMENT_BASES
        )
        metadata = experiment8.expand_metadata(base_metadata, options)
        uniform = {"00": 25, "01": 25, "10": 25, "11": 25}
        counts_by_basis = {
            "Z": {"01": 50, "10": 50},
            "X": uniform,
            "XY": uniform,
        }
        results = tuple(
            sample_result(counts_by_basis[item.basis]) for item in metadata
        )

        analysis = experiment8.calculate_zne_observables(
            results,
            metadata,
            options,
            time_count=1,
        )

        populations, correlations, flows = analysis["measured"]
        np.testing.assert_allclose(populations, 0.5)
        np.testing.assert_allclose(correlations, 0.0)
        np.testing.assert_allclose(flows, 0.0)
        np.testing.assert_allclose(analysis["weights"], (1.5, -0.5))

    def test_no_zne_returns_scale_one_observables_without_extrapolation(self):
        options = SimpleNamespace(
            no_zne=True,
            n_qubits=2,
            trajectories=1,
            fold_repetitions=1,
            zne_scale_factors=(1.0,),
            seed_folding=31,
        )
        base_metadata = tuple(
            (0, 0, basis) for basis in experiment8.MEASUREMENT_BASES
        )
        metadata = experiment8.expand_metadata(base_metadata, options)
        uniform = {"00": 25, "01": 25, "10": 25, "11": 25}
        counts_by_basis = {
            "Z": {"01": 50, "10": 50},
            "X": uniform,
            "XY": uniform,
        }
        results = tuple(
            sample_result(counts_by_basis[item.basis]) for item in metadata
        )

        analysis = experiment8.calculate_zne_observables(
            results,
            metadata,
            options,
            time_count=1,
        )

        populations, correlations, flows = analysis["measured"]
        np.testing.assert_allclose(populations, 0.5)
        np.testing.assert_allclose(correlations, 0.0)
        np.testing.assert_allclose(flows, 0.0)
        np.testing.assert_allclose(analysis["weights"], (1.0,))

    def test_alternating_basis_reconstructs_even_and_odd_flow_signs(self):
        options = SimpleNamespace(
            n_qubits=3,
            trajectories=1,
            fold_repetitions=1,
            zne_scale_factors=(1.0,),
        )
        grouped = [[[[{
            "Z": {"000": 100},
            "X": {"000": 100},
            "XY": {"000": 100},
        }]]]]

        values, variances = experiment8._raw_observable_arrays(
            grouped,
            options,
            time_count=1,
        )

        np.testing.assert_allclose(values[1][:, 0, 0, 0, 0], (2.0, 2.0))
        np.testing.assert_allclose(
            values[2][:, 0, 0, 0, 0],
            (-experiment8.experiment5.J, experiment8.experiment5.J),
        )
        np.testing.assert_allclose(variances[1], 0.0)
        np.testing.assert_allclose(variances[2], 0.0)


if __name__ == "__main__":
    unittest.main()
