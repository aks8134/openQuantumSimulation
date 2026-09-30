import ast
import unittest
from dataclasses import FrozenInstanceError
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

from mitiq_qem import (
    MitiqUnavailable,
    combine_ddd,
    combine_pauli_twirling,
    combine_zne,
    construct_ddd,
    construct_pauli_twirling,
    construct_zne,
    ddd_plan,
    diagonal_expectation,
    inference,
    mitigate_counts,
    pauli_twirling_plan,
    scaling,
    uncorrelated_readout_plan,
    zne_plan,
)
from mitiq_qem.qiskit import scale_unitary_regions, validate_mitiq_circuit


class CoreAndZNETests(unittest.TestCase):
    def test_zne_construction_is_reproducible_and_immutable(self):
        plan = zne_plan(
            (1.0, 2.0, 3.0),
            repetitions=2,
            scaling=scaling("random_local"),
            inference=inference("richardson"),
            seed=10,
        )
        batch = construct_zne(
            "circuit",
            plan,
            scale_noise=lambda circuit, factor, seed: (
                circuit,
                factor,
                seed,
            ),
        )

        self.assertEqual(len(batch), 6)
        self.assertEqual(
            batch.circuits,
            (
                ("circuit", 1.0, 10),
                ("circuit", 1.0, 11),
                ("circuit", 2.0, 12),
                ("circuit", 2.0, 13),
                ("circuit", 3.0, 14),
                ("circuit", 3.0, 15),
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            plan.repetitions = 3

    def test_richardson_recovers_quadratic_zero_noise_value(self):
        plan = zne_plan((1.0, 2.0, 3.0), inference=inference("richardson"))
        values = tuple(map(lambda factor: 0.8 + 0.2 * factor + 0.05 * factor**2, plan.scale_factors))

        result = combine_zne(plan, values)

        self.assertAlmostEqual(result.estimate, 0.8)
        self.assertAlmostEqual(result.residual_rms, 0.0, places=12)

    def test_linear_repetitions_report_standard_errors(self):
        plan = zne_plan(
            (1.0, 3.0),
            repetitions=2,
            inference=inference("linear"),
        )
        result = combine_zne(plan, (0.91, 0.89, 0.71, 0.69))

        self.assertAlmostEqual(result.estimate, 1.0)
        self.assertTrue(all(error > 0.0 for error in result.scale_standard_errors))


class EnsembleConstructionTests(unittest.TestCase):
    def test_twirling_and_ddd_compose_as_immutable_batches(self):
        twirled = construct_pauli_twirling(
            "base",
            pauli_twirling_plan(variants=2, seed=4),
            twirl=lambda circuit, count, seed: tuple(
                map(lambda index: f"{circuit}-pt{index}-s{seed}", range(count))
            ),
        )
        decoupled = construct_ddd(
            twirled,
            ddd_plan("xyxy", trials=2),
            insert=lambda circuit, rule, spacing, trials: tuple(
                map(lambda index: f"{circuit}-{rule}{index}", range(trials))
            ),
        )

        self.assertEqual(len(twirled), 2)
        self.assertEqual(len(decoupled), 4)
        self.assertTrue(
            all(
                tuple(stage.technique for stage in variant.stages)
                == ("pauli_twirling", "ddd")
                for variant in decoupled.variants
            )
        )
        self.assertAlmostEqual(combine_pauli_twirling((0.8, 1.0)).estimate, 0.9)
        self.assertAlmostEqual(combine_ddd((0.7, 0.9)).estimate, 0.8)


class ReadoutTests(unittest.TestCase):
    def test_uncorrelated_confusion_inversion_recovers_distribution(self):
        plan = uncorrelated_readout_plan(2, p0=0.25, p1=0.25)
        result = mitigate_counts(
            plan,
            {"00": 625, "01": 1875, "10": 1875, "11": 5625},
        )

        self.assertTrue(
            np.allclose(result.mitigated_probabilities, (0.0, 0.0, 0.0, 1.0))
        )
        self.assertAlmostEqual(
            diagonal_expectation(result, (1.0, -1.0, -1.0, 1.0)),
            1.0,
        )

    def test_invalid_readout_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            uncorrelated_readout_plan(2, p0=0.6, p1=0.4)
        with self.assertRaises(ValueError):
            mitigate_counts(
                uncorrelated_readout_plan(1, p0=0.1, p1=0.1),
                {"bad": 1},
            )


class QiskitBoundaryTests(unittest.TestCase):
    @staticmethod
    def _triple(region, _factor):
        output = region.copy_empty_like()
        for item in region.data:
            for _ in range(3):
                output.append(item.operation, item.qubits, item.clbits)
        return output

    def test_resets_split_unitary_folding_regions(self):
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.reset(0)
        circuit.x(0)
        circuit.measure(0, 0)

        scaled = scale_unitary_regions(circuit, 3.0, self._triple)

        self.assertEqual(
            tuple(item.operation.name for item in scaled.data),
            ("h", "h", "h", "reset", "x", "x", "x", "measure"),
        )

    def test_mid_circuit_measurement_is_rejected(self):
        circuit = QuantumCircuit(1, 1)
        circuit.measure(0, 0)
        circuit.x(0)

        with self.assertRaisesRegex(ValueError, "mid-circuit"):
            validate_mitiq_circuit(circuit)

    @unittest.skipIf(find_spec("mitiq") is not None, "Mitiq is installed")
    def test_missing_optional_dependency_has_actionable_error(self):
        circuit = QuantumCircuit(1)
        circuit.x(0)

        with self.assertRaisesRegex(MitiqUnavailable, r"mitiq\[qiskit\]"):
            construct_zne(circuit, zne_plan((1.0, 3.0)))


class FunctionalStyleTests(unittest.TestCase):
    def test_package_avoids_imperative_traversal_syntax(self):
        prohibited = (
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.AugAssign,
        )
        package = Path(__file__).parents[1] / "mitiq_qem"
        violations = tuple(
            (source.name, type(node).__name__, node.lineno)
            for source in package.rglob("*.py")
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
            if isinstance(node, prohibited)
        )

        self.assertEqual(violations, ())


if __name__ == "__main__":
    unittest.main()
