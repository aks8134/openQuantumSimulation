import ast
import unittest
from pathlib import Path

import numpy as np
from qiskit.circuit.library import RXGate
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from qdrift import (
    add_term,
    empty_hamiltonian,
    qdrift,
    sampling_probabilities,
    to_qiskit,
)


class QDriftCoreTests(unittest.TestCase):
    def test_functional_construction_preserves_previous_values(self):
        empty = empty_hamiltonian()
        first = add_term(empty, 2.0, "X", label="x")
        complete = add_term(
            first,
            -1.0,
            np.diag((2.0, -2.0)),
            operator_norm=2.0,
            label="scaled z",
        )

        self.assertEqual(empty.terms, ())
        self.assertEqual(len(first.terms), 1)
        self.assertEqual(len(complete.terms), 2)
        self.assertEqual(complete.lambda_norm, 4.0)
        self.assertEqual(sampling_probabilities(complete), (0.5, 0.5))

    def test_seeded_sampling_is_reproducible_and_uses_generalized_times(self):
        model = add_term(empty_hamiltonian(), 2.0, "X", label="x")
        model = add_term(
            model,
            -1.0,
            np.diag((2.0, -2.0)),
            operator_norm=2.0,
            label="scaled z",
        )
        first = qdrift(model, evolution_time=3.0, sample_count=12, seed=19)
        second = qdrift(model, evolution_time=3.0, sample_count=12, seed=19)

        self.assertEqual(first.sampled_term_indices, second.sampled_term_indices)
        self.assertEqual(len(first.steps), 12)
        self.assertTrue(
            all(
                map(
                    lambda step: np.isclose(
                        step.operator_time,
                        1.0 if step.term_index == 0 else -0.5,
                    ),
                    first.steps,
                )
            )
        )

    def test_zero_coefficient_terms_are_not_sampled(self):
        model = add_term(empty_hamiltonian(), 0.0, "Z", label="inactive")
        model = add_term(model, 1.0, "X", label="active")
        program = qdrift(model, evolution_time=1.0, sample_count=20, seed=4)

        self.assertEqual(program.probabilities, (0.0, 1.0))
        self.assertEqual(program.sampled_term_indices, (1,) * 20)

    def test_invalid_sampling_inputs_are_rejected(self):
        model = add_term(empty_hamiltonian(), 1.0, "X")

        with self.assertRaises(ValueError):
            qdrift(model, evolution_time=1.0, sample_count=0)
        with self.assertRaises(ValueError):
            qdrift(empty_hamiltonian(), evolution_time=1.0, sample_count=1)
        with self.assertRaises(ValueError):
            qdrift(model, evolution_time=-1.0, sample_count=1)


class QDriftQiskitTests(unittest.TestCase):
    def test_one_term_program_matches_exact_evolution(self):
        model = add_term(empty_hamiltonian(), -0.7, "X", targets=(0,), label="x")
        program = qdrift(model, evolution_time=1.3, sample_count=11, seed=3)
        circuit = to_qiskit(program, num_qubits=1)
        expected = expm(-1j * 1.3 * (-0.7 * np.array(((0.0, 1.0), (1.0, 0.0)))))

        self.assertTrue(np.allclose(Operator(circuit).data, expected))
        self.assertEqual(circuit.metadata["sample_count"], 11)
        self.assertEqual(circuit.metadata["sampled_term_indices"], (0,) * 11)

    def test_local_targets_are_retained_in_circuit(self):
        model = add_term(
            empty_hamiltonian(),
            1.0,
            "XX",
            targets=(1, 3),
            label="nonlocal xx",
        )
        circuit = to_qiskit(
            qdrift(model, evolution_time=0.2, sample_count=1, seed=1),
            num_qubits=4,
        )

        self.assertEqual(circuit.num_qubits, 4)
        self.assertEqual(tuple(map(circuit.find_bit, circuit.data[0].qubits)), (circuit.find_bit(circuit.qubits[1]), circuit.find_bit(circuit.qubits[3])))

    def test_ambiguous_local_operator_requires_targets(self):
        model = add_term(empty_hamiltonian(), 1.0, "X")
        program = qdrift(model, evolution_time=0.2, sample_count=1, seed=1)

        with self.assertRaises(ValueError):
            to_qiskit(program, num_qubits=2)

    def test_custom_operator_representation_uses_factory(self):
        custom_operator = object()
        model = add_term(
            empty_hamiltonian(),
            1.0,
            custom_operator,
            targets=(1,),
            label="custom x",
        )
        program = qdrift(model, evolution_time=0.3, sample_count=2, seed=5)
        circuit = to_qiskit(
            program,
            num_qubits=2,
            evolution_factory=lambda selected, time: RXGate(2.0 * time),
        )

        expected = expm(-1j * 0.3 * np.array(((0.0, 1.0), (1.0, 0.0))))
        reduced = Operator(circuit).data.reshape(2, 2, 2, 2)
        self.assertTrue(np.allclose(reduced[:, 0, :, 0], expected))


class QDriftStyleTests(unittest.TestCase):
    def test_package_uses_ocaml_style_immutable_traversals(self):
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
        package = Path(__file__).parents[1] / "qdrift"
        violations = tuple(
            (source.name, type(node).__name__, node.lineno)
            for source in package.rglob("*.py")
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
            if isinstance(node, prohibited)
        )

        self.assertEqual(violations, ())


if __name__ == "__main__":
    unittest.main()
