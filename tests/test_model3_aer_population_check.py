"""Compare exact bath elimination with the explicit system+bath density."""
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import DensityMatrix, Kraus, Pauli
from qiskit_aer import AerSimulator

from Model3.Experiment5 import check_aer_populations as check
from Model3.Experiment5 import circuit_depth as circuits
from tests.test_model3_experiment5 import nonselective_density, source


class AerPopulationCheckTests(unittest.TestCase):
    def test_bath_matrices_are_contiguous_and_aer_preserves_trace_and_coherences(self):
        for occupied in (0, 1):
            instruction = check.bath_channel(XXPlusYYGate(1.3), occupied)
            self.assertTrue(all(matrix.flags.c_contiguous for matrix in instruction.params))
            expected = DensityMatrix.from_label("+").evolve(Kraus(instruction.params)).data
            circuit = QuantumCircuit(1)
            circuit.h(0)
            circuit.append(instruction, [0])
            circuit.save_density_matrix()
            result = AerSimulator(method="density_matrix", max_parallel_threads=1).run(circuit, shots=1).result()
            actual = np.asarray(result.data()["density_matrix"])
            np.testing.assert_allclose(actual, expected, atol=1e-14)
            self.assertAlmostEqual(np.trace(actual).real, 1.)

    def test_reduced_channel_matches_full_density_for_all_jump_types_and_orders(self):
        simulator = AerSimulator(method="density_matrix", max_parallel_threads=1)
        for order in ("hamiltonian-first", "jump-first"):
            for policy in ("rate-weighted", "uniform"):
                with self.subTest(order=order, policy=policy):
                    data = source(order=order, policy=policy)
                    body = circuits.build_body(data, 0)
                    full = nonselective_density(body)
                    reduced = simulator.run(check.reduced_density_circuit(body, 3), shots=1).result().data()
                    for site in range(3):
                        label = ["I"]*body.num_qubits
                        label[-1-site] = "Z"
                        self.assertAlmostEqual(reduced[f"z{site}"], full.expectation_value(Pauli("".join(label))).real, places=12)

    def test_outcome_marginal_uses_actual_classical_indices(self):
        counts = {"101 01": 7, "010 11": 3}
        self.assertEqual(check.outcome_counts(counts, [2, 3, 4]), {"101": 7, "010": 3})
        self.assertEqual(check.outcome_counts(counts, [4, 2]), {"11": 7, "00": 3})

    def test_full_native_snapshots_retain_measurements_resets_and_readout(self):
        data = source()
        native, mapping = check.compile_native(circuits.build_body(data, 0), 3, seed=11, optimization=3)
        diagnostic, indices = check.sampled_circuit(native, 3)
        self.assertEqual(indices, [3, 4, 5])
        self.assertEqual(native.metadata["physical_sites"], mapping[:3])
        self.assertEqual(diagnostic.count_ops()["measure"], 6)
        self.assertEqual(diagnostic.count_ops()["reset"], 4)
        self.assertEqual(diagnostic.count_ops()["save_expval"], 3)
        simulator = AerSimulator(method="statevector", max_parallel_threads=1)
        result = simulator.run(diagnostic, shots=16, seed_simulator=9).result()
        self.assertTrue(result.success)
        self.assertEqual(sum(check.outcome_counts(result.get_counts(), indices).values()), 16)
        for j in range(3):
            values = np.asarray(result.data()[f"z{j}"])
            self.assertEqual(values.shape, (16,))
            self.assertTrue(np.all(np.abs(values) <= 1+1e-12))

    def test_reduction_rejects_nonlocal_bath_use_and_classical_control(self):
        circuit = QuantumCircuit(3)
        circuit.cx(0, 2)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            check.reduced_density_circuit(circuit, 2)
        circuit = QuantumCircuit(3, 1)
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.x(0)
        with self.assertRaisesRegex(ValueError, "classical control"):
            check.reduced_density_circuit(circuit, 2)


if __name__ == "__main__":
    unittest.main()
