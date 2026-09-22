import unittest

import numpy as np
from qiskit.circuit import IfElseOp
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace

from IBMRuntime import (
    Circuit,
    ClassicallyControlledXXPlusYY,
    Measure,
    Reset,
    XXPlusYY,
)
from IBMRuntime.interpreters.qiskit import _to_qiskit
from Model1.Experiment3 import dynamic_lie_trotter as dynamic
from Model1.Experiment3 import unitary_lie_trotter as unitary


class GeneralNDynamicLieCircuitTests(unittest.TestCase):
    dt = 0.2

    def test_four_site_layout_removes_all_quantum_controls(self):
        number_of_qubits = 4
        jump_angle = 2.0 * np.sqrt(dynamic.gamma * self.dt)
        system_qubits = unitary.system_qubits_by_site(number_of_qubits)
        ancilla_low, ancilla_high = unitary.ancilla_qubits(
            number_of_qubits
        )
        circuit = dynamic.build_one_step_circuit(
            number_of_qubits,
            self.dt,
            jump_angle,
        )
        operations = circuit.operations

        self.assertEqual(circuit.bit_count, 1)
        self.assertEqual(len(operations), 8)
        self.assertEqual(
            operations[:4],
            (
                XXPlusYY(
                    2.0 * dynamic.J * self.dt,
                    0.0,
                    system_qubits[0],
                    system_qubits[1],
                    "H_even[0,1]",
                ),
                XXPlusYY(
                    2.0 * dynamic.J * self.dt,
                    0.0,
                    system_qubits[2],
                    system_qubits[3],
                    "H_even[2,3]",
                ),
                XXPlusYY(
                    2.0 * dynamic.J * self.dt,
                    0.0,
                    system_qubits[1],
                    system_qubits[2],
                    "H_odd[1,2]",
                ),
                XXPlusYY(
                    jump_angle,
                    0.0,
                    system_qubits[0],
                    ancilla_low,
                    "K_1",
                ),
            ),
        )
        self.assertEqual(
            operations[4],
            Measure(ancilla_low, dynamic.LOW_ANCILLA_MEASUREMENT_BIT),
        )
        self.assertEqual(
            operations[5],
            ClassicallyControlledXXPlusYY(
                jump_angle,
                0.0,
                dynamic.LOW_ANCILLA_MEASUREMENT_BIT,
                0,
                system_qubits[-1],
                ancilla_high,
                "K_2",
            ),
        )
        self.assertEqual(
            operations[-2:],
            (Reset(ancilla_low), Reset(ancilla_high)),
        )

        native_circuit = _to_qiskit(circuit)
        self.assertIsInstance(native_circuit.data[5].operation, IfElseOp)
        self.assertEqual(native_circuit.data[5].operation.condition[1], 0)

    def test_measurement_feed_forward_preserves_three_site_lie_channel(self):
        number_of_qubits = 3
        total_qubits = number_of_qubits + 2
        jump_angle = 2.0 * np.sqrt(dynamic.gamma * self.dt)
        system_qubits = unitary.system_qubits_by_site(number_of_qubits)
        shared_ancilla = unitary.ancilla_qubits(number_of_qubits)
        ancilla_low, ancilla_high = shared_ancilla
        initial_label = format(
            1 << system_qubits[number_of_qubits // 2],
            f"0{total_qubits}b",
        )

        dynamic_circuit = dynamic.build_one_step_circuit(
            number_of_qubits,
            self.dt,
            jump_angle,
        )
        measurement_index = next(
            index
            for index, operation in enumerate(dynamic_circuit.operations)
            if isinstance(operation, Measure)
        )
        unitary_prefix = Circuit(
            total_qubits,
            0,
            dynamic_circuit.operations[:measurement_index],
            "dynamic_general_n_unitary_prefix",
        )
        density_matrix = np.asarray(
            DensityMatrix.from_label(initial_label).evolve(
                _to_qiskit(unitary_prefix)
            ).data
        )

        low_ancilla_zero = np.diag(
            tuple(
                1.0
                if ((basis_state >> ancilla_low) & 1) == 0
                else 0.0
                for basis_state in range(2**total_qubits)
            )
        )
        low_ancilla_one = np.eye(2**total_qubits) - low_ancilla_zero
        second_jump = Circuit(
            total_qubits,
            0,
            (
                XXPlusYY(
                    jump_angle,
                    0.0,
                    system_qubits[-1],
                    ancilla_high,
                    "K_2",
                ),
            ),
            "uncontrolled_second_boundary_jump",
        )
        second_jump_unitary = np.asarray(
            Operator(_to_qiskit(second_jump)).data
        )
        zero_branch = (
            low_ancilla_zero @ density_matrix @ low_ancilla_zero
        )
        one_branch = low_ancilla_one @ density_matrix @ low_ancilla_one
        dynamic_output = (
            second_jump_unitary
            @ zero_branch
            @ second_jump_unitary.conj().T
            + one_branch
        )
        actual = np.asarray(
            partial_trace(
                DensityMatrix(dynamic_output),
                shared_ancilla,
            ).data
        )

        unitary_circuit = unitary.build_one_step_circuit(
            number_of_qubits,
            self.dt,
            jump_angle,
        )
        expected = np.asarray(
            partial_trace(
                DensityMatrix.from_label(initial_label).evolve(
                    _to_qiskit(unitary_circuit)
                ),
                shared_ancilla,
            ).data
        )

        np.testing.assert_allclose(actual, expected, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
