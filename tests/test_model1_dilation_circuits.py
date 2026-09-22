import unittest

import numpy as np
from qiskit.circuit import IfElseOp
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace

from IBMRuntime import (
    Circuit,
    ClassicallyControlledXXPlusYY,
    ControlledXXPlusYY,
    Measure,
    MultiControlledXXPlusYY,
    Reset,
    XXPlusYY,
)
from IBMRuntime.interpreters.qiskit import _to_qiskit
from library import classical
from Model1.Experiment2 import dynamic_lie_trotter as dynamic
from Model1.Experiment2 import unitary_lie_trotter as lie
from Model1.Experiment2 import unitary_strang_trotter as symmetric


class ModelProblemOneDilationCircuitTests(unittest.TestCase):
    dt = 0.2

    def _expected_density_matrix(self, product_formula):
        hamiltonian, jump_operators, *_ = classical.build_linear_chain(
            number_of_qubits=2,
            J=lie.J,
            h=lie.h,
            gamma=lie.gamma,
        )
        initial_state = classical.computational_state(2, [1])
        initial_density_matrix = np.outer(
            initial_state,
            initial_state.conj(),
        )
        return classical.hamiltonian_dilation_evolve(
            initial_density_matrix,
            hamiltonian,
            jump_operators,
            np.asarray([0.0, self.dt]),
            product_formula=product_formula,
        )[-1]

    def _reduced_aer_density_matrix(self, circuit):
        joint_density_matrix = DensityMatrix.from_label("0001").evolve(
            _to_qiskit(circuit)
        )
        return np.asarray(
            partial_trace(
                joint_density_matrix,
                lie.ANCILLA_QUBITS,
            ).data
        )

    def test_lie_gate_layout_contains_both_required_open_controls(self):
        jump_angle = 2.0 * np.sqrt(lie.gamma * self.dt)
        operations = lie.build_one_step_circuit(
            self.dt,
            jump_angle,
        ).operations

        self.assertEqual(len(operations), 5)
        self.assertIsInstance(operations[0], MultiControlledXXPlusYY)
        self.assertEqual(operations[0].controls, lie.ANCILLA_QUBITS)
        self.assertEqual(operations[0].control_state, 0)
        self.assertEqual(
            operations[1],
            ControlledXXPlusYY(
                jump_angle,
                0.0,
                lie.ANCILLA_HIGH_QUBIT,
                lie.SYSTEM_QUBITS_BY_SITE[0],
                lie.ANCILLA_LOW_QUBIT,
                0,
                "K_1",
            ),
        )
        self.assertEqual(
            operations[2],
            ControlledXXPlusYY(
                jump_angle,
                0.0,
                lie.ANCILLA_LOW_QUBIT,
                lie.SYSTEM_QUBITS_BY_SITE[1],
                lie.ANCILLA_HIGH_QUBIT,
                0,
                "K_2",
            ),
        )
        self.assertEqual(
            operations[-2:],
            (
                Reset(lie.ANCILLA_LOW_QUBIT),
                Reset(lie.ANCILLA_HIGH_QUBIT),
            ),
        )

    def test_lie_step_matches_the_classical_dilation(self):
        jump_angle = 2.0 * np.sqrt(lie.gamma * self.dt)
        actual = self._reduced_aer_density_matrix(
            lie.build_one_step_circuit(self.dt, jump_angle)
        )
        expected = self._expected_density_matrix("lie")

        np.testing.assert_allclose(actual, expected, atol=1.0e-12)

    def test_dynamic_lie_replaces_quantum_controls_with_feed_forward(self):
        jump_angle = 2.0 * np.sqrt(lie.gamma * self.dt)
        circuit = dynamic.build_one_step_circuit(self.dt, jump_angle)
        operations = circuit.operations

        self.assertEqual(len(operations), 6)
        self.assertEqual(
            operations[:3],
            (
                XXPlusYY(
                    2.0 * lie.J * self.dt,
                    0.0,
                    lie.SYSTEM_QUBITS_BY_SITE[0],
                    lie.SYSTEM_QUBITS_BY_SITE[1],
                    "K_H",
                ),
                XXPlusYY(
                    jump_angle,
                    0.0,
                    lie.SYSTEM_QUBITS_BY_SITE[0],
                    lie.ANCILLA_LOW_QUBIT,
                    "K_1",
                ),
                Measure(
                    lie.ANCILLA_LOW_QUBIT,
                    dynamic.LOW_ANCILLA_MEASUREMENT_BIT,
                ),
            ),
        )
        self.assertEqual(
            operations[3],
            ClassicallyControlledXXPlusYY(
                jump_angle,
                0.0,
                dynamic.LOW_ANCILLA_MEASUREMENT_BIT,
                0,
                lie.SYSTEM_QUBITS_BY_SITE[1],
                lie.ANCILLA_HIGH_QUBIT,
                "K_2",
            ),
        )
        self.assertEqual(
            operations[-2:],
            (
                Reset(lie.ANCILLA_LOW_QUBIT),
                Reset(lie.ANCILLA_HIGH_QUBIT),
            ),
        )

        native_circuit = _to_qiskit(circuit)
        self.assertIsInstance(native_circuit.data[3].operation, IfElseOp)
        self.assertEqual(native_circuit.data[3].operation.condition[1], 0)
        self.assertEqual(
            native_circuit.data[3].operation.blocks[0].data[0].operation.name,
            "xx_plus_yy",
        )

    def test_dynamic_measurement_preserves_the_reduced_lie_channel(self):
        jump_angle = 2.0 * np.sqrt(lie.gamma * self.dt)
        dynamic_circuit = dynamic.build_one_step_circuit(
            self.dt,
            jump_angle,
        )
        unitary_prefix = Circuit(
            lie.TOTAL_CIRCUIT_QUBITS,
            0,
            dynamic_circuit.operations[:2],
            "dynamic_unitary_prefix",
        )
        density_matrix = np.asarray(
            DensityMatrix.from_label("0001").evolve(
                _to_qiskit(unitary_prefix)
            ).data
        )

        low_ancilla_zero = np.diag(
            tuple(
                1.0
                if ((basis_state >> lie.ANCILLA_LOW_QUBIT) & 1) == 0
                else 0.0
                for basis_state in range(2**lie.TOTAL_CIRCUIT_QUBITS)
            )
        )
        low_ancilla_one = np.eye(
            2**lie.TOTAL_CIRCUIT_QUBITS
        ) - low_ancilla_zero
        second_jump = Circuit(
            lie.TOTAL_CIRCUIT_QUBITS,
            0,
            (
                XXPlusYY(
                    jump_angle,
                    0.0,
                    lie.SYSTEM_QUBITS_BY_SITE[1],
                    lie.ANCILLA_HIGH_QUBIT,
                    "K_2",
                ),
            ),
            "uncontrolled_second_jump",
        )
        second_jump_unitary = np.asarray(
            Operator(_to_qiskit(second_jump)).data
        )
        zero_branch = low_ancilla_zero @ density_matrix @ low_ancilla_zero
        one_branch = low_ancilla_one @ density_matrix @ low_ancilla_one
        output_density_matrix = (
            second_jump_unitary
            @ zero_branch
            @ second_jump_unitary.conj().T
            + one_branch
        )
        actual = np.asarray(
            partial_trace(
                DensityMatrix(output_density_matrix),
                lie.ANCILLA_QUBITS,
            ).data
        )
        expected = self._expected_density_matrix("lie")

        np.testing.assert_allclose(actual, expected, atol=1.0e-12)

    def test_symmetric_step_has_system_between_jump_half_steps(self):
        jump_full_angle = 2.0 * np.sqrt(lie.gamma * self.dt)
        jump_half_angle = 0.5 * jump_full_angle
        circuit = symmetric.build_one_step_circuit(
            self.dt,
            jump_half_angle,
        )
        operations = circuit.operations

        self.assertEqual(len(operations), 7)
        self.assertEqual(
            tuple(operation.label for operation in operations[:5]),
            ("K_1/2", "K_2/2", "K_H", "K_2/2", "K_1/2"),
        )

        actual = self._reduced_aer_density_matrix(circuit)
        expected = self._expected_density_matrix("strang")

        np.testing.assert_allclose(actual, expected, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
