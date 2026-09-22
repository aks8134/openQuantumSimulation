import unittest

import numpy as np
from qiskit.quantum_info import DensityMatrix, partial_trace
from scipy.linalg import expm
from scipy.sparse import csr_matrix

from IBMRuntime import (
    ControlledXXPlusYY,
    MultiControlledXXPlusYY,
    Reset,
)
from IBMRuntime.interpreters.qiskit import _to_qiskit
from library import classical
from Model1.Experiment3 import unitary_lie_trotter as experiment


class GeneralNUnitaryLieCircuitTests(unittest.TestCase):
    dt = 0.2

    def test_four_site_layout_has_even_then_odd_bond_layers(self):
        number_of_qubits = 4
        jump_angle = 2.0 * np.sqrt(experiment.gamma * self.dt)
        circuit = experiment.build_one_step_circuit(
            number_of_qubits,
            self.dt,
            jump_angle,
        )
        operations = circuit.operations
        system_qubits = experiment.system_qubits_by_site(number_of_qubits)
        shared_ancilla = experiment.ancilla_qubits(number_of_qubits)

        self.assertEqual(len(operations), 7)
        self.assertEqual(
            tuple(operation.label for operation in operations[:5]),
            (
                "H_even[0,1]",
                "H_even[2,3]",
                "H_odd[1,2]",
                "K_1",
                "K_2",
            ),
        )
        self.assertTrue(
            all(
                isinstance(operation, MultiControlledXXPlusYY)
                and operation.controls == shared_ancilla
                and operation.control_state == 0
                for operation in operations[:3]
            )
        )
        self.assertEqual(
            operations[3],
            ControlledXXPlusYY(
                jump_angle,
                0.0,
                shared_ancilla[1],
                system_qubits[0],
                shared_ancilla[0],
                0,
                "K_1",
            ),
        )
        self.assertEqual(
            operations[4],
            ControlledXXPlusYY(
                jump_angle,
                0.0,
                shared_ancilla[0],
                system_qubits[-1],
                shared_ancilla[1],
                0,
                "K_2",
            ),
        )
        self.assertEqual(
            operations[-2:],
            (Reset(shared_ancilla[0]), Reset(shared_ancilla[1])),
        )

    def test_three_site_circuit_matches_even_odd_dilation_product(self):
        number_of_qubits = 3
        jump_angle = 2.0 * np.sqrt(experiment.gamma * self.dt)
        system_qubits = experiment.system_qubits_by_site(number_of_qubits)
        shared_ancilla = experiment.ancilla_qubits(number_of_qubits)
        circuit = experiment.build_one_step_circuit(
            number_of_qubits,
            self.dt,
            jump_angle,
        )
        initial_qiskit_qubit = system_qubits[number_of_qubits // 2]
        initial_label = format(
            1 << initial_qiskit_qubit,
            f"0{number_of_qubits + 2}b",
        )
        actual = np.asarray(
            partial_trace(
                DensityMatrix.from_label(initial_label).evolve(
                    _to_qiskit(circuit)
                ),
                shared_ancilla,
            ).data
        )

        (
            _,
            jump_operators,
            x_operators,
            y_operators,
            _,
        ) = classical.build_linear_chain(
            number_of_qubits=number_of_qubits,
            J=experiment.J,
            h=experiment.h,
            gamma=experiment.gamma,
        )
        dimension = 2**number_of_qubits

        def layer_hamiltonian(bonds):
            return sum(
                (
                    0.5
                    * experiment.J
                    * (
                        x_operators[bond] @ x_operators[bond + 1]
                        + y_operators[bond] @ y_operators[bond + 1]
                    )
                    for bond in bonds
                ),
                start=csr_matrix((dimension, dimension), dtype=complex),
            )

        even_hamiltonian = layer_hamiltonian(
            experiment.even_bonds(number_of_qubits)
        )
        odd_hamiltonian = layer_hamiltonian(
            experiment.odd_bonds(number_of_qubits)
        )
        even_term, jump_terms = classical.build_hamiltonian_dilation_terms(
            even_hamiltonian,
            jump_operators,
            self.dt,
        )
        odd_term, _ = classical.build_hamiltonian_dilation_terms(
            odd_hamiltonian,
            jump_operators,
            self.dt,
        )

        isometry = np.zeros((3 * dimension, dimension), dtype=complex)
        isometry[:dimension] = np.eye(dimension, dtype=complex)
        for generator in (even_term, odd_term, *jump_terms):
            isometry = expm(-1j * generator.toarray()) @ isometry

        kraus_operators = isometry.reshape(3, dimension, dimension)
        initial_state = classical.computational_state(
            number_of_qubits,
            (number_of_qubits // 2,),
        )
        initial_density_matrix = np.outer(
            initial_state,
            initial_state.conj(),
        )
        expected = sum(
            (
                kraus_operator
                @ initial_density_matrix
                @ kraus_operator.conj().T
                for kraus_operator in kraus_operators
            )
        )

        np.testing.assert_allclose(actual, expected, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
