import unittest
from itertools import product

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix

from Model1.Experiment6 import classical_sample_complexity as experiment6
from Model1.Experiment6 import strategy3
from library import classical


class Experiment6Strategy3Tests(unittest.TestCase):
    def test_exact_branch_average_equals_mean_channel(self):
        problem = strategy3.build_problem(
            3,
            np.asarray((0.0, 0.1, 0.2)),
            0.1,
        )
        choices = np.asarray(tuple(product((0, 1), repeat=3)))
        enumerated_mean = np.mean(
            strategy3.simulate_branch_choices(problem, choices),
            axis=0,
        )
        expected = strategy3.expected_randomized_observables(problem)

        self.assertTrue(np.allclose(enumerated_mean, expected, atol=1e-12))

    def test_reduced_branches_match_full_hilbert_space_channels(self):
        number_of_qubits = 3
        dt = 0.1
        problem = strategy3.build_problem(
            number_of_qubits,
            np.asarray((0.0, dt)),
            dt,
        )
        reduced = strategy3.simulate_branch_choices(
            problem,
            np.asarray(((0,), (1,))),
        )[:, :, -1]
        _, _, x_operators, y_operators, _ = classical.build_linear_chain(
            number_of_qubits,
            experiment6.J,
            experiment6.h_field,
            experiment6.gamma,
        )
        dimension = 2**number_of_qubits

        def layer_hamiltonian(bonds):
            result = csr_matrix((dimension, dimension), dtype=complex)
            for bond in bonds:
                result = result + 0.5 * experiment6.J * (
                    x_operators[bond] @ x_operators[bond + 1]
                    + y_operators[bond] @ y_operators[bond + 1]
                )
            return result.toarray()

        odd_unitary = expm(-1j * 2.0 * dt * layer_hamiltonian((1,)))
        even_unitary = expm(-1j * 2.0 * dt * layer_hamiltonian((0,)))
        cosine = np.cos(np.sqrt(2.0 * experiment6.gamma * dt))
        sine = np.sin(np.sqrt(2.0 * experiment6.gamma * dt))
        initial = problem.initial_density_matrix

        def branch_density(unitary, jump_site):
            after_layer = unitary @ initial @ unitary.conj().T
            no_jump = classical.operator_on_site(
                csr_matrix(np.diag((1.0, cosine))),
                jump_site,
                number_of_qubits,
            ).toarray()
            jump = classical.operator_on_site(
                csr_matrix(np.asarray(((0.0, sine), (0.0, 0.0)))),
                jump_site,
                number_of_qubits,
            ).toarray()
            return (
                no_jump @ after_layer @ no_jump.conj().T
                + jump @ after_layer @ jump.conj().T
            )

        full_densities = np.asarray(
            (
                branch_density(odd_unitary, 0),
                branch_density(even_unitary, number_of_qubits - 1),
            )
        )
        full = experiment6._observable_batch(
            full_densities,
            problem.observables,
        )

        self.assertTrue(np.allclose(reduced, full, atol=1e-12))

    def test_repository_bond_convention_and_enhancement(self):
        dt = 0.1
        problem = strategy3.build_problem(4, np.asarray((0.0, dt)), dt)
        channel = problem.step_channels[0]
        expected_odd = strategy3._layer_unitary(4, (1,), 2.0 * dt)
        expected_even = strategy3._layer_unitary(4, (0, 2), 2.0 * dt)

        self.assertTrue(np.allclose(channel.odd_unitary, expected_odd))
        self.assertTrue(np.allclose(channel.even_unitary, expected_even))
        self.assertTrue(
            np.isclose(
                channel.damping_cosine,
                np.cos(np.sqrt(2.0 * experiment6.gamma * dt)),
            )
        )

    def test_study_is_reproducible_and_selects_from_requested_grid(self):
        arguments = dict(
            number_of_qubits=3,
            times=np.asarray((0.0, 0.1, 0.2)),
            delta_ts=(0.2, 0.1),
            sample_counts=(1, 2, 4),
            replicates=4,
            tolerance=10.0,
            confidence=0.75,
            seed=37,
        )
        first = strategy3.run_joint_complexity_study(**arguments)
        second = strategy3.run_joint_complexity_study(**arguments)

        self.assertEqual(first.optimal_delta_t, 0.2)
        self.assertEqual(first.optimal_sample_count, 1)
        self.assertTrue(
            np.allclose(
                first.studies[0].replicate_estimates,
                second.studies[0].replicate_estimates,
            )
        )


if __name__ == "__main__":
    unittest.main()
