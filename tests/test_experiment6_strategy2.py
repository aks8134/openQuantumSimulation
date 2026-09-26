import unittest
from itertools import product

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix

from Model1.Experiment6 import classical_sample_complexity as experiment6
from Model1.Experiment6 import strategy2
from library import classical


class Experiment6Strategy2Tests(unittest.TestCase):
    def test_exact_component_average_equals_mean_channel(self):
        problem = strategy2.build_problem(
            2,
            np.asarray((0.0, 0.1, 0.2)),
            0.1,
        )
        choices = np.asarray(tuple(product((0, 1, 2), repeat=3)))
        enumerated_mean = np.mean(
            strategy2.simulate_component_choices(problem, choices),
            axis=0,
        )
        expected = strategy2.expected_randomized_observables(problem)

        self.assertTrue(np.allclose(enumerated_mean, expected, atol=1e-12))

    def test_reduced_branches_match_full_hilbert_space_channels(self):
        dt = 0.1
        problem = strategy2.build_problem(2, np.asarray((0.0, dt)), dt)
        reduced = strategy2.simulate_component_choices(
            problem,
            np.asarray(((0,), (1,), (2,))),
        )[:, :, -1]
        hamiltonian, _, _, _, _ = classical.build_linear_chain(
            2,
            experiment6.J,
            experiment6.h_field,
            experiment6.gamma,
        )
        unitary = expm(-1j * 3.0 * dt * hamiltonian.toarray())
        initial = problem.initial_density_matrix
        hamiltonian_density = unitary @ initial @ unitary.conj().T
        cosine = np.cos(np.sqrt(3.0 * experiment6.gamma * dt))
        sine = np.sin(np.sqrt(3.0 * experiment6.gamma * dt))

        def jump_density(site):
            no_jump = classical.operator_on_site(
                csr_matrix(np.diag((1.0, cosine))),
                site,
                2,
            ).toarray()
            jump = classical.operator_on_site(
                csr_matrix(np.asarray(((0.0, sine), (0.0, 0.0)))),
                site,
                2,
            ).toarray()
            return (
                no_jump @ initial @ no_jump.conj().T
                + jump @ initial @ jump.conj().T
            )

        full_densities = np.asarray(
            (
                hamiltonian_density,
                jump_density(0),
                jump_density(1),
            )
        )
        full = experiment6._observable_batch(
            full_densities,
            problem.observables,
        )

        self.assertTrue(np.allclose(reduced, full, atol=1e-12))

    def test_enhancement_constants_are_three(self):
        dt = 0.1
        problem = strategy2.build_problem(3, np.asarray((0.0, dt)), dt)
        channel = problem.step_channels[0]

        self.assertEqual(strategy2.NUMBER_OF_COMPONENTS, 3)
        self.assertTrue(
            np.isclose(
                channel.damping_cosine,
                np.cos(np.sqrt(3.0 * experiment6.gamma * dt)),
            )
        )

    def test_study_is_reproducible_and_selects_from_requested_grid(self):
        arguments = dict(
            number_of_qubits=2,
            times=np.asarray((0.0, 0.1, 0.2)),
            delta_ts=(0.2, 0.1),
            sample_counts=(1, 2, 4),
            replicates=4,
            tolerance=10.0,
            confidence=0.75,
            seed=31,
        )
        first = strategy2.run_joint_complexity_study(**arguments)
        second = strategy2.run_joint_complexity_study(**arguments)

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
