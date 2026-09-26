import unittest
from itertools import product

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix

from Model1.Experiment6 import classical_sample_complexity as experiment6
from library import classical


class Experiment6ClassicalTests(unittest.TestCase):
    def test_initial_state_and_observable_families_match_experiment5(self):
        problem = experiment6.build_problem(
            4,
            np.asarray((0.0, 0.1)),
            0.1,
        )
        exact = experiment6.exact_lindblad_observables(problem)
        populations, correlations, flows = experiment6.split_observable_families(
            exact,
            problem.family_sizes,
        )

        self.assertTrue(np.allclose(populations[:, 0], (0.0, 0.0, 1.0, 0.0)))
        self.assertTrue(np.allclose(correlations[:, 0], 0.0))
        self.assertTrue(np.allclose(flows[:, 0], 0.0))

    def test_exact_boundary_average_equals_deterministic_mean_channel(self):
        problem = experiment6.build_problem(
            2,
            np.asarray((0.0, 0.1, 0.2)),
            0.1,
        )
        choices = np.asarray(tuple(product((0, 1), repeat=3)))
        enumerated_mean = np.mean(
            experiment6.simulate_boundary_choices(problem, choices),
            axis=0,
        )
        expected = experiment6.expected_randomized_observables(problem)

        self.assertTrue(np.allclose(enumerated_mean, expected, atol=1e-12))

    def test_reduced_trajectory_matches_full_hilbert_space_channel(self):
        dt = 0.1
        problem = experiment6.build_problem(2, np.asarray((0.0, dt)), dt)
        reduced = experiment6.simulate_boundary_choices(
            problem,
            np.asarray(((0,), (1,))),
        )[:, :, -1]
        hamiltonian, _, _, _, _ = classical.build_linear_chain(
            2,
            experiment6.J,
            experiment6.h_field,
            experiment6.gamma,
        )
        unitary = expm(-1j * dt * hamiltonian.toarray())
        after_system = (
            unitary
            @ problem.initial_density_matrix
            @ unitary.conj().T
        )
        cosine = np.cos(np.sqrt(2.0 * experiment6.gamma * dt))
        sine = np.sin(np.sqrt(2.0 * experiment6.gamma * dt))

        def branch_values(site):
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
            density = (
                no_jump @ after_system @ no_jump.conj().T
                + jump @ after_system @ jump.conj().T
            )
            return experiment6._observable_batch(
                density[np.newaxis, :, :],
                problem.observables,
            )[0]

        full = np.asarray((branch_values(0), branch_values(1)))
        self.assertTrue(np.allclose(reduced, full, atol=1e-12))

    def test_study_is_reproducible_and_has_requested_shapes(self):
        problem = experiment6.build_problem(
            2,
            np.asarray((0.0, 0.1, 0.2)),
            0.1,
        )
        first = experiment6.run_complexity_study(
            problem,
            (1, 2, 4),
            replicates=4,
            tolerance=0.5,
            confidence=0.75,
            seed=17,
        )
        second = experiment6.run_complexity_study(
            problem,
            (1, 2, 4),
            replicates=4,
            tolerance=0.5,
            confidence=0.75,
            seed=17,
        )

        self.assertEqual(first.sample_counts, (1, 2, 4))
        self.assertEqual(first.replicate_estimates.shape, (4, 3, 4, 3))
        self.assertTrue(np.allclose(first.replicate_estimates, second.replicate_estimates))
        self.assertEqual(len(first.summaries_vs_exact), 3)

    def test_non_multiple_interval_uses_short_remainder(self):
        schedule = experiment6.build_time_schedule(
            np.asarray((0.0, 0.7)),
            0.2,
        )

        self.assertEqual(schedule.target_substep_dts[0], ())
        self.assertTrue(
            np.allclose(schedule.target_substep_dts[1], (0.2, 0.2, 0.2, 0.1))
        )

    def test_trotter_candidates_are_independent_of_observation_spacing(self):
        schedule = experiment6.build_time_schedule(
            np.asarray((0.0, 0.2, 0.4, 0.8)),
            0.3,
        )

        self.assertEqual(schedule.target_substep_dts[0], ())
        self.assertTrue(np.allclose(schedule.target_substep_dts[1], (0.2,)))
        self.assertTrue(np.allclose(schedule.target_substep_dts[2], (0.3, 0.1)))
        self.assertTrue(np.allclose(schedule.target_substep_dts[3], (0.3, 0.3, 0.2)))

    def test_joint_search_maximizes_delta_t_then_minimizes_R(self):
        joint = experiment6.run_joint_complexity_study(
            number_of_qubits=2,
            times=np.asarray((0.0, 0.1, 0.2)),
            delta_ts=(0.2, 0.1),
            sample_counts=(1, 2, 4),
            replicates=4,
            tolerance=10.0,
            confidence=0.75,
            seed=23,
        )

        self.assertEqual(joint.delta_ts, (0.2, 0.1))
        self.assertEqual(len(joint.pair_records), 6)
        self.assertTrue(
            all(
                record["total_sampled_substeps"]
                == record["sample_count"]
                * record["physical_substeps_per_trajectory"]
                for record in joint.pair_records
            )
        )
        self.assertIsNotNone(joint.optimal_delta_t)
        self.assertEqual(
            joint.optimal_delta_t,
            max(
                record["delta_t"] for record in joint.pair_records if record["feasible"]
            ),
        )
        self.assertEqual(
            joint.optimal_sample_count,
            min(
                record["sample_count"]
                for record in joint.pair_records
                if record["feasible"]
                and record["delta_t"] == joint.optimal_delta_t
            ),
        )


if __name__ == "__main__":
    unittest.main()
