"""Validate randomized covariance reduction against full ancilla dilations."""

import itertools
import unittest

import numpy as np

from library import classical
from Model2.Experiment1 import simulation
from Model2.Experiment2 import randomized


def covariance(rho, n):
    annihilators = []
    for site in range(n):
        operator = np.ones((1, 1), complex)
        for j in range(n):
            local = (classical.Z if j < site else
                     classical.SIGMA_MINUS if j == site else classical.I2)
            operator = np.kron(operator, local.toarray())
        annihilators.append(operator)
    return np.asarray([[np.trace(b.conj().T @ a @ rho)
                        for b in annihilators] for a in annihilators])


class RandomizedDilationTests(unittest.TestCase):
    def test_continuous_reference_matches_full_liouvillian(self):
        H, jumps, _, populations, currents, _ = simulation.build_model(3, J=2, gamma=4, p=0.3)
        _, initial = simulation.initial_states(3)
        _, initial_C = randomized.initial_covariances(3)
        times = np.array([0, 0.0357, 0.42, 2.0])
        rho = simulation.exact_evolve(initial, classical.build_liouvillian(H, jumps), times)
        C = randomized.exact_covariances(initial_C, 3, 2, 4, 0.3, 0.25, times)
        population, current = randomized.observables(C, 1.0)
        np.testing.assert_allclose(population, simulation.expectations(rho, populations), atol=1e-13, rtol=0)
        np.testing.assert_allclose(current, simulation.expectations(rho, currents), atol=1e-13, rtol=0)

    def test_each_covariance_branch_matches_full_density_dilation(self):
        # Include mixed states with coherences across excitation sectors.
        rng = np.random.default_rng(183)
        matrix = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        rho = matrix @ matrix.conj().T
        rho /= np.trace(rho)
        H, jumps, _, pop_ops, current_ops, _ = simulation.build_model(3, J=2, gamma=4, p=0.3)
        C = covariance(rho, 3)
        for policy, order in itertools.product(randomized.POLICIES, randomized.ORDERS):
            branches = randomized.covariance_branches(3, 2, 4, 0.3, 0.25, 0.03, policy, order)
            for a in range(4):
                with self.subTest(policy=policy, order=order, jump=a):
                    kraus = randomized.sampled_kraus(H, jumps[a], branches["mu"][a], 0.03, order)
                    evolved = sum(K @ rho @ K.conj().T for K in kraus)
                    np.testing.assert_allclose(sum(K.conj().T @ K for K in kraus), np.eye(8), atol=1e-14)
                    reduced = branches["A"][a] @ C @ branches["A"][a].conj().T + branches["B"][a]
                    np.testing.assert_allclose(reduced, covariance(evolved, 3), atol=1e-14, rtol=0)
                    population, current = randomized.observables(reduced, branches["hopping"])
                    np.testing.assert_allclose(population, [np.trace(op @ evolved).real for op in pop_ops], atol=1e-14)
                    np.testing.assert_allclose(current, [np.trace(op @ evolved).real for op in current_ops], atol=1e-14)

    def test_ensemble_powers_match_all_two_step_sequences(self):
        H, jumps, *_ = simulation.build_model(3, J=2, gamma=4, p=0.3)
        _, initial = simulation.initial_states(3)
        _, initial_C = randomized.initial_covariances(3)
        for policy, order in itertools.product(randomized.POLICIES, randomized.ORDERS):
            branches = randomized.covariance_branches(3, 2, 4, 0.3, 0.25, 0.03, policy, order)
            kraus = [randomized.sampled_kraus(H, jump, mu, 0.03, order)
                     for jump, mu in zip(jumps, branches["mu"])]
            average = np.zeros_like(initial)
            for a, b in itertools.product(range(4), repeat=2):
                first = sum(K @ initial @ K.conj().T for K in kraus[a])
                second = sum(K @ first @ K.conj().T for K in kraus[b])
                average += branches["mu"][a] * branches["mu"][b] * second
            predicted, _ = randomized.mean_trajectory(initial_C, branches, np.array([0, 2, 100]))
            np.testing.assert_allclose(predicted[:, 1], [covariance(rho, 3) for rho in average], atol=1e-13, rtol=0)

    def test_weighted_mean_preserves_full_product_target(self):
        H, jumps, *_, target = simulation.build_model(3, J=2, gamma=4, p=0.25)
        for order in randomized.ORDERS:
            for policy in randomized.POLICIES:
                mu = randomized.probabilities(0.25, policy)
                average = np.zeros_like(target)
                for jump, probability in zip(jumps, mu):
                    kraus = randomized.sampled_kraus(H, jump, probability, 0.03, order)
                    average += probability * sum(K @ target @ K.conj().T for K in kraus)
                if policy == "rate-weighted":
                    np.testing.assert_allclose(average, target, atol=1e-14, rtol=0)
                else:
                    self.assertGreater(np.max(np.abs(average - target)), 1e-4)

    def test_seed_reproducibility_and_sampling_mean(self):
        _, initial = randomized.initial_covariances(3)
        branches = randomized.covariance_branches(3, 2, 4, 0.25, 0.25, 0.003, "rate-weighted", "jump-first")
        counts = np.array([0, 10, 100])
        pop, current, _ = randomized.sample_trajectories(initial, branches, counts, 4000, 2, "jump-first")
        repeat, _, _ = randomized.sample_trajectories(initial, branches, counts, 4000, 2, "jump-first")
        np.testing.assert_array_equal(pop, repeat)
        mean_C, _ = randomized.mean_trajectory(initial, branches, counts)
        mean_pop, mean_current = randomized.observables(mean_C, branches["hopping"])
        for values, expected in ((pop, mean_pop), (current, mean_current)):
            error = np.abs(values.mean(axis=1) - expected)
            standard_error = values.std(axis=1, ddof=1) / np.sqrt(4000)
            self.assertTrue(np.all(error <= 6 * standard_error + 1e-13))


if __name__ == "__main__":
    unittest.main()
