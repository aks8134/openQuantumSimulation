import unittest

import numpy as np
from scipy.linalg import expm

from library import classical
from Model2.Experiment1 import simulation as model


class ModelTwoExperimentOneTests(unittest.TestCase):
    def test_isolated_reservoir_targets_excitation_population(self):
        loss = classical.SIGMA_MINUS
        gamma, p = 2.0, 0.25
        jumps = [np.sqrt(gamma * p) * loss.getH(), np.sqrt(gamma * (1 - p)) * loss]
        L = classical.build_liouvillian(0 * classical.Z, jumps)
        for population in (0.0, 0.3, 1.0):
            rho = np.diag([1 - population, population])
            derivative = (L @ rho.reshape(-1, order="F")).reshape(2, 2, order="F")
            self.assertAlmostEqual(derivative[1, 1].real, gamma * (p - population))

    def test_product_state_is_stationary_with_interactions_and_both_prefactors(self):
        for prefactor in (0.25, 0.5):
            for p in (0.25, 0.75):
                H, jumps, _, _, _, target = model.build_model(
                    4, delta=1.3, p=p, exchange_prefactor=prefactor,
                )
                L = classical.build_liouvillian(H, jumps)
                self.assertEqual(L.shape, (256, 256))
                np.testing.assert_allclose(L @ target.reshape(-1, order="F"), 0, atol=1e-14)

    def test_continuity_is_an_operator_identity(self):
        # Check every matrix component, not only the three benchmark states.
        for prefactor in (0.25, 0.5):
            H, jumps, _, populations, currents, _ = model.build_model(
                delta=0.7, exchange_prefactor=prefactor,
            )
            L = classical.build_liouvillian(H, jumps)
            identity = np.eye(8)
            rhs = [2 * (0.25 * identity - populations[0].toarray()) - currents[0].toarray(),
                   currents[0].toarray() - currents[1].toarray(),
                   currents[1].toarray() + 2 * (0.25 * identity - populations[2].toarray())]
            for population, expected in zip(populations, rhs):
                adjoint = (L.getH() @ population.toarray().reshape(-1, order="F")).reshape(8, 8, order="F")
                np.testing.assert_allclose(adjoint, expected, atol=1e-14)

    def test_exact_arbitrary_times_match_independent_dense_exponentials(self):
        H, jumps, *_ = model.build_model()
        L = classical.build_liouvillian(H, jumps)
        _, initial = model.initial_states(3)
        for times in ([0, 0.013, 0.17, 0.63], [0.17], [0.17, 0.63]):
            evolved = model.exact_evolve(initial, L, times)
            for index, time in enumerate(times):
                propagator = expm(time * L.toarray())
                for state, rho in enumerate(initial):
                    expected = (propagator @ rho.reshape(-1, order="F")).reshape(8, 8, order="F")
                    np.testing.assert_allclose(evolved[state, index], expected, atol=2e-14)

    def test_dilation_matches_existing_model1_engine_for_four_jumps(self):
        H, jumps, *_ = model.build_model()
        _, initial = model.initial_states(3)
        for formula, existing in (("unsplit", "exact"), ("lie", "lie"), ("strang", "strang")):
            evolved = model.dilation_evolve(initial, H, jumps, [0, 0.08, 0.16],
                                            max_dt=0.04, formula=formula)
            for state, rho in enumerate(initial):
                reference = classical.hamiltonian_dilation_evolve(
                    rho, H, jumps, [0, 0.08, 0.16], product_formula=existing,
                    steps_per_interval=2,
                )
                np.testing.assert_allclose(evolved[state], reference, atol=2e-14)

    def test_dilation_is_cptp_and_converges_under_refinement(self):
        H, jumps, *_ = model.build_model()
        L = classical.build_liouvillian(H, jumps)
        _, initial = model.initial_states(3)
        times = [0.13, 0.37, 1.0]
        exact = model.exact_evolve(initial, L, times)
        for formula in ("unsplit", "lie", "strang"):
            errors = []
            for dt in (0.1, 0.05, 0.025):
                evolved = model.dilation_evolve(initial, H, jumps, times, max_dt=dt, formula=formula)
                np.testing.assert_allclose(np.trace(evolved, axis1=-2, axis2=-1), 1, atol=2e-13)
                np.testing.assert_allclose(evolved, evolved.conj().swapaxes(-1, -2), atol=2e-14)
                self.assertGreater(np.linalg.eigvalsh(evolved).min(), -2e-14)
                errors.append(np.max(np.linalg.norm(evolved - exact, axis=(-2, -1))))
            self.assertLess(errors[1], errors[0])
            self.assertLess(errors[2], errors[1])

    def test_invalid_times_are_rejected(self):
        for times in ([], [-1], [np.nan], [np.inf], [0.3, 0.2], [0, 0]):
            with self.assertRaises(ValueError):
                model.validate_times(times)


if __name__ == "__main__":
    unittest.main()
