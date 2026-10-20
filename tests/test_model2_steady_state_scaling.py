"""Physical checks of exact population convergence and threshold crossings."""

import unittest

import numpy as np
from scipy.linalg import expm

from library import classical
from Model2.Experiment1 import simulation
from Model2.Experiment1.analysis import steady_state_scaling as scaling


class ExactPopulationSettlingTests(unittest.TestCase):
    def test_two_spin_time_matches_analytic_solution(self):
        # Both sites have reservoirs; the uniform extreme preparation decays
        # as max(p,1-p)*exp(-Gamma*t), independent of exchange strength.
        for p in (.25, .75):
            for gamma in (2., 20.):
                result = scaling.settling_time(2, J=3.7, gamma=gamma, p=p)
                exact = np.log(max(p, 1-p)/1e-3)/gamma
                self.assertAlmostEqual(result['settling_time'], exact, delta=1e-9)
                self.assertGreater(result['population_error_at_lower'], 1e-3)
                self.assertLessEqual(result['population_error_at_upper'], 1e-3)

    def test_covariance_populations_match_full_liouvillian(self):
        for n in (2, 3, 4):
            H, jumps, _, operators, *_ = simulation.build_model(n, J=10, gamma=20)
            L = classical.build_liouvillian(H, jumps)
            _, initial = simulation.initial_states(n)
            crossing = scaling.settling_time(n)['settling_time']
            times = np.asarray([0, crossing/3, crossing, 1.1*crossing])
            rhos = simulation.exact_evolve(initial, L, times)
            observed = simulation.expectations(rhos, operators)
            A = scaling.generator(n)
            reduced = np.stack([scaling.populations(expm(t*A)) for t in times], axis=1)
            np.testing.assert_allclose(reduced, observed, atol=5e-13, rtol=0)

    def test_worst_case_is_monotone_and_covers_all_three_preparations(self):
        for n in (3, 7, 10, 20):
            result = scaling.settling_time(n)
            crossing = result['settling_time']
            A = scaling.generator(n)
            times = np.linspace(0, 1.5*crossing, 31)
            errors = []
            for t in times:
                E = expm(t*A)
                worst = scaling.population_distance(E)
                actual = float(np.max(np.abs(scaling.populations(E)-.25)))
                self.assertAlmostEqual(worst, actual, delta=5e-15)
                errors.append(worst)
            self.assertLessEqual(np.max(np.diff(errors)), 5e-14)
            for t in (crossing, 1.1*crossing, 2*crossing):
                self.assertLessEqual(scaling.population_distance(expm(t*A)), 1e-3+5e-15)

    def test_rescaling_all_rates_rescales_settling_time(self):
        for n in (7, 20):
            fast = scaling.settling_time(n, J=10, gamma=20)
            original = scaling.settling_time(n, J=1, gamma=2)
            self.assertAlmostEqual(original['settling_time'], 10*fast['settling_time'], delta=1e-8)


if __name__ == '__main__':
    unittest.main()
