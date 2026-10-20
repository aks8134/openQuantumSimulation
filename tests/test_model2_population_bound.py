"""Independent checks of the covariance certificate and polynomial enclosure."""

from fractions import Fraction
from math import comb, factorial
import unittest

import numpy as np
from scipy.linalg import expm

from Model2.Experiment2 import randomized
from Model2.Experiment2.analysis import population_bound as bound
from Model2.Experiment2.analysis.tolerance_sweep import threshold


class PopulationBoundTests(unittest.TestCase):
    def test_cosine_large_angles_enclose_independent_rational_series(self):
        for x in (Fraction(12, 7), Fraction(20), Fraction(75)):
            result = bound.cosine_sqrt(bound.Ball.rational(x))
            reference = sum((-x)**k / factorial(2 * k) for k in range(101))
            reference_tail = x**101 / factorial(202)
            discrepancy = abs(Fraction.from_float(float(result.center)) - reference)
            self.assertLessEqual(discrepancy + reference_tail,
                                 Fraction.from_float(float(result.radius)))
        with self.assertRaises(ValueError):
            bound.cosine_sqrt(bound.Ball.exact(-1))

    def test_tolerance_search_finds_adjacent_certificates(self):
        def evaluate(n, steps):
            return dict(steps=steps, population_upper_bound=16 / steps,
                        lower_witness=dict(error=16 / steps))
        for tolerance in (.1, .03, .003):
            passing, failing = threshold(7, tolerance, evaluate, 600)
            self.assertEqual(passing['steps'], failing['steps'] + 1)
            self.assertLessEqual(passing['population_upper_bound'], tolerance)
            self.assertGreater(failing['lower_witness']['error'], tolerance)

    def test_large_steps_certificate_covers_original_channel(self):
        n, steps = 3, 281
        result = bound.certificate(n, steps, tolerance=.1, block_steps=1, degree=1, bernstein_degree=1)
        _, initial = randomized.initial_covariances(n)
        dt = 12 / steps
        channel = randomized.covariance_branches(n, 10, 20, .25, .25, dt, 'rate-weighted', 'jump-first')
        counts = np.arange(steps + 1)
        C, _ = randomized.mean_trajectory(initial, channel, counts)
        exact = randomized.exact_covariances(initial, n, 10, 20, .25, .25, counts * dt)
        population, _ = randomized.observables(C - exact, 5)
        observed = float(np.max(np.abs(population)))
        self.assertLessEqual(observed, result['population_upper_bound'])
        self.assertLess(result['population_upper_bound'] - observed, 1e-8)
        self.assertTrue(result['certified_population'])

    def test_ball_matrix_products_contain_exact_rational_results(self):
        left = [[Fraction(1, 3), Fraction(-2, 7)], [Fraction(5, 11), Fraction(13, 17)]]
        right = [[Fraction(-3, 5), Fraction(7, 19)], [Fraction(23, 29), Fraction(-31, 37)]]
        def balls(values):
            packed = [[bound.Ball.rational(value) for value in row] for row in values]
            return bound.Ball([[x.center for x in row] for row in packed],
                              [[x.radius for x in row] for row in packed])
        result = balls(left) @ balls(right)
        for i in range(2):
            for j in range(2):
                exact = sum(left[i][k] * right[k][j] for k in range(2))
                error = abs(Fraction.from_float(float(result.center[i, j])) - exact)
                self.assertLessEqual(error, Fraction.from_float(float(result.radius[i, j])))

    def test_bernstein_coefficients_enclose_binomial_polynomial(self):
        length, degree, bern_degree = 16, 10, 32
        change = bound.bernstein_transform(length, degree, bern_degree)
        coefficients = [Fraction((-1)**j, 13**j) for j in range(degree + 1)]
        packed = [bound.Ball.rational(x) for x in coefficients]
        values = change @ bound.Ball(np.asarray([x.center for x in packed])[:, None],
                                    np.asarray([x.radius for x in packed])[:, None])
        lower = np.min(values.center - values.radius)
        upper = np.max(values.center + values.radius)
        for k in range(length + 1):
            exact = sum(Fraction(comb(k, j)) * coefficients[j] for j in range(min(k, degree) + 1))
            self.assertGreaterEqual(float(exact), lower)
            self.assertLessEqual(float(exact), upper)
            u = Fraction(k, length)
            interpolated = sum(comb(bern_degree, j) * float(u**j * (1 - u)**(bern_degree - j)) * values.center[j, 0]
                               for j in range(bern_degree + 1))
            self.assertAlmostEqual(interpolated, float(exact), places=12)

    def test_validated_maps_match_original_complex_channel(self):
        n, steps = 3, 6000
        P, Q, initial, *_ = bound.model_maps(n, steps)
        dt = 12 / steps
        channel = randomized.covariance_branches(n, 10, 20, .25, .25, dt, 'rate-weighted', 'jump-first')
        phases = np.diag((1j)**np.arange(n))
        _, initial_C = randomized.initial_covariances(n)
        result = randomized.apply_mean(initial_C, channel) - .25 * np.eye(n)
        gauged = phases.conj().T @ result @ phases
        expected = gauged.reshape(3, -1).T.real
        actual = P @ initial
        self.assertTrue(np.all(np.abs(actual.center - expected) < actual.radius + 2e-15))
        k = np.diag(np.full(n - 1, 5), 1) - np.diag(np.full(n - 1, 5), -1)
        a = k - np.diag([10, 0, 10])
        generator = np.kron(a, np.eye(n)) + np.kron(np.eye(n), a)
        np.testing.assert_array_less(np.abs(Q.center - expm(dt * generator)), Q.radius)

    def test_bounds_contain_every_step_and_are_tight(self):
        # Independent complex eigensolver/reference; certificate uses only
        # real Taylor arithmetic and Bernstein polynomial enclosures.
        for n in (3, 7):
            steps = 6000
            result = bound.certificate(n, steps)
            _, initial = randomized.initial_covariances(n)
            dt = 12 / steps
            channel = randomized.covariance_branches(n, 10, 20, .25, .25, dt, 'rate-weighted', 'jump-first')
            counts = np.arange(steps + 1)
            C, _ = randomized.mean_trajectory(initial, channel, counts)
            exact = randomized.exact_covariances(initial, n, 10, 20, .25, .25, counts * dt)
            population, current = randomized.observables(C - exact, 5)
            observed_pop = float(np.max(np.abs(population)))
            observed_current = float(np.max(np.abs(current)) / 10)
            self.assertLessEqual(observed_pop, result['population_upper_bound'])
            self.assertLessEqual(observed_current, result['normalized_current_upper_bound'])
            self.assertLess(result['population_upper_bound'] - observed_pop, 5e-6)
            self.assertTrue(result['certified_population'])
            self.assertLess(result['diagnostics']['final_propagation_radius_F'], 1e-8)

    def test_lower_witness_certifies_an_actual_violation(self):
        n, steps = 7, 5400
        result = bound.certificate(n, steps)
        self.assertTrue(result['certified_violation'])
        witness = result['lower_witness']
        _, initial = randomized.initial_covariances(n)
        dt = 12 / steps
        channel = randomized.covariance_branches(n, 10, 20, .25, .25, dt, 'rate-weighted', 'jump-first')
        C, _ = randomized.mean_trajectory(initial, channel, np.array([witness['step']]))
        exact = randomized.exact_covariances(initial, n, 10, 20, .25, .25, [witness['step'] * dt])
        site = witness['site'] - 1
        error = abs((C - exact)[1, 0, site, site].real)
        self.assertGreater(error, .003)
        self.assertGreaterEqual(error, witness['error'])


if __name__ == '__main__':
    unittest.main()
