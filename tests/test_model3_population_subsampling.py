"""Finite-pool resampling checks using exactly enumerable small examples."""
import unittest

import numpy as np

from Model3.Experiment5 import subsample_populations as analysis


class PopulationSubsamplingTests(unittest.TestCase):
    def test_whole_bitstring_expansion_respects_little_endian_sites(self):
        pool = analysis.expand_counts({"0 01": 2, "110": 1}, 3)
        np.testing.assert_array_equal(pool, [[1, 0, 0], [1, 0, 0], [0, 1, 1]])
        for counts in ({}, {"001": 0}, {"001": -1}, {"001": 1.5}, {"00x": 2}, {"00": 2}):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                analysis.expand_counts(counts, 3)

    def test_uniform_subsets_match_enumerated_distribution_and_site_correlations(self):
        # Six possible two-shot subsets: means 0, 1/2, 1 with weights 1,4,1.
        pools = np.array([[[0, 0], [0, 0], [1, 1], [1, 1]]], dtype=np.uint8)
        draws = analysis.subsample(pools, [1, 2, 3, 4], 20000, seed=27)
        np.testing.assert_array_equal(draws[..., 0], draws[..., 1])
        two_shot = draws[:, 1, 0, 0]
        for value, expected in ((0., 1/6), (.5, 4/6), (1., 1/6)):
            self.assertAlmostEqual(np.mean(two_shot == value), expected, delta=.012)
        self.assertAlmostEqual(two_shot.mean(), .5, delta=.006)
        self.assertAlmostEqual(two_shot.var(ddof=1), 1/12, delta=.003)
        covariance = analysis.finite_population_covariance(pools, [1, 2, 3, 4])
        np.testing.assert_allclose(covariance[1], np.full((2, 2), 1/12))
        self.assertTrue(np.isin(draws[:, 2, 0], [1/3, 2/3]).all())
        np.testing.assert_array_equal(draws[:, -1], np.full((20000, 1, 2), .5))
        np.testing.assert_array_equal(covariance[-1], np.zeros((2, 2)))

    def test_independent_path_sampling_reduces_covariance(self):
        pools = np.array([[[0, 1], [0, 1], [1, 0], [1, 0]]]*2, dtype=np.uint8)
        draws = analysis.subsample(pools, [2, 4], 12000, seed=19)
        pooled = draws[:, 0].mean(axis=1)
        expected = np.array([[1, -1], [-1, 1]])/24
        np.testing.assert_allclose(analysis.finite_population_covariance(pools, [2])[0], expected)
        np.testing.assert_allclose(np.cov(pooled, rowvar=False), expected, atol=.002)
        self.assertLess(abs(np.cov(draws[:, 0, :, 0], rowvar=False)[0, 1]), .003)

    def test_invalid_levels_and_nonbinary_pools_rejected(self):
        pools = np.zeros((1, 4, 2), dtype=np.uint8)
        for levels in ([0], [5], [2, 2], [3, 1], [1.5], []):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                analysis.subsample(pools, levels, 10, seed=1)
        with self.assertRaises(ValueError):
            analysis.subsample(pools+2, [1], 10, seed=1)


if __name__ == "__main__":
    unittest.main()
