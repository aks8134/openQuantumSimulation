"""One-shot estimators and job grouping; synthetic counts only, no hardware."""
import unittest
import numpy as np
from Model3.Experiment5 import analyze_one_shot as analysis


class OneShotAnalysisTests(unittest.TestCase):
    def test_single_shot_bit_order_both_current_parities_and_correlated_mean(self):
        records, observations = [], []
        for path, counts in enumerate((("001", "000", "100"), ("010", "001", "000"))):
            for basis, bits in zip(("z", "xy", "yx"), counts):
                records.append({"trajectory": path, "basis": basis})
                observations.append({"outcome_counts": {bits: 1}})
        indices, values = analysis.reconstruct(records, observations, 3, .5, 2.)
        self.assertEqual(indices, [0, 1])
        np.testing.assert_allclose(values, [[1, 0, 0, 0, 2], [0, 1, 0, 2, 0]])
        mean, covariance = analysis.statistics(values)
        np.testing.assert_allclose(mean, [.5, .5, 0, 1, 1])
        np.testing.assert_allclose(np.sqrt(np.diag(covariance)), [.5, .5, 0, 1, 1])
        np.testing.assert_allclose(covariance[3:, 3:], [[1, -1], [-1, 1]])
        self.assertAlmostEqual(covariance[3:, 3:].sum()/4, 0.)

    def test_bernoulli_variance_is_estimated_across_paths(self):
        values = np.array([[0], [1], [1], [0], [1]], float)
        mean, covariance = analysis.statistics(values)
        self.assertAlmostEqual(covariance[0, 0], mean[0]*(1-mean[0])/(len(values)-1))
        with self.assertRaisesRegex(ValueError, "two independent"):
            analysis.statistics([[1.]])

    def test_invalid_counts_and_incomplete_or_duplicate_bases_are_rejected(self):
        records = [{"trajectory": 0, "basis": basis} for basis in ("z", "xy", "yx")]
        for counts in ({}, {"000": 0}, {"000": 2}, {"00x": 1}, {"00": 1}, {"000": True}, {"000": -1}):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                analysis.reconstruct(records, [{"outcome_counts": counts}]*3, 3, .5, 1.)
        rows = [{"outcome_counts": {"000": 1}}]*3
        with self.assertRaisesRegex(ValueError, "three bases"):
            analysis.reconstruct(records[:2], rows[:2], 3, .5, 1.)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analysis.reconstruct([records[0]]*3, rows, 3, .5, 1.)

    def test_job_clustering_excludes_only_split_trajectories_and_retains_shift(self):
        def triple(path):
            return [{"trajectory": path, "basis": basis} for basis in ("z", "xy", "yx")]
        records = triple(0)+triple(1)+triple(6)[:1]+triple(6)[1:]+triple(2)+triple(3)+triple(4)+triple(5)
        batches = [{"start": a, "stop": b, "job_id": str(i)} for i, (a, b) in enumerate(((0, 7), (7, 15), (15, 21)))]
        values = np.zeros((7, 3)); values[:, 2] = [0, 0, 1, 1, 2, 2, 99]
        result = analysis.batch_diagnostics(values, np.zeros_like(values), list(range(7)), records, batches, 2)
        self.assertEqual(result["complete_paths"], 6)
        self.assertEqual(result["excluded_split_paths"], 1)
        self.assertEqual([row["complete_paths"] for row in result["jobs"]], [2, 2, 2])
        self.assertAlmostEqual(result["complete_path_hardware_mean"][2], 1.)
        self.assertAlmostEqual(result["mean_current_cluster_se"], 1/np.sqrt(3))
        self.assertAlmostEqual(result["paired_mean_current_cluster_se"], 1/np.sqrt(3))
        self.assertIsNone(result["job_spread_to_within_job_se_ratio"])


if __name__ == "__main__":
    unittest.main()
