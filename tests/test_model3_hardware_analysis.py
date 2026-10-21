"""Saved-count reconstruction and uncertainty propagation without Runtime."""
import hashlib
import unittest

import numpy as np

from Model3.Experiment5 import analyze_hardware as analysis


class HardwareAnalysisTests(unittest.TestCase):
    def test_source_integrity_allows_only_proven_line_ending_conversion(self):
        lf = b'{\n  "value": 0.9\n}\n'
        crlf = lf.replace(b'\n', b'\r\n')
        digest = hashlib.sha256(crlf).hexdigest()
        self.assertEqual(analysis.source_integrity(crlf, digest), 'exact_bytes')
        self.assertEqual(analysis.source_integrity(lf, digest), 'matches_original_CRLF_line_endings')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            analysis.source_integrity(lf.replace(b'0.9', b'0.8'), digest)

    def test_histogram_covariance_matches_explicit_samples(self):
        counts = {'001': 2, '110': 3, '010': 1, '111': 2}
        for pairs in (False, True):
            samples = []
            for key, count in counts.items():
                bits = np.array([int(bit) for bit in key[::-1]])
                signs = 1-2*bits
                values = signs[:-1]*signs[1:] if pairs else bits
                samples.extend([values]*count)
            samples = np.asarray(samples)
            mean, covariance, shots = analysis.moments(counts, 3, pairs=pairs)
            np.testing.assert_allclose(mean, samples.mean(axis=0), atol=1e-14)
            np.testing.assert_allclose(covariance, np.cov(samples.T, ddof=1)/len(samples), atol=1e-14)
            self.assertEqual(shots, len(samples))

    def test_current_signs_both_parities_and_fixed_path_averaging(self):
        records, rows = [], []
        for path in (0, 1):
            for basis, counts in (('z', {'001': 1, '010': 3}),
                                  ('xy', {'000': 3, '001': 1}),
                                  ('yx', {'100': 2, '000': 2})):
                records.append({'trajectory': path, 'basis': basis})
                rows.append({'outcome_counts': counts})
        result = analysis.reconstruct(records, rows, 3, .5, 2.)
        np.testing.assert_allclose(result['populations'], [[.25, .75, 0.]]*2)
        np.testing.assert_allclose(result['excitation_flows'], [[.5, 1.]]*2)
        np.testing.assert_allclose(np.diag(result['current_shot_covariance']), [.125, 1/6])
        np.testing.assert_allclose(np.diag(result['population_shot_covariance']), [.03125, .03125, 0.])
        self.assertEqual(result['indices'], [0, 1])

    def test_mean_current_uncertainty_retains_bond_correlations(self):
        records = [{'trajectory': 0, 'basis': basis} for basis in ('z', 'xy', 'yx')]
        rows = [{'outcome_counts': counts} for counts in (
            {'000': 4}, {'000': 2, '101': 2}, {'000': 4})]
        result = analysis.reconstruct(records, rows, 3, .5, 2.)
        np.testing.assert_allclose(result['current_shot_covariance'], [[1/3, -1/3], [-1/3, 1/3]])
        self.assertAlmostEqual(result['current_shot_covariance'].sum(), 0.)

    def test_invalid_or_incomplete_data_rejected(self):
        for counts in ({}, {'000': 1}, {'00x': 4}, {'000': -1}, {'00': 3}):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                analysis.moments(counts, 3)
        with self.assertRaisesRegex(ValueError, 'all three bases'):
            analysis.reconstruct([{'trajectory': 0, 'basis': 'z'}], [{'outcome_counts': {'000': 4}}], 3, .5, 1.)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            analysis.reconstruct([{'trajectory': 0, 'basis': 'z'}]*2, [{'outcome_counts': {'000': 4}}]*2, 3, .5, 1.)

    def test_original_metric_excludes_duplicate_magnetization_scaling(self):
        left = analysis.as_observables([.51, .49], [.3])
        right = analysis.as_observables([.5, .5], [.32])
        values = analysis.metric(left, right, 2.)
        self.assertAlmostEqual(values['maximum'], .01)
        self.assertEqual(values['worst_bond_1based'], [1, 2])


if __name__ == '__main__':
    unittest.main()
