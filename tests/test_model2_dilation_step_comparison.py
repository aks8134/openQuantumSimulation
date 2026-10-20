"""Check the accelerated channel on mixed states with number-sector coherences."""

import unittest

import numpy as np

from Model2.Experiment1 import simulation
from Model2.Experiment1.analysis import dilation_step_comparison as comparison


class DilationStepComparisonTests(unittest.TestCase):
    def test_channel_powers_match_direct_iteration(self):
        H, jumps, *_ = simulation.build_model(3, J=2, gamma=4, delta=0.4)
        sectors = comparison.number_sectors(3)
        _, initial = simulation.initial_states(3)
        vectors = np.concatenate([block.reshape(3, -1) for block in comparison.extract_blocks(initial, sectors)], axis=1)
        for formula in ('lie', 'strang'):
            with self.subTest(formula=formula):
                prepared = comparison.prepare_sparse_channel(comparison.analytical_kraus(H, jumps, 0.003, formula), sectors)
                powered, diagnostics = comparison.channel_powers(prepared['operator'], vectors, np.arange(101), 0.003, sectors)
                direct = vectors.T.copy()
                for index in range(101):
                    np.testing.assert_allclose(powered[:, index], direct.T, atol=2e-12, rtol=0)
                    direct = prepared['operator'] @ direct
                self.assertLess(diagnostics['residual_trace_norm_bound'], 1e-9)

    def test_accelerated_channels_preserve_mixed_state_evolution(self):
        n = 3
        H, jumps, *_ = simulation.build_model(n, J=2, gamma=4, delta=0.4)
        sectors = comparison.number_sectors(n)
        random = np.random.default_rng(20261011)
        initial = np.zeros((3, 2**n, 2**n), complex)
        for state in initial:
            for indices in sectors:
                matrix = random.normal(size=(len(indices), len(indices))) + 1j * random.normal(size=(len(indices), len(indices)))
                state[np.ix_(indices, indices)] = matrix @ matrix.conj().T
            state /= np.trace(state)
        for formula in ("lie", "strang"):
            for dt in (0.003, 0.03):
                with self.subTest(formula=formula, dt=dt):
                    original = simulation.dilation_kraus(H, jumps, dt, formula)
                    analytical = comparison.analytical_kraus(H, jumps, dt, formula)
                    np.testing.assert_allclose(analytical, original, atol=2e-14, rtol=0)
                    prepared = comparison.prepare_sparse_channel(analytical, sectors)
                    blocks = comparison.extract_blocks(initial, sectors)
                    expected = initial.copy()
                    for _ in range(5):
                        blocks = comparison.apply_channel(blocks, prepared)
                        expected = sum(K @ expected @ K.conj().T for K in original)
                    actual = comparison.full_density(blocks, sectors)
                    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=0)
                    np.testing.assert_allclose(np.trace(actual, axis1=-2, axis2=-1), 1, atol=2e-13, rtol=0)
                    self.assertGreater(np.linalg.eigvalsh(actual).min(), -2e-13)


if __name__ == "__main__":
    unittest.main()
