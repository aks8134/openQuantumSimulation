import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Model1.Experiment11 import physics
from Model1.Experiment11 import classical_randomized_lie_trotter as classical


class ClassicalDynamicsTests(unittest.TestCase):
    def test_matches_small_ideal_circuit_reset_channels(self):
        from qiskit.quantum_info import DensityMatrix, partial_trace
        from IBMRuntime.interpreters.qiskit import _to_qiskit
        for n, gamma, choices, dts in ((2, 20.0, (0, 1, 0), (0.1, 0.1, 0.03)),
                                      (3, 0.7, (1, 0, 1), (0.13, 0.07, 0.02))):
            model = physics.Model(n, J=-0.8, h=0.3, gamma=gamma, initial_site=0)
            amplitude = np.zeros((1, n), dtype=complex)
            amplitude[0, 0] = 1
            for dt, boundary in zip(dts, choices):
                amplitude = classical.advance(amplitude, model, dt, [boundary])
            expected = np.zeros((2**n, 2**n), dtype=complex)
            slots = [1 << (n-1-site) for site in range(n)]
            expected[np.ix_(slots, slots)] = np.outer(amplitude[0], amplitude[0].conj())
            expected[0, 0] = 1 - np.vdot(amplitude[0], amplitude[0]).real
            compact = physics.build_circuit(model, dts, choices, "Z")
            for circuit in (compact, physics.prepare_for_compilation(compact)):
                native = _to_qiskit(circuit).remove_final_measurements(inplace=False)
                density = DensityMatrix.from_label("0"*(n+2)).evolve(native)
                reduced = partial_trace(density, (0, n+1))
                np.testing.assert_allclose(reduced.data, expected, atol=1e-12)

    def test_large_angle_keeps_negative_cosine_and_does_not_normalize(self):
        model = physics.Model(2, J=0, gamma=20, initial_site=0)
        initial = np.array([[1, 0]], dtype=complex)
        result = classical.advance(initial, model, 0.1, [0])
        self.assertLess(result[0, 0].real, 0)
        np.testing.assert_allclose(result, [[np.cos(2), 0]], atol=1e-14)
        np.testing.assert_array_equal(initial, [[1, 0]])
        self.assertLess(np.linalg.norm(result), 1)

    def test_loss_free_two_site_evolution_matches_exact_reference(self):
        model = physics.Model(2, J=0.8, h=-0.7, gamma=0, initial_site=1)
        result = classical.evolve(model, [0, 0.13, 0.7], trajectories=4, maximum_dt=0.1)
        for actual, expected in zip(result.means, physics.exact_reference(model, result.times)):
            np.testing.assert_allclose(actual, expected, atol=1e-12)
        for error in result.standard_errors:
            np.testing.assert_allclose(error, 0, atol=1e-14)
        np.testing.assert_allclose(result.trajectory_excitation, 1, atol=1e-12)

    def test_same_seed_schedule_remainders_and_no_baseline(self):
        times = [0.7]
        result = classical.evolve(physics.Model(5), times, trajectories=4, seed=29, maximum_dt=0.2)
        schedule = physics.schedule(times, 4, 29, 0.2)
        self.assertEqual((result.step_dts, result.step_counts, result.boundary_choices), schedule)
        np.testing.assert_allclose(result.step_dts, (0.2, 0.2, 0.2, 0.1), atol=1e-15)
        self.assertEqual(result.means[0].shape, (5, 1))
        self.assertEqual(result.times.tolist(), [0.7])

    def test_trajectory_statistics_match_analytic_no_hopping_case(self):
        model = physics.Model(2, J=0, gamma=0.7, initial_site=0)
        result = classical.evolve(model, [0.4], trajectories=8, maximum_dt=0.1)
        expected = np.array([np.prod([np.cos(np.sqrt(2*model.gamma*dt))**2
            if boundary == 0 else 1 for dt, boundary in zip(result.step_dts, row)])
            for row in result.boundary_choices])
        np.testing.assert_allclose(result.trajectory_excitation[:, 0], expected, atol=1e-14)
        self.assertAlmostEqual(result.means[0][0, 0], expected.mean())
        self.assertAlmostEqual(result.standard_errors[0][0, 0], expected.std(ddof=1)/np.sqrt(8))

    def test_large_N_state_is_only_R_by_N_and_norm_is_monotone(self):
        result = classical.evolve(physics.Model(256, gamma=20), [0, 0.1, 0.2], trajectories=2)
        self.assertEqual(result.final_amplitudes.shape, (2, 256))
        self.assertTrue(np.all(np.diff(result.trajectory_excitation, axis=1) <= 1e-12))
        for data in result.means:
            self.assertTrue(np.all(np.isfinite(data)))

    def test_invalid_inputs(self):
        for extra in (("--gamma", "-1"), ("--J", "nan"), ("--trajectories", "0"),
                      ("--trotter-delta-t", "0"), ("--steady-tolerance", "1"),
                      ("--times", "0.2", "0.1"), ("--times", "0.1", "--time-points", "3")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                classical.parse_arguments(extra)


class PersistenceTests(unittest.TestCase):
    def test_append_replace_times_and_configuration_guards(self):
        options = classical.parse_arguments(("--n-qubits", "2"))
        cfg = classical.configuration(options)
        first = classical.evolve(options.model, [0, 0.1], trajectories=4)
        payload = classical.make_payload(first, cfg)
        second = classical.evolve(options.model, [0.1, 0.2], trajectories=4)
        updated = classical.make_payload(second, cfg, existing=payload)
        self.assertEqual(updated["times"], [0, 0.1, 0.2])
        self.assertEqual(len(updated["samples"]), 3)
        self.assertEqual(updated["samples"][0], payload["samples"][0])
        np.testing.assert_allclose(np.asarray(updated["observables"]["populations"])[:, 1:], second.means[0])
        self.assertEqual(len(updated["runs"]), 2)
        changed = {**cfg, "physics": {**cfg["physics"], "gamma": 5}}
        with self.assertRaisesRegex(ValueError, "incompatible"):
            classical.make_payload(second, changed, existing=payload)

    def test_reference_and_data_saved_before_failed_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            options = classical.parse_arguments(("--n-qubits", "2", "--times", "0.1", "--no-progress",
                "--optimized-classical-reference", "--output-directory", directory))
            with patch.object(classical, "plot_result", side_effect=RuntimeError("plot failed")):
                with self.assertRaisesRegex(RuntimeError, "plot failed"):
                    classical.main(options)
            path = classical.output_paths(options, classical.configuration(options))["result"]
            payload = json.loads(path.read_text())
            self.assertEqual(payload["figure_generation_status"], "pending")
            self.assertEqual(payload["reference_generation_status"], "complete")
            self.assertEqual(payload["times"], [0.1])
            self.assertIn("exact_reference_observables", payload)

    def test_single_trajectory_does_not_claim_estimated_uncertainty(self):
        options = classical.parse_arguments(("--trajectories", "1"))
        payload = classical.make_payload(classical.evolve(options.model, [0.1], trajectories=1),
                                         classical.configuration(options))
        self.assertFalse(payload["uncertainty"]["trajectory_standard_error_estimable"])
        self.assertFalse(payload["uncertainty"]["measurement_shot_noise"])


if __name__ == "__main__":
    unittest.main()
