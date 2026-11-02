"""Model2 annealing: independent channel reference and acceptance checks."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from library import reservoir_scaling as physics
from Model2.Experiment2 import randomized
from Model2.Experiment4 import cosine_search as experiment
from Model2.Experiment4 import analyze_results as verifier
from Model2.Experiment4 import refine_warped
from Model2.Experiment4 import parameter_study


class Model2CosineTests(unittest.TestCase):
    def config(self, *extra):
        return experiment.parse_arguments(["--n-qubits", "3", "--iterations", "3",
            "--optimizer-seeds", "1729", "--neighbor-counts", "1", "--local-iterations", "10",
            "--step-budget", "24", "--forward-window", ".5", "--no-plots", *extra])[1]

    def test_defaults_match_requested_preparation_and_model2_normalization(self):
        _, config = experiment.parse_arguments([])
        self.assertEqual(config["preparation"], "half_mixed")
        self.assertEqual(config["p"], .25)
        self.assertEqual(config["exchange_prefactor"], .25)
        self.assertEqual(config["gamma"]/config["J"], 2)
        self.assertAlmostEqual(config["mean_tolerance"], .016)
        for args in (["--gamma", "nan"], ["--p", "1"], ["--sampling-tolerance", ".03"],
                     ["--minimum-dt", "3"], ["--trajectories", "1"], ["--optimizer-seeds", "-1"]):
            with self.subTest(args=args), self.assertRaises(ValueError):
                experiment.parse_arguments(args)

    def test_equal_target_is_fixed_even_for_negative_collision_cosines(self):
        model = physics.Model(5, 2, .25, .25, .25)
        for order in ("jump-first", "hamiltonian-first"):
            for formula in ("exact", "even-odd"):
                config = self.config("--order", order, "--system-formula", formula)
                for dt in (.03, .7, 2.):
                    channel = experiment.branch(model, dt, config)
                    target = .25*np.eye(5)
                    np.testing.assert_allclose(physics.mean_step(target, channel), target, atol=1e-14)

    def test_variable_steps_match_independent_complex_model2_reference(self):
        model = physics.Model(4, 2, .25, .25, .25)
        phase = (-1j)**np.arange(4)
        for order in ("jump-first", "hamiltonian-first"):
            config = self.config("--system-formula", "exact", "--order", order)
            dts = [.13, 1.7, .31, .07]
            real, _ = experiment.evolve(model, dts, config)
            complex_state = .5*np.eye(4, dtype=complex)
            for dt in dts:
                channel = randomized.covariance_branches(4, 1, 2, .25, .25, dt, "rate-weighted", order)
                complex_state = randomized.apply_mean(complex_state, channel)
            gauge = phase[:, None]*phase.conj()[None, :]
            np.testing.assert_allclose(real*gauge, complex_state, atol=3e-15)

    def test_batched_parameter_physics_matches_affine_channel_for_wide_steps_and_rates(self):
        rng = np.random.default_rng(73)
        for n in (2, 3, 10):
            for rate in (.25, 2., 8.):
                for order in ("jump-first", "hamiltonian-first"):
                    for p in (.1, .25, .4):
                        config = self.config("--n-qubits", str(n), "--p", str(p), "--order", order)
                        config["gamma"] = rate
                        model = physics.Model(n, rate, p, p, .25)
                        dts = rng.uniform(.001, 8., 20)
                        actual, _ = experiment.evolve(model, dts, config)
                        fast = p*np.eye(n)+(.5-p)*parameter_study.fast_output(dts, config)
                        np.testing.assert_allclose(fast, actual, atol=2e-14)

    def test_rate_tuned_swap_channel_prepares_uniform_error_in_twenty_steps(self):
        config = self.config("--n-qubits", "10")
        config["gamma"] = np.pi/8
        model = physics.Model(10, np.pi/8, .25, .25, .25)
        result = experiment.evaluate(model, np.full(20, np.pi), config)
        np.testing.assert_allclose(result["output_covariance"], np.eye(10)*.265625, atol=2e-15)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["permanent_local_error_bound"], .015625)

    def test_determinant_contraction_bound_for_random_positive_covariances(self):
        rng = np.random.default_rng(17)
        for rate in (.25, 2., 8.):
            U, W = parameter_study.matrices([.03, .7, 3., 7.], 5, rate)
            for u, w in zip(U, W):
                A = rng.normal(size=(5, 5))
                K = A@A.T+np.eye(5)
                before = np.linalg.slogdet(K)[1]
                after = np.linalg.slogdet(parameter_study.advance(K, u, w))[1]
                self.assertGreaterEqual(after, before-np.log(4)-1e-12)

    def test_forward_validation_catches_intermediate_failure(self):
        model = physics.Model(3, 2, .25, .25, .25)
        config = self.config()
        target = .25*np.eye(3)
        with patch.object(physics, "mean_step", side_effect=[target+.2*np.eye(3), target]):
            result = experiment.continuation(model, target, .25, config)
        self.assertFalse(result["passed"])
        self.assertEqual(result["errors"], [0., .2, 0.])

    def test_permanent_bound_covers_later_variable_steps(self):
        model = physics.Model(3, 2, .25, .25, .25)
        config = self.config()
        result = experiment.evaluate(model, np.full(40, .7), config)
        state = np.asarray(result["output_covariance"])
        for dt in [.01, 2., .1, 1.3]*10:
            state = physics.mean_step(state, experiment.branch(model, dt, config))
            self.assertLessEqual(experiment.local_error(state, .25*np.eye(3), model),
                                 result["permanent_local_error_bound"]+1e-14)

    def test_permanent_order_check_accepts_decrease_and_rejects_zero_dissipation(self):
        config = self.config()
        model = physics.Model(3, 2, .25, .25, .25)
        result = experiment.evaluate(model, np.full(30, .7), config)
        state = np.asarray(result["output_covariance"])
        check = verifier.permanent_order_check(state, .7, config)
        self.assertTrue(check["matrix_order_established"])
        for _ in range(100):
            state = verifier.explicit_step(state, .7, config)
            self.assertLessEqual(verifier.distance(state, config), check["permanent_local_error_bound"]+1e-14)
        # theta=2pi removes all bath population transfer and gives M(I)=I.
        no_transfer_dt = np.pi**2
        check = verifier.permanent_order_check(.5*np.eye(3), no_transfer_dt, config)
        self.assertFalse(check["matrix_order_established"])

    def test_sampling_reproducible_and_trajectory_prefixes_preserved(self):
        model = physics.Model(3, 2, .25, .25, .25)
        config = self.config()
        selected = experiment.evaluate(model, np.array([.7, .3, .1]), config)
        first = experiment.sampled_replay(model, selected, config, 29, 8)
        self.assertEqual(first, experiment.sampled_replay(model, selected, config, 29, 8))
        saved = []
        original = physics.sampled_step
        def capture(states, choices, channel):
            value = original(states, choices, channel)
            saved.append(value.copy())
            return value
        with patch.object(physics, "sampled_step", side_effect=capture):
            experiment.sampled_replay(model, selected, config, 29, 4)
            small = saved[-1]
            experiment.sampled_replay(model, selected, config, 29, 9)
        np.testing.assert_array_equal(small, saved[-1][:, :4])

    def test_failed_search_retains_no_unvalidated_selection(self):
        model = physics.Model(3, 2, .25, .25, .25)
        config = self.config("--step-budget", "1", "--total-tolerance", "1e-8",
                             "--sampling-tolerance", "1e-9")
        result = experiment.search_family(model, "cosine", config)
        self.assertIsNone(result["selected"])
        self.assertEqual(result["status"], "no_passing_schedule_found")

    def test_warped_refinement_resume_preserves_completed_trials(self):
        config = self.config("--n-qubits", "2", "--plan-only")
        model = physics.Model(2, 2, .25, .25, .25)
        baseline = experiment.evaluate(model, np.full(20, .7), config)
        baseline.update(family="cosine", parameters=[float(np.log(.7))]*2)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            source = Path(folder)/"source"
            source.mkdir()
            physics.atomic_json(source/"study.json", {"configuration": config,
                "families": {"cosine": {"selected": baseline}}})
            args = [str(source), "--iterations", "1", "--counts-below", "1"]
            with patch.object(experiment, "plot"):
                first = refine_warped.main(args)
                with patch.object(refine_warped, "differential_evolution", side_effect=AssertionError("completed trials must be reused")):
                    second = refine_warped.main(args+["--resume"])
            self.assertEqual(first, second)
            self.assertTrue(second["complete"])
            self.assertEqual(len(second["families"]["warped_cosine"]["trials"]), 1)

    def test_saved_small_search_validates_forward_and_retains_failed_sampling(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            result = experiment.main(["--n-qubits", "2", "--families", "cosine", "--step-budget", "20",
                "--iterations", "4", "--optimizer-seeds", "1729", "--neighbor-counts", "0",
                "--forward-window", ".5", "--trajectories", "8", "--sampling-seeds", "29",
                "--no-plots", "--output-directory", folder])
            selection = result["families"]["cosine"]["selected"]
            self.assertTrue(selection["passed"])
            self.assertGreaterEqual(selection["forward"]["duration"], .5)
            saved = json.loads(next(Path(folder).glob("*/study.json")).read_text())
            self.assertEqual(saved, result)
            self.assertEqual(len(saved["families"]["cosine"]["replays"]), 1)
            self.assertFalse(saved["families"]["cosine"]["sampling_passed"])


if __name__ == "__main__":
    unittest.main()
