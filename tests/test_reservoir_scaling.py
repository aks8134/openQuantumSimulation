import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from library import classical
from library import reservoir_scaling as study
from Model2.Experiment2 import randomized
from Model2.Experiment3 import study_randomized_scaling as model2
from Model3.Experiment2 import study_randomized_scaling as model3


class ReservoirScalingTests(unittest.TestCase):
    def options(self, *extra):
        return model2.parse_arguments(("--sizes", "2", "--ratios", "2", "--steps", "0.005",
            "--R-values", "4", "8", "--seeds", "29", "--steady-tolerance", "0.05",
            "--dynamic-tolerance", "0.2", "--timing-tolerance", "0.2",
            "--step-refinements", "0", "--progress-every", "0", *extra))

    def gauge(self, real):
        phases = (-1j)**np.arange(real.shape[-1])
        return phases[:, None] * real * phases.conj()[None, :]

    def test_defaults_preserve_distinct_models_and_normalizations(self):
        left, right = model2.parse_arguments([]).config, model3.parse_arguments([]).config
        self.assertEqual((left["p_left"], left["p_right"], left["exchange_prefactor"]), (.25, .25, .25))
        self.assertEqual((right["p_left"], right["p_right"], right["exchange_prefactor"]), (.9, .1, .5))
        self.assertEqual(left["system_formula"], "exact")
        self.assertTrue(left["finite_R_steady_check"])

    def test_exact_equal_reference_matches_existing_solver(self):
        for n in (2, 3, 5):
            model = study.Model(n, 2.7, .25, .25, .25)
            initial = study.initial_covariances(n)
            expected = randomized.exact_covariances(initial, n, 1, 2.7, .25, .25, [0, .17, 2])
            actual = np.asarray([self.gauge(study.exact_at(model, t)) for t in (0, .17, 2)])
            np.testing.assert_allclose(actual.swapaxes(0, 1), expected, atol=2e-13)

    def test_exact_unequal_reference_matches_model3_and_stationary_targets(self):
        from Model3.Experiment1 import simulation
        for n in (2, 3, 5):
            model = study.Model(n, 2, .9, .1, .5)
            target = study.exact_stationary(model)
            analytic = simulation.xx_stationary_targets(n)
            np.testing.assert_allclose(np.diag(target), analytic["populations"], atol=2e-14)
            np.testing.assert_allclose(2*np.diag(target, 1), analytic["I"], atol=2e-14)
            times = [0, .13, 1.7]
            labels = ["0"*n, "1"*n, "1"+"0"*(n-1)]
            expected_z, expected_i = simulation.xx_correlations(times, labels)
            trajectory = np.asarray([study.exact_at(model, t) for t in times]).swapaxes(0, 1)
            np.testing.assert_allclose(1-2*np.diagonal(trajectory, axis1=-2, axis2=-1), expected_z, atol=3e-13)
            np.testing.assert_allclose(2*np.diagonal(trajectory, offset=1, axis1=-2, axis2=-1), expected_i, atol=3e-13)

    def test_mean_and_sampled_channels_match_existing_equal_reservoir_branches(self):
        for n, ratio, step in ((2, 1.7, .03), (4, 20, .2)):
            model = study.Model(n, ratio, .25, .25, .25)
            C = study.initial_covariances(n)
            for policy in ("uniform", "rate-weighted"):
                for order in ("jump-first", "hamiltonian-first"):
                    branch = study.channel(model, step, policy, order)
                    expected = randomized.covariance_branches(n, 1, ratio, .25, .25, step, policy, order)
                    np.testing.assert_allclose(self.gauge(study.mean_step(C, branch)),
                        randomized.apply_mean(self.gauge(C), expected), atol=2e-14)
                    paths = np.repeat(C[:, None], 4, axis=1)
                    actual = self.gauge(study.sampled_step(paths, np.arange(4), branch))
                    for jump in range(4):
                        A, B = expected["A"][jump], expected["B"][jump]
                        np.testing.assert_allclose(actual[:, jump], A@self.gauge(C)@A.conj().T+B, atol=2e-14)
            if ratio == 20:
                self.assertLess(study.channel(model, step).cosines[0], 0)

    def test_unequal_sampled_covariances_match_full_density_dilation(self):
        from Model3.Experiment1 import simulation
        n = 3
        model = study.Model(n, 1.7, .8, .3, .5)
        H, jumps, *_ = simulation.build_model(n, gamma=model.ratio, p_left=.8, p_right=.3)
        annihilators = []
        for site in range(n):
            matrix = np.array([[1.]])
            for j in range(n):
                term = classical.Z.toarray() if j < site else classical.SIGMA_MINUS.toarray() if j == site else np.eye(2)
                matrix = np.kron(matrix, term)
            annihilators.append(matrix)
        covariance_ops = [[annihilators[j].conj().T@annihilators[i] for j in range(n)] for i in range(n)]
        # Successive different channels create nontrivial coherences before later branches.
        for order in ("jump-first", "hamiltonian-first"):
            branch = study.channel(model, .13, order=order)
            rho = np.zeros((2**n, 2**n), complex); rho[0, 0] = 1
            C = study.initial_covariances(n)[0][None, None]
            for jump in (0, 2, 1, 0, 3):
                kraus = randomized.sampled_kraus(H, jumps[jump], branch.probabilities[jump], .13, order)
                rho = sum(K@rho@K.conj().T for K in kraus)
                C = study.sampled_step(C, [jump], branch)
                expected = np.asarray([[np.trace(rho@op) for op in row] for row in covariance_ops])
                np.testing.assert_allclose(self.gauge(C[0, 0]), expected, atol=3e-14)

    def test_mean_stationary_bias_and_refinement(self):
        equal = study.Model(3, 2, .25, .25, .25)
        fixed, checks = study.mean_stationary(study.channel(equal, .05), 3)
        np.testing.assert_allclose(fixed, .25*np.eye(3), atol=1e-12)
        self.assertLess(checks["residual"], 1e-12)
        unequal = study.Model(3, 2, .9, .1, .5)
        target = study.exact_stationary(unequal)
        biases = []
        for step in (.08, .04, .02):
            fixed, checks = study.mean_stationary(study.channel(unequal, step), 3)
            biases.append(max(study.observable_error(fixed, target, .5).values()))
        self.assertGreater(biases[0], biases[1])
        self.assertGreater(biases[1], biases[2])

    def test_even_odd_is_optional_and_has_the_right_two_site_limit(self):
        n2 = study.Model(2, 2, .25, .25, .25)
        np.testing.assert_allclose(study.system_unitary(n2, .1, "even-odd"),
                                   study.system_unitary(n2, .1, "exact"), atol=1e-14)
        n4 = study.Model(4, 2, .9, .1, .5)
        large = np.linalg.norm(study.system_unitary(n4, .1, "even-odd")-study.system_unitary(n4, .1, "exact"))
        small = np.linalg.norm(study.system_unitary(n4, .05, "even-odd")-study.system_unitary(n4, .05, "exact"))
        self.assertGreater(large, 3.9*small)

    def test_permanent_envelope_is_monotone_and_bounds_local_errors(self):
        model = study.Model(5, 2.3, .9, .1, .5)
        target = study.exact_stationary(model)
        distances = []
        for tau in (0, .1, 1, 10, 100):
            C = study.exact_at(model, tau)
            value = study.envelope(C, target, model.prefactor)
            distances.append(value)
            self.assertLessEqual(max(study.observable_error(C, target, .5).values()), value+1e-14)
        self.assertTrue(np.all(np.diff(distances) <= 1e-12))
        self.assertIsNone(study.exact_crossing(model, .001, .01))
        crossing = study.exact_crossing(model, .001, 10000)
        self.assertAlmostEqual(study.envelope(study.exact_at(model, crossing), target, .5), .001, places=9)

    def test_evaluated_steps_include_tail_bound_and_caps(self):
        cfg = self.options().config
        model = study.Model(2, 2, .25, .25, .25)
        t = study.exact_crossing(model, cfg["steady_tolerance"], cfg["time_cap"])
        result = study.evaluate_step(model, .005, t, cfg)
        self.assertTrue(result["passed_recommended"])
        self.assertLessEqual(result["tail_error_bound"], cfg["bias_tolerance"])
        self.assertAlmostEqual(np.diff(result["steady_time_bracket"])[0], .005)
        capped = study.evaluate_step(model, .005, t, {**cfg, "step_cap": 1})
        self.assertFalse(capped["passed_recommended"])
        self.assertEqual(capped["status"], "step_cap_reached")

    def test_ensemble_seed_prefixes_and_explicit_caps(self):
        cfg = {**self.options("--no-finite-R-steady-check").config, "progress_every": 0}
        model = study.Model(2, 2, .25, .25, .25)
        t = study.exact_crossing(model, cfg["steady_tolerance"], cfg["time_cap"])
        with contextlib.redirect_stdout(io.StringIO()):
            first = study.evaluate_ensemble(model, .005, t, 16, cfg)
            second = study.evaluate_ensemble(model, .005, t, 16, cfg)
            capped = study.evaluate_ensemble(model, .005, t, 16, {**cfg, "step_cap": 1})
        self.assertEqual(first, second)
        self.assertFalse(capped["passed_all_seeds"])
        memory = study.evaluate_ensemble(model, .005, t, 100000, {**cfg, "memory_limit_mb": .01})
        self.assertEqual(memory["status"], "memory_cap_reached")
        # Actual sampled evolution and draw blocks preserve the first R paths.
        branch = study.channel(model, .005)
        streams4 = [np.random.default_rng(np.random.SeedSequence([29, path])) for path in range(4)]
        streams8 = [np.random.default_rng(np.random.SeedSequence([29, path])) for path in range(8)]
        paths4 = np.repeat(study.initial_covariances(2)[:, None], 4, axis=1)
        paths8 = np.repeat(study.initial_covariances(2)[:, None], 8, axis=1)
        for _ in range(2):
            choices4 = study.draw_choice_block(streams4, branch.probabilities)
            choices8 = study.draw_choice_block(streams8, branch.probabilities)
            np.testing.assert_array_equal(choices4, choices8[:4])
            for k in range(3):
                paths4 = study.sampled_step(paths4, choices4[:, k], branch)
                paths8 = study.sampled_step(paths8, choices8[:, k], branch)
                np.testing.assert_allclose(paths4, paths8[:, :4], atol=1e-14)

    def test_persistence_resume_and_new_tolerance_directory(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args = ("--output-directory", directory, "--no-plots", "--no-finite-R-steady-check")
            result = study.main(self.options(*args))
            paths = list(Path(directory).glob("*/study.json"))
            self.assertEqual(len(paths), 1)
            saved = json.loads(paths[0].read_text())
            self.assertEqual(saved["recommendations"], result["recommendations"])
            with patch.object(study, "evaluate_step", side_effect=AssertionError("must reuse mean cases")), \
                 patch.object(study, "exact_crossing", side_effect=AssertionError("must reuse exact cases")), \
                 patch.object(study, "evaluate_ensemble", side_effect=AssertionError("must reuse R cases")):
                resumed = study.main(self.options(*args, "--resume"))
            self.assertEqual(resumed["recommendations"], result["recommendations"])
            with self.assertRaises(ValueError):
                study.main(self.options(*args))
            study.main(self.options(*args, "--steady-tolerance", "0.06"))
            self.assertEqual(len(list(Path(directory).glob("*/study.json"))), 2)

    def test_report_and_plots_including_unresolved_cases(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = study.main(self.options("--output-directory", directory, "--step-cap", "1"))
            self.assertEqual(result["status"], "complete_with_search_limits")
            folder = next(Path(directory).glob("scaling_*"))
            self.assertIn("NOT full-state trace distance", (folder/"Study.md").read_text())
            for filename in ("scaling.png", "ratio_landscape.png", "target_workload.png"):
                self.assertGreater((folder/filename).stat().st_size, 1000)

    def test_invalid_configuration_and_missing_resume(self):
        for args in (("--sizes", "1"), ("--p", "0"), ("--ratios", "nan"),
                     ("--steady-tolerance", "0"), ("--horizon-factor", "1.1"),
                     ("--seeds", "2", "2"), ("--memory-limit-mb", "0")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                model2.parse_arguments(args)
        with self.assertRaises(ValueError):
            model3.parse_arguments(("--p-left", "1"))
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            study.main(self.options("--output-directory", directory, "--resume"))

    def test_already_settled_initial_envelope_does_not_generate_zero_rates(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = study.main(self.options("--p", "0.5", "--steady-tolerance", "0.9",
                "--output-directory", directory, "--no-plots"))
        self.assertEqual(result["exact_landscape"][0]["steady_time"], 0)
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["status"], "complete_with_search_limits")


if __name__ == "__main__":
    unittest.main()
