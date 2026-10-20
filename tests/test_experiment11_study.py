import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Model1.Experiment11 import physics
from Model1.Experiment11.classical_randomized_lie_trotter import advance
from Model1.Experiment11 import study_randomized_scaling as study


class StudyTests(unittest.TestCase):
    def config(self, *extra):
        return study.parse_arguments(("--sizes", "2", "--ratios", "2", "--steps", "0.04", "0.02", "0.01",
            "--R-values", "4", "8", "--seeds", "29", "101", *extra)).config

    def test_averaged_map_is_average_of_random_alternatives_not_composition(self):
        for n, ratio, step in ((2, 20, 0.1), (3, 0.7, 0.13), (5, 2, 0.02)):
            model = physics.Model(n, J=1, gamma=ratio)
            x = np.arange(1, n+1, dtype=float)
            x /= np.linalg.norm(x)
            phases = (-1j)**(np.arange(n)-model.initial_site)
            a = phases*x
            branches = [advance(a[None, :], model, step, [boundary])[0]/phases for boundary in (0, 1)]
            expected = sum(np.outer(v, v.conj()) for v in branches)/2
            operator, _, _ = study.averaged_map(n, ratio, step)
            actual = (operator @ np.outer(x, x).ravel()).reshape(n, n)
            np.testing.assert_allclose(actual, expected, atol=1e-14)
            np.testing.assert_allclose(actual, actual.T, atol=1e-14)
            self.assertLessEqual(float(actual.trace()), 1+1e-14)

    def test_real_gauge_exact_reference(self):
        from scipy.linalg import expm
        for n in (2, 5):
            a = expm(0.7*study.generator(n, 1.7))[:, n//2]
            expected = physics.exact_reference(physics.Model(n, gamma=1.7), [0.7])
            np.testing.assert_allclose(a*a, expected[0][:, 0], atol=1e-12)
            np.testing.assert_allclose(2*a[:-1]*a[1:], expected[2][:, 0], atol=1e-12)

    def test_exact_crossing_two_site_analytic_and_cap(self):
        expected = np.log(1000)/4
        self.assertAlmostEqual(study.exact_crossing(2, 4, 1e-3, 10), expected, places=9)
        self.assertIsNone(study.exact_crossing(2, 4, 1e-3, 0.1))

    def test_timestep_results_have_crossing_bracket_and_tail_certificate(self):
        cfg = self.config()
        t = study.exact_crossing(2, 2, 1e-3, 100)
        case = study.evaluate_step(2, 2, 0.002, t, cfg)
        self.assertTrue(case["passed"])
        self.assertEqual(case["status"], "complete")
        self.assertAlmostEqual(case["steady_time_bracket"][1]-case["steady_time_bracket"][0], 0.002)
        self.assertIsNotNone(case["tail_error_certified_below"])
        self.assertLessEqual(case["relative_steady_time_error"], cfg["timing_tolerance"])
        bad = study.evaluate_step(2, 2, 0.5, t, cfg)
        self.assertFalse(bad["passed"])

    def test_finite_R_is_seeded_and_nested_and_reports_missing_pass(self):
        cfg = self.config("--steady-tolerance", "0.05", "--dynamic-tolerance", "0.1", "--timing-tolerance", "0.2")
        t = study.exact_crossing(2, 2, cfg["steady_tolerance"], 100)
        first = study.evaluate_ensembles(2, 2, 0.01, t, cfg)
        second = study.evaluate_ensembles(2, 2, 0.01, t, cfg)
        self.assertEqual(first["results"], second["results"])
        extended = study.evaluate_ensembles(2, 2, 0.01, t, {**cfg, "R_values": [4, 8, 16]})
        self.assertEqual(first["results"], extended["results"][:2])
        tiny = study.evaluate_ensembles(2, 2, 0.01, t, {**cfg, "step_cap": 1})
        self.assertIsNone(tiny["minimum_tested_R_passing_all_seeds"])

    def test_early_rejection_preserves_decision_without_full_evolution(self):
        cfg = {**self.config(), "R_values": [4], "sampling_tolerance": 1e-5}
        t = study.exact_crossing(2, 2, cfg["steady_tolerance"], 100)
        early = study.evaluate_ensembles(2, 2, 0.01, t, cfg, early_reject=True)
        full = study.evaluate_ensembles(2, 2, 0.01, t, cfg)
        self.assertEqual(early["status"], "early_rejected")
        self.assertIsNone(early["minimum_tested_R_passing_all_seeds"])
        self.assertIsNone(full["minimum_tested_R_passing_all_seeds"])
        self.assertLess(early["completed_steps"], full["completed_steps"])

    def test_incremental_save_resume_without_recomputing(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ("--sizes", "2", "--ratios", "2", "--steps", "0.005", "--R-values", "4", "8",
                "--seeds", "29", "--dynamic-tolerance", "0.1", "--timing-tolerance", "0.2",
                "--output-directory", directory, "--no-plots")
            result = study.main(study.parse_arguments(args))
            self.assertEqual(len(result["recommendations"]), 1)
            paths = list(Path(directory).glob("*/study.json"))
            self.assertEqual(len(paths), 1)
            self.assertEqual(json.loads(paths[0].read_text())["status"], "complete")
            with patch.object(study, "exact_crossing", side_effect=AssertionError("must not rerun")), \
                 patch.object(study, "evaluate_step", side_effect=AssertionError("must not rerun")), \
                 patch.object(study, "evaluate_ensembles", side_effect=AssertionError("must not rerun")):
                resumed = study.main(study.parse_arguments((*args, "--resume")))
            self.assertEqual(resumed["recommendations"], result["recommendations"])

    def test_invalid_parameters(self):
        for args in (("--sizes", "1"), ("--ratios", "0"), ("--steps", "nan"),
                     ("--dynamic-tolerance", "0"), ("--rate-margin", "0.9"),
                     ("--seeds", "1", "1")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                study.parse_arguments(args)


if __name__ == "__main__":
    unittest.main()
