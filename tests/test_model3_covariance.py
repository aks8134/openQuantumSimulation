"""Exact covariance solver, observable bounds, and Experiment1 CLI routing."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from library import classical
from Model2.Experiment1 import simulation as engines
from Model3.Experiment1 import covariance
from Model3.Experiment1 import simulation as model


class CovarianceTests(unittest.TestCase):
    def test_observables_match_full_lindblad_with_both_normalizations(self):
        for n, J, c in ((2, 1.7, .5), (3, -.7, .25), (4, .8, .5)):
            with self.subTest(n=n, J=J, prefactor=c):
                parameters = dict(number_of_qubits=n, J=J, gamma=1.3,
                                  p_left=.8, p_right=.3, exchange_prefactor=c)
                times = [.17, .83, 2.]
                exact = model.run_experiment(times, **parameters, methods=())
                reduced = covariance.run_experiment(times, **parameters)
                np.testing.assert_allclose(reduced["populations"], (1-exact["magnetizations"]["exact"])/2, atol=3e-13)
                np.testing.assert_allclose(reduced["currents"], exact["currents"]["exact"], atol=3e-13)
                np.testing.assert_allclose(reduced["stationary"]["populations"], exact["stationary"]["populations"], atol=3e-13)
                np.testing.assert_allclose(reduced["stationary"]["I"], exact["stationary"]["I"], atol=3e-13)
                self.assertLess(reduced["diagnostics"]["continuity_residual"], 1e-13)

    def test_half_mixed_and_custom_bitstrings_match_full_density_evolution(self):
        times = [0., .23, 1.]
        H, jumps, _, pops, currents, _ = model.build_model(J=1., gamma=1.)
        initial = np.array([np.eye(8)/8, np.diag([0, 0, 0, 0, 0, 1., 0, 0])])  # half mixture and |101>
        states = engines.exact_evolve(initial, classical.build_liouvillian(H, jumps), times)
        reduced = covariance.run_experiment(times, J=1., gamma=1., initial_states=["half_mixed", "101"])
        np.testing.assert_allclose(reduced["populations"], engines.expectations(states, pops), atol=2e-13)
        np.testing.assert_allclose(reduced["currents"], engines.expectations(states, currents), atol=2e-13)
        np.testing.assert_allclose(reduced["populations"][0, 0], .5)
        np.testing.assert_allclose(reduced["currents"][:, 0], 0.)

    def test_future_error_bound_controls_all_later_local_observables(self):
        result = covariance.run_experiment([0., .2, 1., 4., 10., 30.], J=1.7, gamma=.8)
        bounds, errors = result["future_error_bounds"], result["local_observable_errors"]
        self.assertTrue(np.all(errors <= bounds+1e-13))
        self.assertTrue(np.all(np.diff(bounds, axis=1) <= 1e-13))
        for k in range(bounds.shape[1]):
            self.assertTrue(np.all(errors[:, k:] <= bounds[:, k, None]+1e-13))
        self.assertGreaterEqual(result["diagnostics"]["minimum_covariance_eigenvalue"], -1e-13)
        self.assertLessEqual(result["diagnostics"]["maximum_covariance_eigenvalue"], 1+1e-13)

    def test_stop_at_initial_steady_state_and_explicit_cap_failure(self):
        stationary = covariance.run_experiment([0., 1., 10.], p_left=.5, p_right=.5,
            initial_states=["half_mixed"], until_steady=True)
        np.testing.assert_array_equal(stationary["times"], [0.])
        self.assertEqual(stationary["steady_state"]["stop_reason"], "future_local_error_bound_met")
        capped = covariance.run_experiment([0., .001], number_of_qubits=10, gamma=1., until_steady=True)
        self.assertEqual(capped["steady_state"]["stop_reason"], "time_cap_reached_without_bound")
        self.assertFalse(any(capped["steady_state"]["future_bound_met"]))
        self.assertEqual(capped["times"][-1], .001)

    def test_n10_stationary_targets_and_until_steady_for_all_preparations(self):
        result = covariance.run_experiment(np.linspace(0, 200, 201), number_of_qubits=10,
                                          J=1., gamma=1., until_steady=True)
        np.testing.assert_allclose(result["stationary"]["populations"], [.58]+[.5]*8+[.42], atol=2e-14)
        np.testing.assert_allclose(result["stationary"]["I"], .32, atol=2e-14)
        self.assertTrue(all(result["steady_state"]["future_bound_met"]))
        self.assertTrue(all(result["steady_state"]["final_local_within_tolerance"]))
        self.assertLess(result["times"][-1], 200.)

    def test_invalid_parameters_preparations_and_times_are_rejected(self):
        for kwargs in (dict(delta=.1), dict(J=0.), dict(gamma=0.), dict(p_left=1.1), dict(p_right=np.nan),
                       dict(number_of_qubits=1), dict(steady_tolerance=-.1), dict(exchange_prefactor=0.),
                       dict(initial_states=[]), dict(initial_states=["00"]),
                       dict(initial_states=["000", "000"]), dict(initial_states=["+++" ])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                covariance.run_experiment([0., 1.], **kwargs)
        for times in ([], [-1.], [0., 0.], [1., 0.], [np.inf], [[0., 1.]]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                covariance.run_experiment(times)
        for p in (0., 1.):
            result = covariance.run_experiment([0., 1.], p_left=p, p_right=p)
            np.testing.assert_allclose(result["stationary"]["populations"], p, atol=2e-14)

    def test_n10_cli_avoids_many_body_solver_and_saves_observables(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(model, "build_model", side_effect=AssertionError("no many-body solve")), \
             patch.object(engines, "initial_states", side_effect=AssertionError("no density matrices")):
            model.main("exact", ["--solver", "covariance", "--n-qubits", "10", "--J", "1", "--gamma", "1",
                "--initial-states", "half_mixed", "--t-final", "200", "--num-times", "201", "--until-steady",
                "--output-dir", folder])
            summary_path = next(Path(folder).glob("*.json"))
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["solver"], "exact_covariance")
            self.assertEqual(summary["initial_states"], ["half_mixed"])
            self.assertTrue(summary["steady_state"]["future_bound_met"][0])
            for suffix in ("png", "pdf", "md", "csv", "npz"):
                self.assertTrue(summary_path.with_suffix("."+suffix).is_file())
            with np.load(summary_path.with_suffix(".npz")) as saved:
                self.assertEqual(saved["C_exact"].shape[-2:], (10, 10))
                self.assertEqual(saved["n_exact"].shape[0], 1)
                self.assertFalse(any(key.startswith("rho_") for key in saved.files))
            with summary_path.with_suffix(".csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(summary["times"]))
            self.assertAlmostEqual(float(rows[0]["n_1"]), .5)
            self.assertAlmostEqual(float(rows[0]["I_1"]), 0.)

    def test_cli_rejects_incompatible_modes_before_any_evolution(self):
        invalid = (["--solver", "covariance", "--mode", "lie"],
                   ["--solver", "covariance", "--delta", ".1"], ["--until-steady"],
                   ["--initial-states", "half_mixed"])
        with contextlib.redirect_stderr(io.StringIO()), patch.object(model, "run_experiment", side_effect=AssertionError("no evolution")):
            for args in invalid:
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    model.main("exact", args)


if __name__ == "__main__":
    unittest.main()
