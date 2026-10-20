"""Schedule optimization, accelerated predictions, and independent acceptance."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from library import reservoir_scaling as physics
from Model3.Experiment4 import planning
from Model3.Experiment5.circuit_depth import validate_source
from Model3.Experiment7 import compare_schedules as experiment
from Model3.Experiment7 import schedules


class ScheduleComparisonTests(unittest.TestCase):
    def options(self, *extra):
        sampling = [] if "--R-values" in extra else ["--trajectories", "4"]
        return experiment.parse_arguments(["--n-qubits", "2", "--step-budget", "30",
            "--minimum-dt", ".01", "--maximum-dt", ".8", "--steady-window", ".5",
            "--steady-tolerance", ".1", "--sampling-tolerance", ".02",
            "--search-iterations", "20", "--feedback-iterations", "5", "--population-multiplier", "5",
            "--seeds", "29", *sampling, "--no-plots", "--no-progress", *extra])

    def predictor(self, options):
        return schedules.Predictor(options.model, options.J, options.settings,
            options.sampling_policy, options.order, options.system_formula)

    def plan(self, options, method):
        prep = planning.preparations(options.model, physics.exact_stationary(options.model), [options.preparation])[0]
        return schedules.plan(options.model, options.J, prep, options.settings, method=method,
            policy=options.sampling_policy, order=options.order, formula=options.system_formula)

    def test_parsing_is_lazy_and_rejects_invalid_policy(self):
        with patch.object(physics, "exact_stationary", side_effect=AssertionError("parsing must not solve")):
            options = experiment.parse_arguments([])
        self.assertEqual(options.methods, ["two_stage", "cosine"])
        self.assertEqual(options.mean_tolerance, .016)
        self.assertEqual(options.seeds, [29, 30, 31])
        self.assertFalse(hasattr(options, "step_grid"))
        self.assertFalse(hasattr(options, "baseline_study"))
        for extra in (("--J", "0"), ("--gamma", "nan"), ("--n-qubits", "1"),
                      ("--minimum-dt", "0"), ("--minimum-dt", "2"), ("--forecast-nodes", "0"),
                      ("--forecast-margin", "1.1"), ("--search-iterations", "0"),
                      ("--optimizer-seed", "-1"), ("--sampling-tolerance", ".2"),
                      ("--R-values", "2", "2"), ("--methods", "cosine", "cosine")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.options(*extra)

    def test_cosine_has_exact_positive_endpoints_and_no_restart(self):
        dts, parameters = schedules.decode("cosine", [17, np.log(.7), np.log(.03)])
        self.assertEqual(len(dts), 17)
        self.assertEqual(dts[0], parameters["initial_dt"])
        self.assertEqual(dts[-1], parameters["final_dt"])
        self.assertTrue(np.all(np.diff(dts) <= 0))
        np.testing.assert_allclose(dts, .03+.335*(1+np.cos(np.pi*np.arange(17)/16)))
        single, _ = schedules.decode("cosine", [1, np.log(.7), np.log(.03)])
        np.testing.assert_allclose(single, [.03])

    def test_affine_acceleration_matches_direct_channel_for_all_orders_and_policies(self):
        rng = np.random.default_rng(91)
        for order in ("hamiltonian-first", "jump-first"):
            for policy in ("uniform", "rate-weighted"):
                options = self.options("--n-qubits", "4", "--order", order, "--sampling-policy", policy)
                predictor = self.predictor(options)
                Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
                state = Q @ np.diag([.1, .3, .6, .9]) @ Q.T
                original = state.copy()
                branch = predictor.channel(.17)
                vector = np.r_[predictor.pack(state), 1.]
                accelerated = predictor.unpack(predictor.affine_matrix(branch) @ vector)
                np.testing.assert_allclose(accelerated, physics.mean_step(state, branch), atol=1e-14)
                reference = state.copy()
                for _ in range(137):
                    reference = physics.mean_step(reference, branch)
                np.testing.assert_allclose(predictor.constant(state, .17, 137), reference, atol=2e-13)
                np.testing.assert_array_equal(state, original)

    def test_two_stage_forecast_matches_all_committed_steps(self):
        options = self.options("--n-qubits", "4")
        predictor = self.predictor(options)
        state = np.eye(4)*.5
        dts, _ = schedules.decode("two_stage", [181, np.log(.68), np.log(.04), .2])
        reference = state.copy()
        for dt in dts:
            reference = physics.mean_step(reference, predictor.channel(float(dt)))
        np.testing.assert_allclose(predictor.forecast(state, dts, "two_stage"), reference, atol=2e-13)

    def test_sparse_window_uses_exact_advancement_and_covers_endpoint(self):
        options = self.options("--n-qubits", "4", "--forecast-nodes", "7", "--steady-window", "1.13")
        predictor = self.predictor(options)
        state, dt = np.eye(4)*.5, .011
        count = planning.holding_steps(dt, options.steady_window)
        stride = int(np.ceil(count/7))
        observed = [planning.distance(state, predictor.target, options.model)]
        for step in range(1, count+1):
            state = physics.mean_step(state, predictor.channel(dt))
            if step % stride == 0 or step == count:
                observed.append(planning.distance(state, predictor.target, options.model))
        proxy = predictor.window_proxy(np.eye(4)*.5, dt)
        self.assertAlmostEqual(proxy, max(observed), places=12)

    def test_actual_window_rejects_an_intermediate_failure_and_leaves_input_unchanged(self):
        options = self.options()
        predictor = self.predictor(options)
        state = predictor.target.copy()
        with patch.object(physics, "mean_step", return_value=state+.2*np.eye(2)):
            passed, validation = predictor.validate(state, .1, 3.)
        self.assertFalse(passed)
        self.assertEqual(len(validation["step_dts"]), 1)
        self.assertEqual(validation["times"][0], 3.)
        np.testing.assert_array_equal(state, predictor.target)

    def test_both_planners_pass_without_loading_prior_schedules_or_stationary_floors(self):
        options = self.options()
        with patch.object(planning, "load_baselines", side_effect=AssertionError("no baselines")), \
             patch.object(physics, "mean_stationary", side_effect=AssertionError("no stationary floor veto")):
            for method in ("two_stage", "cosine"):
                result = self.plan(options, method)
                selected = result["selected"]
                self.assertEqual(result["status"], "mean_target_met")
                self.assertLessEqual(selected["steps"], options.step_budget)
                self.assertLessEqual(selected["tail_mean_error"], options.mean_tolerance)
                self.assertGreaterEqual(selected["window_end"]-selected["final_time"], options.steady_window-1e-12)
                state = np.eye(2)*.5
                for dt in selected["step_dts"]:
                    state = physics.mean_step(state, physics.channel(options.model, options.J*dt,
                        options.sampling_policy, options.order, options.system_formula))
                np.testing.assert_allclose(state, selected["final_mean_covariance_real_gauge"], atol=1e-14)
                self.assertEqual(len(result["preparation_mean_errors"]), selected["steps"]+1)
                if method == "cosine":
                    self.assertEqual(len(result["decisions"]), 1)

    def test_feedback_retains_own_feasible_continuation_when_search_finds_nothing(self):
        options = self.options()
        predictor = self.predictor(options)
        state = predictor.target.copy()
        incumbent = np.full(3, .01)
        with patch.object(schedules, "differential_evolution", return_value=SimpleNamespace(nit=0, success=False, fun=1000.)):
            best, audit = predictor.optimize(state, "two_stage", 20, 10., decision=1, incumbent=incumbent)
        np.testing.assert_array_equal(best["dts"], incumbent)
        self.assertEqual(audit["selected_origin"], "retained_own_forecast")
        self.assertEqual(audit["remaining_step_cap"], len(incumbent))

    def test_planning_is_independent_of_R_and_sampling_seeds(self):
        first = self.plan(self.options(), "cosine")
        second = self.plan(self.options("--trajectories", "99", "--seeds", "101", "102"), "cosine")
        self.assertEqual(first, second)

    def test_failed_forecast_commits_no_speculative_state_and_skips_sampling(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--step-budget", "1", "--steady-tolerance", "1e-7",
                "--sampling-tolerance", "1e-8", "--output-directory", folder)
            study = experiment.main(options)
            for record in study["methods"].values():
                self.assertEqual(record["controller_status"], "no_feasible_forecast")
                self.assertEqual(record["selected"]["steps"], 0)
                self.assertEqual(record["replays"], {})
                np.testing.assert_array_equal(record["selected"]["final_mean_covariance_real_gauge"], np.eye(2)*.5)
            self.assertFalse(list(Path(folder).glob("comparison_*/*/replay_*.json")))

    def test_persisted_replay_rechecks_mean_and_preserves_R_prefixes_and_circuit_schema(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--output-directory", folder, "--R-values", "2", "5")
            study = experiment.main(options)
            directory = next(Path(folder).glob("comparison_*"))
            for method in options.methods:
                a = json.loads((directory/method/"replay_seed29_R2.json").read_text())
                b = json.loads((directory/method/"replay_seed29_R5.json").read_text())
                self.assertEqual(a["schedule"]["jump_choices"], b["schedule"]["jump_choices"][:2])
                self.assertEqual(a["mean_forward_validation"], b["mean_forward_validation"])
                self.assertLess(b["max_mean_covariance_replay_difference"], 1e-13)
                self.assertEqual(b["assessment"]["sampling_scope"], "output_only")
                self.assertEqual(b["experiment"], "Model3/Experiment7")
                validate_source(b)
                self.assertEqual(len(study["methods"][method]["replays"]), 2)

    def test_plan_only_and_zero_step_stationary_preparation(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--plan-only", "--output-directory", folder)
            study = experiment.main(options)
            self.assertTrue(all(not record["replays"] for record in study["methods"].values()))
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--p-left", "0", "--p-right", "0", "--preparation", "vacuum", "--output-directory", folder)
            study = experiment.main(options)
            self.assertTrue(all(record["selected"]["steps"] == 0 for record in study["methods"].values()))
            self.assertTrue(all(next(iter(record["replays"].values()))["status"] == "passed" for record in study["methods"].values()))

    def test_plots_and_reports_are_written_after_results(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--output-directory", folder)
            options.no_plots = False
            study = experiment.main(options)
            self.assertEqual(study["figure_generation_status"], "complete")
            directory = next(Path(folder).glob("comparison_*"))
            self.assertTrue((directory/"Study.md").exists())
            self.assertTrue((directory/"comparison.png").exists())
            self.assertTrue((directory/"cosine/figures/validation_seed29_R4.png").exists())


if __name__ == "__main__":
    unittest.main()
