"""Automatic timestep control, independent replay, and circuit compatibility."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from library import reservoir_scaling as physics
from Model3.Experiment4 import planning
from Model3.Experiment5.circuit_depth import validate_source
from Model3.Experiment6 import adaptive_budget as experiment
from Model3.Experiment6 import controller


class AdaptiveBudgetTests(unittest.TestCase):
    def options(self, *extra):
        sampling = [] if "--R-values" in extra else ["--trajectories", "8"]
        return experiment.parse_arguments(["--n-qubits", "2", "--step-budget", "40",
            "--minimum-dt", ".01", "--maximum-dt", ".5", "--steady-window", ".5",
            "--steady-tolerance", ".1", "--sampling-tolerance", ".02",
            "--prediction-steps", "4", "--decision-steps", "2", "--optimizer-iterations", "18",
            "--optimizer-log-tolerance", ".05", *sampling, "--no-plots", "--no-progress", *extra])

    def plan(self, options):
        target = physics.exact_stationary(options.model)
        prep = planning.preparations(options.model, target, [options.preparation])[0]
        return controller.plan(options.model, options.J, prep, options.settings,
            policy=options.sampling_policy, order=options.order, formula=options.system_formula)

    def test_defaults_are_lazy_and_have_no_grid_or_baselines(self):
        with patch.object(physics, "exact_stationary", side_effect=AssertionError("parse must not solve")):
            options = experiment.parse_arguments([])
        self.assertEqual(options.mean_tolerance, .016)
        self.assertEqual(options.sampling_tolerance, .004)
        self.assertEqual((options.R_values, options.seeds), ([8192], [29]))
        self.assertFalse(hasattr(options, "step_grid"))
        self.assertFalse(hasattr(options, "baseline_study"))
        for extra in (("--J", "0"), ("--gamma", "nan"), ("--n-qubits", "1"),
                      ("--minimum-dt", "0"), ("--minimum-dt", "1"), ("--decision-steps", "5"),
                      ("--sampling-tolerance", ".1"), ("--step-budget", "0"), ("--seeds", "-1"),
                      ("--R-values", "2", "2"), ("--prediction-steps", "0"), ("--validation-step-cap", "1")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.options(*extra)

    def test_forecast_score_uses_accuracy_budget_and_predicted_gate_count(self):
        score, crossing = controller.forecast_score([.05, .03], .01, .04)
        self.assertEqual(crossing, 2)
        stricter, crossing = controller.forecast_score([.05, .03], .01, .02)
        self.assertIsNone(crossing)
        self.assertGreater(stricter, score)
        bad_floor, crossing = controller.forecast_score([.01, .005], .1, .02)
        self.assertIsNone(crossing)
        self.assertGreater(bad_floor, score)
        stalled, _ = controller.forecast_score([.04999]*12, .0001, .016, current_error=.05)
        useful, _ = controller.forecast_score([.04]*12, .03, .016, current_error=.05)
        self.assertGreater(stalled, useful*100)

    def test_controller_adapts_to_state_with_continuous_evaluations(self):
        options = self.options()
        predictor = controller.Predictor(options.model, options.J, options.settings,
            options.sampling_policy, options.order, options.system_formula)
        initial = np.eye(2)*.5
        dt_far, far = predictor.choose(initial, .5, 40, 100.)
        dt_near, near = predictor.choose(.95*predictor.target+.05*initial, dt_far, 40, 100.)
        for dt, audit in [(dt_far, far), (dt_near, near)]:
            self.assertTrue(options.minimum_dt <= dt <= options.maximum_dt)
            self.assertTrue(audit["selected"]["valid"])
            self.assertGreater(len(audit["candidates"]), 3)
            self.assertEqual(audit["mean_allowance"], options.mean_tolerance)
        self.assertGreater(dt_far, dt_near)

    def test_forward_check_detects_intermediate_failure(self):
        options = self.options()
        predictor = controller.Predictor(options.model, options.J, options.settings,
            options.sampling_policy, options.order, options.system_formula)
        initial = predictor.target.copy()
        with patch.object(physics, "mean_step", return_value=initial+.2*np.eye(2)):
            passed, validation = controller.forward_check(initial, .1, 2., predictor)
        self.assertFalse(passed)
        self.assertEqual(len(validation["step_dts"]), 1)
        self.assertGreater(max(validation["mean_errors"]), options.mean_tolerance)
        self.assertEqual(validation["times"][0], 2.)

    def test_prediction_horizon_extends_when_short_forecast_exceeds_step_budget(self):
        options = self.options()
        predictor = controller.Predictor(options.model, options.J, options.settings,
            options.sampling_policy, options.order, options.system_formula)
        def score(errors, bias, tolerance, current_error):
            return (1000., None) if len(errors) < 8 else (1., 1)
        with patch.object(controller, "forecast_score", side_effect=score):
            dt, audit = predictor.choose(np.eye(2)*.5, .5, 40, 100.)
        self.assertIsNotNone(dt)
        self.assertEqual([row["prediction_steps"] for row in audit["horizon_trials"]], [4, 8])
        self.assertTrue(audit["predicted_gate_budget_feasible"])

    def test_plan_passes_budget_and_separates_real_from_speculative_steps(self):
        options = self.options()
        with patch.object(planning, "search", side_effect=AssertionError("no beam search")), \
             patch.object(planning, "load_baselines", side_effect=AssertionError("no prior schedules")):
            plan = self.plan(options)
        chosen = plan["selected"]
        self.assertEqual(plan["status"], "mean_target_met")
        self.assertLessEqual(chosen["steps"], options.step_budget)
        self.assertEqual(chosen["steps"], len(chosen["step_dts"]))
        self.assertEqual(chosen["steps"], sum(s["steps"] for s in chosen["stages"]))
        self.assertLessEqual(chosen["tail_mean_error"], options.mean_tolerance)
        self.assertGreaterEqual(chosen["window_end"]-chosen["window_start"], options.steady_window-1e-13)
        self.assertTrue(all(dt == chosen["step_dts"][-1] for dt in chosen["forward_validation"]["step_dts"]))
        state = np.diag(plan["selected_preparation"]["populations"])
        for dt in chosen["step_dts"]:
            branch = physics.channel(options.model, options.J*dt, options.sampling_policy, options.order, options.system_formula)
            state = physics.mean_step(state, branch)
        np.testing.assert_allclose(state, chosen["final_mean_covariance_real_gauge"], atol=1e-14)
        self.assertGreater(plan["prediction_mean_steps"], chosen["steps"])

    def test_plan_is_independent_of_sampling_seeds_and_R(self):
        first = self.plan(self.options("--seeds", "1", "--trajectories", "2"))
        second = self.plan(self.options("--seeds", "99", "--trajectories", "31"))
        self.assertEqual(first, second)

    def test_tightening_the_budget_changes_the_generated_schedule(self):
        loose = self.plan(self.options())
        tight = self.plan(self.options("--steady-tolerance", ".05", "--sampling-tolerance", ".01"))
        self.assertTrue(loose["selected"]["mean_target_met"])
        self.assertTrue(tight["selected"]["mean_target_met"])
        self.assertLessEqual(tight["selected"]["tail_mean_error"], .04)
        self.assertNotEqual(loose["selected"]["step_dts"], tight["selected"]["step_dts"])

    def test_budget_failure_does_not_replay_or_claim_success(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--step-budget", "2", "--steady-tolerance", "1e-8",
                                   "--sampling-tolerance", "1e-9", "--output-directory", folder)
            result = experiment.main(options)
            self.assertEqual(result["controller_status"], "step_budget_reached")
            self.assertFalse(result["selected"]["mean_target_met"])
            self.assertEqual(result["replays"], {})
            self.assertFalse(list(Path(folder).glob("adaptive_*/replay_*.json")))

    def test_time_cap_and_resonance_have_explicit_failure_status(self):
        result = self.plan(self.options("--maximum-time", ".01"))
        self.assertEqual(result["status"], "maximum_time_reached")
        self.assertLessEqual(result["selected"]["final_time"], .01)
        resonance = str(np.pi**2/2)
        result = self.plan(self.options("--minimum-dt", resonance, "--maximum-dt", resonance))
        self.assertEqual(result["status"], "no_valid_timestep")

    def test_no_predicted_progress_is_reported_without_committing_probes(self):
        result = self.plan(self.options("--step-budget", "1", "--steady-tolerance", "1e-8", "--sampling-tolerance", "1e-9"))
        self.assertEqual(result["status"], "no_predicted_progress")
        self.assertEqual(result["selected"]["steps"], 0)

    def test_plan_only_and_zero_step_stationary_preparation(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--plan-only", "--output-directory", folder)
            result = experiment.main(options)
            self.assertEqual(result["controller_status"], "mean_target_met")
            self.assertEqual(result["replays"], {})
            self.assertTrue(list(Path(folder).glob("adaptive_*/plan.json")))
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--preparation", "vacuum", "--p-left", "0", "--p-right", "0", "--output-directory", folder)
            result = experiment.main(options)
            self.assertEqual(result["selected"]["steps"], 0)
            record = next(iter(result["replays"].values()))
            self.assertEqual(record["status"], "passed")
            replay = json.loads(next(Path(folder).glob("adaptive_*/replay_*.json")).read_text())
            validate_source(replay)

    def test_saved_replays_revalidate_mean_and_preserve_R_prefixes(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--output-directory", folder, "--R-values", "2", "5", "--seeds", "29", "30")
            study = experiment.main(options)
            self.assertEqual(len(study["replays"]), 4)
            output = next(Path(folder).glob("adaptive_*"))
            a = json.loads((output/"replay_seed29_R2.json").read_text())
            b = json.loads((output/"replay_seed29_R5.json").read_text())
            for data in (a, b):
                self.assertEqual(data["experiment"], "Model3/Experiment6")
                self.assertLess(data["max_mean_covariance_replay_difference"], 1e-13)
                self.assertEqual(data["assessment"]["sampling_scope"], "output_only")
                validate_source(data)
            self.assertEqual(a["schedule"]["jump_choices"], b["schedule"]["jump_choices"][:2])
            self.assertEqual(a["mean_forward_validation"], b["mean_forward_validation"])

    def test_plotting_does_not_lose_saved_data(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--output-directory", folder)
            options.no_plots = False
            study = experiment.main(options)
            self.assertEqual(study["figure_generation_status"], "complete")
            self.assertTrue(list(Path(folder).glob("adaptive_*/figures/controller.png")))
            self.assertTrue(list(Path(folder).glob("adaptive_*/figures/validation_seed29_R8.png")))


if __name__ == "__main__":
    unittest.main()
