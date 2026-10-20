"""Validate automatic steady-state planning, replay, budgets, and bounded failure."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from library import reservoir_scaling as physics
from Model3.Experiment3 import adaptive_schedule as controller
from Model3.Experiment3 import classical_randomized_lie_trotter as experiment


class AutomaticScheduleTests(unittest.TestCase):
    def options(self, *extra):
        return experiment.parse_arguments(("--auto-schedule", "--n-qubits", "2",
            "--steady-tolerance", ".2", "--initial-dt", ".5", "--minimum-dt", ".01",
            "--check-interval", ".5", "--steady-window", "1", "--max-time", "30",
            "--no-plots", "--no-progress", *extra))

    def plan(self, options):
        initial = np.asarray([np.diag([int(bit) for bit in label]) for label in options.initial_states], float)
        return controller.plan(options.model, initial, options.J, options.controller_settings,
            policy=options.sampling_policy, order=options.order, formula=options.system_formula,
            step_cap=options.step_cap)

    def test_parsing_is_lazy_defaults_scale_and_manual_configuration_is_preserved(self):
        with patch.object(controller, "plan", side_effect=AssertionError("parsing must not evolve")):
            auto = experiment.parse_arguments(("--auto-schedule", "--J", "2", "--gamma", "4"))
        settings = auto.controller_settings
        self.assertEqual((settings.initial_dt, settings.minimum_dt), (.125, 2.5e-6))
        self.assertEqual((settings.check_interval, settings.steady_window, settings.max_time), (2.5, 5, 100))
        self.assertEqual((settings.bias_tolerance, settings.relaxation_tolerance, settings.sampling_tolerance), (.003, .003, .004))
        np.testing.assert_array_equal(auto.nodes, [0])
        manual = experiment.parse_arguments(("--step-dts", ".1", ".2"))
        config = experiment.configuration(manual)
        self.assertEqual(config["schema_version"], 1)
        self.assertNotIn("automatic_controller", config)
        self.assertEqual(config["schedule_input"]["step_dts"], [.1, .2])

    def test_invalid_controller_settings_and_mode_conflicts(self):
        cases = (("--minimum-dt", "1"), ("--minimum-dt", "0"), ("--initial-dt", "nan"),
                 ("--check-interval", ".1"), ("--steady-window", ".2"),
                 ("--max-time", "0"), ("--step-cap", "0"), ("--refine-factor", "0"),
                 ("--bias-tolerance", ".2"), ("--sampling-tolerance", "-1"),
                 ("--t-final", "2"))
        for extra in cases:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.options(*extra)
        with self.assertRaises(ValueError):
            experiment.parse_arguments(("--max-time", "10"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.options("--dt-schedule", "1:.1")

    def test_largest_passing_tested_step_and_refinement_history(self):
        options = self.options()
        with patch.object(physics, "sampled_step", side_effect=AssertionError("planning must not sample")):
            nodes, plan = self.plan(options)
        self.assertEqual(plan["status"], "converged")
        candidates = plan["candidates"]
        self.assertTrue(candidates[-1]["within_bias_budget"])
        self.assertTrue(all(not row["within_bias_budget"] for row in candidates[:-1]))
        np.testing.assert_allclose([row["dt"] for row in candidates], .5/2**np.arange(len(candidates)))
        self.assertEqual(plan["selected_final_dt"], candidates[-1]["dt"])
        self.assertEqual(plan["final_nominal_dt"], plan["selected_final_dt"])
        self.assertTrue(any(entry["action"] == "refine" for entry in plan["checkpoints"]))
        self.assertTrue(np.all(np.diff(nodes) > 0))
        self.assertLessEqual(nodes[-1], options.controller_settings.max_time)
        self.assertGreaterEqual(plan["window_end"]-plan["window_start"], .999999999)
        for entry in plan["checkpoints"]:
            self.assertAlmostEqual(entry["time"]/.5, round(entry["time"]/.5))
        exact = physics.exact_stationary(options.model)
        for row in candidates:
            if row["status"] == "valid":
                branch = physics.channel(options.model, options.J*row["dt"], options.sampling_policy,
                                         options.order, options.system_formula)
                target, _ = physics.mean_stationary(branch, options.n_qubits)
                self.assertAlmostEqual(row["bias"], controller.distance(target, exact, options.model))

    def test_schedule_does_not_depend_on_seed_or_trajectory_count(self):
        left, right = self.options("--trajectories", "2", "--seed", "1"), self.options("--trajectories", "19", "--seed", "107")
        nodes_left, plan_left = self.plan(left)
        nodes_right, plan_right = self.plan(right)
        np.testing.assert_array_equal(nodes_left, nodes_right)
        self.assertEqual(plan_left, plan_right)

    def test_three_site_noncommuting_bonds_meet_the_default_tight_budgets(self):
        options = experiment.parse_arguments(("--auto-schedule", "--n-qubits", "3",
            "--check-interval", "1", "--steady-window", "2", "--max-time", "40"))
        nodes, plan = self.plan(options)
        self.assertEqual(plan["status"], "converged")
        self.assertLessEqual(plan["final_channel_bias"], .003)
        self.assertLessEqual(plan["final_relaxation_distance"], .003)
        self.assertLessEqual(plan["passing_span_max_physical_distance"], .006+1e-12)
        self.assertLessEqual(nodes[-1], 40)

    def test_replay_matches_planning_and_checks_every_tail_node_when_thinned(self):
        options = self.options("--record-every", "10000", "--J", "2.3", "--gamma", "1.7",
                               "--initial-dt", ".17", "--check-interval", ".6", "--steady-window", "1.2")
        options.nodes, options.controller_plan = self.plan(options)
        plan = options.controller_plan
        self.assertEqual(plan["status"], "converged")
        payload = experiment.evolve(options)
        planned_observables = experiment.observables(np.asarray(plan["final_mean_covariances_real_gauge"]), options)
        for family in experiment.FAMILIES:
            np.testing.assert_allclose(np.asarray(payload["mean_channel_observables"][family])[:, -1], planned_observables[family], atol=1e-13)
        required = options.nodes[options.nodes >= plan["window_start"]-1e-10]
        self.assertTrue(set(required) <= set(payload["times"]))
        self.assertTrue({entry["time"] for entry in plan["checkpoints"]} <= set(payload["times"]))
        self.assertTrue(np.any(np.diff(options.nodes) < plan["final_nominal_dt"]-1e-9))
        # Independently check the budget at every integration node in the tail.
        state = np.asarray([np.diag([int(bit) for bit in label]) for label in options.initial_states], float)
        fixed = np.asarray(plan["final_channel_stationary_real_gauge"])
        target = physics.exact_stationary(options.model)
        for step, dt in enumerate(np.diff(options.nodes)):
            branch = physics.channel(options.model, options.J*dt, options.sampling_policy, options.order, options.system_formula)
            state = physics.mean_step(state, branch)
            if options.nodes[step+1] >= plan["window_start"]-1e-10:
                self.assertLessEqual(controller.distance(state, fixed, options.model), options.controller_settings.relaxation_tolerance+1e-12)
                self.assertLessEqual(controller.distance(state, target, options.model),
                    options.controller_settings.bias_tolerance+options.controller_settings.relaxation_tolerance+1e-12)

    def test_time_and_step_limits_are_explicit_nonconvergence(self):
        for extra, status in ((("--max-time", ".1"), "max_time_reached"),
                              (("--step-cap", "2"), "step_cap_reached")):
            with self.subTest(status=status):
                options = self.options(*extra)
                nodes, plan = self.plan(options)
                self.assertEqual(plan["status"], status)
                self.assertIsNone(plan["window_start"])
                self.assertLessEqual(len(nodes)-1, options.step_cap)
                self.assertLessEqual(nodes[-1], options.controller_settings.max_time)

    def test_minimum_step_failure_is_saved_without_claiming_convergence(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--minimum-dt", ".2", "--bias-tolerance", "1e-12", "--output-directory", directory)
            options.no_plots = False
            with patch.object(experiment, "plot_result", side_effect=AssertionError("no evolution to plot")):
                payload = experiment.main(options)
            self.assertEqual(payload["run_status"], "minimum_dt_reached")
            self.assertEqual(payload["sampling_assessment"]["status"], "not_assessed")
            self.assertEqual(payload["times"], [0])
            self.assertEqual(payload["schedule"]["step_dts"], [])
            self.assertEqual(payload["figure_generation_status"], "not_generated_no_evolution")
            self.assertTrue(list(Path(directory).glob("plans/*_plan.json")))
            saved = json.loads(next(Path(directory).glob("results/*.json")).read_text())
            self.assertEqual(saved, payload)

    def test_invalid_fixed_point_candidate_is_rejected_then_halved(self):
        options = self.options()
        solve = physics.mean_stationary
        calls = 0
        def reject_first(branch, n):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("test numerical solve failure")
            return solve(branch, n)
        with patch.object(physics, "mean_stationary", side_effect=reject_first):
            nodes, plan = self.plan(options)
        self.assertEqual(plan["candidates"][0]["status"], "rejected")
        self.assertIn("solve failure", plan["candidates"][0]["reason"])
        self.assertEqual(plan["stages"][0]["dt"], .25)
        self.assertEqual(plan["status"], "converged")

    def test_resonant_step_with_no_endpoint_damping_is_rejected(self):
        options = self.options("--initial-dt", str(np.pi**2/2), "--check-interval", "5", "--steady-window", "10")
        _, plan = self.plan(options)
        self.assertEqual(plan["candidates"][0]["status"], "rejected")
        self.assertIn("endpoint population damping", plan["candidates"][0]["reason"])

    def test_complete_successful_run_and_configuration_separation(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args = ("--p-left", "0", "--p-right", "0", "--initial-states", "00",
                    "--trajectories", "2", "--output-directory", directory)
            payload = experiment.main(self.options(*args))
            self.assertEqual(payload["run_status"], "passed")
            self.assertEqual(payload["automatic_controller"]["status"], "converged")
            self.assertEqual(payload["automatic_controller"]["max_population_replay_difference"], 0)
            self.assertEqual(payload["sampling_assessment"]["max_realized_sampling_distance"], 0)
            saved = json.loads(next(Path(directory).glob("results/*.json")).read_text())
            self.assertEqual(saved, payload)
            experiment.main(self.options(*args, "--refine-factor", "2"))
            self.assertEqual(len(list(Path(directory).glob("results/*.json"))), 2)

    def test_mean_convergence_with_R1_is_saved_as_sampling_failure(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--p-left", "0", "--p-right", "0", "--initial-states", "00",
                                   "--trajectories", "1", "--output-directory", directory)
            payload = experiment.main(options)
            self.assertEqual(payload["automatic_controller"]["status"], "converged")
            self.assertEqual(payload["run_status"], "sampling_budget_not_met")
            self.assertFalse(payload["sampling_assessment"]["uncertainty_estimable"])
            self.assertTrue(any("R=1" in reason for reason in payload["sampling_assessment"]["reasons"]))


class SamplingBudgetTests(unittest.TestCase):
    def payload(self):
        zero = np.zeros((1, 3, 10)).tolist()
        currents = np.zeros((1, 3, 9)).tolist()
        return {"times": [0, 1, 2], "configuration": {"physics": {"J": 2},
                "trajectories": 64, "steady_tolerance": .01},
            "observables": {"populations": copy.deepcopy(zero), "excitation_flows": copy.deepcopy(currents)},
            "mean_channel_observables": {"populations": copy.deepcopy(zero), "excitation_flows": copy.deepcopy(currents)},
            "trajectory_standard_errors": {"populations": copy.deepcopy(zero), "excitation_flows": copy.deepcopy(currents)},
            "stationary_observables": {"populations": [0]*10, "excitation_flows": [0]*9}}

    def controller(self):
        return {"status": "converged", "window_start": 0, "window_end": 2,
                "settings": {"sampling_tolerance": .004}}

    def test_all_sites_bonds_and_entire_window_are_checked(self):
        payload = self.payload()
        payload["observables"]["populations"][0][1][3] = .006
        result = controller.assess_sampling(payload, self.controller())
        self.assertEqual(result["status"], "sampling_budget_not_met")
        self.assertEqual(result["max_realized_sampling_distance"], .006)
        payload = self.payload()
        payload["observables"]["excitation_flows"][0][1][6] = .01
        result = controller.assess_sampling(payload, self.controller())
        self.assertEqual(result["max_realized_sampling_distance"], .005)
        self.assertEqual(result["status"], "sampling_budget_not_met")

    def test_uncertainty_has_its_own_check_and_R1_cannot_pass(self):
        payload = self.payload()
        payload["trajectory_standard_errors"]["populations"][0][1][0] = .003
        result = controller.assess_sampling(payload, self.controller())
        self.assertEqual(result["max_realized_sampling_distance"], 0)
        self.assertEqual(result["status"], "sampling_budget_not_met")
        self.assertEqual(result["max_two_standard_errors"], .006)
        payload = self.payload()
        payload["configuration"]["trajectories"] = 1
        result = controller.assess_sampling(payload, self.controller())
        self.assertFalse(result["uncertainty_estimable"])
        self.assertEqual(result["status"], "sampling_budget_not_met")

    def test_controller_failure_cannot_be_overridden_by_sampling(self):
        plan = self.controller()
        plan["status"] = "max_time_reached"
        self.assertEqual(controller.assess_sampling(self.payload(), plan)["status"], "not_assessed")


if __name__ == "__main__":
    unittest.main()
