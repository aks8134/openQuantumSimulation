"""Budget-first planning and mixed-product replay validation on small systems."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from library import classical
from library import reservoir_scaling as physics
from Model2.Experiment2 import randomized
from Model3.Experiment1 import simulation
from Model3.Experiment4 import budgeted_steady_state as experiment
from Model3.Experiment4 import planning


class BudgetedExperimentTests(unittest.TestCase):
    def options(self, *extra):
        return experiment.parse_arguments(("--n-qubits", "2", "--step-budget", "8", "--step-grid", ".5", ".25",
            "--steady-window", ".5", "--preparations", "vacuum", "profile_mixed", "--block-steps", "2",
            "--beam-width", "2", "--max-stages", "2", "--trajectories", "4", "--no-progress", "--no-plots", *extra))

    def test_defaults_lazy_parsing_and_validation(self):
        with patch.object(planning, "search", side_effect=AssertionError("parse must not evolve")):
            options = experiment.parse_arguments([])
        self.assertEqual((options.n_qubits, options.step_budget, options.R_values), (10, 500, [64]))
        self.assertEqual((options.mean_tolerance, options.sampling_tolerance), (.006, .004))
        for args in (("--step-budget", "0"), ("--J", "0"), ("--gamma", "nan"),
            ("--step-grid", ".1", ".1"), ("--step-grid", "0"), ("--steady-window", "0"),
            ("--sampling-tolerance", ".01"), ("--search-node-cap", "-1"), ("--beam-width", "0"),
            ("--estimated-step-depth", "3"), ("--depth-budget", "8"), ("--p-left", "2"),
            ("--preparations", "vacuum", "vacuum"), ("--trajectories", "0")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.options(*args)
        with self.assertRaises(ValueError):
            experiment.parse_arguments(("--R-values", "2", "2"))

    def test_warm_starts_are_product_states_with_explicit_preparation_cost(self):
        options = self.options("--n-qubits", "3", "--preparations", *planning.PREPARATIONS)
        target = physics.exact_stationary(options.model)
        preps = planning.preparations(options.model, target, options.preparations)
        profile = next(p for p in preps if p["name"] == "profile_mixed")
        np.testing.assert_allclose(profile["populations"], [.58, .5, .42], atol=2e-14)
        C = np.diag(profile["populations"])
        self.assertTrue(profile["reference_assisted_populations"])
        self.assertTrue(profile["classical_bit_randomization"])
        self.assertEqual(profile["local_preparation_depth"], 1)
        self.assertFalse(np.any(experiment.observables(C, options)["excitation_flows"]))
        self.assertGreater(np.max(np.abs(C-target)), .1)
        self.assertEqual(preps[0]["local_preparation_depth"], 0)

    def test_unrepresentable_channel_is_rejected_before_ranking(self):
        options = self.options("--step-grid", "1e308")
        with self.assertRaisesRegex(ValueError, "nonfinite channel coefficients"):
            planning.search(options)

    def test_every_candidate_obeys_step_and_modeled_depth_budgets(self):
        options = self.options("--step-budget", "9", "--estimated-step-depth", "3", "--depth-budget", "8")
        search = planning.search(options)
        for row in search["best_per_preparation"]+search["constant_baselines"]+[search["selected"]]:
            self.assertLessEqual(row["steps"], 9)
            self.assertLessEqual(row["estimated_total_depth"], 8)
        self.assertEqual(search["preparation_step_limits"]["profile_mixed"], 2)
        self.assertEqual(search["selected"]["estimated_total_depth"],
            search["selected_preparation"]["local_preparation_depth"]+3*search["selected"]["steps"])

    def test_best_tested_candidate_is_at_least_as_good_as_all_constants(self):
        options = self.options()
        search = planning.search(options)
        for best in search["best_per_preparation"]:
            for constant in search["constant_baselines"]:
                if constant["preparation"] != best["preparation"]:
                    continue
                if constant["mean_target_met"]:
                    self.assertTrue(best["mean_target_met"])
                    self.assertLessEqual(best["steps"], constant["steps"])
                elif constant["tail_window_covered"] and not best["mean_target_met"]:
                    self.assertLessEqual(best["tail_mean_error"], constant["tail_mean_error"])
        stages = search["selected"]["stages"]
        self.assertLessEqual(len(stages), options.max_stages)
        self.assertTrue(all(left["dt"] >= right["dt"] for left, right in zip(stages, stages[1:])))
        self.assertEqual(sum(s["steps"] for s in stages), search["selected"]["steps"])

    def test_seed_and_R_do_not_change_mean_search(self):
        first = planning.search(self.options("--seed", "1", "--trajectories", "2"))
        second = planning.search(self.options("--seed", "99", "--trajectories", "17"))
        self.assertEqual(first, second)

    def test_wide_beam_matches_an_exhaustive_small_schedule_grid(self):
        options = self.options("--step-budget", "6", "--preparations", "vacuum", "--beam-width", "64",
                               "--steady-tolerance", "1e-8", "--sampling-tolerance", "0")
        result = planning.search(options)
        target = physics.exact_stationary(options.model)
        branches = [physics.channel(options.model, options.J*dt, options.sampling_policy,
                                    options.order, options.system_formula) for dt in options.step_grid]
        oracle = []
        for length in range(1, 7):
            for seq in itertools.product(range(2), repeat=length):
                if any(a > b for a, b in zip(seq, seq[1:])):
                    continue
                changes = [i for i in range(1, length) if seq[i] != seq[i-1]]
                if changes and (length % 2 or changes[0] % 2):
                    continue
                C = np.zeros((2, 2)); times = [0.]; errors = [max(abs(np.diag(C-target)).max(), 2*abs((C-target)[0, 1]))]
                for index in seq:
                    C = physics.mean_step(C, branches[index])
                    times.append(times[-1]+options.step_grid[index])
                    errors.append(max(abs(np.diag(C-target)).max(), 2*abs((C-target)[0, 1])))
                if times[-1] >= options.steady_window:
                    boundary = times[-1]-options.steady_window
                    anchor = max(i for i, t in enumerate(times) if t <= boundary)
                    oracle.append((max(errors[anchor:]), length, errors[-1]))
        expected = min(oracle)
        self.assertAlmostEqual(result["selected"]["tail_mean_error"], expected[0])
        self.assertEqual(result["selected"]["steps"], expected[1])

    def test_window_coverage_and_search_limits_cannot_claim_success(self):
        options = self.options("--step-budget", "2", "--steady-window", "10", "--search-node-cap", "1")
        search = planning.search(options)
        self.assertEqual(search["status"], "tail_window_not_covered")
        self.assertTrue(search["search_limit_reached"])
        result = experiment.replay(options, search, 4)
        self.assertEqual(result["run_status"], "tail_window_not_covered")
        self.assertFalse(result["assessment"]["tail_window_covered"])
        self.assertEqual(len(search["constant_baselines"]), 4)

    def test_tail_metric_catches_hidden_error_and_includes_boundary_anchor(self):
        options = self.options()
        prep = {"name": "vacuum", "local_preparation_depth": 0}
        node = planning.Node(np.zeros((2, 2)), (0, 0, 0), (0., .4, .8, 1.2), (.5, .08, .02, .001), 1)
        result = planning.evaluate(node, prep, options)
        self.assertAlmostEqual(result["tail_mean_error"], .08)
        self.assertEqual(result["checked_from_step"], 1)
        self.assertFalse(result["mean_target_met"])

    def test_floating_window_boundary_does_not_require_an_extra_step(self):
        options = self.options("--step-grid", ".1", "--step-budget", "10", "--steady-window", "1",
            "--p-left", "0", "--p-right", "0", "--preparations", "vacuum", "--search-node-cap", "0")
        search = planning.search(options)
        self.assertEqual(search["status"], "mean_target_met")
        self.assertEqual(search["selected"]["steps"], 10)

    def test_mixed_product_replay_matches_full_density_channels_and_exact_reference(self):
        options = self.options("--J", "1.7", "--gamma", ".8", "--preparations", "profile_mixed")
        search = planning.search(options)
        result = experiment.replay(options, search, 3)
        H, jumps, _, pop_ops, current_ops, _ = simulation.build_model(2, J=options.J, gamma=options.gamma)
        rho0 = np.ones((1, 1))
        for p in search["selected_preparation"]["populations"]:
            rho0 = np.kron(rho0, np.diag([1-p, p]))
        L = classical.build_liouvillian(H, jumps).toarray()
        for ti, t in enumerate(result["times"]):
            rho = (expm(t*L)@rho0.reshape(-1, order="F")).reshape(4, 4, order="F")
            for family, ops in (("populations", pop_ops), ("excitation_flows", current_ops)):
                expected = [np.trace(op@rho).real for op in ops]
                np.testing.assert_allclose(result["exact_reference_observables"][family][0][ti], expected, atol=2e-13)
        finals = []
        for choices in result["schedule"]["jump_choices"]:
            rho = rho0.copy()
            for dt, choice in zip(result["schedule"]["step_dts"], choices):
                probability = result["schedule"]["jump_probabilities"][choice]
                kraus = randomized.sampled_kraus(H, jumps[choice], probability, dt, options.order)
                rho = sum(K@rho@K.conj().T for K in kraus)
            finals.append([np.trace(op@rho).real for op in pop_ops])
        expected = np.asarray(finals)
        np.testing.assert_allclose(result["observables"]["populations"][0][-1], expected.mean(axis=0), atol=2e-13)
        np.testing.assert_allclose(result["trajectory_standard_errors"]["populations"][0][-1], expected.std(axis=0, ddof=1)/np.sqrt(3), atol=2e-13)
        self.assertLess(result["max_mean_covariance_replay_difference"], 1e-13)
        self.assertAlmostEqual(result["assessment"]["tail_mean_error"], search["selected"]["tail_mean_error"])

    def test_R_prefixes_and_unestimable_uncertainty(self):
        options = self.options()
        search = planning.search(options)
        a, b = experiment.replay(options, search, 2), experiment.replay(options, search, 5)
        self.assertEqual(a["schedule"]["jump_choices"], b["schedule"]["jump_choices"][:2])
        np.testing.assert_array_equal(a["final_covariances_real_gauge"], np.asarray(b["final_covariances_real_gauge"])[:, :2])
        one = experiment.replay(options, search, 1)
        self.assertFalse(one["uncertainty"]["trajectory_standard_error_estimable"])
        self.assertFalse(one["assessment"]["sampling_budget_met"])
        self.assertNotEqual(one["run_status"], "passed")

    def test_successful_run_persists_search_and_shared_schedule_R_sweep(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            options = experiment.parse_arguments(("--n-qubits", "2", "--step-budget", "3", "--step-grid", ".5", ".25",
                "--steady-window", ".5", "--p-left", "0", "--p-right", "0", "--preparations", "vacuum",
                "--R-values", "2", "4", "--output-directory", directory, "--no-plots", "--no-progress"))
            study = experiment.main(options)
            folder = next(Path(directory).glob("budget_*"))
            a, b = (json.loads((folder/f"replay_R{R}.json").read_text()) for R in (2, 4))
            self.assertEqual((a["run_status"], b["run_status"]), ("passed", "passed"))
            self.assertEqual(a["schedule"]["step_dts"], b["schedule"]["step_dts"])
            self.assertEqual(a["workload"]["per_trajectory_steps"], 1)
            self.assertIsNone(a["workload"]["compiled_depth"])
            self.assertTrue((folder/"search.json").exists())
            self.assertTrue((folder/"Study.md").exists())
            self.assertEqual(json.loads((folder/"study.json").read_text()), study)

    def test_plotting_and_data_preservation_before_plot_failure(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--step-budget", "4", "--output-directory", directory)
            options.no_plots = False
            study = experiment.main(options)
            folder = next(Path(directory).glob("budget_*"))
            self.assertEqual(study["figure_generation_status"], "complete")
            self.assertGreater((folder/"figures/observables_R4.png").stat().st_size, 1000)
            self.assertTrue((folder/"figures/preparation_comparison.png").exists())
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--output-directory", directory)
            options.no_plots = False
            with patch.object(experiment, "plot_search", side_effect=RuntimeError("plot failed")):
                with self.assertRaisesRegex(RuntimeError, "plot failed"):
                    experiment.main(options)
            folder = next(Path(directory).glob("budget_*"))
            saved = json.loads((folder/"replay_R4.json").read_text())
            self.assertEqual(saved["figure_generation_status"], "pending")
            self.assertTrue((folder/"search.json").exists())


if __name__ == "__main__":
    unittest.main()
