"""Forward validation must not charge its mean-only continuation to a circuit."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from library import reservoir_scaling as physics
from Model3.Experiment4 import budgeted_steady_state as experiment
from Model3.Experiment4 import planning


class ForwardWindowTests(unittest.TestCase):
    def options(self, *extra):
        return experiment.parse_arguments(["--n-qubits", "2", "--step-budget", "8", "--step-grid", ".5", ".25",
            "--preparations", "half_mixed", "--block-steps", "2", "--beam-width", "8",
            "--steady-window", "1", "--steady-tolerance", ".15", "--sampling-tolerance", ".02",
            "--validation-mode", "forward", "--trajectories", "4", "--no-progress", "--no-plots", *extra])

    def test_validation_uses_future_errors_and_includes_output(self):
        options = self.options("--steady-window", ".5")
        prep = {"name": "test", "local_preparation_depth": 1}
        node = planning.Node(np.zeros((2, 2)), (0, 0, 0), (0., .5, 1., 1.5), (.3, .2, .1, .01), 1)
        result = planning.evaluate(node, prep, options)
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["classical_validation_steps"], 1)
        self.assertAlmostEqual(result["tail_mean_error"], .1)
        self.assertTrue(result["mean_target_met"])
        bad = planning.Node(node.covariance, node.indices, node.times, (.3, .2, .14, .01), 1)
        self.assertFalse(planning.evaluate(bad, prep, options)["mean_target_met"])

    def test_validation_cannot_include_future_refinement(self):
        options = self.options("--steady-window", ".75")
        prep = {"name": "test", "local_preparation_depth": 1}
        node = planning.Node(np.zeros((2, 2)), (0, 0, 1, 1, 1),
            (0., .5, 1., 1.25, 1.5, 1.75), (.3, .2, .1, .05, .02, .01), 2, stage_start=2)
        # Output would precede all fine steps: that is not a constant-final-step continuation.
        self.assertFalse(planning.evaluate(node, prep, options)["tail_window_covered"])
        extended = planning.Node(node.covariance, node.indices+(1,), node.times+(2.,), node.errors+(.01,), 2, stage_start=2)
        result = planning.evaluate(extended, prep, options)
        self.assertTrue(result["tail_window_covered"])
        self.assertEqual(result["steps"], 3)

    def test_validation_length_rounding_and_budget_accounting(self):
        self.assertEqual(planning.holding_steps(.1, 1.), 10)
        self.assertEqual(planning.holding_steps(.075, 10.), 134)
        options = self.options("--step-budget", "2", "--estimated-step-depth", "4", "--depth-budget", "9")
        search = planning.search(options)
        for row in search["best_per_preparation"]+search["constant_baselines"]:
            self.assertLessEqual(row["steps"], 2)
            self.assertLessEqual(row["estimated_total_depth"], 9)
        chosen = search["selected"]
        self.assertEqual(len(chosen["step_dts"]), chosen["steps"])
        self.assertEqual(len(chosen["forward_validation"]["step_dts"]), chosen["classical_validation_steps"])
        self.assertGreaterEqual(chosen["window_end"]-chosen["window_start"], options.steady_window-1e-13)

    def test_wide_beam_matches_exhaustive_two_stage_forward_search(self):
        options = self.options("--step-budget", "5", "--block-steps", "1", "--max-stages", "2", "--beam-width", "4096")
        result = planning.search(options)
        target = np.asarray(result["physical_stationary_real_gauge"])
        prep = result["preparations"][0]
        branches = [physics.channel(options.model, options.J*dt, options.sampling_policy,
                                    options.order, options.system_formula) for dt in options.step_grid]
        oracle = []
        for count in range(6):
            for indices in itertools.product(range(2), repeat=count):
                if any(a > b for a, b in zip(indices, indices[1:])):
                    continue
                state = np.diag(prep["populations"])
                for index in indices:
                    state = physics.mean_step(state, branches[index])
                # Zero-step preparation can be validated with either final step.
                for last in ([indices[-1]] if indices else range(2)):
                    errors = [planning.distance(state, target, options.model)]
                    future = state
                    for _ in range(planning.holding_steps(options.step_grid[last], options.steady_window)):
                        future = physics.mean_step(future, branches[last])
                        errors.append(planning.distance(future, target, options.model))
                    score = max(errors)
                    key = ((0, count, score, 1) if score <= options.mean_tolerance else (1, score, count, errors[0]))
                    oracle.append(key)
        selected = result["selected"]
        chosen_key = ((0, selected["steps"], selected["tail_mean_error"], 1) if selected["mean_target_met"]
                      else (1, selected["tail_mean_error"], selected["steps"], selected["final_mean_error"]))
        np.testing.assert_allclose(chosen_key, min(oracle), atol=1e-13)

    def test_independent_mean_continuation_and_output_only_sampling(self):
        options = self.options()
        search = planning.search(options)
        result = experiment.replay(options, search, 4)
        self.assertEqual(result["assessment"]["sampling_scope"], "output_only")
        self.assertEqual(result["assessment"]["validation_mode"], "forward")
        chosen = search["selected"]
        self.assertEqual(result["workload"]["per_trajectory_steps"], chosen["steps"])
        self.assertEqual(len(result["schedule"]["jump_choices"][0]), chosen["steps"])
        self.assertEqual(result["times"][-1], chosen["final_time"])
        self.assertGreater(result["assessment"]["window_end"], result["times"][-1])
        self.assertAlmostEqual(max(result["mean_forward_validation"]["local_observable_distance"]), chosen["tail_mean_error"])
        for a, b in (("populations", 1), ("excitation_flows", options.J)):
            mean = np.array(result["mean_channel_observables"][a])[0, -1]
            sample = np.array(result["observables"][a])[0, -1]
            self.assertLessEqual(abs(sample-mean).max()/b, result["assessment"]["max_sampling_distance"]+1e-14)
        self.assertLess(result["max_mean_covariance_replay_difference"], 1e-13)

    def test_baseline_winner_is_retained_and_incompatible_physics_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            options = self.options("--block-steps", "1")
            seed = planning.search(options)
            path = Path(folder)/"study.json"
            study = {"experiment": "Model3/Experiment4", "configuration": experiment.configuration(options),
                     "selected": seed["selected"], "best_per_preparation": seed["best_per_preparation"]}
            path.write_text(json.dumps(study))
            coarse = self.options("--block-steps", "4", "--beam-width", "1", "--search-node-cap", "0", "--baseline-study", str(path))
            rows, manifest = planning.load_baselines(coarse)
            pooled = planning.search(coarse, baselines=rows)
            self.assertEqual(len(pooled["incumbent_baselines"]), 1)
            self.assertEqual(len(manifest[0]["sha256"]), 64)
            self.assertLessEqual(pooled["selected"]["steps"], seed["selected"]["steps"])
            changed = self.options("--J", "2", "--baseline-study", str(path))
            with self.assertRaisesRegex(ValueError, "physics/channel mismatch"):
                planning.load_baselines(changed)

    def test_plan_only_does_not_replay_and_forward_plots_persist(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--output-directory", folder, "--plan-only")
            study = experiment.main(options)
            self.assertEqual(study["replays"], {})
            self.assertEqual(study["configuration"]["schema_version"], 2)
            self.assertEqual(study["configuration"]["validation_mode"], "forward")
            self.assertFalse(list(Path(folder).glob("budget_*/replay_R*.json")))
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--output-directory", folder)
            options.no_plots = False
            study = experiment.main(options)
            self.assertEqual(study["figure_generation_status"], "complete")
            self.assertTrue(list(Path(folder).glob("budget_*/figures/forward_validation_R4.png")))

    def test_zero_preparation_steps_can_be_validated_beyond_budget(self):
        options = self.options("--preparations", "vacuum", "--p-left", "0", "--p-right", "0", "--step-budget", "1")
        search = planning.search(options)
        self.assertEqual(search["selected"]["steps"], 0)
        self.assertTrue(search["selected"]["mean_target_met"])
        self.assertGreater(search["selected"]["classical_validation_steps"], options.step_budget)
        result = experiment.replay(options, search, 4)
        self.assertEqual(result["run_status"], "passed")
        self.assertEqual(result["workload"]["total_channel_applications"], 0)


if __name__ == "__main__":
    unittest.main()
