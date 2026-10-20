"""Check manual schedules against independent full-density channels."""
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
from Model2.Experiment2 import randomized
from Model3.Experiment1 import simulation
from Model3.Experiment3 import classical_randomized_lie_trotter as experiment


def full_branches(options, dt):
    H, jumps, *_ = simulation.build_model(options.n_qubits, J=options.J,
        gamma=options.gamma, p_left=options.p_left, p_right=options.p_right,
        exchange_prefactor=options.exchange_prefactor)
    fractions = np.array([options.p_left, 1-options.p_left, options.p_right, 1-options.p_right])
    probabilities = fractions/2 if options.sampling_policy == "rate-weighted" else np.full(4, .25)
    if options.system_formula == "exact":
        U = expm(-1j*dt*H.toarray())
    else:
        U = np.eye(H.shape[0], dtype=complex)
        for parity in (0, 1):
            half = np.zeros(H.shape, dtype=complex)
            for site in range(parity, options.n_qubits-1, 2):
                for pauli in (classical.X, classical.Y):
                    left = classical.operator_on_site(pauli, site, options.n_qubits)
                    right = classical.operator_on_site(pauli, site+1, options.n_qubits)
                    half += options.exchange_prefactor*options.J*(left@right).toarray()
            U = expm(-1j*dt*half)@U
    kraus = []
    for probability, jump in zip(probabilities, jumps):
        if probability == 0:
            kraus.append(np.array([U]))
            continue
        reset = randomized.sampled_kraus(np.zeros(H.shape), jump, probability, dt)
        kraus.append(U@reset if options.order == "jump-first" else reset@U)
    return probabilities, kraus


def covariance(rho, n):
    annihilators = []
    for site in range(n):
        op = np.ones((1, 1))
        for j in range(n):
            term = classical.Z if j < site else classical.SIGMA_MINUS if j == site else classical.I2
            op = np.kron(op, term.toarray())
        annihilators.append(op)
    return np.asarray([[np.trace(b.conj().T@a@rho) for b in annihilators] for a in annihilators])


class ManualScheduleTests(unittest.TestCase):
    def options(self, *args):
        return experiment.parse_arguments(args)

    def test_interval_remainders_and_explicit_steps(self):
        nodes = experiment.build_schedule(segments=[(.25, .1), (.6, .2)])
        np.testing.assert_allclose(nodes, [0, .1, .2, .25, .45, .6], atol=1e-15)
        np.testing.assert_allclose(np.diff(nodes), [.1, .1, .05, .2, .15], atol=1e-15)
        explicit = self.options("--step-dts", ".01", ".02", ".04")
        np.testing.assert_allclose(explicit.nodes, [0, .01, .03, .07])
        baseline = self.options("--t-final", ".3", "--trotter-delta-t", ".1", "--step-cap", "3")
        self.assertEqual(len(baseline.nodes), 4)
        self.assertEqual(baseline.nodes[-1], .3)

    def test_defaults_and_supplied_physics(self):
        default = self.options()
        self.assertEqual((default.J, default.gamma, default.p_left, default.p_right), (1, 1, .9, .1))
        self.assertEqual((default.order, default.system_formula), ("hamiltonian-first", "even-odd"))
        self.assertEqual(default.initial_states, ["000", "111", "100"])
        self.assertEqual(default.nodes[-1], 20)
        changed = self.options("--J", "2", "--gamma", "3", "--initial-states", "010")
        self.assertEqual(changed.model.ratio, 1.5)
        self.assertEqual(changed.initial_states, ["010"])

    def test_invalid_inputs_and_conflicting_modes(self):
        cases = (("--J", "0"), ("--gamma", "nan"), ("--gamma", "-1"),
            ("--p-left", "1.1"), ("--n-qubits", "1"), ("--trajectories", "0"),
            ("--seed", "-1"), ("--record-every", "0"), ("--steady-tolerance", "0"),
            ("--dt-schedule", "2"), ("--dt-schedule", "1:.1", "1:.2"),
            ("--dt-schedule", "1:0"), ("--step-dts", "nan"),
            ("--dt-schedule", "1:.1", "--t-final", "2"),
            ("--step-dts", ".1", "--t-final", ".2"),
            ("--t-final", "1", "--trotter-delta-t", ".01", "--step-cap", "10"),
            ("--initial-states", "1011"), ("--initial-states", "abc"),
            ("--initial-states", "100", "100"))
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.options(*args)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.options("--step-dts", ".1", "--trotter-delta-t", ".2")

    def test_variable_step_sampled_paths_match_full_density(self):
        for order, formula in itertools.product(("jump-first", "hamiltonian-first"), ("exact", "even-odd")):
            options = self.options("--J", "1.7", "--gamma", "20", "--step-dts", ".1", ".03", ".2",
                "--order", order, "--system-formula", formula, "--initial-states", "100", "--trajectories", "3")
            result = experiment.evolve(options)
            choices = np.asarray(result["schedule"]["jump_choices"])
            final = np.asarray(result["final_covariances_real_gauge"])[0]
            phases = (-1j)**np.arange(options.n_qubits)
            actual = phases[:, None]*final*phases.conj()[None, :]
            for trajectory in range(options.trajectories):
                rho = np.zeros((8, 8), complex)
                rho[4, 4] = 1
                for step, dt in enumerate(result["schedule"]["step_dts"]):
                    _, branches = full_branches(options, dt)
                    kraus = branches[choices[trajectory, step]]
                    rho = sum(K@rho@K.conj().T for K in kraus)
                with self.subTest(order=order, formula=formula, trajectory=trajectory):
                    np.testing.assert_allclose(actual[trajectory], covariance(rho, 3), atol=2e-13)

    def test_mean_channel_matches_all_variable_two_step_sequences(self):
        for policy in ("uniform", "rate-weighted"):
            options = self.options("--J", "2.3", "--gamma", "1.4", "--step-dts", ".07", ".13",
                "--sampling-policy", policy, "--initial-states", "010")
            result = experiment.evolve(options)
            branches = [full_branches(options, dt) for dt in result["schedule"]["step_dts"]]
            initial = np.zeros((8, 8), complex)
            initial[2, 2] = 1
            average = np.zeros_like(initial)
            for a, b in itertools.product(range(4), repeat=2):
                first = sum(K@initial@K.conj().T for K in branches[0][1][a])
                second = sum(K@first@K.conj().T for K in branches[1][1][b])
                average += branches[0][0][a]*branches[1][0][b]*second
            _, _, _, pop_ops, current_ops, _ = simulation.build_model(3, J=options.J, gamma=options.gamma)
            for family, ops in (("populations", pop_ops), ("excitation_flows", current_ops)):
                expected = [np.trace(op@average).real for op in ops]
                np.testing.assert_allclose(result["mean_channel_observables"][family][0][-1], expected, atol=1e-13)

    def test_exact_reference_and_stationary_targets_use_physical_units(self):
        options = self.options("--J", "2.5", "--gamma", "1.3", "--step-dts", ".03", ".2", ".07")
        result = experiment.evolve(options)
        z, currents = simulation.xx_correlations(result["times"], options.initial_states, J=options.J,
            gamma=options.gamma, p_left=options.p_left, p_right=options.p_right)
        np.testing.assert_allclose(result["exact_reference_observables"]["magnetizations"], z, atol=2e-13)
        np.testing.assert_allclose(result["exact_reference_observables"]["excitation_flows"], currents, atol=2e-13)
        target = simulation.xx_stationary_targets(J=options.J, gamma=options.gamma)
        np.testing.assert_allclose(result["stationary_observables"]["populations"], target["populations"], atol=2e-14)
        np.testing.assert_allclose(result["stationary_observables"]["excitation_flows"], target["I"], atol=2e-14)

    def test_recording_preserves_evolution_and_final_sample(self):
        args = ("--dt-schedule", ".2:.03", ".5:.07")
        full = experiment.evolve(self.options(*args))
        thin = experiment.evolve(self.options(*args, "--record-every", "5"))
        self.assertEqual(thin["schedule"]["recorded_step_counts"], [0, 5, 10, 12])
        self.assertEqual(full["schedule"]["jump_choices"], thin["schedule"]["jump_choices"])
        self.assertEqual(full["final_covariances_real_gauge"], thin["final_covariances_real_gauge"])
        for family in experiment.FAMILIES:
            np.testing.assert_array_equal(np.asarray(full["observables"][family])[:, [0, 5, 10, 12]], thin["observables"][family])

    def test_seed_repeatability_and_stable_trajectory_prefix(self):
        args = ("--step-dts", ".02", ".05", ".01")
        first = experiment.evolve(self.options(*args, "--trajectories", "2"))
        repeat = experiment.evolve(self.options(*args, "--trajectories", "2"))
        extended = experiment.evolve(self.options(*args, "--trajectories", "4"))
        self.assertEqual(first, repeat)
        self.assertEqual(first["schedule"]["jump_choices"], extended["schedule"]["jump_choices"][:2])
        np.testing.assert_array_equal(first["final_covariances_real_gauge"], np.asarray(extended["final_covariances_real_gauge"])[:, :2])

    def test_endpoint_reservoirs_skip_zero_probability_jumps(self):
        for policy in ("uniform", "rate-weighted"):
            options = self.options("--p-left", "1", "--p-right", "0", "--step-dts", ".1", ".03",
                "--sampling-policy", policy)
            with np.errstate(invalid="raise", divide="raise"):
                result = experiment.evolve(options)
            if policy == "rate-weighted":
                self.assertTrue(set(np.asarray(result["schedule"]["jump_choices"]).ravel()) <= {0, 3})
            self.assertTrue(np.all(np.isfinite(result["final_covariances_real_gauge"])))
            eigenvalues = np.linalg.eigvalsh(result["final_covariances_real_gauge"])
            self.assertGreaterEqual(eigenvalues.min(), -1e-13)
            self.assertLessEqual(eigenvalues.max(), 1+1e-13)

    def test_single_trajectory_uncertainty_and_standard_errors(self):
        args = ("--step-dts", ".03", ".07")
        single = experiment.evolve(self.options(*args, "--trajectories", "1"))
        self.assertFalse(single["uncertainty"]["trajectory_standard_error_estimable"])
        for error in single["trajectory_standard_errors"].values():
            self.assertFalse(np.any(error))
        options = self.options(*args, "--trajectories", "8")
        result = experiment.evolve(options)
        values = experiment.observables(np.asarray(result["final_covariances_real_gauge"]), options)
        for family, data in values.items():
            np.testing.assert_allclose(np.asarray(result["trajectory_standard_errors"][family])[:, -1],
                                       data.std(axis=1, ddof=1)/np.sqrt(8), atol=1e-15)

    def test_persistence_plots_and_schedule_configuration_separation(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args = ("--output-directory", directory, "--initial-states", "100", "--no-progress")
            first = experiment.main(self.options(*args, "--step-dts", ".03", ".07"))
            self.assertEqual(first["figure_generation_status"], "complete")
            for path in Path(directory).glob("figures/*.png"):
                self.assertGreater(path.stat().st_size, 1000)
            self.assertEqual(len(list(Path(directory).glob("figures/*.png"))), 2)
            result_file = next(Path(directory).glob("results/*.json"))
            self.assertEqual(json.loads(result_file.read_text()), first)
            experiment.main(self.options(*args, "--step-dts", ".05", ".05", "--no-plots"))
            self.assertEqual(len(list(Path(directory).glob("results/*.json"))), 2)

    def test_data_persists_before_failed_plot(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            options = self.options("--output-directory", directory, "--step-dts", ".03", "--no-progress")
            with patch.object(experiment, "plot_result", side_effect=RuntimeError("plot failed")):
                with self.assertRaisesRegex(RuntimeError, "plot failed"):
                    experiment.main(options)
            saved = json.loads(next(Path(directory).glob("results/*.json")).read_text())
            self.assertEqual(saved["figure_generation_status"], "pending")
            self.assertIn("exact_reference_observables", saved)


if __name__ == "__main__":
    unittest.main()
