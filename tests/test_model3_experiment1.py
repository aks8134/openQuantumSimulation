import unittest

import numpy as np

from library import classical
from Model2.Experiment1 import simulation as engines
from Model3.Experiment1 import simulation as model


class ModelThreeExperimentOneTests(unittest.TestCase):
    def test_reduced_stationary_solve_matches_full_solve(self):
        for delta in (0.0, 0.7):
            H, jumps, _, _, _, reduced = model.build_model(delta=delta)
            full = model.stationary_state(classical.build_liouvillian(H, jumps), H.shape[0])
            np.testing.assert_allclose(reduced, full, atol=2e-14)

    def test_sparse_dilation_action_matches_dense_joint_exponentials(self):
        H, jumps, *_ = model.build_model(delta=0.7, p_left=0.8, p_right=0.3)
        for formula in ("unsplit", "lie", "strang"):
            actual = model.dilation_kraus(H, jumps, 0.1, formula)
            expected = engines.dilation_kraus(H, jumps, 0.1, formula)
            np.testing.assert_allclose(actual, expected, atol=2e-14)
            _, initial = engines.initial_states(3)
            actual_trajectory = engines.dilation_evolve(initial, H, jumps, [0, 0.1, 0.3],
                max_dt=0.1, formula=formula, kraus_builder=model.dilation_kraus)
            expected_trajectory = engines.dilation_evolve(initial, H, jumps, [0, 0.1, 0.3],
                max_dt=0.1, formula=formula)
            np.testing.assert_allclose(actual_trajectory, expected_trajectory, atol=2e-14)

    def test_full_stationary_solve_matches_all_three_quoted_targets(self):
        cases = ((0.25, 0.25, (0.5, 0.5, 0.5), 0),
                 (0.9, 0.1, (-0.4, 0, 0.4), 0.4),
                 (0.1, 0.9, (0.4, 0, -0.4), -0.4))
        for p_left, p_right, expected_Z, expected_I in cases:
            H, jumps, Z, _, I, rho = model.build_model(p_left=p_left, p_right=p_right)
            L = classical.build_liouvillian(H, jumps)
            np.testing.assert_allclose(L @ rho.reshape(-1, order="F"), 0, atol=2e-14)
            np.testing.assert_allclose(np.trace(rho), 1, atol=2e-14)
            np.testing.assert_allclose(rho, rho.conj().T, atol=2e-14)
            self.assertGreater(np.linalg.eigvalsh(rho).min(), -2e-14)
            np.testing.assert_allclose(engines.expectations(rho[None, None], Z)[0, 0], expected_Z, atol=2e-14)
            np.testing.assert_allclose(engines.expectations(rho[None, None], I)[0, 0], expected_I, atol=2e-14)

    def test_prefactor_and_chain_length_have_correct_stationary_targets(self):
        for n in (2, 3, 4):
            for c in (0.25, 0.5):
                H, jumps, Z, _, I, rho = model.build_model(n, J=-0.7, gamma=1.3,
                    p_left=0.8, p_right=0.3, exchange_prefactor=c)
                target = model.xx_stationary_targets(n, J=-0.7, gamma=1.3,
                    p_left=0.8, p_right=0.3, exchange_prefactor=c)
                np.testing.assert_allclose(engines.expectations(rho[None, None], Z)[0, 0], target["Z"], atol=3e-14)
                np.testing.assert_allclose(engines.expectations(rho[None, None], I)[0, 0], target["I"], atol=3e-14)
        target = model.xx_stationary_targets(exchange_prefactor=0.25)
        np.testing.assert_allclose(target["Z"], (-0.64, 0, 0.64), atol=1e-14)
        np.testing.assert_allclose(target["I"], (0.16, 0.16), atol=1e-14)

    def test_reservoir_reversal_reflects_the_full_stationary_density(self):
        forward = model.build_model(p_left=0.8, p_right=0.3)[-1]
        reverse = model.build_model(p_left=0.3, p_right=0.8)[-1]
        reflected = forward.reshape((2,) * 6).transpose(2, 1, 0, 5, 4, 3).reshape(8, 8)
        np.testing.assert_allclose(reverse, reflected, atol=2e-14)

    def test_boundary_continuity_is_an_operator_identity_with_interactions(self):
        H, jumps, _, populations, currents, _ = model.build_model(delta=0.7)
        L = classical.build_liouvillian(H, jumps)
        identity = np.eye(8)
        rhs = (2 * (0.9 * identity - populations[0].toarray()) - currents[0].toarray(),
               currents[0].toarray() - currents[1].toarray(),
               currents[1].toarray() + 2 * (0.1 * identity - populations[2].toarray()))
        for n, expected in zip(populations, rhs):
            adjoint = (L.getH() @ n.toarray().reshape(-1, order="F")).reshape(8, 8, order="F")
            np.testing.assert_allclose(adjoint, expected, atol=2e-14)

    def test_many_body_evolution_matches_independent_correlations(self):
        for n, times, c in ((3, [0, 0.13, 0.6, 2, 20], 0.5),
                            (4, [0.17, 0.83], 0.25), (2, [0.23], 0.5)):
            result = model.run_experiment(times, number_of_qubits=n,
                exchange_prefactor=c, p_left=0.7, p_right=0.2, methods=())
            self.assertLess(result["diagnostics"]["maximum_Z_error_vs_correlations"], 2e-13)
            self.assertLess(result["diagnostics"]["maximum_I_error_vs_correlations"], 2e-13)
            self.assertLess(result["diagnostics"]["continuity_residual"], 2e-13)
            if n == 3:
                np.testing.assert_allclose(result["magnetizations"]["exact"][:, -1],
                    np.tile(result["stationary"]["Z"], (3, 1)), atol=1e-7)

    def test_all_dilation_methods_are_cptp_and_improve_under_refinement(self):
        H, jumps, *_ = model.build_model()
        L = classical.build_liouvillian(H, jumps)
        _, initial = engines.initial_states(3)
        times = [0.13, 0.37, 1.0]
        exact = engines.exact_evolve(initial, L, times)
        for formula in ("unsplit", "lie", "strang"):
            errors = []
            for dt in (0.1, 0.05, 0.025):
                rhos = engines.dilation_evolve(initial, H, jumps, times, max_dt=dt, formula=formula)
                np.testing.assert_allclose(np.trace(rhos, axis1=-2, axis2=-1), 1, atol=2e-13)
                np.testing.assert_allclose(rhos, rhos.conj().swapaxes(-1, -2), atol=2e-14)
                self.assertGreater(np.linalg.eigvalsh(rhos).min(), -2e-14)
                errors.append(np.max(np.linalg.norm(rhos - exact, axis=(-2, -1))))
            self.assertLess(errors[1], errors[0])
            self.assertLess(errors[2], errors[1])

    def test_fully_polarized_reservoirs_and_interacting_model(self):
        for p_left, p_right in ((0, 0), (1, 1), (1, 0)):
            result = model.run_experiment([0, 0.1], p_left=p_left, p_right=p_right, methods=())
            self.assertLess(result["diagnostics"]["stationary_residual"], 2e-14)
        interacting = model.run_experiment([0, 0.1], delta=0.7, methods=())
        self.assertNotIn("maximum_stationary_Z_error_vs_analytic", interacting["diagnostics"])
        self.assertLess(interacting["diagnostics"]["stationary_residual"], 2e-14)
        for p in (-0.1, 1.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                model.build_model(p_left=p)
            with self.assertRaises(ValueError):
                model.build_model(p_right=p)


if __name__ == "__main__":
    unittest.main()
