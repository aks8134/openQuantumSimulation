import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply

from Model1.Experiment10 import exact
from library import classical


class ExactSectorTests(unittest.TestCase):
    def test_matches_full_lindblad_reference_with_nonuniform_times(self):
        times = np.asarray((0.0, 0.13, 0.7, 2.0))
        for n in (2, 3, 4):
            with self.subTest(n=n):
                model = exact.Model(n, J=0.8, h=0.37, gamma=5.0)
                result = exact.evolve(model, times)
                H, jumps, X, Y, Z = classical.build_linear_chain(n, model.J, model.h, model.gamma)
                L = classical.build_liouvillian(H, jumps)
                psi = classical.computational_state(n, [model.initial_site])
                rho0 = np.outer(psi, psi.conj())
                operators = classical.build_observables(eye(2**n, format="csr"), X, Y, Z, model.J)
                full = np.asarray([expm_multiply(t * L, rho0.reshape(-1, order="F"),
                    traceA=t * L.diagonal().sum()).reshape(rho0.shape, order="F") for t in times])
                for reduced, family in zip((result.populations, result.xy_correlations,
                                             result.excitation_flows), operators):
                    reference = np.asarray([classical.expectation_series(full, operator) for operator in family])
                    np.testing.assert_allclose(reduced, reference, atol=2e-12, rtol=2e-12)
                np.testing.assert_allclose(result.total_excitation,
                    1.0 - full[:, 0, 0].real, atol=2e-12)
                self.assertLess(result.max_continuity_residual, 1e-12)
                self.assertLess(result.max_excitation_balance_residual, 1e-12)

    def test_no_loss_conserves_excitation(self):
        result = exact.evolve(exact.Model(5, gamma=0), (0, 0.3, 3))
        np.testing.assert_allclose(result.total_excitation, 1, atol=1e-12)

    def test_no_hopping_boundary_and_interior_limits(self):
        times = (0, 0.2, 2)
        boundary = exact.evolve(exact.Model(4, J=0, gamma=5, initial_site=0), times)
        np.testing.assert_allclose(boundary.total_excitation, np.exp(-5 * np.asarray(times)), atol=1e-12)
        interior = exact.evolve(exact.Model(4, J=0, gamma=5, initial_site=2), times)
        np.testing.assert_allclose(interior.total_excitation, 1, atol=1e-12)

    def test_nonzero_single_time_evolves_from_initial_state(self):
        model = exact.Model(4)
        single = exact.evolve(model, (0.8,))
        sequence = exact.evolve(model, (0, 0.2, 0.8))
        np.testing.assert_allclose(single.populations[:, 0], sequence.populations[:, -1], atol=1e-12)
        self.assertEqual(single.times.tolist(), [0.8])

    def test_high_n_uses_only_sparse_n_component_generator(self):
        model = exact.Model(128)
        generator = exact.build_generator(model)
        self.assertEqual(generator.shape, (128, 128))
        self.assertLessEqual(generator.nnz, 3 * 128)
        with patch.object(classical, "build_liouvillian", side_effect=AssertionError("must not construct Liouvillian")):
            result = exact.evolve(model, (0, 0.1, 2))
        self.assertEqual(result.populations.shape, (128, 3))
        self.assertEqual(result.final_amplitude.shape, (128,))

    def test_until_steady_stops_on_grid_and_respects_horizon(self):
        model = exact.Model(2, J=0, gamma=5, initial_site=0)
        result = exact.evolve(model, (0, 1, 2, 3, 4), steady_tolerance=1e-4, until_steady=True)
        self.assertEqual(result.times.tolist(), [0, 1, 2])
        self.assertLessEqual(result.total_excitation[-1], 1e-4)
        unfinished = exact.evolve(exact.Model(4, J=0, initial_site=2), (0, 1), until_steady=True)
        self.assertEqual(unfinished.times.tolist(), [0, 1])
        self.assertGreater(unfinished.total_excitation[-1], 1e-8)


class ConfigurationAndOutputTests(unittest.TestCase):
    def test_defaults_match_requested_long_time_problem(self):
        options = exact.parse_arguments(())
        self.assertEqual(options.n_qubits, 10)
        self.assertEqual(options.gamma, 5)
        self.assertEqual(options.times[-1], 1000)
        self.assertEqual(len(options.times), 1001)

    def test_invalid_physics_times_and_tolerances_fail(self):
        for args in (("--n-qubits", "1"), ("--gamma", "-1"), ("--h", "nan"),
                     ("--initial-site", "10"), ("--t-final", "0"), ("--time-points", "1"),
                     ("--times", "1", "0"), ("--times", "0", "0"),
                     ("--times", "0", "--time-points", "2"),
                     ("--steady-tolerance", "1")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                exact.parse_arguments(args)
        with self.assertRaises(ValueError):
            exact.evolve(exact.Model(), ())

    def test_hash_changes_with_physics_and_time_grid(self):
        model = exact.Model(10)
        def path(model, times):
            config = exact.configuration(model, times, steady_tolerance=1e-8, until_steady=False)
            return exact.output_paths("unused", config)["result"]
        first = path(model, (0, 1, 2))
        self.assertEqual(first, path(model, (0, 1, 2)))
        self.assertNotEqual(first, path(model, (0, 0.5, 2)))
        self.assertNotEqual(first, path(exact.Model(10, gamma=1), (0, 1, 2)))

    def test_json_saved_before_plot_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            options = exact.parse_arguments(("--n-qubits", "2", "--times", "0", "0.3",
                "--no-progress", "--output-directory", directory))
            with patch.object(exact, "plot_result", side_effect=RuntimeError("render failed")):
                with self.assertRaisesRegex(RuntimeError, "render failed"):
                    exact.main(options)
            files = list((Path(directory) / "results").glob("*.json"))
            self.assertEqual(len(files), 1)
            payload = json.loads(files[0].read_text())
            self.assertEqual(payload["figure_generation_status"], "pending")
            self.assertEqual(payload["propagated_vector_dimension"], 2)

    def test_cli_no_plots_and_headless_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            options = exact.parse_arguments(("--n-qubits", "3", "--times", "0.3",
                "--no-progress", "--no-plots", "--output-directory", directory))
            payload = exact.main(options)
            self.assertEqual(payload["figure_generation_status"], "disabled")
            self.assertFalse(payload["steady_state"]["reached"])
            figure = Path(directory) / "test.png"
            import matplotlib.pyplot as plt
            with patch.object(plt, "close", wraps=plt.close) as close:
                exact.plot_result(payload, figure)
            plotted = close.call_args.args[0]
            self.assertEqual(len(plotted.axes), 4)
            self.assertEqual([len(axis.lines) for axis in plotted.axes], [3, 2, 2, 2])
            for axis in plotted.axes:
                self.assertEqual(len(axis.collections), 0)  # No heatmaps/colorbars.
            for axis, key in zip(plotted.axes[:3],
                                 ("populations", "xy_correlations", "excitation_flows")):
                for line, curve in zip(axis.lines, payload["observables"][key]):
                    np.testing.assert_array_equal(line.get_xdata(), payload["times"])
                    np.testing.assert_array_equal(line.get_ydata(), curve)
                    self.assertEqual(line.get_marker(), "o")
            self.assertEqual(plotted.axes[3].get_yscale(), "log")
            self.assertTrue(figure.is_file())
            self.assertGreater(figure.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
