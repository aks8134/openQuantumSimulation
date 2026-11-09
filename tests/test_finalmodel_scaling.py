"""Physics, acceptance and resource checks for the offline FinalModel study."""
import copy
import json
import unittest

import numpy as np
from scipy.linalg import expm

from FinalModel.Experiment1 import study
from library import reservoir_scaling as physics


class FinalModelTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((study.DIRECTORY/"config.json").read_text())

    def test_fast_mean_matches_existing_channel_including_n2_and_zero_pumps(self):
        for n in (2, 3, 7):
            for spec in self.config["models"].values():
                model = study.model_for(n, spec)
                # Physical, nondiagonal real covariance exercises all matrix entries.
                Q, _ = np.linalg.qr(np.random.default_rng(17).normal(size=(n, n)))
                state = Q@np.diag(np.linspace(.1, .9, n))@Q.T
                for dt in (.001, .2, 1.7):
                    with self.subTest(n=n, dt=dt, spec=spec):
                        branch = physics.channel(model, spec["J"]*dt, "rate-weighted", "hamiltonian-first", "even-odd")
                        actual = study.fast_step(state, model, dt, spec["J"])
                        np.testing.assert_allclose(actual, physics.mean_step(state, branch), atol=5e-15)
                        np.testing.assert_allclose(actual, actual.T, atol=1e-15)
                        self.assertGreaterEqual(np.linalg.eigvalsh(actual).min(), -1e-13)

    def test_continuous_certificate_controls_later_local_errors(self):
        for spec in self.config["models"].values():
            model = study.model_for(4, spec)
            result = study.exact_case(4, spec, self.config)
            t = result["settling_time_upper_bound"]
            self.assertEqual(result["status"], "permanent_bound_met")
            self.assertLessEqual(result["bound_at_upper"], self.config["mean_tolerance"])
            self.assertLess(result["stationary_residual"], 1e-12)
            target = physics.exact_stationary(model)
            delta = study.initial(4, spec["preparation"])-target
            for later in (t, 1.1*t, 3*t):
                E = expm(later*spec["J"]*physics.generator(model))
                self.assertLessEqual(study.error(target+E@delta@E.T, target, model), result["bound_at_upper"]+1e-12)

    def test_cap_is_not_reported_as_settling(self):
        self.config["time_cap"] = 1e-5
        result = study.exact_case(10, self.config["models"]["model1"], self.config)
        self.assertEqual(result["status"], "time_cap_reached")
        self.assertIsNone(result["settling_time_upper_bound"])

    def test_warped_endpoints_and_constant_subset(self):
        dts = study.schedule("warped_cosine", 11, np.log([.03, .7, 2.4]))
        self.assertAlmostEqual(dts[0], .7)
        self.assertAlmostEqual(dts[-1], .03)
        self.assertTrue(np.all(np.diff(dts) <= 0))
        np.testing.assert_allclose(study.schedule("warped_cosine", 8, np.log([.2, .2, 3.])), .2)
        np.testing.assert_allclose(study.schedule("warped_cosine", 1, np.log([.03, .7, 2.])), [.03])

    def test_dense_independent_schedule_replay_and_forward_cap(self):
        spec = self.config["models"]["model2"]
        model = study.model_for(2, spec)
        dts = np.full(100, .1)
        fast = study.evaluate(dts, model, spec, self.config)
        dense = study.evaluate(dts, model, spec, self.config, independent=True)
        self.assertTrue(dense["passed"])
        self.assertAlmostEqual(fast["maximum_forward_error"], dense["maximum_forward_error"], places=13)
        self.config["validation_step_cap"] = 1
        self.assertFalse(study.evaluate(dts, model, spec, self.config)["passed"])

    def test_circuit_channel_matches_reduced_mean_for_each_model(self):
        from qiskit.quantum_info import DensityMatrix, partial_trace
        # Enumerate the four branches rather than using a finite trajectory sample.
        from unittest.mock import patch
        for spec in self.config["models"].values():
            n, dt = 3, .23
            model = study.model_for(n, spec)
            expected = study.fast_step(study.initial(n, spec["preparation"]), model, dt, spec["J"])
            weights = np.array([spec["p_left"], 1-spec["p_left"], spec["p_right"], 1-spec["p_right"]])/2
            density = np.zeros((2**n, 2**n), complex)
            for choice, weight in enumerate(weights):
                if weight == 0:
                    continue
                class FixedRng:
                    def choice(self, *args, **kwargs):
                        return np.array([choice])
                with patch.object(study.np.random, "default_rng", return_value=FixedRng()):
                    circuit = study.build_circuit(n, spec, [dt], 0)
                # Replace ignored preparation measurements by dephasing channels.
                from qiskit.quantum_info import Kraus
                dm = DensityMatrix.from_int(0, 2**(n+2))
                for instruction in circuit.data:
                    qubits = [circuit.find_bit(q).index for q in instruction.qubits]
                    if instruction.operation.name == "measure":
                        if circuit.find_bit(instruction.clbits[0]).index < n:
                            dm = dm.evolve(Kraus([np.diag([1, 0]), np.diag([0, 1])]), qubits)
                    elif instruction.operation.name == "reset":
                        dm = dm.evolve(Kraus([np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])]), qubits)
                    else:
                        dm = dm.evolve(instruction.operation, qubits)
                density += weight*partial_trace(dm, [0, n+1]).data
            populations = [sum(density[k, k].real for k in range(2**n) if (k >> j)&1) for j in range(n)]
            np.testing.assert_allclose(populations, np.diag(expected), atol=2e-14)
            X = np.array([[0, 1], [1, 0]])
            Y = np.array([[0, -1j], [1j, 0]])
            def local(operator, site):
                factors = [operator if j == site else np.eye(2) for j in reversed(range(n))]
                result = factors[0]
                for factor in factors[1:]:
                    result = np.kron(result, factor)
                return result
            currents = [np.trace(density@(local(Y, j)@local(X, j+1)-local(X, j)@local(Y, j+1))).real*spec["prefactor"]
                        for j in range(n-1)]
            np.testing.assert_allclose(currents, 4*spec["prefactor"]*np.diag(expected, 1), atol=2e-14)

    def test_resources_measure_real_compilation_and_label_T_estimates(self):
        spec = self.config["models"]["model1"]
        config = copy.deepcopy(self.config)
        config["resource_paths"] = 1
        source = {"families": {"fixed": {"selected_steps": 2, "step_dts": [.1, .1]},
                               "constant": {"selected_steps": None}}}
        result = study.resources_case(3, spec, config, source)
        compiled = [r for r in result["records"] if r["status"] == "compiled"]
        self.assertEqual(len(compiled), 2)
        for row in compiled:
            self.assertEqual(row["logical_qubits"], 5)
            self.assertEqual(row["logical_exchanges"], 6)
            self.assertEqual(row["native_operations"]["reset"], 2)
            self.assertGreater(row["native_operations"]["cx"], 0)
            self.assertIsNone(row["fault_tolerant"]["T_depth"])
        config["maximum_compile_exchanges"] = 1
        self.assertEqual(study.resources_case(3, spec, config, source)["records"][0]["status"], "compile_size_cap")

    def test_exact_angles_do_not_create_generic_T_estimates(self):
        result = study.rotation_estimate([0, np.pi/2, np.pi/4, -np.pi/4], .001, .5)
        self.assertEqual(result["generic_rotations"], 0)
        self.assertEqual(result["T_count_leading_order_estimate"], 2)
        result = study.rotation_estimate([.123, .234], .001, .5)
        self.assertAlmostEqual(result["per_rotation_operator_error_budget"], .001/8)
        self.assertGreater(result["T_count_leading_order_estimate"], 0)

    def test_search_cannot_discard_a_passing_baseline(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        config = copy.deepcopy(self.config)
        config.update(step_budget=40, optimizer_seeds=[1729])
        with patch.object(study, "differential_evolution", return_value=SimpleNamespace(success=False, fun=99.)):
            result = study.schedules_case(2, config["models"]["model2"], config)
        counts = [result["families"][family]["selected_steps"] for family in ("fixed", "constant", "warped_cosine")]
        self.assertIsNotNone(counts[0])
        self.assertEqual(counts, [counts[0]]*3)

    def test_bad_inputs_are_rejected(self):
        for key, value in (("sizes", [101]), ("minimum_dt", 3.), ("mean_tolerance", float("nan")), ("step_budget", 0)):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                study.validate(config)


if __name__ == "__main__":
    unittest.main()
