"""Physical channel equivalence and target/layout validation for Experiment5."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import DensityMatrix, Kraus, Pauli, partial_trace

from library import reservoir_scaling as physics
from Model3.Experiment5 import circuit_depth as experiment


def source(n=3, order="hamiltonian-first", policy="rate-weighted"):
    p = {"n_qubits": n, "J": 1.7, "gamma": .8, "exchange_prefactor": .5,
         "p_left": .9, "p_right": .1, "delta": 0.}
    dts = [.5, 3., .125, .25]
    fractions = np.array([.9, .1, .1, .9])
    probabilities = fractions/2 if policy == "rate-weighted" else np.full(4, .25)
    return {"experiment": "Model3/Experiment4", "run_status": "test_fixture", "assessment": {},
        "configuration": {"physics": p, "trajectories": 2, "order": order,
                          "sampling_policy": policy, "system_formula": "even-odd"},
        "preparation": {"name": "test_product", "populations": np.linspace(.12, .81, n).tolist()},
        "times": np.r_[0, np.cumsum(dts)].tolist(),
        "schedule": {"step_dts": dts, "jump_choices": [[0, 1, 2, 3], [3, 2, 1, 0]],
                     "jump_probabilities": probabilities.tolist()}}


def nonselective_density(circuit):
    """Exactly average ignored measurement outcomes; no sampling/shot error."""
    density = DensityMatrix.from_label("0"*circuit.num_qubits)
    dephase = Kraus([np.diag([1, 0]), np.diag([0, 1])])
    for instruction in circuit.data:
        args = [circuit.find_bit(q).index for q in instruction.qubits]
        if instruction.operation.name == "measure":
            density = density.evolve(dephase, args)
        elif instruction.operation.name == "reset":
            density = density.reset(args)
        else:
            density = density.evolve(instruction.operation, args)
    return density


def jw_covariance(density, n):
    lowering, Z = np.array([[0, 1], [0, 0]]), np.diag([1, -1])
    operators = []
    for j in range(n):
        factors = [Z if k < j else (lowering if k == j else np.eye(2)) for k in range(n)]
        value = np.ones((1, 1))
        for factor in reversed(factors):
            value = np.kron(value, factor)
        operators.append(value)
    return np.array([[np.trace(operators[j].conj().T @ operators[i] @ density.data)
                      for j in range(n)] for i in range(n)])


class CircuitDepthTests(unittest.TestCase):
    def test_parse_is_lazy_and_rejects_bad_requests(self):
        options = experiment.parse_arguments([])
        self.assertEqual(options.ancillas, "boundary")
        self.assertEqual(options.trajectory_indices, [0, 1, 2, 3])
        for arguments in (("--trajectory-indices", "-1"), ("--prefix-steps", "2", "2"),
                          ("--seed-transpiler", "-1"), ("--layout", "automatic", "--physical-path", "0", "1")):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                experiment.parse_arguments(arguments)

    def test_source_rejects_inconsistent_physics_and_choices(self):
        experiment.validate_source(source())
        for mutate in (lambda d: d["configuration"].update(system_formula="exact"),
                       lambda d: d["schedule"]["jump_choices"][0].__setitem__(0, 4),
                       lambda d: d["schedule"]["jump_probabilities"].__setitem__(0, .1),
                       lambda d: d["times"].__setitem__(1, .6),
                       lambda d: d["preparation"]["populations"].__setitem__(0, float("nan"))):
            data = source(); mutate(data)
            with self.assertRaises(ValueError):
                experiment.validate_source(data)

    def test_mixed_preparation_is_diagonal_and_has_no_free_current(self):
        data = source()
        circuit = experiment.build_body(data, 0, 0)
        reduced = partial_trace(nonselective_density(circuit), [3, 4])
        expected = np.ones((1, 1))
        for p in reversed(data["preparation"]["populations"]):
            expected = np.kron(expected, np.diag([1-p, p]))
        np.testing.assert_allclose(reduced.data, expected, atol=2e-15)
        self.assertEqual(circuit.count_ops()["measure"], 3)
        self.assertNotIn("reset", circuit.count_ops())

    def test_all_jump_types_orders_and_probabilities_match_covariance_channels(self):
        for order in ("hamiltonian-first", "jump-first"):
            for policy in ("rate-weighted", "uniform"):
                for ancillas in ("boundary", "shared"):
                    with self.subTest(order=order, policy=policy, ancillas=ancillas):
                        data = source(order=order, policy=policy)
                        p = data["configuration"]["physics"]
                        model = physics.Model(3, p["gamma"]/p["J"], p["p_left"], p["p_right"], p["exchange_prefactor"])
                        covariance = np.diag(data["preparation"]["populations"])[None, None]
                        for prefix, (dt, choice) in enumerate(zip(data["schedule"]["step_dts"], data["schedule"]["jump_choices"][0]), 1):
                            branch = physics.channel(model, p["J"]*dt, policy, order, "even-odd")
                            covariance = physics.sampled_step(covariance, [choice], branch)
                            circuit = experiment.build_body(data, 0, prefix, ancillas)
                            reduced = partial_trace(nonselective_density(circuit), list(range(3, circuit.num_qubits)))
                            expected = ((-1j)**np.arange(3))[:, None] * ((1j)**np.arange(3))[None, :] * covariance[0, 0]
                            np.testing.assert_allclose(jw_covariance(reduced, 3), expected, atol=3e-14)
                        self.assertLess(physics.channel(model, p["J"]*3, policy, order, "even-odd").cosines[0], 0)
                        self.assertEqual(circuit.count_ops()["reset"], 4)

    def test_current_readout_rotations_and_site_mapping(self):
        body = QuantumCircuit(2)
        body.ry(.8, 0); body.rx(.4, 1); body.cx(0, 1); body.rz(.7, 0)
        density = DensityMatrix.from_instruction(body)
        for basis, pauli in (("xy", "YX"), ("yx", "XY")):
            measured = experiment.append_readout(body, [0, 1], basis)
            rotated = DensityMatrix.from_instruction(measured.remove_final_measurements(inplace=False))
            self.assertAlmostEqual(rotated.expectation_value(Pauli("ZZ")).real,
                                   density.expectation_value(Pauli(pauli)).real, places=13)
        reversed_readout = experiment.append_readout(body, [1, 0], "xy")
        rotated = DensityMatrix.from_instruction(reversed_readout.remove_final_measurements(inplace=False))
        self.assertAlmostEqual(rotated.expectation_value(Pauli("ZZ")).real,
                               density.expectation_value(Pauli("XY")).real, places=13)
        with self.assertRaises(ValueError):
            experiment.append_readout(body, [0, 1], "bad")

    def test_current_sign_and_normalization(self):
        data = source(n=2)
        circuit = experiment.build_body(data, 0, 1)
        reduced = partial_trace(nonselective_density(circuit), [2, 3])
        C = jw_covariance(reduced, 2)
        p = data["configuration"]["physics"]
        current = p["exchange_prefactor"]*p["J"]*(reduced.expectation_value(Pauli("XY"))-reduced.expectation_value(Pauli("YX")))
        self.assertAlmostEqual(current.real, 4*p["exchange_prefactor"]*p["J"]*C[0, 1].imag, places=13)

    def test_native_target_compilation_and_qpy_preserve_layout_and_reset(self):
        backend = GenericBackendV2(6, basis_gates=["cz", "rz", "sx", "x"],
            coupling_map=[[i, i+1] for i in range(5)] + [[i+1, i] for i in range(5)], seed=17)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(temp); src = folder / "replay.json"; data = source()
            src.write_text(json.dumps(data)); original = src.read_bytes()
            options = experiment.parse_arguments(["--source", str(src), "--trajectory-indices", "0", "1",
                "--prefix-steps", "1", "4", "--output-directory", str(folder), "--no-plots"])
            result = experiment.main(options, backend)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["records"]), 4)
            self.assertEqual(src.read_bytes(), original)
            output = next(folder.glob("depth_*"))
            self.assertTrue((output / "Depth.md").exists())
            for row in result["records"]:
                with (output / row["native_qpy"]).open("rb") as handle:
                    circuits = qpy.load(handle)
                self.assertEqual(len(circuits), 4)
                for circuit in circuits:
                    experiment.assert_native(circuit, backend)
                self.assertEqual(row["native_body"]["operation_counts"]["reset"], row["steps"])
                self.assertEqual(len(row["final_logical_to_physical"]), 5)
                # Compare compiled density after its output permutation with the logical channel.
                mapped = row["final_logical_to_physical"]
                native_density = nonselective_density(circuits[0])
                # Read physical populations directly; includes a nontrivial logical mapping.
                expected = partial_trace(nonselective_density(experiment.build_body(data, row["trajectory"], row["steps"])), [3, 4])
                for site in range(3):
                    label = ["I"]*6; label[5-mapped[site]] = "Z"
                    label_logical = ["I"]*3; label_logical[2-site] = "Z"
                    self.assertAlmostEqual(native_density.expectation_value(Pauli("".join(label))).real,
                        expected.expectation_value(Pauli("".join(label_logical))).real, places=11)
            with self.assertRaises(ValueError):
                experiment.chain_layout(backend, 3, supplied=[0, 2, 1, 3, 4])


if __name__ == "__main__":
    unittest.main()
