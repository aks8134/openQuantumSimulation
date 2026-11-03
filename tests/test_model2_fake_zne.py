"""Model2 snapshot remapping and mitigation preserve the intended channels."""
import unittest
import copy
import json

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Kraus
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeKingston

from Model2.Experiment6 import fake_kingston_zne as experiment


def probabilities(circuit, n):
    density = DensityMatrix.from_label("0" * circuit.num_qubits)
    dephase = Kraus([np.diag([1, 0]), np.diag([0, 1])])
    for item in circuit.data:
        args = [circuit.find_bit(q).index for q in item.qubits]
        if item.operation.name == "barrier":
            continue
        if item.operation.name == "measure":
            density = density.evolve(dephase, args)
        elif item.operation.name == "reset":
            density = density.reset(args)
        else:
            density = density.evolve(item.operation, args)
    return density.probabilities(qargs=list(range(n)))


class Model2FakeZNETests(unittest.TestCase):
    def test_subset_errors_exactly_match_the_full_156_qubit_noise_model(self):
        original = FakeKingston()
        compact, active = experiment.compact_backend(original, experiment.DEFAULT_PATH)
        lookup = {physical: logical for logical, physical in enumerate(active)}
        full_errors = experiment.canonical_noise(NoiseModel.from_backend(original))["errors"]
        selected = []
        for error in full_errors:
            if (set(error["operations"]) <= set(compact.target.operation_names)
                    and all(set(args) <= lookup.keys() for args in error["gate_qubits"])):
                value = copy.deepcopy(error)
                value["gate_qubits"] = [[lookup[q] for q in args] for args in value["gate_qubits"]]
                selected.append(value)
        actual = experiment.canonical_noise(NoiseModel.from_backend(compact))["errors"]
        self.assertEqual(len(actual), 82)
        self.assertEqual(sorted(json.dumps(row, sort_keys=True) for row in selected),
                         sorted(json.dumps(row, sort_keys=True) for row in actual))

    def test_subset_preserves_physical_properties_and_noise_fingerprint(self):
        original = FakeKingston()
        compact, active = experiment.compact_backend(original, experiment.DEFAULT_PATH)
        self.assertEqual(active, experiment.DEFAULT_PATH[1:-1] + [95, 52])
        self.assertEqual(compact.num_qubits, 12)
        self.assertEqual(compact.dt, original.dt)
        for logical, physical in enumerate(active):
            a, b = compact.target.qubit_properties[logical], original.target.qubit_properties[physical]
            self.assertEqual((a.t1, a.t2, a.frequency), (b.t1, b.t2, b.frequency))
        for name in compact.target.operation_names:
            for args, prop in compact.target[name].items():
                physical_args = tuple(active[q] for q in args)
                expected = original.target[name][physical_args]
                self.assertEqual(None if prop is None else (prop.duration, prop.error),
                                 None if expected is None else (expected.duration, expected.error))
        first = experiment.canonical_noise(NoiseModel.from_backend(compact))
        second = experiment.canonical_noise(NoiseModel.from_backend(compact))
        self.assertEqual(first, second)
        self.assertTrue(any(error["type"] == "roerror" for error in first["errors"]))
        self.assertTrue(any("cz" in error["operations"] for error in first["errors"]))
        with self.assertRaisesRegex(RuntimeError, "local Aer"):
            compact.run([])

    def test_final_decoder_traces_out_preparation_records(self):
        circuit = QuantumCircuit(3)
        circuit.add_register(ClassicalRegister(3, "preparation"))
        circuit.add_register(ClassicalRegister(3, "outcome"))
        self.assertEqual(experiment.final_counts({"101 011": 2, "101 111": 3, "010 101": 4}, circuit, 3),
                         {"101": 5, "010": 4})

    def test_large_angle_model2_twirl_and_folding_preserve_exact_density(self):
        spec = experiment.input_data.load_spec()
        source = {"configuration": {"physics": {**spec["physics"], "n_qubits": 3},
                                    "trajectories": 1, "order": "jump-first"},
                  "preparation": {"populations": [.5] * 3},
                  "schedule": {"step_dts": [spec["schedule"]["step_dts"][0],
                                             spec["schedule"]["step_dts"][-1]],
                               "jump_choices": [[0, 3]], "jump_probabilities": [.125, .375, .125, .375]}}
        body = experiment.run_hardware.build_body(source, 0)
        native = transpile(body, basis_gates=["cz", "rz", "sx", "x", "reset", "measure"],
                           optimization_level=3, seed_transpiler=11)
        native = experiment.run_hardware.circuits.append_readout(native, list(range(3)), "z")
        expected = probabilities(native, 3)
        np.testing.assert_allclose(expected, probabilities(body, 3), atol=2e-14)
        base = native.count_ops()["cz"]
        for factor in [1., 1.25, 1.5]:
            mask = 5
            folds = experiment.mit.fold_count(factor, base)
            variant = experiment.mit.transformed(native, seed=29, folds=folds, measurement_mask=mask)
            actual = probabilities(variant, 3)[np.arange(8) ^ mask]
            np.testing.assert_allclose(actual, expected, atol=3e-14)
            self.assertEqual(variant.count_ops()["reset"], 2)
            self.assertEqual(variant.count_ops()["cz"], base + 2 * folds)

    def test_matched_plan_uses_equal_per_factor_shots_and_exact_cz_scaling(self):
        cfg = dict(seed=29, trajectory_count=32, randomizations=4,
                   physical_path=experiment.DEFAULT_PATH, steps=44, fold_factors=[1., 1.25, 1.5],
                   shots=128, control_shots=2048, batch_size=10**9, controls_per_batch=True)
        records, _ = experiment.protocol.plan(cfg)
        self.assertEqual(len(records), 431)
        self.assertEqual(sum(row["shots"] for row in records), 28672)
        for path in range(32):
            selected = [row for row in records if row.get("trajectory") == path]
            self.assertEqual(sum(row["shots"] for row in selected if row["arm"] == "raw"), 128)
            for factor in range(3):
                variants = [row for row in selected if row["arm"] == "twirled" and row["factor_index"] == factor]
                self.assertEqual(sum(row["shots"] for row in variants), 128)
                self.assertEqual({880 + 2 * row["folds"] for row in variants}, [{880}, {1100}, {1320}][factor])
                self.assertEqual(len({experiment.protocol.transform_seed(cfg, row) for row in variants}), 4)


if __name__ == "__main__":
    unittest.main()
