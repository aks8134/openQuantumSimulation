"""Portable schedules and whole-circuit target compilation, without IBM access."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from qiskit.circuit import Measure, Parameter, Reset
from qiskit.circuit.library import CZGate, RZGate, SXGate, XGate
from qiskit.transpiler import InstructionProperties, Target

from FinalModel.Experiment1 import phoenix_resources as p


def fixture_backend():
    target = Target(num_qubits=8, dt=1e-9)
    for gate in (RZGate(Parameter("theta")), SXGate(), XGate(), Reset(), Measure()):
        duration = {"rz": 0., "sx": 40e-9, "x": 40e-9, "reset": 2e-6, "measure": 1e-6}[gate.name]
        target.add_instruction(gate, {(q,): InstructionProperties(duration=duration, error=0.) for q in range(8)})
    target.add_instruction(CZGate(), {pair: InstructionProperties(duration=120e-9, error=.001)
        for q in range(7) for pair in ((q, q+1), (q+1, q))})
    return SimpleNamespace(name="synthetic_unit_test_target", backend_version="fixture", num_qubits=8,
        target=target, get_translation_stage_plugin=lambda: "translator")


class PhoenixResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = p.read_inputs(p.DEFAULT_INPUT)

    def case(self, model="model1"):
        return next(copy.deepcopy(c) for c in self.payload["cases"]
                    if c["N"] == 2 and c["model"] == model and c["family"] == "warped_cosine")

    def test_portable_bundle_preserves_all_selected_lengths_and_times(self):
        self.assertEqual(len(self.payload["cases"]), 117)
        for case in self.payload["cases"]:
            dts = p.steps(case)
            self.assertEqual(len(dts), case["steps"])
            self.assertAlmostEqual(sum(dts), case["physical_time"], places=8)
        case = next(c for c in self.payload["cases"] if c["id"] == "model2_N100_fixed")
        self.assertEqual(case["steps"], 51280)
        self.assertEqual(case["constant_dt"], .05)

    def test_input_tampering_and_step_hash_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"inputs.json"
            data = copy.deepcopy(self.payload)
            data["cases"][0]["steps"] += 1
            p.atomic_json(path, data)
            with self.assertRaisesRegex(ValueError, "Content hash"):
                p.read_inputs(path)
        case = self.case()
        case["constant_dt"] = .1234
        case.pop("step_dts", None)
        with self.assertRaisesRegex(ValueError, "Schedule hash"):
            p.steps(case)

    def test_full_target_cache_preserves_all_qubits_and_compiler_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = p.capture_snapshot(fixture_backend(), Path(directory)/"target")
            target, loaded = p.load_snapshot(Path(directory)/"target")
            self.assertEqual(metadata, loaded)
            self.assertEqual(target.num_qubits, 8)
            self.assertTrue(target.instruction_supported(operation_name="cz", qargs=(7, 6)))
            self.assertEqual(loaded["compiler_methods"], {"translation_method": "translator"})
            with self.assertRaisesRegex(ValueError, "must be empty"):
                p.capture_snapshot(fixture_backend(), Path(directory)/"target")
            with (Path(directory)/"target/target.pkl").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                p.load_snapshot(Path(directory)/"target")

    def test_version_mismatch_does_not_deserialize_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/"target"
            metadata = p.capture_snapshot(fixture_backend(), folder)
            metadata.pop("sha256")
            metadata["versions"]["qiskit"] = "different"
            p.atomic_json(folder/"target.json", p.sealed(metadata))
            with patch.object(p.pickle, "load", side_effect=AssertionError("must reject first")):
                with self.assertRaisesRegex(ValueError, "same Python"):
                    p.load_snapshot(folder)

    def test_capture_command_only_resolves_backend_with_fractional_gates_disabled(self):
        backend = fixture_backend()
        service = SimpleNamespace(backend=Mock(return_value=backend))
        with tempfile.TemporaryDirectory() as directory, patch.object(p, "make_service", return_value=service), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(p.main(["capture-target", "--backend", backend.name, "--target-dir", str(Path(directory)/"target")]), 0)
            service.backend.assert_called_once_with(backend.name, use_fractional_gates=False)

    def test_chain_selection_validates_physical_connectivity_and_search_limit(self):
        target = fixture_backend().target
        layout, path = p.chain_layout(target, 2, supplied=[0, 1, 2, 3])
        self.assertEqual(layout, [1, 2, 0, 3])
        self.assertEqual(path, [0, 1, 2, 3])
        for bad in ([0, 1, 1, 3], [0, 1, 5, 6], [0, 1, 2]):
            with self.assertRaises(ValueError):
                p.chain_layout(target, 2, supplied=bad)
        with self.assertRaisesRegex(ValueError, "search limit"):
            p.chain_layout(target, 2, limit=1)
        with self.assertRaisesRegex(ValueError, "eligible"):
            p.chain_layout(target, 7)

    def test_bodies_match_the_existing_hardware_construction(self):
        from Model3.Experiment5 import circuit_depth
        from qiskit.quantum_info import DensityMatrix, Kraus
        def density(circuit):
            result = DensityMatrix.from_int(0, 2**circuit.num_qubits)
            for instruction in circuit.data:
                qubits = [circuit.find_bit(q).index for q in instruction.qubits]
                if instruction.operation.name == "measure":
                    operation = Kraus([np.diag([1, 0]), np.diag([0, 1])])
                elif instruction.operation.name == "reset":
                    operation = Kraus([np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])])
                else:
                    operation = instruction.operation
                result = result.evolve(operation, qubits)
            return result.data
        for model in p.MODELS:
            case = self.case(model)
            body, choices = p.build_body(case, 29)
            spec, n = case["physics"], case["N"]
            populations = [0.] * n if spec["preparation"] == "center_excitation" else [.5] * n
            if spec["preparation"] == "center_excitation":
                populations[n//2] = 1.
            source = dict(configuration=dict(trajectories=1, order="hamiltonian-first",
                physics=dict(n_qubits=n, exchange_prefactor=spec["prefactor"], **{k: spec[k] for k in ("J", "gamma", "p_left", "p_right")})),
                preparation=dict(populations=populations), schedule=dict(step_dts=p.steps(case),
                jump_choices=[choices.tolist()], jump_probabilities=[spec["p_left"]/2, (1-spec["p_left"])/2, spec["p_right"]/2, (1-spec["p_right"])/2]))
            reference = circuit_depth.build_body(source, 0)
            np.testing.assert_allclose(density(body), density(reference), atol=3e-13)
            self.assertEqual(body.count_ops()["xx_plus_yy"], n * case["steps"])
            self.assertEqual(body.count_ops()["reset"], case["steps"])

    def test_whole_circuit_compilation_uses_target_native_gates_and_dynamic_timing(self):
        from qiskit import qpy
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(directory)
            metadata = p.capture_snapshot(fixture_backend(), folder/"target")
            target, metadata = p.load_snapshot(folder/"target")
            layout, path = p.chain_layout(target, 2, supplied=[0, 1, 2, 3])
            settings = dict(optimization_level=3, transpiler_seed=11, bases=list(p.BASES), timing=True, save_qpy=True)
            result = p.compile_one(self.case("model3"), target, metadata, settings, 29,
                initial_layout=layout, physical_path=path, output=folder)
            self.assertEqual(result["final_logical_to_physical"], layout)
            for row in result["readouts"]:
                self.assertEqual(set(row["metrics"]["two_qubit_counts"]), {"cz"})
                self.assertGreater(row["metrics"]["operation_counts"]["cz"], 0)
                self.assertEqual(row["metrics"]["operation_counts"]["measure"], 4)
                self.assertEqual(row["timing"]["status"], "modeled")
                with (folder/row["qpy"]["file"]).open("rb") as stream:
                    circuit = qpy.load(stream)[0]
                self.assertEqual(p.native_metrics(circuit, target), row["metrics"])
                self.assertEqual(p.file_digest(folder/row["qpy"]["file"]), row["qpy"]["sha256"])

    def test_offline_cli_checkpoints_resumes_and_exposes_explicit_caps(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(directory)
            p.capture_snapshot(fixture_backend(), folder/"target")
            arguments = ["compile", "--target-dir", str(folder/"target"), "--output", str(folder/"runs"),
                "--models", "model1", "--sizes", "2", "--families", "warped_cosine", "--seeds", "29"]
            with patch.object(p, "make_service", side_effect=AssertionError("offline compilation must not authenticate")):
                self.assertEqual(p.main(arguments), 0)
                output = next((folder/"runs").glob("phoenix_*"))
                with (output/"resources.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["status"], "compiled")
                self.assertGreater(int(rows[0]["CZ"]), 0)
                with patch.object(p.subprocess, "Popen", side_effect=AssertionError("completed paths must be reused")):
                    self.assertEqual(p.main(arguments), 0)
                    self.assertEqual(p.main(arguments + ["--max-source-exchanges", "1"]), 1)
                statuses = [p.read_sealed(f)["status"] for f in (folder/"runs").glob("phoenix_*/cases/*.json")]
                self.assertCountEqual(statuses, ["compiled", "source_size_cap"])

    def test_plan_uses_no_backend_or_account(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(p, "make_service", side_effect=AssertionError("no IBM access")), \
                patch.object(p, "load_snapshot", side_effect=AssertionError("no target needed")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(p.main(["plan", "--sizes", "100", "--families", "fixed", "--output", directory]), 0)
            with (Path(directory)/"plan.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(max(int(r["source_exchanges"]) for r in rows), 5128000)

    def test_automatic_routing_records_actual_initial_and_final_maps(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            backend = fixture_backend()
            metadata = p.capture_snapshot(backend, Path(directory)/"target")
            settings = dict(optimization_level=3, transpiler_seed=11, bases=["z"], timing=False, save_qpy=False)
            result = p.compile_one(self.case("model2"), backend.target, metadata, settings, 29)
            self.assertEqual(result["status"], "compiled")
            self.assertEqual(len(result["initial_logical_to_physical"]), 4)
            self.assertEqual(len(result["final_logical_to_physical"]), 4)
            self.assertEqual(len(set(result["final_logical_to_physical"])), 4)
            self.assertIsNone(result["physical_path"])
            self.assertGreater(result["readouts"][0]["metrics"]["operation_counts"]["cz"], 0)


if __name__ == "__main__":
    unittest.main()
