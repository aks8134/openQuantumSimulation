"""Integration checks across the experiment/Mitiq interpreter boundary."""

import os
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_ibm_runtime.fake_provider import FakeFez

from mitiq_qem import (
    WorkerError, close_worker, combine_ddd, configure_worker, construct_ddd,
    construct_pauli_twirling, ddd_plan, pauli_twirling_plan,
    scale_circuit, scaling, worker_info,
)


class WorkerIntegrationTests(unittest.TestCase):
    def test_persistent_worker_is_a_different_process(self):
        first, second = worker_info(), worker_info()
        self.assertEqual(first["pid"], second["pid"])
        self.assertNotEqual(first["pid"], os.getpid())
        self.assertIn(".venv-mitiq", first["interpreter"])
        self.assertTrue(first["mitiq"])
        self.assertNotIn("mitiq", sys.modules)

    def test_folding_preserves_phase_physical_wires_and_classical_registers(self):
        circuit = QuantumCircuit(2, 2, name="physical-test")
        circuit.global_phase = 0.37
        circuit.sx(0)
        circuit.cz(0, 1)
        circuit.measure(1, 0)
        circuit.measure(0, 1)
        native = transpile(circuit, FakeFez(), optimization_level=1, seed_transpiler=3)
        native.metadata = {"purpose": "worker-roundtrip"}
        native.global_phase = 0.37
        scaled = scale_circuit(native, 3.0, seed=7,
                               scaling=scaling("random_local", fidelities=(("rz", 1.0),)))
        self.assertEqual(scaled.num_qubits, native.num_qubits)
        self.assertEqual(scaled.num_clbits, native.num_clbits)
        self.assertEqual(scaled.cregs, native.cregs)
        self.assertEqual(scaled.metadata, native.metadata)
        self.assertIs(scaled.layout, native.layout)

        def measurement_wires(value):
            return tuple((value.find_bit(item.qubits[0]).index,
                          value.find_bit(item.clbits[0]).index)
                         for item in value.data if item.operation.name == "measure")

        self.assertEqual(measurement_wires(scaled), measurement_wires(native))
        # Compact the active physical wires to check the unitary without
        # allocating an Operator on all 156 hardware qubits.
        active = sorted({native.find_bit(bit).index for item in native.data for bit in item.qubits})

        def compact(value):
            result = QuantumCircuit(len(active))
            result.global_phase = value.global_phase
            for item in value.data:
                if item.operation.name != "measure":
                    result.append(item.operation, [active.index(value.find_bit(bit).index)
                                                   for bit in item.qubits])
            return result

        np.testing.assert_allclose(Operator(compact(native)).data,
                                   Operator(compact(scaled)).data, atol=1e-10)

    def test_twirling_and_ddd_use_the_same_worker(self):
        circuit = QuantumCircuit(3, 3)
        circuit.x(0)
        circuit.cx(0, 1)
        circuit.measure(range(3), range(3))
        pid = worker_info()["pid"]
        twirled = construct_pauli_twirling(circuit, pauli_twirling_plan(variants=1, seed=12))
        padded = construct_ddd(circuit, ddd_plan("xx"))
        self.assertEqual(len(twirled), 1)
        self.assertEqual(len(padded), 1)
        self.assertEqual(worker_info()["pid"], pid)
        self.assertNotIn("mitiq", sys.modules)

    def test_worker_exception_is_visible_and_subsequent_request_still_works(self):
        circuit = QuantumCircuit(1, 1)
        circuit.reset(0)
        pid = worker_info()["pid"]
        with self.assertRaisesRegex(ValueError, "global ZNE folding"):
            scale_circuit(circuit, 3.0, scaling=scaling("global"))
        self.assertAlmostEqual(combine_ddd((0.2, 0.4)).estimate, 0.3)
        self.assertEqual(worker_info()["pid"], pid)

    @unittest.skipIf(find_spec("mitiq") is not None,
                     "requires a main interpreter without upstream Mitiq")
    def test_worker_missing_dependency_fails_clearly_without_main_fallback(self):
        close_worker()
        try:
            configure_worker(sys.executable)
            with self.assertRaisesRegex(WorkerError, "could not import upstream dependencies.*mitiq"):
                worker_info()
            self.assertNotIn("mitiq", sys.modules)
        finally:
            configure_worker()

    def test_missing_worker_interpreter_has_no_fallback(self):
        close_worker()
        try:
            configure_worker(Path("/tmp/oqs-nonexistent-mitiq-env/python"))
            with self.assertRaisesRegex(WorkerError, "No local mitigation fallback"):
                worker_info()
            self.assertNotIn("mitiq", sys.modules)
        finally:
            configure_worker()


if __name__ == "__main__":
    unittest.main()
