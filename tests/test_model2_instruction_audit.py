"""Instruction accounting, indivisible-path partitions and read-only archives."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2

from Model2.Experiment5 import instruction_audit as audit


class InstructionAuditTests(unittest.TestCase):
    def test_weights_charge_both_cz_operands_and_include_unused_qubit_init(self):
        circuit = QuantumCircuit(3, 2)
        circuit.rz(.5, 0)
        circuit.sx(0)
        circuit.delay(12, 0, unit="dt")
        circuit.cz(0, 1)
        circuit.x(1)
        circuit.reset(1)
        circuit.barrier()
        circuit.measure(0, 0)
        circuit.measure(1, 1)
        np.testing.assert_array_equal(audit.footprint(circuit), [69, 84, 50])
        np.testing.assert_array_equal(audit.footprint(circuit, init_qubits=False), [19, 34, 0])
        circuit.h(2)
        with self.assertRaisesRegex(ValueError, "no published weight"):
            audit.footprint(circuit)

    def test_controls_count_in_every_job_and_whole_paths_are_preserved(self):
        result = audit.partition([[10, 6]]*3, [3, 3, 3], [2, 3], 5,
                                 instruction_limit=22, execution_limit=1000)
        self.assertEqual(result["lower_bound_jobs"], 2)
        self.assertEqual(result["feasible_jobs"], 2)
        self.assertTrue(result["minimum_certified_for_counting_model"])
        self.assertEqual(result["single_job_maximum"], 32)
        self.assertEqual(result["single_job_executions"], 14)
        self.assertEqual(result["total_executions_after_repartition"], 19)
        self.assertEqual([(j["path_start"], j["path_stop"]) for j in result["jobs"]], [(0, 2), (2, 3)])
        self.assertEqual(result["jobs"][0]["maximum_instructions"], 22)

    def test_shots_affect_execution_limit_without_multiplying_instruction_memory(self):
        small = audit.partition([[10]]*3, [1]*3, [2], 1, instruction_limit=1000, execution_limit=100)
        large = audit.partition([[10]]*3, [40]*3, [2], 1, instruction_limit=1000, execution_limit=100)
        self.assertEqual(small["single_job_maximum"], large["single_job_maximum"])
        self.assertEqual(small["feasible_jobs"], 1)
        self.assertEqual(large["feasible_jobs"], 2)
        self.assertEqual(large["total_executions_after_repartition"], 122)
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            audit.partition([[999]], [1], [2], 1, instruction_limit=1000)

    def test_greedy_does_not_claim_a_minimum_when_lower_bound_is_smaller(self):
        result = audit.partition([[8, 2], [8, 2], [2, 8], [2, 8]], [1]*4, [0, 0], 0,
                                 instruction_limit=10, execution_limit=100)
        self.assertEqual(result["lower_bound_jobs"], 2)
        self.assertEqual(result["feasible_jobs"], 3)
        self.assertFalse(result["minimum_certified_for_counting_model"])

    def test_projection_repeats_only_footprints_and_handles_partial_cycles(self):
        np.testing.assert_array_equal(audit.repeat_rows([[1, 2], [3, 4]], 5),
                                      [[1, 2], [3, 4], [1, 2], [3, 4], [1, 2]])

    def test_actual_archive_audit_is_offline_and_preserves_manifest_and_qpy(self):
        backend = GenericBackendV2(12, basis_gates=["cz", "rz", "sx", "x"],
            coupling_map=[[j, j+1] for j in range(11)]+[[j+1, j] for j in range(11)], seed=17)
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(audit.run_hardware.runtime, "make_service", side_effect=AssertionError("no IBM access")):
            options = audit.run_hardware.parse_arguments(["--compile-only", "--backend", backend.name,
                "--trajectory-count", "2", "--shots", "32", "--bases", "z", "--batch-size", "6",
                "--physical-path", *map(str, range(12)), "--min-t1-us", "1", "--min-t2-us", "1",
                "--max-cz-error", ".1", "--max-readout-error", ".1", "--output-directory", temporary])
            audit.run_hardware.main(options, backend=backend)
            folder = next(Path(temporary).glob("cosine44_*"))
            manifest = folder/"manifest.json"
            before = manifest.read_bytes()
            qpy_hash = audit.qpy_storage.file_sha256(folder/"isa_circuits.qpy")
            with patch.object(audit.run_hardware, "transpile", side_effect=AssertionError("no new compilation")):
                data = audit.main(["--manifest", str(manifest)])
            result = data["workloads"]["raw"]
            self.assertEqual(result["kind"], "exact_saved_raw_archive")
            self.assertEqual(result["original_total_executions"], 6208)
            self.assertEqual(result["at_limit"]["feasible_jobs"], 1)
            self.assertEqual(before, manifest.read_bytes())
            self.assertEqual(qpy_hash, audit.qpy_storage.file_sha256(folder/"isa_circuits.qpy"))
            self.assertTrue((folder/"instruction_audit/InstructionAudit.md").exists())


if __name__ == "__main__":
    unittest.main()
