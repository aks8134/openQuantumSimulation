"""Small native-QEM tests: no providers, credentials or experiment workloads."""

from dataclasses import FrozenInstanceError
from math import pi, sqrt
from pathlib import Path
import subprocess
import sys
import unittest

from qem import (
    DDPlan, Err, FoldPlan, Ok, ReadoutBlock, ReadoutModel, ReadoutPlan,
    TwirlingPlan, ZNEPlan, apply_dd, bind, calibration_circuits, extrapolate,
    fit_readout_model, fold, map_result, mitigate_marginal, mitigate_parity,
    native_pauli, pipe, pipe_result, select_folds, twirl,
)


class CoreTests(unittest.TestCase):
    def test_import_is_lightweight(self):
        command = (
            "import sys,qem; "
            "assert not {'numpy','qiskit','mitiq','cirq','mitiq_qem'} & set(sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-S", "-c", command],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())

    def test_functional_composition(self):
        self.assertEqual(pipe(2, lambda x: x + 1, lambda x: x * 2), 6)
        self.assertEqual(map_result(Ok(2), lambda x: x + 1), Ok(3))
        self.assertEqual(pipe_result(Ok(2), lambda x: Ok(x + 1)), Ok(3))
        failure = Err("bad")
        self.assertIs(bind(failure, lambda x: self.fail("must short circuit")), failure)

    def test_plans_are_immutable(self):
        plan = FoldPlan(gate_names=["rx", "rzz"])
        self.assertEqual(plan.gate_names, ("rx", "rzz"))
        with self.assertRaises(FrozenInstanceError):
            plan.seed = 2

    def test_fold_rounding_and_reproducibility(self):
        plan = FoldPlan(1.5, seed=21)
        result = select_folds((1, 1, 1, 1), plan).value
        self.assertEqual(result, select_folds((1, 1, 1, 1), plan).value)
        self.assertEqual(sum(result.repetitions), 1)
        self.assertEqual(result.achieved_scale, 1.5)
        rounded = select_folds((1, 1, 1), FoldPlan(2)).value
        self.assertAlmostEqual(rounded.achieved_scale, 5 / 3)

    def test_weighted_selection(self):
        result = select_folds((2, 1, 1), FoldPlan(2, order="left")).value
        self.assertEqual(result.repetitions, (1, 0, 0))
        self.assertEqual(result.achieved_scale, 2)

    def test_selection_empty_and_resource_limits(self):
        self.assertIsInstance(select_folds((), FoldPlan(1)), Ok)
        self.assertIsInstance(select_folds((), FoldPlan(2)), Err)
        self.assertIsInstance(select_folds((1, 1), FoldPlan(9, max_inserted_operations=2)), Err)
        self.assertIsInstance(select_folds((1e308, 1e308), FoldPlan(2)), Err)

    def test_invalid_plans(self):
        for construct in (
            lambda: FoldPlan(float("nan")), lambda: FoldPlan(0.9),
            lambda: FoldPlan(seed=True), lambda: FoldPlan(weights=(("rx", 0),)),
            lambda: TwirlingPlan(gate_names=("crx",)), lambda: DDPlan("invalid"),
            lambda: DDPlan(spacing=(0.1, 0.1, 0.1)),
            lambda: ZNEPlan((1, 1)), lambda: ZNEPlan((1,), "linear"),
        ):
            with self.subTest(construct=construct), self.assertRaises((TypeError, ValueError)):
                construct()


class NumericalTests(unittest.TestCase):
    def test_linear_extrapolation(self):
        result = extrapolate(ZNEPlan((1, 1.5, 2)), (0.8, 0.7, 0.6)).value
        self.assertAlmostEqual(result.estimate, 1)
        self.assertIsNone(result.standard_error)
        self.assertAlmostEqual(sum(result.weights), 1)
        self.assertAlmostEqual(sum(w * s for w, s in zip(result.weights, result.scales)), 0)

    def test_richardson_quadratic_and_uncertainty(self):
        plan = ZNEPlan((1, 2, 3), "richardson")
        result = extrapolate(plan, (1.3, 2, 3.1), standard_errors=(0.1, 0.1, 0.1)).value
        self.assertAlmostEqual(result.estimate, 1)
        self.assertAlmostEqual(result.standard_error, 0.1 * sqrt(19))

    def test_polynomial_and_none(self):
        result = extrapolate(ZNEPlan((1, 2, 3, 4), "polynomial", 2), (1.1, 1.4, 1.9, 2.6)).value
        self.assertAlmostEqual(result.estimate, 1)
        baseline = extrapolate(ZNEPlan((1,), "none"), (0.5,)).value
        self.assertAlmostEqual(baseline.estimate, 0.5)

    def test_correlated_covariance(self):
        result = extrapolate(
            ZNEPlan((1, 2)), (0.8, 0.7), covariance=((0.01, 0.005), (0.005, 0.01)),
        ).value
        self.assertAlmostEqual(result.standard_error, sqrt(0.03))

    def test_invalid_numerical_inputs(self):
        self.assertIsInstance(extrapolate(ZNEPlan(), (1,)), Err)
        self.assertIsInstance(extrapolate(ZNEPlan((1, 2)), (1, 1), covariance=((1, 2), (2, 1))), Err)
        self.assertIsInstance(extrapolate(ZNEPlan((1, 2)), (1, float("nan"))), Err)
        self.assertIsInstance(extrapolate(ZNEPlan((1, 1.000000000000001)), (1, 1)), Err)

    def test_readout_matrix_validation(self):
        with self.assertRaises(ValueError):
            ReadoutBlock((0,), ((0.8, 0.1), (0.1, 0.8)))
        block = ReadoutBlock((0,), ((1, 0), (0, 1)))
        with self.assertRaises(ValueError):
            ReadoutModel((block, block))

    def test_asymmetric_readout_recovery(self):
        model = ReadoutModel((ReadoutBlock((0,), ((0.9, 0.2), (0.1, 0.8))),))
        # Ideal probabilities (.7,.3) -> observed (.69,.31).
        result = mitigate_marginal({"0": 690, "1": 310}, model, (0,)).value
        self.assertAlmostEqual(result.probabilities[0], 0.7)
        corrected = mitigate_parity({"0": 690, "1": 310}, model, (0,)).value
        self.assertAlmostEqual(corrected.estimate, 0.4)
        self.assertGreater(corrected.standard_error, 0)

    def test_correlated_block_and_reordered_marginal(self):
        matrix = ((0.8, 0.1, 0, 0.1), (0.1, 0.8, 0.1, 0),
                  (0, 0.1, 0.8, 0.1), (0.1, 0, 0.1, 0.8))
        model = ReadoutModel((ReadoutBlock((0, 1), matrix),))
        result = mitigate_marginal({"00": 140, "01": 200, "10": 300, "11": 360}, model, (1, 0)).value
        for actual, expected in zip(result.probabilities, (0.1, 0.3, 0.2, 0.4)):
            self.assertAlmostEqual(actual, expected)
        parity = mitigate_parity({"00": 140, "01": 200, "10": 300, "11": 360}, model, (0, 1)).value
        self.assertAlmostEqual(parity.estimate, 0)

    def test_product_block_correction(self):
        matrix = ((0.9, 0.1), (0.1, 0.9))
        model = ReadoutModel(tuple(ReadoutBlock((bit,), matrix) for bit in (0, 1)))
        counts = {"00": 340, "01": 160, "10": 160, "11": 340}
        marginal = mitigate_marginal(counts, model, (0, 1)).value
        expected = (0.390625, 0.109375, 0.109375, 0.390625)
        for actual, probability in zip(marginal.probabilities, expected):
            self.assertAlmostEqual(actual, probability)
        self.assertAlmostEqual(mitigate_parity(counts, model, (0, 1)).value.estimate, 0.5625)

    def test_large_system_parity_avoids_full_distribution(self):
        model = ReadoutModel(tuple(ReadoutBlock((bit,), ((1, 0), (0, 1))) for bit in range(50)))
        result = mitigate_parity({"0" * 50: 9, "1" * 50: 1}, model, tuple(range(50))).value
        self.assertAlmostEqual(result.estimate, 1)
        self.assertIsInstance(mitigate_marginal({"0" * 50: 1}, model, tuple(range(50))), Err)

    def test_negative_probabilities_are_not_hidden(self):
        model = ReadoutModel((ReadoutBlock((0,), ((0.9, 0.1), (0.1, 0.9))),))
        result = mitigate_marginal({"0": 100}, model, (0,)).value
        self.assertLess(result.probabilities[1], 0)
        projected = mitigate_marginal({"0": 100}, model, (0,), ReadoutPlan(project_probabilities=True)).value
        self.assertEqual(projected.probabilities, (1.0, 0.0))
        self.assertGreater(mitigate_parity({"0": 100}, model, (0,)).value.estimate, 1)

    def test_singular_calibration_and_invalid_counts(self):
        model = ReadoutModel((ReadoutBlock((0,), ((0.5, 0.5), (0.5, 0.5))),))
        self.assertIsInstance(mitigate_parity({"0": 1}, model, (0,)), Err)
        self.assertIsInstance(mitigate_marginal({"0": 0}, model, (0,)), Err)
        self.assertIsInstance(mitigate_marginal({"0": 1.5}, model, (0,)), Err)
        self.assertIsInstance(mitigate_parity({"0": 1}, model, (1,)), Err)

    def test_bitstrings_spaces_and_hex(self):
        model = ReadoutModel((ReadoutBlock((2,), ((1, 0), (0, 1))),))
        for counts in ({"1 00": 1}, {"0x4": 1}, {4: 1}):
            self.assertEqual(mitigate_parity(counts, model, (2,)).value.estimate, -1)


class CircuitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator, SuperOp
        cls.np, cls.QC, cls.Operator, cls.SuperOp = np, QuantumCircuit, Operator, SuperOp

    def assertSameOperator(self, first, second):
        self.np.testing.assert_allclose(self.Operator(first).data, self.Operator(second).data, atol=1e-10)

    def test_fold_native_rotations_without_bloat(self):
        circuit = self.QC(2)
        circuit.rx(0.3, 0)
        circuit.rzz(0.2, 0, 1)
        circuit.rz(0.7, 1)
        result = fold(circuit, FoldPlan(2, seed=7)).value
        self.assertEqual(result.before.operations, 3)
        self.assertEqual(result.after.operations, 5)
        self.assertEqual(result.after.qubits, 2)
        self.assertSameOperator(circuit, result.circuit)

    def test_fold_resets_measurements_and_metadata(self):
        circuit = self.QC(2, 1)
        circuit.metadata = {"nested": {"value": 1}}
        circuit.global_phase = 0.2
        circuit.h(0)
        circuit.reset(0)
        circuit.rzz(0.3, 0, 1)
        circuit.measure(1, 0)
        result = fold(circuit, FoldPlan(3)).value
        self.assertEqual(result.after.reset_count, 1)
        self.assertEqual(result.after.measurement_count, 1)
        self.assertEqual(tuple(result.circuit.count_ops()), ("h", "rzz", "reset", "measure"))
        first = circuit.remove_final_measurements(inplace=False)
        second = result.circuit.remove_final_measurements(inplace=False)
        self.np.testing.assert_allclose(self.SuperOp(first).data, self.SuperOp(second).data, atol=1e-10)
        result.circuit.metadata["nested"]["value"] = 2
        self.assertEqual(circuit.metadata["nested"]["value"], 1)

    def test_scale_one_preserves_operation_sequence(self):
        circuit = self.QC(2)
        circuit.rxx(0.4, 0, 1)
        circuit.reset(1)
        result = fold(circuit, FoldPlan(1)).value
        self.assertEqual(result.circuit, circuit)
        self.assertIsNot(result.circuit, circuit)

    def test_layout_and_physical_qargs_are_preserved(self):
        from qiskit.circuit import Parameter
        from qiskit.circuit.library import RXGate, RZZGate
        from qiskit.transpiler import Layout, Target
        from qiskit.transpiler.layout import TranspileLayout
        circuit = self.QC(3)
        circuit.rx(0.3, 2)
        circuit.rzz(0.2, 2, 0)
        circuit._layout = TranspileLayout(
            Layout({circuit.qubits[0]: 2, circuit.qubits[1]: 1, circuit.qubits[2]: 0}),
            {bit: index for index, bit in enumerate(circuit.qubits)},
        )
        target = Target(num_qubits=3)
        target.add_instruction(RXGate(Parameter("a")), {(2,): None})
        target.add_instruction(RZZGate(Parameter("b")), {(2, 0): None})
        result = fold(circuit, FoldPlan(3), target=target).value
        self.assertEqual(result.circuit.layout.initial_layout, circuit.layout.initial_layout)
        self.assertEqual(result.circuit.num_qubits, 3)
        self.assertTrue(all(result.circuit.find_bit(item.qubits[0]).index == 2 for item in result.circuit.data))
        self.assertSameOperator(circuit, result.circuit)

    def test_modified_circuits_invalidate_old_schedule(self):
        circuit = self.QC(1)
        circuit.rx(0.4, 0)
        circuit._op_start_times = [123]
        folded = fold(circuit, FoldPlan(3)).value.circuit
        randomized = twirl(circuit, TwirlingPlan(gate_names=("rx",))).value.circuit
        self.assertEqual(circuit.op_start_times, [123])
        for changed in (folded, randomized):
            with self.assertRaises(AttributeError):
                _ = changed.op_start_times

    def test_custom_gate_alias_is_not_incorrectly_twirled(self):
        from qiskit.circuit import Gate
        circuit = self.QC(2)
        circuit.append(Gate("rzz", 2, [0.2]), (0, 1))
        self.assertIsInstance(twirl(circuit), Err)

    def test_unsupported_inverse_is_not_lowered(self):
        from qiskit.circuit.library import SXGate
        from qiskit.transpiler import Target
        target = Target(num_qubits=1)
        target.add_instruction(SXGate(), {(0,): None})
        circuit = self.QC(1)
        circuit.sx(0)
        self.assertIsInstance(fold(circuit, FoldPlan(3), target=target), Err)
        result = fold(circuit, FoldPlan(3), target=target,
                      inverse_factory=lambda gate: (SXGate(), SXGate(), SXGate())).value
        self.assertEqual(result.selection.achieved_scale, 3)
        self.assertEqual(result.represented_weight_scale, 5)
        self.assertSameOperator(circuit, result.circuit)

    def test_native_parameter_restrictions_are_checked(self):
        from qiskit.circuit.library import RXGate
        from qiskit.transpiler import Target
        target = Target(num_qubits=1)
        target.add_instruction(RXGate(0.3), {(0,): None})
        circuit = self.QC(1)
        circuit.rx(0.3, 0)
        self.assertIsInstance(fold(circuit, FoldPlan(1), target=target), Ok)
        self.assertIsInstance(fold(circuit, FoldPlan(3), target=target), Err)

    def test_clifford_twirling_preserves_exact_phase(self):
        for name in ("cx", "cz"):
            circuit = self.QC(2)
            getattr(circuit, name)(0, 1)
            for seed in range(64):
                with self.subTest(name=name, seed=seed):
                    result = twirl(circuit, TwirlingPlan(seed, (name,))).value
                    self.assertSameOperator(circuit, result.circuit)

    def test_rotation_twirling_and_signs(self):
        signs = set()
        for name in ("rx", "ry", "rz", "rxx", "ryy", "rzz"):
            width = 1 if len(name) == 2 else 2
            circuit = self.QC(width)
            getattr(circuit, name)(0.3, *range(width))
            for seed in range(16):
                with self.subTest(name=name, seed=seed):
                    result = twirl(circuit, TwirlingPlan(seed, (name,))).value
                    self.assertSameOperator(circuit, result.circuit)
                    signs.add(result.choices[0].angle_sign)
        self.assertEqual(signs, {-1, 1})

    def test_native_paulis_and_target_twirling(self):
        from qiskit.circuit import Parameter
        from qiskit.circuit.library import RZGate, RZZGate, XGate
        from qiskit.transpiler import Target
        target = Target(num_qubits=2)
        target.add_instruction(XGate(), {(0,): None, (1,): None})
        target.add_instruction(RZGate(Parameter("a")), {(0,): None, (1,): None})
        target.add_instruction(RZZGate(Parameter("b")), {(0, 1): None})
        for label in "XYZ":
            implementation = native_pauli(label, 0, target)
            actual, expected = self.QC(1), self.QC(1)
            actual.global_phase = implementation.global_phase
            for gate in implementation.gates:
                actual.append(gate, (0,))
            getattr(expected, label.lower())(0)
            self.assertSameOperator(actual, expected)
        circuit = self.QC(2)
        circuit.rzz(0.4, 0, 1)
        for seed in range(16):
            result = twirl(circuit, TwirlingPlan(seed, ("rzz",)), target=target).value
            self.assertSameOperator(circuit, result.circuit)
            self.assertFalse(set(result.circuit.count_ops()) - {"x", "rz", "rzz"})

    def test_rx_and_sx_only_native_pauli_fallbacks(self):
        from qiskit.circuit import Parameter
        from qiskit.circuit.library import RXGate, RZGate, SXGate
        from qiskit.transpiler import Target
        for pulse in (RXGate(Parameter("a")), SXGate()):
            target = Target(num_qubits=1)
            target.add_instruction(pulse, {(0,): None})
            target.add_instruction(RZGate(Parameter("b")), {(0,): None})
            for label in "XYZ":
                implementation = native_pauli(label, 0, target)
                actual, expected = self.QC(1), self.QC(1)
                actual.global_phase = implementation.global_phase
                for gate in implementation.gates:
                    actual.append(gate, (0,))
                getattr(expected, label.lower())(0)
                self.assertSameOperator(actual, expected)

    def test_twirling_preserves_resets_and_reports_skips(self):
        circuit = self.QC(2, 1)
        circuit.rzz(0.3, 0, 1)
        circuit.reset(0)
        circuit.crx(0.2, 0, 1)
        circuit.measure(1, 0)
        result = twirl(circuit).value
        self.assertEqual(result.after.reset_count, 1)
        self.assertEqual(result.untwirled_two_qubit_gates, (("crx", 1),))
        first = circuit.remove_final_measurements(inplace=False)
        second = result.circuit.remove_final_measurements(inplace=False)
        self.np.testing.assert_allclose(self.SuperOp(first).data, self.SuperOp(second).data, atol=1e-10)

    def test_control_flow_is_explicitly_rejected(self):
        circuit = self.QC(1, 1)
        with circuit.if_test((circuit.clbits[0], 0)):
            circuit.x(0)
        self.assertIsInstance(fold(circuit), Err)
        self.assertIsInstance(twirl(circuit), Err)

    def test_calibration_build_and_fit(self):
        batch = calibration_circuits(((0,), (1,)), physical_qubits=(3, 5), num_qubits=6).value
        self.assertEqual(len(batch.circuits), 2)
        self.assertEqual(batch.prepared_states, ((0, 0), (1, 1)))
        self.assertEqual(batch.circuits[0].num_qubits, 6)
        model = fit_readout_model(batch, ({"00": 81, "01": 9, "10": 9, "11": 1},
                                         {"00": 4, "01": 16, "10": 16, "11": 64})).value
        self.assertEqual(model.blocks[0].assignment, ((0.9, 0.2), (0.1, 0.8)))
        self.assertIsInstance(calibration_circuits(((0,), (0,))), Err)
        correlated = calibration_circuits(((0, 1), (2,))).value
        self.assertEqual(len(correlated.circuits), 4)

    def test_dd_requires_timing(self):
        self.assertIsInstance(apply_dd(self.QC(1)), Err)

    def test_dd_sequences_with_local_durations(self):
        from qiskit.transpiler import InstructionDurations
        durations = InstructionDurations([("x", None, 16), ("y", None, 16), ("measure", None, 32)], dt=1e-9)
        for sequence in ("xx", "yy", "xy4", "xy8"):
            circuit = self.QC(2, 2)
            circuit.x(0)
            circuit.delay(512, 0)
            circuit.x(0)
            circuit.x(1)
            circuit.measure([0, 1], [0, 1])
            result = apply_dd(circuit, DDPlan(sequence), durations=durations)
            self.assertIsInstance(result, Ok, result)
            self.assertGreater(result.value.inserted_gate_count, 0)
            first, second = self.QC(2), self.QC(2)
            for source, stripped in ((circuit, first), (result.value.circuit, second)):
                stripped.global_phase = source.global_phase
                for item in source.data:
                    if item.operation.name not in ("delay", "measure", "barrier"):
                        qargs = tuple(source.find_bit(bit).index for bit in item.qubits)
                        stripped.append(item.operation, qargs)
            self.assertSameOperator(first, second)

    def test_dd_target_native_y_frames_and_alignment(self):
        from qiskit.circuit import Delay, Measure, Parameter
        from qiskit.circuit.library import RZGate, XGate
        from qiskit.transpiler import InstructionProperties, Target
        target = Target(num_qubits=2, dt=1e-9, pulse_alignment=4, acquire_alignment=4)
        target.add_instruction(XGate(), {(q,): InstructionProperties(duration=16e-9) for q in (0, 1)})
        target.add_instruction(RZGate(Parameter("a")), {(q,): InstructionProperties(duration=0) for q in (0, 1)})
        target.add_instruction(Measure(), {(q,): InstructionProperties(duration=32e-9) for q in (0, 1)})
        target.add_instruction(Delay(Parameter("t")), {(q,): None for q in (0, 1)})
        circuit = self.QC(2, 2)
        circuit.x(0)
        circuit.delay(512, 0)
        circuit.x(0)
        circuit.x(1)
        circuit.measure([0, 1], [0, 1])
        result = apply_dd(circuit, DDPlan("xy4"), target=target)
        self.assertIsInstance(result, Ok, result)
        self.assertGreater(result.value.circuit.count_ops().get("rz", 0), 0)
        self.assertEqual(result.value.pulse_alignment, 4)
        self.assertEqual(result.value.acquire_alignment, 4)
        self.assertTrue(all(start % 4 == 0 for start in result.value.circuit.op_start_times))

    def test_dd_reschedules_only_for_nontrivial_alignment(self):
        from unittest.mock import patch
        from qiskit.transpiler import InstructionDurations
        from qiskit.transpiler.passes import ConstrainedReschedule
        durations = InstructionDurations([("x", None, 16), ("measure", None, 32)], dt=1e-9)
        circuit = self.QC(1, 1)
        circuit.x(0)
        circuit.delay(512, 0)
        circuit.measure(0, 0)
        for pulse, acquire in ((1, 1), (4, 1), (1, 4), (4, 4)):
            with self.subTest(pulse=pulse, acquire=acquire):
                with patch("qiskit.transpiler.passes.ConstrainedReschedule",
                           wraps=ConstrainedReschedule) as reschedule:
                    result = apply_dd(circuit, DDPlan(pulse_alignment=pulse,
                        acquire_alignment=acquire), durations=durations)
                self.assertIsInstance(result, Ok, result)
                if pulse == acquire == 1:
                    reschedule.assert_not_called()
                else:
                    reschedule.assert_called_once_with(pulse_alignment=pulse,
                                                       acquire_alignment=acquire)
                self.assertGreater(result.value.inserted_gate_count, 0)


if __name__ == "__main__":
    unittest.main()
