import unittest
from collections import Counter

import numpy as np
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Operator

from IBMRuntime import Circuit, RX, RZ, RZZ, Reset, XXPlusYY
from IBMRuntime.interpreters.qiskit import _to_qiskit
from Model1.Experiment5 import randomized_lie_trotter as experiment5
from Model1.Experiment7 import randomized_lie_trotter as experiment7


class Experiment7ExplicitDecompositionTests(unittest.TestCase):
    def test_default_working_point_does_not_change_experiment5_defaults(self):
        experiment5_options = experiment5.parse_arguments(())
        experiment7_options = experiment7.parse_arguments(())

        self.assertEqual(experiment5_options.trajectories, 16)
        self.assertIsNone(experiment5_options.trotter_delta_t)
        self.assertEqual(experiment5_options.t_final, 10.0)
        self.assertEqual(experiment5_options.time_points, 51)

        self.assertEqual(experiment7_options.trajectories, 4)
        self.assertEqual(experiment7_options.trotter_delta_t, 0.1)
        self.assertEqual(experiment7_options.t_final, 2.0)
        self.assertEqual(experiment7_options.time_points, 11)
        self.assertFalse(experiment7_options.no_circuit_plots)
        self.assertFalse(experiment7_options.optimized_classical_reference)
        self.assertFalse(
            hasattr(
                experiment5_options,
                "optimized_classical_reference",
            )
        )

        disabled = experiment7.parse_arguments(("--no-circuit-plots",))
        self.assertTrue(disabled.no_circuit_plots)

        optimized = experiment7.parse_arguments(
            ("--optimized-classical-reference",)
        )
        self.assertTrue(optimized.optimized_classical_reference)

        with self.assertRaisesRegex(ValueError, "choose either"):
            experiment7.parse_arguments(
                (
                    "--classical-reference",
                    "--optimized-classical-reference",
                )
            )

    def test_optimized_reference_model_uses_n_plus_one_basis(self):
        hamiltonian, jumps, initial_density = (
            experiment7.build_optimized_reference_model(10)
        )

        self.assertEqual(hamiltonian.shape, (11, 11))
        self.assertEqual(tuple(jump.shape for jump in jumps), ((11, 11),) * 2)
        self.assertEqual(initial_density.shape, (11, 11))
        self.assertAlmostEqual(np.trace(initial_density).real, 1.0)
        self.assertEqual(np.count_nonzero(initial_density), 1)

    def test_optimized_reference_matches_full_reference(self):
        times = np.asarray((0.0, 0.1, 0.2))
        for number_of_qubits in (2, 3):
            with self.subTest(number_of_qubits=number_of_qubits):
                optimized = (
                    experiment7.calculate_optimized_classical_reference(
                        times,
                        number_of_qubits,
                    )
                )
                full = experiment5.calculate_exact_reference(
                    times,
                    number_of_qubits,
                )
                for optimized_family, full_family in zip(
                    optimized,
                    full,
                    strict=True,
                ):
                    np.testing.assert_allclose(
                        optimized_family,
                        full_family,
                        rtol=1e-12,
                        atol=1e-12,
                    )

    def test_optimized_reference_accepts_one_nonzero_time(self):
        optimized = experiment7.calculate_optimized_classical_reference(
            np.asarray((0.37,)),
            3,
        )
        full = experiment5.calculate_exact_reference(
            np.asarray((0.0, 0.37)),
            3,
        )
        for optimized_family, full_family in zip(
            optimized,
            full,
            strict=True,
        ):
            np.testing.assert_allclose(
                optimized_family[:, 0],
                full_family[:, -1],
                rtol=1e-12,
                atol=1e-12,
            )

    def test_single_exchange_has_requested_counts_and_depth(self):
        circuit = Circuit(
            qubit_count=2,
            bit_count=0,
            operations=experiment7.explicit_xx_plus_yy_operations(
                0.4,
                0,
                1,
            ),
            name="explicit_exchange",
        )
        qiskit_circuit = _to_qiskit(circuit)

        self.assertEqual(
            qiskit_circuit.count_ops(),
            {"rz": 6, "rx": 6, "rzz": 2},
        )
        self.assertEqual(qiskit_circuit.size(), 14)
        self.assertEqual(qiskit_circuit.depth(), 8)
        self.assertEqual(
            experiment7.single_exchange_resource_summary(),
            {
                "qiskit_operation_count": 14,
                "gate_counts": {"rz": 6, "rx": 6, "rzz": 2},
                "virtual_gate_count": 6,
                "physical_gate_count": 8,
                "standard_circuit_depth": 8,
                "physical_pulse_depth": 5,
                "two_qubit_depth": 2,
                "virtual_gate_types": ["rz"],
                "physical_gate_types": ["rx", "rzz"],
            },
        )

    def test_single_exchange_is_exact_xx_plus_yy_up_to_global_phase(self):
        for theta in (0.1, 0.4, 1.2, -0.7):
            with self.subTest(theta=theta):
                circuit = Circuit(
                    qubit_count=2,
                    bit_count=0,
                    operations=experiment7.explicit_xx_plus_yy_operations(
                        theta,
                        0,
                        1,
                    ),
                )
                self.assertTrue(
                    Operator(_to_qiskit(circuit)).equiv(
                        Operator(XXPlusYYGate(theta, 0.0))
                    )
                )

    def test_one_step_stays_logical_until_compiler_boundary(self):
        for boundary in (
            experiment7.LEFT_BOUNDARY,
            experiment7.RIGHT_BOUNDARY,
        ):
            with self.subTest(boundary=boundary):
                logical = experiment7.build_one_step_circuit(
                    number_of_qubits=4,
                    dt=0.1,
                    boundary=boundary,
                )
                logical_counts = Counter(
                    type(operation) for operation in logical.operations
                )

                self.assertEqual(logical_counts[XXPlusYY], 4)
                self.assertEqual(logical_counts[Reset], 1)
                self.assertEqual(logical_counts[RX], 0)
                self.assertEqual(logical_counts[RZZ], 0)

                prepared = experiment7.prepare_circuit_for_compilation(
                    logical
                )
                counts = Counter(
                    type(operation) for operation in prepared.operations
                )

                self.assertFalse(
                    any(
                        isinstance(operation, XXPlusYY)
                        for operation in prepared.operations
                    )
                )
                # N-1 system bonds plus one selected boundary jump gives N
                # exchange blocks per randomized substep.
                self.assertEqual(counts[RZ], 4 * 6)
                self.assertEqual(counts[RX], 4 * 6)
                self.assertEqual(counts[RZZ], 4 * 2)
                self.assertEqual(counts[Reset], 1)

    def test_sample_circuits_remain_logical_but_schedule_is_unchanged(self):
        circuits, metadata, schedule = experiment7.build_sample_circuits(
            times=np.asarray((0.0, 0.2)),
            number_of_qubits=3,
            trajectories=2,
            seed_trajectories=29,
            trotter_delta_t=0.1,
        )

        self.assertEqual(len(circuits), 2 * 2 * len(experiment7.MEASUREMENT_BASES))
        self.assertEqual(len(metadata), len(circuits))
        np.testing.assert_allclose(schedule.substep_dts, (0.1, 0.1))
        final_circuits = tuple(
            circuit
            for circuit, (time_index, _, _) in zip(
                circuits,
                metadata,
                strict=True,
            )
            if time_index == 1
        )
        self.assertTrue(
            all(
                any(
                    isinstance(operation, XXPlusYY)
                    for operation in circuit.operations
                )
                for circuit in final_circuits
            )
        )
        self.assertTrue(
            all(
                not any(
                    isinstance(operation, XXPlusYY)
                    for operation in (
                        experiment7.prepare_circuit_for_compilation(
                            circuit
                        ).operations
                    )
                )
                for circuit in final_circuits
            )
        )


if __name__ == "__main__":
    unittest.main()
