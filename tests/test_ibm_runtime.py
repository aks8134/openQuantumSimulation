import unittest
from dataclasses import FrozenInstanceError

from IBMRuntime import (
    Aer,
    Circuit,
    CompilationMetrics,
    DensityMatrixResult,
    Err,
    Estimate,
    EstimateResult,
    FakeIBMBackend,
    H,
    IBMHardware,
    Ok,
    Sample,
    SimulateDensityMatrices,
    counts_dict,
    compile_circuit_batch_sync,
    draw_transpiled_circuit_layout_sync,
    empty,
    execution_plan,
    h,
    observable,
    pauli,
    pauli_term,
    pipe,
    run_async,
    run_sample_batch_sync,
    run_sync,
    save_density_matrix,
    snapshots_dict,
    validate,
    x,
)
from examples.bell_pair import bell_circuit, bell_plan


class CircuitTests(unittest.TestCase):
    def test_construction_is_immutable(self):
        original = empty(1, 0)
        transformed = pipe(original, h(0))

        self.assertEqual(original.operations, ())
        self.assertEqual(transformed.operations, (H(0),))
        with self.assertRaises(FrozenInstanceError):
            transformed.name = "changed"

    def test_bell_pair_is_valid(self):
        plan = execution_plan(bell_circuit(), Aer(), workload=Sample(shots=32))
        self.assertEqual(validate(plan), Ok(plan))

    def test_validation_returns_error_data(self):
        plan = execution_plan(Circuit(1, 0, (H(2),)), Aer(), workload=Sample(shots=0))
        result = validate(plan)

        self.assertIsInstance(result, Err)
        self.assertGreaterEqual(len(result.error.problems), 3)

    def test_estimator_validation_rejects_measurement_and_bad_observable(self):
        invalid_observable = observable(
            "invalid",
            pauli_term(1.0, pauli("Z", 2)),
        )
        plan = execution_plan(
            bell_circuit(),
            Aer(),
            workload=Estimate((invalid_observable,), precision=-1.0),
        )

        result = validate(plan)

        self.assertIsInstance(result, Err)
        self.assertGreaterEqual(len(result.error.problems), 3)


class AerIntegrationTests(unittest.TestCase):
    def test_bell_pair_runs_synchronously(self):
        result = run_sync(bell_plan(shots=256, seed_simulator=7))

        self.assertIsInstance(result, Ok)
        counts = counts_dict(result.value)
        self.assertEqual(sum(counts.values()), 256)
        self.assertLessEqual(set(counts), {"00", "11"})
        self.assertIsInstance(result.value.counts, tuple)

    def test_density_matrix_snapshot_runs_synchronously(self):
        circuit = pipe(
            empty(1, 0, name="excited_state"),
            x(0),
            save_density_matrix((0,), label="rho"),
        )
        plan = execution_plan(
            circuit,
            Aer(method="density_matrix"),
            workload=SimulateDensityMatrices(("rho",)),
        )

        result = run_sync(plan)

        self.assertIsInstance(result, Ok)
        self.assertIsInstance(result.value, DensityMatrixResult)
        matrix = snapshots_dict(result.value)["rho"]
        self.assertAlmostEqual(matrix[0][0].real, 0.0)
        self.assertAlmostEqual(matrix[1][1].real, 1.0)

    def test_estimator_returns_named_expectation_values(self):
        observables = (
            observable("x", pauli_term(1.0, pauli("X", 0))),
            observable("z", pauli_term(1.0, pauli("Z", 0))),
        )
        plan = execution_plan(
            pipe(empty(1, 0, name="plus"), h(0)),
            Aer(),
            workload=Estimate(
                observables,
                precision=0.02,
                seed_simulator=7,
            ),
        )

        result = run_sync(plan)

        self.assertIsInstance(result, Ok)
        self.assertIsInstance(result.value, EstimateResult)
        self.assertEqual(
            tuple(value.observable for value in result.value.values),
            ("x", "z"),
        )
        self.assertAlmostEqual(result.value.values[0].value, 1.0, delta=0.1)
        self.assertAlmostEqual(result.value.values[1].value, 0.0, delta=0.1)
        self.assertEqual(result.value.values[0].standard_error, 0.02)

    def test_sampler_batch_runs_in_one_immutable_result_tuple(self):
        result = run_sample_batch_sync(
            (bell_circuit(), bell_circuit()),
            Aer(),
            workload=Sample(shots=64, seed_simulator=19),
        )

        self.assertIsInstance(result, Ok)
        self.assertEqual(len(result.value), 2)
        self.assertTrue(all(item.shots == 64 for item in result.value))
        self.assertTrue(
            all(sum(counts_dict(item).values()) == 64 for item in result.value)
        )
        self.assertEqual(result.value[0].job_id, result.value[1].job_id)

    def test_compile_batch_returns_metrics_without_sampling(self):
        result = compile_circuit_batch_sync(
            (bell_circuit(), bell_circuit()),
            Aer(),
        )

        self.assertIsInstance(result, Ok)
        self.assertEqual(len(result.value), 2)
        self.assertTrue(
            all(isinstance(item, CompilationMetrics) for item in result.value)
        )
        self.assertTrue(
            all(item.original_gate_count > 0 for item in result.value)
        )
        self.assertTrue(
            all(item.compiled_depth > 0 for item in result.value)
        )

    def test_aer_transpiled_layout_uses_unconstrained_identity_view(self):
        result = draw_transpiled_circuit_layout_sync(
            bell_circuit(),
            Aer(),
            view="virtual",
            logical_labels=("control", "target"),
        )

        self.assertIsInstance(result, Ok)
        self.assertEqual(len(result.value.axes), 2)
        self.assertIn(
            "identity placement",
            result.value.axes[0].get_title(),
        )
        self.assertEqual(
            result.value.axes[1].get_title(),
            "Logical → physical mapping",
        )
        table_text = tuple(
            cell.get_text().get_text()
            for table in result.value.axes[1].tables
            for cell in table.get_celld().values()
        )
        self.assertIn("control", table_text)
        self.assertIn("target", table_text)
        result.value.clear()

    def test_fake_fez_layout_transpiles_without_credentials(self):
        result = draw_transpiled_circuit_layout_sync(
            bell_circuit(),
            FakeIBMBackend("fake_fez"),
            view="physical",
            logical_labels=("control", "target"),
        )

        self.assertIsInstance(result, Ok)
        self.assertEqual(len(result.value.axes), 2)
        table_text = tuple(
            cell.get_text().get_text()
            for table in result.value.axes[1].tables
            for cell in table.get_celld().values()
        )
        self.assertIn("Logical", table_text)
        self.assertIn("Physical", table_text)
        self.assertIn("control", table_text)
        self.assertIn("target", table_text)
        node_labels = tuple(
            text.get_text() for text in result.value.axes[0].texts
        )
        self.assertEqual(len(node_labels), 2)
        self.assertTrue(all(label.isdecimal() for label in node_labels))
        self.assertTrue(set(node_labels).issubset(set(table_text)))
        result.value.clear()

    def test_ibm_target_without_account_returns_error_data(self):
        result = run_sync(bell_plan(IBMHardware("unused-backend"), shots=1))

        self.assertIsInstance(result, Err)
        self.assertEqual(result.error.stage, "resolve_backend")


class AsyncAerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_bell_plan_runs_asynchronously(self):
        result = await run_async(bell_plan(shots=128, seed_simulator=11))

        self.assertIsInstance(result, Ok)
        counts = counts_dict(result.value)
        self.assertEqual(sum(counts.values()), 128)
        self.assertLessEqual(set(counts), {"00", "11"})

    async def test_estimator_runs_asynchronously(self):
        plan = execution_plan(
            pipe(empty(1, 0, name="plus_async"), h(0)),
            Aer(),
            workload=Estimate(
                (observable("x", pauli_term(1.0, pauli("X", 0))),),
                precision=0.02,
                seed_simulator=13,
            ),
        )

        result = await run_async(plan)

        self.assertIsInstance(result, Ok)
        self.assertIsInstance(result.value, EstimateResult)
        self.assertAlmostEqual(result.value.values[0].value, 1.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
