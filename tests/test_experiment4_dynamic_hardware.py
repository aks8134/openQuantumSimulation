import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from IBMRuntime import Err, ExecutionFailure, Measure, Ok, SampleResult
from Model1.Experiment4 import dynamic_lie_trotter as experiment
from Model1.Experiment4 import recover_dynamic_lie_trotter as recovery


def sample_result(counts):
    return SampleResult(
        counts=tuple(counts.items()),
        shots=sum(counts.values()),
        backend_name="test",
        job_id="test-job",
        original_gate_count=0,
        original_depth=0,
        compiled_gate_count=0,
        compiled_depth=0,
    )


class Experiment4CircuitTests(unittest.TestCase):
    def test_builds_five_terminal_measurement_circuits_per_time(self):
        times = np.asarray((0.0, 0.2))
        circuits, metadata, schedule = (
            experiment.build_sample_circuits(times, 4)
        )

        self.assertEqual(len(circuits), 10)
        self.assertEqual(
            metadata,
            tuple(
                (time_index, basis)
                for time_index in range(2)
                for basis in experiment.MEASUREMENT_BASES
            ),
        )
        self.assertEqual(schedule.substep_dts, (0.2,))
        self.assertAlmostEqual(schedule.jump_angles[0], 0.4)

        for circuit in circuits:
            self.assertEqual(circuit.qubit_count, 6)
            self.assertEqual(circuit.bit_count, 5)
            terminal_measurements = tuple(
                operation
                for operation in circuit.operations[-4:]
                if isinstance(operation, Measure)
            )
            self.assertEqual(len(terminal_measurements), 4)
            self.assertEqual(
                tuple(value.bit for value in terminal_measurements),
                (1, 2, 3, 4),
            )

    def test_arbitrary_times_produce_interval_specific_gate_angles(self):
        times = np.asarray((0.0, 0.1, 0.4, 1.0))
        circuits, metadata, schedule = experiment.build_sample_circuits(
            times,
            2,
        )

        self.assertEqual(len(circuits), 20)
        self.assertEqual(len(metadata), 20)
        np.testing.assert_allclose(schedule.substep_dts, (0.1, 0.3, 0.6))
        np.testing.assert_allclose(
            schedule.jump_angles,
            tuple(2.0 * np.sqrt(experiment.reference_tools.gamma * value)
                  for value in (0.1, 0.3, 0.6)),
        )
        final_z_circuit = circuits[3 * len(experiment.MEASUREMENT_BASES)]
        implemented_jump_angles = tuple(
            operation.angle
            for operation in final_z_circuit.operations
            if getattr(operation, "label", None) == "K_1"
        )
        np.testing.assert_allclose(
            implemented_jump_angles,
            schedule.jump_angles,
        )

    def test_trotter_delta_t_uses_full_steps_and_short_remainder(self):
        times = np.asarray((0.0, 0.7))
        circuits, _, schedule = experiment.build_sample_circuits(
            times,
            2,
            trotter_delta_t=0.2,
        )

        np.testing.assert_allclose(
            schedule.interval_substep_dts[0],
            (0.2, 0.2, 0.2, 0.1),
        )
        final_z_circuit = circuits[len(experiment.MEASUREMENT_BASES)]
        implemented_jump_angles = tuple(
            operation.angle
            for operation in final_z_circuit.operations
            if getattr(operation, "label", None) == "K_1"
        )
        np.testing.assert_allclose(
            implemented_jump_angles,
            tuple(
                2.0 * np.sqrt(experiment.reference_tools.gamma * dt)
                for dt in (0.2, 0.2, 0.2, 0.1)
            ),
        )

    def test_explicit_time_list_overrides_uniform_grid_options(self):
        options = experiment.parse_arguments(
            (
                "--times",
                "0,0.1",
                "0.4",
                "1.0",
                "--t-final",
                "20",
                "--time-points",
                "101",
            )
        )

        np.testing.assert_allclose(
            experiment.time_grid_from_options(options),
            (0.0, 0.1, 0.4, 1.0),
        )

    def test_command_line_accepts_maximum_trotter_step(self):
        options = experiment.parse_arguments(
            ("--trotter-delta-t", "0.05")
        )

        self.assertAlmostEqual(options.trotter_delta_t, 0.05)

    def test_reconstructs_observables_from_grouped_counts(self):
        z_counts = {"100": 100}
        unbiased_pair_counts = {
            "000": 25,
            "010": 25,
            "100": 25,
            "110": 25,
        }
        results = (
            sample_result(z_counts),
            sample_result(unbiased_pair_counts),
            sample_result(unbiased_pair_counts),
            sample_result(unbiased_pair_counts),
            sample_result(unbiased_pair_counts),
        )
        metadata = tuple((0, basis) for basis in experiment.MEASUREMENT_BASES)

        measured, standard_errors, _ = experiment.observables_from_results(
            results,
            metadata,
            number_of_qubits=2,
            time_count=1,
        )

        np.testing.assert_allclose(measured[0][:, 0], (0.0, 1.0))
        np.testing.assert_allclose(measured[1][:, 0], (0.0,))
        np.testing.assert_allclose(measured[2][:, 0], (0.0,))
        np.testing.assert_allclose(standard_errors[0][:, 0], (0.0, 0.0))

    def test_command_line_selects_system_size_and_backend(self):
        options = experiment.parse_arguments(
            ("--n-qubits", "7", "--backend", "ibm_kingston")
        )

        self.assertEqual(options.n_qubits, 7)
        self.assertEqual(options.backend, "ibm_kingston")
        self.assertEqual(options.account_file, experiment.DEFAULT_ACCOUNT_FILE)

    def test_hardware_memory_error_is_reported_without_retry(self):
        circuits = tuple(range(5))
        options = SimpleNamespace(
            backend="ibm_kingston",
            aer_method="automatic",
            account_file=experiment.DEFAULT_ACCOUNT_FILE,
            optimization_level=1,
            seed_transpiler=11,
            shots=128,
            seed_simulator=17,
            batch_size=None,
        )

        simulated_error = Err(
            ExecutionFailure(
                stage="collect",
                message="Error code 6073",
            )
        )

        with (
            patch.object(
                experiment,
                "_runtime_target",
                return_value=(object(), object()),
            ),
            patch.object(
                experiment,
                "run_sample_batch_sync",
                return_value=simulated_error,
            ) as run_batch,
            self.assertRaisesRegex(
                RuntimeError,
                "Re-run with a smaller --batch-size",
            ),
        ):
            experiment.execute_sample_circuits(circuits, options)

        run_batch.assert_called_once()

    def test_completed_batch_callback_receives_durable_prefixes(self):
        circuits = tuple(range(5))
        options = SimpleNamespace(
            backend="aer",
            aer_method="automatic",
            account_file=experiment.DEFAULT_ACCOUNT_FILE,
            optimization_level=1,
            seed_transpiler=11,
            shots=128,
            seed_simulator=17,
            batch_size=2,
        )
        prefixes = []

        def simulated_execution(circuit_batch, *_):
            return Ok(tuple(sample_result({"0": 128}) for _ in circuit_batch))

        with (
            patch.object(
                experiment,
                "_runtime_target",
                return_value=(object(), object()),
            ),
            patch.object(
                experiment,
                "run_sample_batch_sync",
                side_effect=simulated_execution,
            ),
        ):
            results = experiment.execute_sample_circuits(
                circuits,
                options,
                on_batch_complete=lambda prefix: prefixes.append(len(prefix)),
            )

        self.assertEqual(len(results), 5)
        self.assertEqual(prefixes, [2, 4, 5])

    def test_checkpoint_round_trip_validates_and_restores_results(self):
        times = np.asarray((0.0, 0.2))
        _, metadata, schedule = (
            experiment.build_sample_circuits(times, 2)
        )
        options = SimpleNamespace(
            backend="ibm_kingston",
            n_qubits=2,
            shots=128,
            optimization_level=1,
            seed_transpiler=11,
            seed_simulator=17,
            aer_method="automatic",
        )
        results = tuple(sample_result({"000": 128}) for _ in range(5))

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            experiment.save_checkpoint(
                checkpoint,
                options,
                times,
                schedule,
                results,
                metadata,
            )
            restored = experiment.load_checkpoint(
                checkpoint,
                options,
                times,
                metadata,
            )

        self.assertEqual(restored, results)

    def test_recovery_downloads_only_completed_sampler_jobs(self):
        class CountsData:
            def get_counts(self):
                return {"000": 128}

        class ProviderResult:
            def join_data(self):
                return CountsData()

        class FakeJob:
            primitive_id = "sampler"
            creation_date = datetime(2026, 9, 21, tzinfo=timezone.utc)

            def __init__(self, job_id, status):
                self._job_id = job_id
                self._status = status

            def job_id(self):
                return self._job_id

            def status(self):
                return SimpleNamespace(name=self._status)

            def result(self):
                return (ProviderResult(),)

        completed = FakeJob("completed-job", "DONE")
        pending = FakeJob("pending-job", "QUEUED")
        results, records = recovery.recover_completed_results(
            (completed, pending),
            "ibm_kingston",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].job_id, "completed-job")
        self.assertEqual(results[0].compiled_depth, -1)
        self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
