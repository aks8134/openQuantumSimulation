import unittest
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from IBMRuntime import (
    CompilationMetrics,
    Err,
    ExecutionFailure,
    Measure,
    Ok,
    SampleResult,
)
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

    def test_single_explicit_time_saves_only_target(self):
        options = experiment.parse_arguments(("--times", "0.7"))

        np.testing.assert_allclose(
            experiment.time_grid_from_options(options),
            (0.7,),
        )

    def test_single_zero_time_is_allowed(self):
        options = experiment.parse_arguments(("--times", "0"))

        np.testing.assert_allclose(
            experiment.time_grid_from_options(options),
            (0.0,),
        )

    def test_single_target_has_only_target_circuits_with_internal_steps(self):
        circuits, metadata, schedule = experiment.build_sample_circuits(
            (0.7,),
            2,
            trotter_delta_t=0.2,
        )

        self.assertEqual(len(circuits), len(experiment.MEASUREMENT_BASES))
        self.assertEqual(
            metadata,
            tuple((0, basis) for basis in experiment.MEASUREMENT_BASES),
        )
        np.testing.assert_allclose(
            schedule.substep_dts,
            (0.2, 0.2, 0.2, 0.1),
        )
        self.assertFalse(schedule.includes_initial_time)

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

    def test_transpilation_summary_covers_one_step_and_full_circuit(self):
        one_step = CompilationMetrics(
            original_gate_count=12,
            original_depth=8,
            compiled_gate_count=31,
            compiled_depth=22,
        )
        full_result = SampleResult(
            counts=(("0", 10),),
            shots=10,
            backend_name="aer",
            job_id="job",
            original_gate_count=50,
            original_depth=40,
            compiled_gate_count=140,
            compiled_depth=95,
        )

        summary = experiment.transpilation_summary(
            one_step,
            full_result,
        )

        self.assertEqual(
            tuple(record["label"] for record in summary),
            (
                "one dynamic Lie substep",
                "complete final-time circuit (Z basis)",
            ),
        )
        self.assertEqual(summary[0]["pre"]["operation_count"], 12)
        self.assertEqual(summary[0]["post"]["depth"], 22)
        self.assertEqual(summary[1]["pre"]["operation_count"], 50)
        self.assertEqual(summary[1]["post"]["depth"], 95)

    def test_compiles_one_step_without_submitting_a_sampler_job(self):
        options = SimpleNamespace(
            n_qubits=4,
            backend="aer",
            aer_method="automatic",
            account_file=experiment.DEFAULT_ACCOUNT_FILE,
            optimization_level=1,
            seed_transpiler=11,
        )
        metrics = CompilationMetrics(
            original_gate_count=12,
            original_depth=8,
            compiled_gate_count=31,
            compiled_depth=22,
        )

        with (
            patch.object(
                experiment,
                "_runtime_target",
                return_value=(object(), object()),
            ),
            patch.object(
                experiment,
                "compile_circuit_batch_sync",
                return_value=Ok((metrics,)),
            ) as compile_batch,
            patch.object(
                experiment,
                "run_sample_batch_sync",
                side_effect=AssertionError("must not submit"),
            ) as sample_batch,
        ):
            observed = experiment.compile_one_step_metrics(
                options,
                dt=0.2,
                jump_angle=0.4,
            )

        self.assertEqual(observed, metrics)
        self.assertEqual(len(compile_batch.call_args.args[0]), 1)
        sample_batch.assert_not_called()

    def test_layout_figure_compiles_without_sampler_submission(self):
        options = SimpleNamespace(
            n_qubits=2,
            backend="aer",
            aer_method="automatic",
            account_file=experiment.DEFAULT_ACCOUNT_FILE,
            optimization_level=1,
            seed_transpiler=11,
        )
        circuit = experiment.build_sample_circuits((0.2,), 2)[0][0]
        figure, _ = experiment.plt.subplots()

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "layout.png"
            with (
                patch.object(
                    experiment,
                    "_runtime_target",
                    return_value=(object(), object()),
                ),
                patch.object(
                    experiment,
                    "draw_transpiled_circuit_layout_sync",
                    return_value=Ok(figure),
                ) as draw_layout,
                patch.object(
                    experiment,
                    "run_sample_batch_sync",
                    side_effect=AssertionError("must not submit"),
                ) as sample_batch,
            ):
                experiment.plot_transpiled_circuit_layout(
                    circuit,
                    options,
                    output_path,
                )
            output_exists = output_path.exists()

        self.assertTrue(output_exists)
        draw_layout.assert_called_once()
        self.assertEqual(draw_layout.call_args.kwargs["view"], "physical")
        self.assertEqual(
            draw_layout.call_args.kwargs["logical_labels"],
            (
                "system site 1",
                "system site 0",
                "low ancilla a_L",
                "high ancilla a_H",
            ),
        )
        sample_batch.assert_not_called()

    def test_layout_only_mode_never_submits_or_writes_results(self):
        with TemporaryDirectory() as directory:
            options = experiment.parse_arguments(
                (
                    "--layout-only",
                    "--n-qubits",
                    "2",
                    "--backend",
                    "fake_fez",
                    "--times",
                    "0.2",
                    "--trotter-delta-t",
                    "0.1",
                    "--output-directory",
                    directory,
                )
            )
            with (
                patch.object(
                    experiment,
                    "plot_transpiled_circuit_layout",
                ) as plot_layout,
                patch.object(
                    experiment,
                    "execute_sample_circuits",
                    side_effect=AssertionError("must not submit"),
                ) as execute,
            ):
                experiment.main(options)

            result_directory_exists = (
                Path(directory) / "results"
            ).exists()

        plot_layout.assert_called_once()
        execute.assert_not_called()
        self.assertFalse(result_directory_exists)

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

    def test_result_archive_appends_and_replaces_matching_times(self):
        options = SimpleNamespace(
            backend="aer",
            n_qubits=2,
            shots=10,
            optimization_level=1,
            seed_transpiler=11,
            seed_simulator=17,
            aer_method="automatic",
            trotter_delta_t=0.05,
            classical_reference=False,
        )

        def save_run(path, times, marker):
            times = np.asarray(times, dtype=float)
            _, metadata, schedule = experiment.build_sample_circuits(
                times,
                options.n_qubits,
                trotter_delta_t=options.trotter_delta_t,
            )
            results = tuple(
                sample_result({str(marker): options.shots})
                for _ in metadata
            )
            measured = (
                np.full((2, len(times)), marker, dtype=float),
                np.full((1, len(times)), marker, dtype=float),
                np.full((1, len(times)), marker, dtype=float),
            )
            standard_errors = tuple(
                np.zeros_like(values) for values in measured
            )
            grouped_counts = tuple(
                {
                    basis: {str(marker): options.shots}
                    for basis in experiment.MEASUREMENT_BASES
                }
                for _ in times
            )
            return experiment.save_results(
                path,
                options,
                times,
                schedule,
                results,
                metadata,
                measured,
                standard_errors,
                grouped_counts,
            )

        with TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            save_run(result_path, (0.1,), 1)
            save_run(result_path, (0.2,), 2)
            combined = save_run(result_path, (0.1,), 9)

        self.assertEqual(combined["times"], [0.1, 0.2])
        np.testing.assert_allclose(
            combined["observables"]["populations"][0],
            (9.0, 2.0),
        )
        self.assertEqual(len(combined["raw_counts"]), 10)
        self.assertEqual(len(combined["circuit_metadata"]), 10)
        replacement_counts = [
            record["counts"]
            for record in combined["raw_counts"]
            if np.isclose(record["time"], 0.1)
        ]
        self.assertEqual(replacement_counts, [{"9": 10}] * 5)
        self.assertEqual(len(combined["run_history"]), 3)

    def test_archive_compatibility_is_checked_before_sampling(self):
        options = SimpleNamespace(
            backend="aer",
            n_qubits=2,
            shots=10,
            optimization_level=1,
            seed_transpiler=11,
            seed_simulator=17,
            aer_method="automatic",
            trotter_delta_t=0.05,
            classical_reference=False,
        )
        times = np.asarray((0.2,), dtype=float)
        _, metadata, schedule = experiment.build_sample_circuits(
            times,
            2,
            trotter_delta_t=options.trotter_delta_t,
        )
        results = tuple(
            sample_result({"0": options.shots}) for _ in metadata
        )
        measured = (
            np.zeros((2, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
        )
        grouped_counts = tuple(
            {
                basis: {"0": options.shots}
                for basis in experiment.MEASUREMENT_BASES
            }
            for _ in times
        )

        with TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            experiment.save_results(
                result_path,
                options,
                times,
                schedule,
                results,
                metadata,
                measured,
                measured,
                grouped_counts,
            )
            incompatible = SimpleNamespace(**{**vars(options), "shots": 20})
            with self.assertRaisesRegex(
                ValueError,
                "shots_per_measurement_circuit",
            ):
                experiment.validate_existing_result_compatibility(
                    result_path,
                    incompatible,
                    schedule,
                )

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

    def test_metadata_only_backfills_negative_metrics_without_submission(self):
        payload = {
            "experiment": "Model1/Experiment4 dynamic Lie-Trotter",
            "backend": "ibm_kingston",
            "number_of_system_qubits": 2,
            "times": [0.0, 0.2],
            "optimization_level": 1,
            "seed_transpiler": 11,
            "circuit_metadata": [
                {
                    "time_index": 0,
                    "basis": "Z",
                    "original_operation_count": -1,
                    "original_depth": -1,
                    "compiled_operation_count": -1,
                    "compiled_depth": -1,
                }
            ],
        }
        metrics = CompilationMetrics(
            original_gate_count=3,
            original_depth=2,
            compiled_gate_count=9,
            compiled_depth=7,
        )

        with TemporaryDirectory() as directory:
            metadata_file = Path(directory) / "recovered.json"
            metadata_file.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            options = SimpleNamespace(
                metadata_file=metadata_file,
                overwrite_metadata=False,
                aer_method="automatic",
                account_file=experiment.DEFAULT_ACCOUNT_FILE,
            )
            with (
                patch.object(
                    experiment,
                    "_runtime_target",
                    return_value=(object(), object()),
                ),
                patch.object(
                    experiment,
                    "compile_circuit_batch_sync",
                    return_value=Ok((metrics,)),
                ) as compile_batch,
                patch.object(
                    experiment,
                    "run_sample_batch_sync",
                    side_effect=AssertionError("must not submit"),
                ) as sample_batch,
            ):
                experiment.backfill_metadata_only(options)

            updated = json.loads(
                metadata_file.read_text(encoding="utf-8")
            )
            backup = metadata_file.with_name(
                "recovered.before_metadata.json"
            )
            backup_exists = backup.exists()

        self.assertEqual(
            updated["circuit_metadata"][0]["original_operation_count"],
            3,
        )
        self.assertEqual(
            updated["circuit_metadata"][0]["compiled_depth"],
            7,
        )
        self.assertFalse(
            updated["metadata_backfill"]["hardware_job_submitted"]
        )
        self.assertTrue(backup_exists)
        compile_batch.assert_called_once()
        sample_batch.assert_not_called()

    def test_metadata_only_rebuilds_each_archived_run_schedule(self):
        run_options = SimpleNamespace(
            backend="aer",
            n_qubits=2,
            shots=10,
            optimization_level=1,
            seed_transpiler=11,
            seed_simulator=17,
            aer_method="automatic",
            trotter_delta_t=0.05,
        )
        measured = (
            np.zeros((2, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
        )

        with TemporaryDirectory() as directory:
            metadata_file = Path(directory) / "archive.json"
            for time in (0.1, 0.2):
                times = np.asarray((time,))
                _, metadata, schedule = experiment.build_sample_circuits(
                    times,
                    2,
                    trotter_delta_t=run_options.trotter_delta_t,
                )
                results = tuple(
                    sample_result({"0": 10}) for _ in metadata
                )
                grouped_counts = (
                    {
                        basis: {"0": 10}
                        for basis in experiment.MEASUREMENT_BASES
                    },
                )
                experiment.save_results(
                    metadata_file,
                    run_options,
                    times,
                    schedule,
                    results,
                    metadata,
                    measured,
                    measured,
                    grouped_counts,
                )

            metrics = CompilationMetrics(
                original_gate_count=3,
                original_depth=2,
                compiled_gate_count=9,
                compiled_depth=7,
            )
            metadata_options = SimpleNamespace(
                metadata_file=metadata_file,
                overwrite_metadata=True,
                aer_method="automatic",
                account_file=experiment.DEFAULT_ACCOUNT_FILE,
            )
            with (
                patch.object(
                    experiment,
                    "_runtime_target",
                    return_value=(object(), object()),
                ),
                patch.object(
                    experiment,
                    "compile_circuit_batch_sync",
                    return_value=Ok((metrics,) * 10),
                ) as compile_batch,
            ):
                experiment.backfill_metadata_only(metadata_options)

            updated = json.loads(metadata_file.read_text(encoding="utf-8"))

        self.assertEqual(len(updated["circuit_metadata"]), 10)
        self.assertEqual(
            {record["time"] for record in updated["circuit_metadata"]},
            {0.1, 0.2},
        )
        compiled_circuits = compile_batch.call_args.args[0]
        self.assertEqual(len(compiled_circuits), 10)


if __name__ == "__main__":
    unittest.main()
