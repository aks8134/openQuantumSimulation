import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from IBMRuntime import (
    ClassicallyControlledXXPlusYY,
    Measure,
    Ok,
    Reset,
    SampleResult,
    XXPlusYY,
)
from Model1.Experiment5 import randomized_lie_trotter as experiment


def sample_result(counts, shots=100):
    return SampleResult(
        counts=tuple(counts.items()),
        shots=shots,
        backend_name="test",
        job_id="test-job",
        original_gate_count=3,
        original_depth=2,
        compiled_gate_count=7,
        compiled_depth=5,
    )


def options_for_checkpoint():
    return SimpleNamespace(
        backend="aer",
        n_qubits=2,
        trajectories=1,
        shots=100,
        optimization_level=1,
        seed_transpiler=11,
        seed_simulator=17,
        seed_trajectories=29,
        aer_method="automatic",
        trotter_delta_t=0.1,
    )


def result_payload(times, marker, job_id):
    time_count = len(times)
    family_values = {
        "populations": [[marker] * time_count, [marker + 1] * time_count],
        "xy_correlations": [[marker + 2] * time_count],
        "excitation_flows": [[marker + 3] * time_count],
    }
    trajectory_values = {
        key: np.asarray(value, dtype=float)[..., None].tolist()
        for key, value in family_values.items()
    }
    errors = {
        key: np.zeros_like(np.asarray(value, dtype=float)).tolist()
        for key, value in family_values.items()
    }
    return {
        "experiment": "Model1/Experiment5 randomized Lie-Trotter",
        "backend": "aer",
        "number_of_system_qubits": 2,
        "number_of_trajectories": 1,
        "total_shots_per_time_basis": 100,
        "measurement_bases": list(experiment.MEASUREMENT_BASES),
        "optimization_level": 1,
        "seed_transpiler": 11,
        "seed_simulator": 17,
        "seed_trajectories": 29,
        "aer_method": "automatic",
        "times": list(times),
        "completed_at_utc": f"marker-{marker}",
        "evolution_schedule": {
            "requested_trotter_delta_t": 0.1,
        },
        "job_ids": [job_id],
        "transpilation_summary": [],
        "observables": family_values,
        "trajectory_observables": trajectory_values,
        "standard_errors": {
            "shot": errors,
            "trajectory": errors,
            "total": errors,
            "trajectory_component_note": "test",
        },
        "raw_counts": [
            {
                "time_index": time_index,
                "time": time,
                "trajectory_index": 0,
                "basis": basis,
                "counts": {str(marker): 100},
            }
            for time_index, time in enumerate(times)
            for basis in experiment.MEASUREMENT_BASES
        ],
        "circuit_metadata": [
            {
                "time_index": time_index,
                "time": time,
                "trajectory_index": 0,
                "basis": basis,
                "job_id": job_id,
                "original_operation_count": marker,
                "original_depth": marker,
                "compiled_operation_count": marker,
                "compiled_depth": marker,
            }
            for time_index, time in enumerate(times)
            for basis in experiment.MEASUREMENT_BASES
        ],
    }


class Experiment5RandomizedLieTests(unittest.TestCase):
    def test_layout_figure_contains_both_randomized_step_choices(self):
        options = experiment.parse_arguments(
            (
                "--n-qubits",
                "2",
                "--trajectories",
                "2",
                "--shots",
                "100",
                "--times",
                "0.2",
                "--trotter-delta-t",
                "0.2",
            )
        )
        circuits, metadata, _ = experiment.build_sample_circuits(
            (0.2,),
            number_of_qubits=2,
            trajectories=2,
            seed_trajectories=29,
            trotter_delta_t=0.2,
        )
        full_circuit = experiment._representative_full_circuit(
            circuits,
            metadata,
            0,
        )
        one_step_circuits = tuple(
            experiment.build_one_step_circuit(2, 0.2, boundary)
            for boundary in (
                experiment.LEFT_BOUNDARY,
                experiment.RIGHT_BOUNDARY,
            )
        )
        layout_figure, _ = experiment.plt.subplots()
        left_figure, _ = experiment.plt.subplots()
        right_figure, _ = experiment.plt.subplots()

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "layout.png"
            with (
                patch.object(
                    experiment.hardware_tools,
                    "_runtime_target",
                    return_value=(object(), object()),
                ),
                patch.object(
                    experiment,
                    "draw_transpiled_circuit_layout_sync",
                    return_value=Ok(layout_figure),
                ) as draw_layout,
                patch.object(
                    experiment,
                    "draw_circuit",
                    side_effect=(left_figure, right_figure),
                ) as draw_step,
            ):
                experiment.plot_transpiled_circuit_layout(
                    full_circuit,
                    options,
                    output_path,
                    one_step_circuits=one_step_circuits,
                )
            output_exists = output_path.exists()

        self.assertTrue(output_exists)
        self.assertEqual(draw_step.call_count, 2)
        self.assertEqual(draw_layout.call_args.kwargs["view"], "physical")
        self.assertEqual(
            draw_layout.call_args.kwargs["logical_labels"],
            (
                "system site 1",
                "system site 0",
                "boundary ancilla a",
            ),
        )

    def test_fake_fez_layout_only_never_samples_or_writes_results(self):
        with TemporaryDirectory() as directory:
            options = experiment.parse_arguments(
                (
                    "--layout-only",
                    "--n-qubits",
                    "2",
                    "--backend",
                    "fake_fez",
                    "--trajectories",
                    "2",
                    "--shots",
                    "100",
                    "--times",
                    "0.2",
                    "--trotter-delta-t",
                    "0.2",
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
        self.assertEqual(
            len(plot_layout.call_args.kwargs["one_step_circuits"]),
            2,
        )
        execute.assert_not_called()
        self.assertFalse(result_directory_exists)

    def test_result_archive_appends_and_replaces_time_points(self):
        existing = result_payload((0.1, 0.2), marker=1, job_id="old")
        new = result_payload((0.2, 0.3), marker=9, job_id="new")

        merged = experiment.merge_result_payloads(existing, new)

        np.testing.assert_allclose(merged["times"], (0.1, 0.2, 0.3))
        np.testing.assert_allclose(
            merged["observables"]["populations"][0],
            (1.0, 9.0, 9.0),
        )
        self.assertEqual(len(merged["raw_counts"]), 15)
        self.assertEqual(len(merged["circuit_metadata"]), 15)
        replacement_counts = tuple(
            record["counts"]
            for record in merged["raw_counts"]
            if np.isclose(record["time"], 0.2)
        )
        self.assertTrue(
            all(value == {"9": 100} for value in replacement_counts)
        )
        self.assertEqual(merged["job_ids"], ("old", "new"))
        self.assertEqual(len(merged["run_history"]), 2)

    def test_default_checkpoint_is_one_shared_active_run_file(self):
        result_path = Path("result.json")

        self.assertEqual(
            experiment.default_checkpoint_path(result_path),
            Path("result_checkpoint.json"),
        )

    def test_single_cli_time_saves_only_target_time(self):
        options = experiment.parse_arguments(("--times", "0.8"))

        np.testing.assert_allclose(
            experiment.hardware_tools.time_grid_from_options(options),
            (0.8,),
        )

        circuits, metadata, schedule = experiment.build_sample_circuits(
            (0.8,),
            number_of_qubits=2,
            trajectories=2,
            seed_trajectories=29,
            trotter_delta_t=0.2,
        )
        self.assertEqual(len(circuits), 2 * len(experiment.MEASUREMENT_BASES))
        self.assertTrue(all(time_index == 0 for time_index, _, _ in metadata))
        np.testing.assert_allclose(
            schedule.substep_dts,
            (0.2, 0.2, 0.2, 0.2),
        )
        self.assertFalse(schedule.includes_initial_time)

    def test_randomized_circuits_use_one_ancilla_and_no_feed_forward(self):
        times = np.asarray((0.0, 0.1, 0.3))
        circuits, metadata, schedule = experiment.build_sample_circuits(
            times,
            number_of_qubits=4,
            trajectories=3,
            seed_trajectories=7,
            trotter_delta_t=0.1,
        )

        self.assertEqual(len(circuits), 3 * 3 * 5)
        self.assertEqual(circuits[0].qubit_count, 5)
        self.assertEqual(circuits[0].bit_count, 4)
        self.assertEqual(len(schedule.substep_dts), 3)
        self.assertEqual(len(schedule.boundary_choices), 3)
        self.assertTrue(
            all(len(choices) == 3 for choices in schedule.boundary_choices)
        )

        final_z_index = metadata.index((2, 0, "Z"))
        final_z = circuits[final_z_index]
        self.assertFalse(
            any(
                isinstance(operation, ClassicallyControlledXXPlusYY)
                for operation in final_z.operations
            )
        )
        self.assertEqual(
            sum(isinstance(operation, Reset) for operation in final_z.operations),
            3,
        )
        measurements = tuple(
            operation
            for operation in final_z.operations
            if isinstance(operation, Measure)
        )
        self.assertEqual(len(measurements), 4)
        self.assertEqual(tuple(value.bit for value in measurements), (0, 1, 2, 3))

    def test_jump_choices_and_sqrt_two_scaling_match_schedule(self):
        times = np.asarray((0.0, 0.7))
        circuits, metadata, schedule = experiment.build_sample_circuits(
            times,
            number_of_qubits=2,
            trajectories=2,
            seed_trajectories=3,
            trotter_delta_t=0.2,
        )
        np.testing.assert_allclose(
            schedule.substep_dts,
            (0.2, 0.2, 0.2, 0.1),
        )
        final_z = circuits[metadata.index((1, 0, "Z"))]
        jump_operations = tuple(
            operation
            for operation in final_z.operations
            if isinstance(operation, XXPlusYY)
            and operation.label in ("K_L", "K_R")
        )
        self.assertEqual(
            tuple(operation.label[-1] for operation in jump_operations),
            tuple(
                experiment.BOUNDARY_NAMES[value]
                for value in schedule.boundary_choices[0]
            ),
        )
        np.testing.assert_allclose(
            tuple(operation.angle for operation in jump_operations),
            tuple(
                2.0 * np.sqrt(2.0 * experiment.gamma * dt)
                for dt in schedule.substep_dts
            ),
        )

    def test_same_trajectory_prefix_is_reused_for_every_basis(self):
        circuits, metadata, _ = experiment.build_sample_circuits(
            (0.0, 0.2),
            number_of_qubits=3,
            trajectories=2,
            seed_trajectories=31,
        )
        labels_by_basis = []
        for basis in experiment.MEASUREMENT_BASES:
            circuit = circuits[metadata.index((1, 1, basis))]
            labels_by_basis.append(
                tuple(
                    operation.label
                    for operation in circuit.operations
                    if isinstance(operation, XXPlusYY)
                    and operation.label in ("K_L", "K_R")
                )
            )
        self.assertTrue(all(value == labels_by_basis[0] for value in labels_by_basis))

    def test_observables_average_trajectories_and_separate_uncertainty(self):
        metadata = tuple(
            (0, trajectory, basis)
            for trajectory in range(2)
            for basis in experiment.MEASUREMENT_BASES
        )
        results = tuple(
            sample_result(
                {"00": 100} if trajectory == 0 or basis != "Z" else {"11": 100}
            )
            for trajectory in range(2)
            for basis in experiment.MEASUREMENT_BASES
        )
        measured, uncertainties, trajectory_values, _ = (
            experiment.observables_from_results(
                results,
                metadata,
                number_of_qubits=2,
                time_count=1,
                trajectories=2,
            )
        )

        np.testing.assert_allclose(measured[0][:, 0], (0.5, 0.5))
        np.testing.assert_allclose(
            trajectory_values[0][:, 0, :],
            ((0.0, 1.0), (0.0, 1.0)),
        )
        np.testing.assert_allclose(uncertainties.shot[0][:, 0], (0.0, 0.0))
        np.testing.assert_allclose(
            uncertainties.trajectory[0][:, 0],
            (0.5, 0.5),
        )

    def test_total_shots_must_divide_evenly_over_trajectories(self):
        self.assertEqual(experiment.shots_per_trajectory(8192, 16), 512)
        with self.assertRaisesRegex(ValueError, "divisible"):
            experiment.shots_per_trajectory(100, 16)

    def test_checkpoint_round_trip(self):
        times = np.asarray((0.0, 0.1))
        _, metadata, schedule = experiment.build_sample_circuits(
            times,
            number_of_qubits=2,
            trajectories=1,
            seed_trajectories=29,
            trotter_delta_t=0.1,
        )
        options = options_for_checkpoint()
        results = tuple(sample_result({"00": 100}) for _ in range(5))
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


if __name__ == "__main__":
    unittest.main()
