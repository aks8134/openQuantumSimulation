"""Population/control separation and statistical response-model diagnostics."""
import copy
import unittest

import numpy as np

from Model3.Experiment5 import analyze_populations as analysis


def manifest():
    records, observations = [], []
    for k, (name, counts) in enumerate((("all_zero", {"00": 4}), (0, {"01": 2}),
                                      ("all_one", {"11": 4}), (1, {"01": 2}),
                                      ("initial_mixture", {"00": 2, "11": 2}))):
        row = {"pub_index": k, "basis": "z", "shots": sum(counts.values()),
               "final_logical_to_physical": [3, 5, 1, 7]}
        if isinstance(name, str):
            row.update(kind="control", control=name,
                       expected_populations={"all_zero": [0., 0.], "all_one": [1., 1.], "initial_mixture": [.5, .5]}[name])
        else:
            row["trajectory"] = name
        records.append(row); observations.append({"outcome_counts": counts})
    return {"experiment": "Model3/Experiment5/hardware", "status": "collected",
            "configuration": {"trajectory_indices": [0, 1], "shots": 2, "total_shots": 16,
                              "total_circuits": 5, "population_controls": True},
            "circuits": records, "batches": [{"status": "collected", "start": 0, "stop": 5,
                "job_id": "fake_job", "observations": observations}]}


class PopulationAnalysisTests(unittest.TestCase):
    def test_control_separation_shot_counts_and_little_endian_sites(self):
        paths, controls, jobs, shots = analysis.reconstruct(manifest(), 2)
        self.assertEqual(list(paths), [0, 1]); self.assertEqual(shots, 16)
        self.assertEqual(jobs, ["fake_job"])
        np.testing.assert_array_equal(np.mean([row["populations"] for row in paths.values()], axis=0), [1., 0.])
        np.testing.assert_array_equal(controls["initial_mixture"]["populations"], [.5, .5])
        np.testing.assert_allclose(controls["initial_mixture"]["covariance_of_mean"], np.full((2, 2), 1/12))

    def test_incomplete_duplicate_mismapped_or_wrong_shot_data_rejected(self):
        changes = [lambda d: d["circuits"][0].update(shots=8),
                   lambda d: d["circuits"][3].update(trajectory=0),
                   lambda d: d["circuits"][2].update(final_logical_to_physical=[5, 3, 1, 7]),
                   lambda d: d["batches"][0].update(stop=4),
                   lambda d: d["configuration"].update(total_shots=20)]
        for change in changes:
            data = manifest(); change(data)
            with self.subTest(change=change), self.assertRaises(ValueError):
                analysis.reconstruct(data, 2)

    def test_affine_response_restores_known_populations_and_predicts_mixture(self):
        low, high = np.array([.02, .04]), np.array([.92, .84])
        target = np.array([.3, .7]); raw = low+(high-low)*target
        covariance = np.array([[2e-5, 3e-6], [3e-6, 4e-5]])
        controls = {name: {"populations": value, "covariance_of_mean": covariance.copy()}
                    for name, value in (("all_zero", low), ("all_one", high), ("initial_mixture", (low+high)/2))}
        controls["initial_mixture"]["expected_populations"] = np.array([.5, .5])
        result = analysis.control_diagnostics(controls, raw, covariance)
        np.testing.assert_allclose(result["exploratory_affine_adjusted_populations"], target)
        np.testing.assert_allclose(result["initial_mixture_residual"], 0., atol=1e-15)
        np.testing.assert_allclose(result["initial_mixture_residual_covariance"], 1.5*covariance)
        changed = copy.deepcopy(controls)
        changed["initial_mixture"]["populations"][0] -= .1
        result = analysis.control_diagnostics(changed, raw, covariance)
        self.assertLess(result["initial_mixture_residual_nominal_z"][0], -10)

    def test_adjustment_covariance_matches_numerical_jacobian(self):
        controls = {"all_zero": {"populations": np.array([.03]), "covariance_of_mean": np.array([[.0001]])},
                    "all_one": {"populations": np.array([.93]), "covariance_of_mean": np.array([[.0002]])},
                    "initial_mixture": {"populations": np.array([.48]), "expected_populations": np.array([.5]),
                                        "covariance_of_mean": np.array([[.0003]])}}
        result = analysis.control_diagnostics(controls, np.array([.4]), np.array([[.0004]]))
        point = np.array([.4, .03, .93])
        def transform(v):
            return (v[0]-v[1])/(v[2]-v[1])
        gradient = []
        for j in range(3):
            step = np.eye(3)[j]*1e-6
            gradient.append((transform(point+step)-transform(point-step))/2e-6)
        expected = np.array(gradient)**2@np.array([.0004, .0001, .0002])
        self.assertAlmostEqual(result["exploratory_adjusted_covariance_of_mean"][0, 0], expected, places=12)


if __name__ == "__main__":
    unittest.main()
