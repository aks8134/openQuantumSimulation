"""Recover real saved measurement arrays without re-executing any circuits."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Model2.Experiment5 import analyze_aer_samples as recovery
from Model2.Experiment5 import aer_sampling, input_data


class AerRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        folder = root / "saved"
        folder.mkdir()
        source = input_data.make_source(input_data.load_spec(), 4, 29)
        (folder / "source.json").write_text(json.dumps(source))
        samples = np.random.default_rng(123).integers(0, 1024, size=(4, 3, 4), dtype=np.uint16)
        chunks = []
        for begin in (0, 2):
            end = begin + 2
            file = folder / f"samples_{begin:05d}_{end:05d}.npz"
            np.savez_compressed(file, samples=samples[begin:end])
            row = {"begin": begin, "end": end, "shots": 4, "seconds": 1.25,
                   "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                   "aer_metadata": {"method": "statevector", "device": "CPU"}}
            file.with_suffix(".json").write_text(json.dumps(row))
            chunks.append(row)
        study = {"configuration": {"paths": 4, "shots": 4, "chunk": 2, "seed": 29,
                                   "threads": 1, "aer_seed": 2026, "full": False,
                                   "spec_sha256": input_data.SPEC_SHA256},
                 "source_sha256": hashlib.sha256((folder / "source.json").read_bytes()).hexdigest(),
                 "versions": {"fixture": "unit test"}, "chunks": chunks, "complete": False}
        path = folder / "study.json"
        path.write_text(json.dumps(study))
        return path, samples

    def test_prefix_recovery_matches_original_estimator_and_preserves_all_inputs(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            path, samples = self.fixture(root)
            hashes = {p: recovery.file_sha256(p) for p in path.parent.iterdir()}
            with patch.object(aer_sampling, "AerSimulator", side_effect=AssertionError("no Aer allowed")):
                result = recovery.analyze(path, [2, 4], root / "analysis")
            self.assertTrue(result["analysis_complete"])
            self.assertFalse(result["original_study_complete_flag"])
            self.assertFalse(result["simulation_rerun"])
            for shots in (2, 4):
                expected = aer_sampling.estimate(samples[:, :, :shots])
                np.testing.assert_array_equal(recovery.bounded_estimate(samples[:, :, :shots], chunk_paths=1), expected)
                profile = result["profiles"][str(shots)]
                np.testing.assert_array_equal(profile["observables"], expected.mean(axis=0))
                np.testing.assert_array_equal(profile["across_path_standard_errors"], expected.std(axis=0, ddof=1)/2)
                self.assertEqual(profile["actual_outcomes_used"], 4*3*shots)
            for p, digest in hashes.items():
                self.assertEqual(recovery.file_sha256(p), digest)
            self.assertTrue((root / "analysis/analysis.json").exists())

    def test_corrupt_samples_are_refused_without_an_analysis_result(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            path, _ = self.fixture(root)
            with next(path.parent.glob("*.npz")).open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Sample hash mismatch"):
                recovery.analyze(path, [4], root / "analysis")
            self.assertFalse((root / "analysis/analysis.json").exists())

    def test_incomplete_checkpoints_and_excess_shots_are_refused(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            path, _ = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "no larger than"):
                recovery.analyze(path, [8], root / "analysis")
            study = json.loads(path.read_text())
            study["chunks"].pop()
            path.write_text(json.dumps(study))
            with self.assertRaisesRegex(ValueError, "Incomplete checkpoints"):
                recovery.analyze(path, [4], root / "analysis")

    def test_noncontiguous_paths_are_refused(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            path, _ = self.fixture(root)
            study = json.loads(path.read_text())
            study["chunks"][1]["begin"] = 1
            path.write_text(json.dumps(study))
            with self.assertRaisesRegex(ValueError, "coverage"):
                recovery.analyze(path, [4], root / "analysis")

    def test_invalid_bit_range_is_refused_even_with_matching_archive_hash(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            path, samples = self.fixture(root)
            samples[0, 0, 0] = 1024
            file = path.parent / "samples_00000_00002.npz"
            np.savez_compressed(file, samples=samples[:2])
            study = json.loads(path.read_text())
            row = study["chunks"][0]
            row["sha256"] = recovery.file_sha256(file)
            file.with_suffix(".json").write_text(json.dumps(row))
            path.write_text(json.dumps(study))
            with self.assertRaisesRegex(ValueError, "Invalid packed samples"):
                recovery.analyze(path, [4], root / "analysis")


if __name__ == "__main__":
    unittest.main()
