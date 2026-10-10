"""Progress presentation tests: no simulation, network or provider jobs."""

import unittest

from Model1.Experiment9.progress import BatchProgress


class FakeBar:
    def __init__(self, **settings):
        self.settings = settings
        self.n = 0
        self.closed = False
        self.description = settings["desc"]

    def update(self, count):
        self.n += count

    def refresh(self):
        pass

    def set_description_str(self, description):
        self.description = description

    def close(self):
        self.closed = True


class ProgressTests(unittest.TestCase):
    def factory(self, **settings):
        bar = FakeBar(**settings)
        self.bars.append(bar)
        return bar

    def setUp(self):
        self.bars = []

    def test_exact_preparation_counts_and_indeterminate_batch_waits(self):
        with BatchProgress(self.factory, label="Batch 1/1", backend="aer", variant_count=6) as progress:
            progress.logical_start(2)
            progress.advance()
            progress.advance()
            progress.stage("resolve_start", 2)
            progress.stage("compile_start", 2)
            progress.stage("transform_start", 2)
            for _ in range(6):
                progress.advance()
            progress.stage("sample_start", 6)
            progress.stage("sample_done", 6)
        self.assertEqual([bar.n for bar in self.bars], [2, 0, 0, 6, 0])
        self.assertEqual([bar.settings["total"] for bar in self.bars], [2, None, None, 6, None])
        self.assertIn("elapsed", self.bars[-1].settings["bar_format"])
        self.assertIn("Sampling complete, 6 results", self.bars[-1].description)
        self.assertTrue(all(bar.closed for bar in self.bars))
        self.assertIsNone(progress.thread)
        self.assertIsNone(progress.bar)

    def test_exception_closes_the_bar_and_refresh_thread(self):
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with BatchProgress(self.factory, label="Batch 1/1", backend="aer", variant_count=1) as progress:
                progress.stage("compile_start", 1)
                raise RuntimeError("failure")
        self.assertTrue(self.bars[0].closed)
        self.assertIsNone(progress.thread)
        self.assertIsNone(progress.stop)

    def test_disabled_progress_creates_no_thread_or_bar(self):
        with BatchProgress(None, label="Batch 1/1", backend="aer", variant_count=1) as progress:
            progress.logical_start(1)
            progress.advance()
            progress.stage("resolve_start", 1)
            progress.stage("sample_start", 1)
            progress.stage("sample_done", 1)
            self.assertIsNone(progress.thread)
            self.assertIsNone(progress.bar)

    def test_ibm_wait_label_does_not_imply_completed_circuits(self):
        with BatchProgress(self.factory, label="Batch 2/3", backend="ibm_kingston", variant_count=6) as progress:
            progress.stage("sample_start", 6)
            self.assertIn("Waiting for IBM job", self.bars[0].description)
            self.assertIn("batch completion pending", self.bars[0].description)
            self.assertIsNone(self.bars[0].settings["total"])
            self.assertEqual(self.bars[0].n, 0)


if __name__ == "__main__":
    unittest.main()
