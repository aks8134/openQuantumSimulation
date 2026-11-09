"""Run the existing Aer sampler with visible progress and a remaining-time estimate.

Accepts every aer_sampling.py option plus --progress-interval (seconds).
The sampler and its checkpoint identity are unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Progress:
    def __init__(self, interval=20., clock=None):
        self.interval = interval
        self.clock = clock or time.monotonic
        self.started = self.last_checkpoint = self.clock()
        self.total = self.shots = self.completed = self.resumed = self.new_paths = 0
        self.new_work_seconds = 0.
        self.seen = set()
        self.fresh = set()
        self.folder = None
        self.phase = "preparing input / first batch"
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.worker = None

    def message(self):
        with self.lock:
            elapsed = duration(self.clock() - self.started)
            if not self.total:
                return f"Aer progress: {self.phase} | elapsed {elapsed} | ETA pending first new checkpoint"
            remaining = self.total - self.completed
            if not remaining:
                eta = "00:00:00"
            elif self.new_paths and self.new_work_seconds > 0:
                eta = duration(remaining * self.new_work_seconds / self.new_paths)
            else:
                eta = "pending first newly simulated batch"
            outcomes = self.completed * 3 * self.shots
            total_outcomes = self.total * 3 * self.shots
            return (f"Aer progress: {self.completed:,}/{self.total:,} trajectories "
                    f"({100*self.completed/self.total:.2f}%) | "
                    f"saved outcomes {outcomes:,}/{total_outcomes:,} | "
                    f"elapsed {elapsed} | ETA {eta} | resumed {self.resumed:,} | {self.phase}")

    def saved(self, path, value):
        """Observe successful writes only; never change a saved object."""
        path = Path(path)
        with self.lock:
            if re.fullmatch(r"samples_\d+_\d+\.json", path.name):
                self.fresh.add((value["begin"], value["end"]))
                return
            if path.name == "source.json":
                self.last_checkpoint = self.clock()
                self.phase = "building / simulating first batch"
                return
            if path.name != "study.json" or "configuration" not in value:
                return
            now = self.clock()
            self.total = value["configuration"]["paths"]
            self.shots = value["configuration"]["shots"]
            for chunk in value["chunks"]:
                key = chunk["begin"], chunk["end"]
                if key in self.seen:
                    continue
                count = chunk["end"] - chunk["begin"]
                if key in self.fresh:
                    self.new_paths += count
                    self.new_work_seconds += max(0., now - self.last_checkpoint)
                    self.fresh.remove(key)
                else:
                    self.resumed += count
                self.seen.add(key)
                self.completed = chunk["end"]
                self.last_checkpoint = now
            if self.folder != path.parent:
                self.folder = path.parent
                print(f"Checkpoint folder: {self.folder}", flush=True)
            self.phase = ("complete" if value.get("complete") else
                          "finalizing analysis" if self.completed == self.total else "next batch running")
            print(self.message(), flush=True)

    def __enter__(self):
        print("Starting/resuming Aer; progress follows each checkpoint and every "
              f"{self.interval:g} seconds. ETA uses newly simulated work in this invocation.", flush=True)

        def heartbeat():
            while not self.stopped.wait(self.interval):
                print(self.message(), flush=True)

        self.worker = threading.Thread(target=heartbeat, name="Aer progress heartbeat", daemon=True)
        self.worker.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stopped.set()
        self.worker.join()
        state = "Finished" if exc_type is None else "Stopped"
        print(f"{state}: Aer invocation | elapsed {duration(self.clock()-self.started)}", flush=True)


def main(arguments=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--progress-interval", type=float, default=20.)
    options, sampler_arguments = parser.parse_known_args(arguments)
    if not options.progress_interval > 0 or not options.progress_interval < float("inf"):
        parser.error("--progress-interval must be a finite positive number")
    if any(flag in sampler_arguments for flag in ("-h", "--help")):
        print(__doc__, flush=True)
        from Model2.Experiment5 import aer_sampling
        return aer_sampling.main(sampler_arguments)
    with Progress(options.progress_interval) as progress:
        from Model2.Experiment5 import aer_sampling
        from Model2.Experiment5.analyze_aer_samples import bounded_estimate
        original_save = aer_sampling.atomic_json
        original_estimate = aer_sampling.estimate

        def save_with_progress(path, value):
            original_save(path, value)
            progress.saved(path, value)

        aer_sampling.atomic_json = save_with_progress
        # Only final postprocessing changes; sampled circuits, seeds and files keep
        # their original identity. Avoid the full paths × bases × shots × bits tensor.
        aer_sampling.estimate = bounded_estimate
        try:
            return aer_sampling.main(sampler_arguments)
        finally:
            aer_sampling.atomic_json = original_save
            aer_sampling.estimate = original_estimate


if __name__ == "__main__":
    main()
