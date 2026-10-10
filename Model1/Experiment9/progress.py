"""Terminal-only progress: exact preparation counts, elapsed provider waits.

No job polling, batch splitting, circuit mutations or fabricated completion
percentages. The refresh thread owns only presentation and stops on all exits.
"""

from __future__ import annotations

from threading import Event, Thread


def tqdm_factory():
    try:
        from tqdm import tqdm
    except ImportError as error:
        raise RuntimeError(
            "Experiment 9 progress requires tqdm. Install with "
            "uv pip install --python .venv/bin/python 'tqdm>=4.67,<5' "
            "(Windows: .venv\\Scripts\\python.exe), or use --no-progress."
        ) from error
    return tqdm


class BatchProgress:
    """Optional UI observer, kept outside the immutable simulation algorithms."""

    def __init__(self, factory, *, label, backend, variant_count):
        self.factory = factory
        self.label = label
        self.backend = backend
        self.variant_count = variant_count
        self.bar = None
        self.stop = None
        self.thread = None

    def __enter__(self):
        return self

    def __exit__(self, *_error):
        self.close()

    def close(self):
        if self.stop is not None:
            self.stop.set()
            self.thread.join(timeout=2)
            self.stop = self.thread = None
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def _start(self, description, *, total=None):
        self.close()
        if self.factory is None:
            return
        settings = dict(desc=f"{self.label}: {description}", total=total,
                        unit="circuit", dynamic_ncols=True, mininterval=0.2)
        if total is None:
            settings["bar_format"] = "{desc} [elapsed {elapsed}]"
        self.bar = self.factory(**settings)
        self.stop = Event()
        bar, stop = self.bar, self.stop

        def refresh():
            while not stop.wait(1.0):
                bar.refresh()

        self.thread = Thread(target=refresh, name="experiment9-progress", daemon=True)
        self.thread.start()

    def logical_start(self, total):
        self._start("Preparing compiler inputs", total=total)

    def advance(self):
        if self.bar is not None:
            self.bar.update(1)

    def stage(self, stage, count):
        if stage == "resolve_start":
            self._start(f"Resolving {self.backend}")
        elif stage == "compile_start":
            self._start(f"Transpiling {count} base circuits (batch completion pending)")
        elif stage == "transform_start":
            self._start("Preparing native variants", total=self.variant_count)
        elif stage == "sample_start":
            operation = "Simulating" if self.backend.lower() == "aer" else "Waiting for IBM job"
            self._start(f"{operation}: {count} circuits (batch completion pending)")
        elif stage == "sample_done":
            if self.bar is not None:
                self.bar.set_description_str(f"{self.label}: Sampling complete, {count} results")
            self.close()
