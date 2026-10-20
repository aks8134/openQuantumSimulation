"""Preparation counts and honest elapsed-time waits; no fabricated job percentage."""
from threading import Event, Thread


class BatchProgress:
    def __init__(self, enabled, label, backend, variants):
        self.enabled, self.label, self.backend, self.variants = enabled, label, backend, variants
        self.bar = self.thread = self.stop = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        if self.stop is not None:
            self.stop.set()
            self.thread.join(timeout=2)
            self.stop = self.thread = None
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def start(self, description, total=None):
        self.close()
        if not self.enabled:
            return
        from tqdm import tqdm
        settings = {"desc": f"{self.label}: {description}", "total": total,
                    "unit": "circuit", "dynamic_ncols": True}
        if total is None:
            settings["bar_format"] = "{desc} [elapsed {elapsed}]"
        self.bar = tqdm(**settings)
        self.stop = Event()
        bar, stop = self.bar, self.stop
        def refresh():
            while not stop.wait(1):
                bar.refresh()
        self.thread = Thread(target=refresh, daemon=True, name="experiment11-progress")
        self.thread.start()

    def advance(self):
        if self.bar is not None:
            self.bar.update(1)

    def stage(self, stage, count):
        if stage == "resolve_start":
            self.start(f"Resolving {self.backend}")
        elif stage == "compile_start":
            self.start(f"Transpiling {count} base circuits (batch completion pending)")
        elif stage == "transform_start":
            self.start("Preparing native variants", self.variants)
        elif stage == "sample_start":
            local = self.backend.lower() in ("aer", "fake_kingston", "fake_miami")
            self.start(f"{'Simulating' if local else 'Waiting for IBM job'} {count} circuits (batch completion pending)")
        elif stage == "sample_done":
            self.close()
