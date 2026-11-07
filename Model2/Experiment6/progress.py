"""Visible stages for local processing and blocking result requests."""
from contextlib import contextmanager
import threading
import time


@contextmanager
def progress(message):
    start = time.monotonic()
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(20):
            print(f"Still waiting: {message} ({time.monotonic()-start:.0f}s elapsed)", flush=True)

    print(f"{message}...", flush=True)
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        stopped.set()
        worker.join()
        state = "Finished" if succeeded else "Stopped"
        print(f"{state}: {message} ({time.monotonic()-start:.1f}s)", flush=True)
