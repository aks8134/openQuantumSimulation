"""Resumable IBM Runtime Batch lifecycle; never collect results or cancel jobs."""
from __future__ import annotations

from datetime import datetime, timezone
import json

from Model2.Experiment6.progress import progress


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def identifier(batch):
    value = batch.session_id
    value = value() if callable(value) else value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("IBM Runtime Batch returned no usable ID; no job was submitted")
    return value


class BatchSubmission:
    def __init__(self, folder, manifest, options, *, backend, service, checkpoint,
                 service_factory, factory=None):
        if factory is None:
            from qiskit_ibm_runtime import Batch
            factory = Batch
        self.factory = factory
        self.folder, self.manifest, self.options = folder, manifest, options
        self.backend, self.service = backend, service
        self.checkpoint, self.service_factory = checkpoint, service_factory
        self.batch = None

    def save(self):
        self.checkpoint(self.folder, self.manifest)

    def restore(self):
        if self.batch is None:
            record = self.manifest["runtime_batch"]
            self.service = self.service or self.service_factory(self.options)
            with progress(f"Reopening recorded Runtime Batch {record['id']}"):
                self.batch = self.factory.from_id(record["id"], service=self.service, use_fractional_gates=False)
            if identifier(self.batch) != record["id"]:
                raise ValueError("provider returned a different Runtime Batch")
        return self.batch

    def details(self):
        value = self.batch.details()
        if not isinstance(value, dict):
            raise ValueError("Runtime Batch details unavailable; recorded IDs retained, retry submission later")
        record = self.manifest["runtime_batch"]
        if (value.get("id") != record["id"] or value.get("backend_name") != self.manifest["configuration"]["backend"]
                or value.get("mode") != "batch"):
            raise ValueError("Runtime Batch backend/ID/execution mode differs from the saved plan")
        record["last_details"] = json.loads(json.dumps(value, default=str))
        record["checked_at_utc"] = utcnow()
        self.save()
        return value

    def check_existing(self):
        """Validate an existing Batch before loading the next large circuit job."""
        record = self.manifest.get("runtime_batch")
        replace = getattr(self.options, "new_batch_for_remaining", False)
        if record is None:
            if replace:
                raise ValueError("there is no recorded Runtime Batch to replace")
            return
        self.restore()
        details = self.details()
        if details.get("accepting_jobs") is True:
            if replace:
                raise ValueError("saved Runtime Batch still accepts jobs; omit --new-batch-for-remaining")
            maximum = getattr(self.options, "batch_max_time", None)
            if maximum is not None and maximum != record["requested_max_time_seconds"]:
                raise ValueError("an existing Runtime Batch TTL cannot be changed; omit --batch-max-time when resuming")
            return
        if details.get("accepting_jobs") is not False:
            raise ValueError("cannot establish whether the saved Runtime Batch accepts jobs; retry later")
        if not replace:
            raise ValueError("saved Runtime Batch no longer accepts jobs; known IDs are retained. "
                             "Use --new-batch-for-remaining explicitly to queue only missing jobs in another Batch")
        self.manifest.setdefault("runtime_batch_history", []).append(record)
        self.manifest.pop("runtime_batch")
        self.batch = None
        self.save()

    def mode(self):
        """Lazily create a Batch only after the first job's circuits are ready."""
        if self.batch is not None:
            return self.batch
        if self.manifest.get("runtime_batch"):
            self.check_existing()
            if self.batch is not None:
                return self.batch
        maximum = getattr(self.options, "batch_max_time", None)
        if maximum is None:
            maximum = self.manifest["configuration"].get("batch_max_time_seconds")
        with progress("Opening one IBM Runtime Batch; no result collection"):
            self.batch = self.factory(backend=self.backend, max_time=maximum)
            batch_id = identifier(self.batch)
            # Durable before the first Sampler.run. If creation itself is interrupted,
            # an unrecorded empty Batch has no jobs; rerunning cannot duplicate a job.
            self.manifest["runtime_batch"] = dict(id=batch_id, backend=self.manifest["configuration"]["backend"],
                created_at_utc=utcnow(), requested_max_time_seconds=maximum, closed_to_submissions=False)
            self.save()
        details = self.details()
        if details.get("accepting_jobs") is not True:
            raise ValueError("new Runtime Batch is not accepting jobs; its ID was retained")
        print(f"Runtime Batch: {batch_id}; provider max_time={details.get('max_time')} seconds", flush=True)
        return self.batch

    @property
    def batch_id(self):
        return self.manifest["runtime_batch"]["id"]

    def check_job(self, job, record):
        """Called only after the job ID is durable, so an association error loses no ID."""
        getter = getattr(job, "session_id", None)
        actual = getter() if callable(getter) else getter
        if isinstance(actual, str):
            record["provider_runtime_batch_id"] = actual
            self.save()
            if actual != record["runtime_batch_id"]:
                raise ValueError("submitted job reports another Runtime Batch; job ID retained, further submissions stopped")

    def close(self):
        if not all(row.get("job_id") for row in self.manifest["batches"]):
            raise ValueError("leave the Runtime Batch open while any job ID is missing")
        record = self.manifest.get("runtime_batch")
        if record is None:
            raise ValueError("batch-mode jobs have no recorded Runtime Batch")
        if record.get("closed_to_submissions"):
            return
        self.restore()
        details = self.details()
        if details.get("accepting_jobs") is False:
            record.update(closed_to_submissions=True, close_confirmed_at_utc=utcnow())
            self.save()
            return
        record["close_requested_at_utc"] = utcnow()
        self.save()
        with progress(f"Closing Runtime Batch {record['id']} to new submissions; queued jobs continue"):
            self.batch.close()
        record.update(closed_to_submissions=True, close_confirmed_at_utc=utcnow())
        self.save()
