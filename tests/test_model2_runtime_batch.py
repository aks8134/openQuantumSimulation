"""Runtime Batch orchestration with mocked jobs; no account or QPU access."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from qiskit import QuantumCircuit

from Model2.Experiment8 import run_hardware as run
from Model2.Experiment8 import estimate_jobs as estimate


class FakeBatchFactory:
    def __init__(self):
        self.created = []
        self.reopened = []
        self.registry = {}
        self.before_create = None

    def __call__(self, *, backend, max_time=None):
        if self.before_create:
            self.before_create()
        identifier = f"runtime_batch_{len(self.created)}"
        batch = SimpleNamespace(session_id=identifier, accepting=True, close_calls=0)
        batch.details = lambda: dict(id=identifier, backend_name=backend.name, mode="batch",
                                     accepting_jobs=batch.accepting, state="open", max_time=max_time or 28800)
        def close():
            batch.close_calls += 1
            batch.accepting = False
        batch.close = close
        self.created.append(batch)
        self.registry[identifier] = batch
        return batch

    def from_id(self, identifier, **kwargs):
        self.reopened.append(identifier)
        return self.registry[identifier]


class Harness:
    """One mock PUB per job isolates queueing from the separately tested physics."""
    def __init__(self, folder, count=24):
        self.folder = folder
        self.backend = SimpleNamespace(name="test_backend", is_mock=False, target=object())
        self.service = Mock()
        self.factory = FakeBatchFactory()
        self.m = dict(experiment=run.EXPERIMENT, status="compiled", files={},
            configuration=dict(backend="test_backend", execution_mode="batch", batch_max_time_seconds=7200,
                physical_path=list(range(12)), timing_model={}, sampler_options=run.sampler_options(False)),
            circuits=[dict(kind="trajectory", shots=32, isa_file=f"shard_{i}", isa_index=0) for i in range(count)],
            batches=[dict(start=i, stop=i+1, compiled=True, isa_file=f"index_{i}", isa_sha256="mock",
                resources=dict(maximum_instructions=123, shots=32), status="not_submitted", job_id=None,
                tag=f"test_{i}") for i in range(count)])
        self.jobs, self.modes, self.events = [], [], []
        self.fail_before_sampler = None
        self.fail_during_run = None
        self.bad_association = False
        self.options = run.parse_arguments(["submit", "--manifest", str(folder/"manifest.json"), "--hardware"])
        self.checkpoints = []
        self.original_checkpoint = run.checkpoint

    def checkpoint(self, folder, manifest):
        self.original_checkpoint(folder, manifest)
        self.checkpoints.append(copy.deepcopy(manifest))

    def sampler(self, *, mode, options):
        index = len(self.jobs)
        if self.fail_before_sampler == index:
            raise RuntimeError("interrupted before next upload")
        self.modes.append(mode)
        sampler = Mock()
        def submit(pubs, **kwargs):
            assert "runtime_batch" in self.m
            assert any(s.get("runtime_batch", {}).get("id") == mode.session_id for s in self.checkpoints)
            if self.fail_during_run == index:
                raise RuntimeError("submission outcome unknown")
            job = Mock()
            job.job_id.return_value = f"job_{index}"
            job.session_id.return_value = "wrong_batch" if self.bad_association else mode.session_id
            job.result.side_effect = AssertionError("must queue all jobs without waiting for results")
            self.jobs.append(job)
            self.events.append(f"submitted_{index}")
            return job
        sampler.run.side_effect = submit
        return sampler

    def execute(self):
        program = QuantumCircuit(1)
        with patch.object(run, "read_saved", return_value=(self.folder, self.m, {})), \
             patch.object(run, "load_shard", return_value=[program]), \
             patch.object(run, "validate_isa_circuits"), \
             patch.object(run.timing, "timing_model", return_value={}), \
             patch.object(run.raw, "backend_readiness"), \
             patch.object(run.base, "resource_check", return_value=dict(maximum_instructions=123, executions=32)), \
             patch.object(run, "checkpoint", side_effect=self.checkpoint), \
             patch.object(run.raw.runtime, "make_service", side_effect=AssertionError("no IBM credentials")):
            return run.submit(self.options, backend=self.backend, service=self.service,
                              sampler_factory=self.sampler, batch_factory=self.factory)


def test_all_24_jobs_share_one_batch_and_never_wait_for_results(tmp_path):
    h = Harness(tmp_path)
    h.execute()
    assert len(h.factory.created) == 1
    batch = h.factory.created[0]
    assert len(h.jobs) == 24 and all(mode is batch for mode in h.modes)
    assert all(b["runtime_batch_id"] == batch.session_id for b in h.m["batches"])
    assert batch.close_calls == 1
    assert h.m["runtime_batch"]["closed_to_submissions"]
    for job in h.jobs:
        job.result.assert_not_called()
        job.cancel.assert_not_called()
    before = h.m["plan_sha256"]
    h.execute()
    assert len(h.jobs) == 24 and len(h.factory.created) == 1 and batch.close_calls == 1
    assert h.m["plan_sha256"] == before


def test_interruption_resumes_same_batch_skipping_known_job_ids(tmp_path):
    h = Harness(tmp_path)
    h.fail_before_sampler = 3
    with pytest.raises(RuntimeError, match="before next upload"):
        h.execute()
    assert len(h.jobs) == 3 and h.factory.created[0].close_calls == 0
    saved = [b["job_id"] for b in h.m["batches"][:3]]
    h.fail_before_sampler = None
    h.execute()
    assert len(h.factory.created) == 1 and h.factory.reopened == ["runtime_batch_0"]
    assert len(h.jobs) == 24 and [b["job_id"] for b in h.m["batches"][:3]] == saved
    assert h.factory.created[0].close_calls == 1


def test_ambiguous_submission_is_not_retried_or_rebatched(tmp_path):
    h = Harness(tmp_path)
    h.fail_during_run = 2
    with pytest.raises(RuntimeError, match="outcome unknown"):
        h.execute()
    assert h.m["batches"][2]["status"] == "submitting" and h.m["batches"][2]["job_id"] is None
    with pytest.raises(ValueError, match="reconcile"):
        h.execute()
    assert len(h.factory.created) == 1 and len(h.jobs) == 2


def test_closed_batch_requires_explicit_recovery_for_missing_jobs(tmp_path):
    h = Harness(tmp_path)
    h.fail_before_sampler = 2
    with pytest.raises(RuntimeError):
        h.execute()
    h.factory.created[0].accepting = False
    h.fail_before_sampler = None
    with pytest.raises(ValueError, match="no longer accepts"):
        h.execute()
    assert len(h.factory.created) == 1
    h.options.new_batch_for_remaining = True
    h.execute()
    assert len(h.factory.created) == 2 and len(h.jobs) == 24
    assert len(h.m["runtime_batch_history"]) == 1
    assert {b["runtime_batch_id"] for b in h.m["batches"][:2]} == {"runtime_batch_0"}
    assert {b["runtime_batch_id"] for b in h.m["batches"][2:]} == {"runtime_batch_1"}


def test_close_failure_is_retried_without_another_job_submission(tmp_path):
    h = Harness(tmp_path, count=2)
    old_create = h.factory.__class__.__call__
    def create(factory, **kwargs):
        batch = old_create(factory, **kwargs)
        batch.close = Mock(side_effect=RuntimeError("close unavailable"))
        return batch
    with patch.object(FakeBatchFactory, "__call__", create):
        with pytest.raises(RuntimeError, match="close unavailable"):
            h.execute()
    assert len(h.jobs) == 2
    h.factory.created[0].close = Mock()
    h.execute()
    h.factory.created[0].close.assert_called_once()
    assert len(h.jobs) == 2


def test_wrong_provider_batch_id_is_durable_and_blocks_further_jobs(tmp_path):
    h = Harness(tmp_path, count=2)
    h.bad_association = True
    with pytest.raises(ValueError, match="another Runtime Batch"):
        h.execute()
    assert h.m["batches"][0]["job_id"] == "job_0"
    assert h.m["batches"][0]["provider_runtime_batch_id"] == "wrong_batch"
    with pytest.raises(ValueError, match="reconcile"):
        h.execute()
    assert len(h.jobs) == 1


def test_batch_creation_failure_does_not_mark_a_job_submitting(tmp_path):
    h = Harness(tmp_path, count=2)
    h.factory.before_create = Mock(side_effect=RuntimeError("Batch unavailable"))
    with pytest.raises(RuntimeError, match="Batch unavailable"):
        h.execute()
    assert not h.jobs and all(b["status"] == "not_submitted" for b in h.m["batches"])


def test_full_16834_plan_matches_reviewed_24_job_accounting_and_default_legacy_limit(tmp_path):
    options = run.parse_arguments(["plan", "--backend", "test", "--trajectory-count", "16834", "--shots", "32",
        "--batch-size", "9141", "--execution-mode", "batch", "--batch-max-time", "2h",
        "--output-directory", str(tmp_path)])
    source = estimate.compilation_source(16834)
    source["experiment"] = "Model2/Experiment5"
    reference = dict(configuration=dict(backend="test", physical_path=list(range(12))), scope="test reference")
    with patch.object(run, "read_reference", return_value=reference), \
         patch.object(run.input_data, "make_source", return_value=source), \
         patch.object(run.raw.runtime, "make_service", side_effect=AssertionError("offline plan")):
        folder, manifest, _ = run.new_plan(options)
    c = manifest["configuration"]
    assert (c["total_jobs"], c["total_circuits"], c["total_shots"]) == (24, 219202, 2449664)
    assert c["execution_mode"] == "batch" and c["batch_max_time_seconds"] == 7200
    run.verify_manifest(manifest, source)
    with pytest.raises(ValueError, match="frozen experiment"):
        run.base.verify_manifest(manifest, source, expected_experiment=run.EXPERIMENT,
                                 expected_sampler_options=run.sampler_options(False))


def test_invalid_batch_flags_and_cohort_limit():
    with pytest.raises(SystemExit):
        run.parse_arguments(["plan", "--trajectory-count", "16835"])
    with pytest.raises(SystemExit):
        run.parse_arguments(["plan", "--batch-max-time", "1h"])
    with pytest.raises(SystemExit):
        run.parse_arguments(["submit", "--manifest", "test", "--dry-run", "--new-batch-for-remaining"])


def test_provider_already_closed_batch_does_not_need_another_close_request(tmp_path):
    h = Harness(tmp_path, count=2)
    h.execute()
    h.m["runtime_batch"]["closed_to_submissions"] = False
    h.execute()
    assert h.factory.created[0].close_calls == 1 and len(h.jobs) == 2
    assert h.m["runtime_batch"]["closed_to_submissions"]


def test_submission_ttl_override_is_recorded_and_cannot_change_existing_batch(tmp_path):
    h = Harness(tmp_path, count=2)
    h.options.batch_max_time = 3600
    h.fail_before_sampler = 1
    with pytest.raises(RuntimeError):
        h.execute()
    assert h.m["runtime_batch"]["requested_max_time_seconds"] == 3600
    assert h.m["configuration"]["batch_max_time_seconds"] == 7200
    h.options.batch_max_time = 1800
    with pytest.raises(ValueError, match="TTL cannot be changed"):
        h.execute()
    assert len(h.jobs) == 1


def test_batch_dry_run_never_opens_a_runtime_batch(tmp_path):
    h = Harness(tmp_path, count=2)
    h.options = run.parse_arguments(["submit", "--manifest", str(tmp_path/"manifest.json"), "--dry-run"])
    # The primitive is still a job-mode compiler dry-run; no Batch/QPU is created.
    def sampler(*, mode, options):
        assert mode is h.backend
        primitive = Mock()
        def submit(pubs, **kwargs):
            assert kwargs == {"dry_run": True}
            job = Mock(); job.job_id.return_value=f"dry_{len(h.jobs)}"
            h.jobs.append(job)
            return job
        primitive.run.side_effect = submit
        return primitive
    h.sampler = sampler
    h.factory.before_create = Mock(side_effect=AssertionError("no Runtime Batch for mock counts"))
    h.execute()
    assert len(h.jobs) == 2 and not h.factory.created
