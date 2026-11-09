"""A visible CPU count must not conceal a smaller inherited compute quota."""
from pathlib import Path

import pytest

from Model2.Experiment8 import benchmark_preparation as benchmark


@pytest.mark.parametrize("membership,limits,capacity", [
    ("7:cpu,cpuacct:/user.slice/user-123.slice\n", {
        "/sys/fs/cgroup/cpu/user.slice/user-123.slice/cpu.cfs_quota_us":"50000",
        "/sys/fs/cgroup/cpu/user.slice/user-123.slice/cpu.cfs_period_us":"100000"}, .5),
    ("0::/allocation/job\n", {"/sys/fs/cgroup/allocation/cpu.max":"200000 100000"}, 2.),
    ("0::/allocation/job\n", {"/sys/fs/cgroup/cpu.max":"max 100000"}, 4),
])
def test_cpu_capacity_accounts_for_ancestor_quota(monkeypatch, membership, limits, capacity):
    files = {"/proc/self/cgroup":membership, **limits}
    def read(path, *args, **kwargs):
        if str(path) not in files:
            raise FileNotFoundError(str(path))
        return files[str(path)]
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(benchmark.os, "cpu_count", lambda:8)
    monkeypatch.setattr(benchmark.os, "sched_getaffinity", lambda _:set(range(4)), raising=False)
    allocation = benchmark.cpu_allocation()
    assert allocation["logical_cpus"] == 8
    assert allocation["effective_cpu_capacity"] == capacity


def test_insufficient_cpu_allocation_stops_before_compilation(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "cpu_allocation", lambda:dict(effective_cpu_capacity=.5))
    destination = tmp_path/"benchmark"
    with pytest.raises(SystemExit):
        benchmark.main(["--trajectories", "4", "--workers", "1", "2", "4", "--require-cpus", "4",
                        "--output-directory", str(destination)])
    assert not destination.exists()
