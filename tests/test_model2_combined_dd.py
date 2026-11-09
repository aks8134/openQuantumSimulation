"""Real local compilation, mocked IBM jobs, and streamed DD-on analysis."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock, patch
import zipfile

import numpy as np
import pytest
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit import QuantumCircuit
from qiskit import ClassicalRegister
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SuperOp

from Model2.Experiment8 import run_hardware as run
from Model2.Experiment8.analyze_hardware import analyze
from Model2.Experiment8.ordering import flat_block_order, block_order_op_nodes


def reference(root, backend):
    options = run.base.parse_arguments(["--backend", backend.name, "--trajectory-count", "2", "--shots", "8",
        "--control-shots", "128", "--bootstrap", "100", "--batch-size", "41", "--output-directory", str(root/"old")])
    _, m, _ = run.base.new_plan(options)
    c = m["configuration"]; c["physical_path"] = list(range(12))
    profiles = {name:dict(populations=[.5]*10, ideal_mae=.1, ideal_maximum=.1, physical_maximum=.25)
                for name in ("raw","raw_spam_corrected","twirled","twirled_spam_corrected","zne","zne_spam_corrected")}
    path = root/"historical.json"
    run.atomic_json(path, dict(experiment=run.base.EXPERIMENT, configuration=c, analysis=dict(complete=True),
        job_ids=["historical_job"], statistics=dict(profiles=profiles), ideal_same_paths=[.4]*10, physical_target=[.25]*10))
    return path


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    root = tmp_path_factory.mktemp("combined_dd")
    b = GenericBackendV2(12, basis_gates=["cz","rz","sx","x"],
        coupling_map=[[q,q+1] for q in range(11)]+[[q+1,q] for q in range(11)], seed=17)
    old = reference(root, b)
    options = run.parse_arguments(["prepare", "--backend", b.name, "--trajectory-count", "2", "--shots", "8",
        "--control-shots", "128", "--bootstrap", "100", "--batch-size", "41", "--capture-timing",
        "--historical-reference", str(old), "--physical-path", *map(str,range(12)),
        "--calibration-policy", "record-only", "--output-directory", str(root/"new")])
    with patch.object(run.raw.runtime,"make_service",side_effect=AssertionError("no account")):
        folder, m = run.prepare(options, backend=b)
    return b, folder, m


def copied(compiled, root):
    b, original, _ = compiled
    folder = root/"run";shutil.copytree(original,folder)
    return b, folder


def job_factory(folder):
    sampler = Mock(); jobs=[]
    def submit(pubs, **kwargs):
        job = Mock(); job.job_id.return_value=f"combined_job_{len(jobs)}"
        saved=json.loads((folder/"manifest.json").read_text()); results=[]
        assert all(b["compiled"] for b in saved["batches"])
        for program, _, shots in pubs:
            row=saved["circuits"][program.metadata["pub_index"]]
            assert row["dd_condition"] == "on"
            mask=row["measurement_mask"];z,o=format(mask,"010b"),format(mask^1023,"010b")
            counts=({z:shots} if row.get("control")=="all_zero" else
                    {o:shots} if row.get("control")=="all_one" else {z:shots//2,o:shots-shots//2})
            metadata=dict(compilation=dict(scheduler_timing=dict(timing="main,sx_0,Qubit 1,0,10,play\nmain,measure_0,AWGR0,20,30,capture\n")))
            results.append(SimpleNamespace(metadata=metadata,data=SimpleNamespace(outcome=SimpleNamespace(get_counts=lambda c=counts:c))))
        job.result.return_value=results;job.status.return_value="DONE";jobs.append(job)
        return job
    sampler.run.side_effect=submit
    return sampler,jobs


def test_offline_plan_matches_protocol_and_never_creates_a_dd_off_arm(tmp_path):
    b=SimpleNamespace(name="test_backend");old=reference(tmp_path,b)
    opts=run.parse_arguments(["plan","--backend",b.name,"--trajectory-count","4","--shots","32",
        "--historical-reference",str(old),"--output-directory",str(tmp_path/"new")])
    with patch.object(run.raw.runtime,"make_service",side_effect=AssertionError("offline")):
        folder,m,_=run.new_plan(opts)
    assert {row["dd_condition"] for row in m["circuits"]}=={"on"}
    assert {row["arm"] for row in m["circuits"]}=={"raw","twirled"}
    assert (len(m["batches"]),len(m["circuits"]),m["configuration"]["total_shots"])==(1,67,12800)
    assert not m["configuration"]["sampler_options"]["twirling"]["enable_measure"]
    assert not m["configuration"]["sampler_options"]["dynamical_decoupling"]["enable"]
    assert m["configuration"]["dd_policy"]["min_idle_ratio"]==10
    assert not m["files"].keys() & {"controls.qpy"}


def test_real_folded_native_circuits_preserve_resets_frames_measurements_and_dd(compiled):
    b,folder,m=compiled
    counts=set();all_dd={i:0 for i in range(3)}
    for name,info in m["files"].items():
        if not name.endswith(".qpy"):continue
        programs=run.load_shard(folder,m,name,b)
        assert len(programs)<=13
        for program,index in zip(programs,info["pub_indices"]):
            row=m["circuits"][index]
            if row["kind"]=="trajectory":
                assert program.count_ops()["reset"]==44
                counts.add(program.count_ops()["cz"])
                assert row["dd_check"]["added_x_pulses"]>0
                if row["arm"]=="twirled":all_dd[row["factor_index"]]+=row["dd_check"]["added_x_pulses"]
            assert row["dd_check"]["original_operations_and_local_times_preserved"]
            assert program.metadata["dd_condition"]=="on"
    assert counts=={880,1100,1320}
    assert all(v>0 for v in all_dd.values())


def test_known_jobs_collect_without_native_loading_and_analyze_six_dd_profiles(compiled,tmp_path):
    b,folder=copied(compiled,tmp_path)
    sampler,jobs=job_factory(folder)
    opts=run.parse_arguments(["submit","--manifest",str(folder/"manifest.json"),"--hardware"])
    with patch.object(run.raw,"backend_readiness"),patch.object(run.raw.runtime,"make_service",side_effect=AssertionError("no live account")):
        run.submit(opts,backend=b,sampler_factory=Mock(return_value=sampler))
    assert len(jobs)==1
    service=Mock();service.job.return_value=jobs[0]
    with patch.object(run,"load_shard",side_effect=AssertionError("no QPY for collection")):
        run.collect(run.parse_arguments(["collect","--manifest",str(folder/"manifest.json")]),service=service)
        with patch.object(run.raw.runtime,"make_service",side_effect=AssertionError("already local")):
            run.collect(run.parse_arguments(["collect","--manifest",str(folder/"manifest.json")]))
            report=analyze(folder/"manifest.json",output=tmp_path/"analysis")
    jobs[0].metrics.assert_not_called()
    assert report["experiment"]==run.EXPERIMENT
    assert len(report["statistics"]["profiles"])==6
    for p in report["statistics"]["profiles"].values():np.testing.assert_allclose(p["populations"],.5,atol=1e-13)
    assert not report["verification"]["native_circuits_loaded"]
    assert report["historical_dd_off_reference"]["execution_dates_and_calibrations_are_unpaired"]
    forbidden=Mock(side_effect=AssertionError("no duplicate submit"))
    run.submit(opts,sampler_factory=forbidden);forbidden.assert_not_called()
    bundle=run.bundle(run.parse_arguments(["bundle","--manifest",str(folder/"manifest.json")]))
    with zipfile.ZipFile(bundle) as z:
        assert "analysis/Analysis.md" in z.namelist()
        assert not any(name.endswith(".qpy") for name in z.namelist())
        assert "job_0000_counts.json" in z.namelist()


def test_mock_mode_is_separate_and_interrupted_id_never_auto_retries(compiled,tmp_path):
    b,folder=copied(compiled,tmp_path)
    sampler,jobs=job_factory(folder)
    opts=run.parse_arguments(["submit","--manifest",str(folder/"manifest.json"),"--dry-run"])
    run.submit(opts,backend=b,sampler_factory=Mock(return_value=sampler))
    with pytest.raises(ValueError,match="separate ledgers"):
        run.submit(run.parse_arguments(["submit","--manifest",str(folder/"manifest.json"),"--hardware"]),backend=b,sampler_factory=Mock())
    with pytest.raises(ValueError,match="randomized mocks"):
        analyze(folder/"manifest.json",output=tmp_path/"analysis")
    _,m,_=run.read_saved(folder/"manifest.json")
    m["batches"][0].update(job_id=None,status="submitting");run.checkpoint(folder,m)
    with patch.object(run.raw.runtime,"make_service",side_effect=AssertionError("no account")),pytest.raises(ValueError,match="reconcile"):
        run.submit(opts)


def test_orphan_count_file_recovers_locally_and_active_jobs_do_not_block(compiled,tmp_path):
    b,folder=copied(compiled,tmp_path)
    sampler,jobs=job_factory(folder)
    run.submit(run.parse_arguments(["submit","--manifest",str(folder/"manifest.json"),"--dry-run"]),backend=b,sampler_factory=Mock(return_value=sampler))
    service=Mock();service.job.return_value=jobs[0]
    jobs[0].status.return_value="RUNNING"
    opts=run.parse_arguments(["collect","--manifest",str(folder/"manifest.json")])
    run.collect(opts,service=service);jobs[0].result.assert_not_called()
    jobs[0].status.return_value="DONE"
    before=(folder/"manifest.json").read_bytes()
    with patch.object(run,"checkpoint",side_effect=KeyboardInterrupt),pytest.raises(KeyboardInterrupt):run.collect(opts,service=service)
    assert (folder/"manifest.json").read_bytes()==before
    assert (folder/"job_0000_counts.json").exists()
    with patch.object(run.raw.runtime,"make_service",side_effect=AssertionError("local journal")):
        final=run.collect(opts)
    assert final["status"]=="collected"
    opts=run.parse_arguments(["render","--manifest",str(folder/"manifest.json"),"--trajectory","0","--view","full"])
    def draw(value,destination,**kwargs):Path(destination).write_text("<html>mock timing</html>")
    with patch.object(run.timing,"draw",side_effect=draw) as drawer:
        paths=run.render(opts)
        assert len(paths)==3 and drawer.call_count==3
        run.render(opts);assert drawer.call_count==3
    clone,ledger=run.clone_for_hardware(run.parse_arguments(["clone-for-hardware","--manifest",str(folder/"manifest.json"),"--output-directory",str(tmp_path/"clone")]))
    assert all(b["job_id"] is None for b in ledger["batches"])
    assert "submission_mode" not in ledger
    assert ledger["configuration"]["dd_condition"]=="on"


def test_capture_timing_size_and_source_tampering_are_rejected(compiled,tmp_path):
    with pytest.raises(SystemExit):run.parse_arguments(["plan","--capture-timing","--trajectory-count","16384"])
    _,folder=copied(compiled,tmp_path)
    with (folder/"source.json").open("ab") as f:f.write(b" ")
    with pytest.raises(ValueError,match="saved input changed"):run.read_saved(folder/"manifest.json")


def test_preparation_resume_preserves_shards_without_recompilation(compiled,tmp_path):
    b,folder=copied(compiled,tmp_path)
    _,old,_=run.read_saved(folder/"manifest.json")
    hashes={name:run.qpy_storage.file_sha256(folder/name) for name in old["files"]}
    options=run.parse_arguments(["prepare","--resume",str(folder/"manifest.json")])
    with patch.object(run,"transpile",side_effect=AssertionError("no recompilation")), \
         patch.object(run.timing,"schedule",side_effect=AssertionError("reuse saved schedules")):
        _,new=run.prepare(options,backend=b)
    assert new["plan_sha256"]==old["plan_sha256"]
    assert {name:run.qpy_storage.file_sha256(folder/name) for name in hashes}==hashes
    name=next(name for name in new["files"] if name.endswith(".qpy"))
    with (folder/name).open("ab") as f:f.write(b" ")
    with pytest.raises(ValueError,match="hash mismatch"):run.load_shard(folder,new,name,b)
    with pytest.raises(SystemExit):run.parse_arguments(["prepare","--resume",str(folder/"manifest.json"),"--shots","64"])


def test_fast_order_exactly_matches_sdk_for_reset_measurement_and_control_flow():
    rng=np.random.default_rng(4)
    for _ in range(200):
        q=QuantumCircuit(4,4)
        for _ in range(60):
            a,b=map(int,rng.choice(4,2,replace=False));op=int(rng.integers(5))
            if op==0:q.sx(a)
            elif op==1:q.cz(a,b)
            elif op==2:q.measure(a,a)
            elif op==3:q.reset(a)
            else:q.barrier(a,b)
        dag=circuit_to_dag(q)
        assert list(flat_block_order(dag))==list(block_order_op_nodes(dag))
    q=QuantumCircuit(2,1);q.h(0);q.measure(0,0)
    with q.if_test((q.clbits[0],1)):q.x(1)
    dag=circuit_to_dag(q)
    assert list(flat_block_order(dag))==list(block_order_op_nodes(dag))


def test_full_folded_variant_has_identical_sdk_block_order(compiled):
    b,folder,m=compiled
    name=next(name for name in m["files"] if name.endswith(".qpy") and "trajectory" in name)
    programs=run.load_shard(folder,m,name,b)
    p=next(p for p in programs if p.count_ops().get("cz")==1320)
    native=p.copy_empty_like()
    for item in p.data:
        if item.operation.name!="delay" and item.operation.label!=run.timing.DD_LABEL:
            native.append(item.operation,item.qubits,item.clbits)
    dag=circuit_to_dag(native)
    assert list(flat_block_order(dag))==list(block_order_op_nodes(dag))


def test_folding_twirling_and_dd_preserve_a_short_open_channel():
    b=GenericBackendV2(3,basis_gates=["cz","rz","sx","x"],coupling_map=[[0,1],[1,0],[0,2],[2,0]],seed=17)
    q=QuantumCircuit(3)
    outcome=ClassicalRegister(2,"outcome");q.add_register(outcome)
    q.metadata=dict(physical_sites=[0,1])
    q.sx(0);q.sx(1);q.cz(0,2);q.reset(2);q.rz(.3,1);q.cz(0,1)
    q.delay(20000,0,unit="dt");q.measure(0,outcome[0]);q.measure(1,outcome[1])
    def channel(program,mask=0):
        stripped=QuantumCircuit(3)
        for item in program.data:
            if item.operation.name not in ("measure","delay","barrier"):
                stripped.append(item.operation,[program.find_bit(bit).index for bit in item.qubits])
        for j in range(2):
            if mask>>j&1:stripped.x(j)
        return SuperOp(stripped).data
    wanted=channel(q)
    policy=dict(physical_qubits=[0,1],min_idle_ratio=10.)
    for folds in (0,1,2):
        variant=run.protocol.mit.transformed(q,seed=19,folds=folds,measurement_mask=1)
        baseline,e=run.timing.schedule(variant,b.target,block_ordering_callable=flat_block_order)
        protected,f=run.timing.schedule(variant,b.target,policy=policy,block_ordering_callable=flat_block_order)
        check=run.timing.verify_pair(baseline,protected,e,f,policy)
        assert check["added_x_pulses"]>0
        np.testing.assert_allclose(channel(protected,mask=1),wanted,atol=2e-12)
