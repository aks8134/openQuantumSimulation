"""Plan, compile, execute and resume the configurable manuscript-protocol run."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import os
import zipfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from FinalModel.Model3Experiment import model,circuits
from Model3.Experiment5 import run_hardware as original_hardware
from Model3.Experiment5.analyze_hardware import source_integrity


def saved_target(source):
    return model.store.load_snapshot(ROOT/source['configuration']['hardware']['saved_target'])


def preview(folder,batch_index=0,paths=1,*,target=None,metadata=None):
    source,manifest,data=model.load_run(folder)
    if target is None:target,metadata=saved_target(source)
    pubs,records=circuits.compile_batch(Path(folder)/'offline_preview',source,data,
        manifest['batches'][batch_index],target,metadata,preview_paths=paths)
    report=dict(scope='Offline saved-target preview only; these circuits are never reused for live submission',
        circuits=len(pubs),trajectory_paths=paths,target_sha256=metadata['target_sha256'],
        operations={name:sorted({row['metrics']['operation_counts'].get(name,0) for row in records if row['kind']=='trajectory'})
                    for name in ('cz','reset','measure')},
        mean_depth=sum(row['metrics']['depth'] for row in records)/len(records))
    model.store.atomic_json(Path(folder)/'offline_preview'/'summary.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


@contextmanager
def execution_lock(path):
    """Nonblocking lock on Linux/macOS and Windows, released on process exit."""
    with Path(path).open('a+b') as lock:
        if os.name=='nt':
            import msvcrt
            lock.seek(0)
            if not lock.read(1):lock.write(b'0');lock.flush()
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            try:yield
            finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:yield
            finally:fcntl.flock(lock,fcntl.LOCK_UN)


def execute(folder,account_file=None,max_jobs=None,*,execution_mode=None,jobs_per_batch=None,
            batch_max_time=None,service=None,backend=None,sampler_factory=None,batch_factory=None):
    if max_jobs is not None and max_jobs<1:raise ValueError('max-jobs must be positive')
    with execution_lock(Path(folder)/'execution.lock'):
        return _execute(folder,account_file,max_jobs,execution_mode=execution_mode,jobs_per_batch=jobs_per_batch,
            batch_max_time=batch_max_time,service=service,backend=backend,
            sampler_factory=sampler_factory,batch_factory=batch_factory)


def native_inputs(folder,source,data,batch,target,metadata,backend):
    pubs,rows=circuits.compile_batch(folder/'native',source,data,batch,target,metadata)
    # Always validate against the connected target, including when loading QPY.
    for qc in pubs:circuits.assert_supported(qc,backend.target)
    batch['records_file']=f'native/batch_{batch["index"]:05d}.json'
    return pubs,rows


def submit_job(folder,source,manifest,data,batch,target,metadata,backend,sampler_factory,mode,*,runtime_batch_id=None):
    cfg=source['configuration'];h=cfg['hardware']
    pubs,rows=native_inputs(folder,source,data,batch,target,metadata,backend)
    options={'dynamical_decoupling':{'enable':False},
        'twirling':{'enable_gates':False,'enable_measure':False},'execution':{'init_qubits':True},
        'max_execution_time':h['max_execution_time'],
        'environment':{'job_tags':['FinalModel-Model3',folder.name,f'batch-{batch["index"]}']}}
    sampler=sampler_factory(mode=mode,options=options)
    batch.update(status='submitting',execution_mode='batch' if runtime_batch_id else 'job')
    if runtime_batch_id:batch['runtime_batch_id']=runtime_batch_id
    model.store.atomic_json(folder/'manifest.json',manifest)
    job=sampler.run([(qc,None,row['shots']) for qc,row in zip(pubs,rows)],shots=cfg['sampling']['shots'])
    job_id=job.job_id()
    if not isinstance(job_id,str) or not job_id.strip():raise ValueError('Submission returned no job ID. Reconcile before resuming.')
    batch.update(status='submitted',job_id=job_id);model.store.atomic_json(folder/'manifest.json',manifest)
    print(f"Submitted job {batch['index']}: {job_id}",flush=True)
    return job


def collect_job(folder,source,manifest,batch,job):
    compiled=model.store.read_sealed(folder/batch['records_file']);rows=compiled['circuits']
    answer=job.result()
    if len(answer)!=len(rows):raise ValueError('Runtime result count mismatch')
    observations=[]
    for pub,row in zip(answer,rows):
        counts={str(k):int(v) for k,v in pub.data.outcome.get_counts().items()}
        if sum(counts.values())!=row['shots']:raise ValueError('Runtime shot count mismatch')
        observations.append({'outcome_counts':counts})
    payload=model.store.sealed(dict(source_sha256=source['sha256'],compiled_sha256=compiled['sha256'],
        job_id=batch['job_id'],observations=observations,collected_utc=datetime.now(timezone.utc).isoformat()))
    output=folder/'counts'/f"batch_{batch['index']:05d}.json"
    model.store.atomic_json(output,payload)
    batch.update(status='collected',result_file=str(output.relative_to(folder)),result_sha256=payload['sha256'])
    try:batch['job_metrics']=json.loads(json.dumps(job.metrics(),default=str))
    except Exception as exc:batch['job_metrics_unavailable']=type(exc).__name__
    model.store.atomic_json(folder/'manifest.json',manifest)
    if not source['configuration']['hardware'].get('retain_native_qpy',True):
        native_file=folder/'native'/f"batch_{batch['index']:05d}.qpy.gz"
        native_file.unlink(missing_ok=True)
        batch['native_qpy_retained']=False
        model.store.atomic_json(folder/'manifest.json',manifest)
    print(f"Collected job {batch['index']}; {len(rows)} PUBs",flush=True)


def close_runtime_batch(folder,manifest,record,remote):
    try:remote.close()  # Stop accepting submissions; queued/running jobs are preserved.
    except Exception as exc:
        record.update(status='close_failed',close_error=type(exc).__name__)
        model.store.atomic_json(folder/'manifest.json',manifest)
        raise
    record.update(status='closed',closed_utc=datetime.now(timezone.utc).isoformat())
    record.pop('close_error',None)
    model.store.atomic_json(folder/'manifest.json',manifest)


def submit_runtime_batch(folder,source,manifest,data,window,target,metadata,backend,sampler_factory,batch_factory,policy):
    # Compile the entire window BEFORE opening a Runtime batch. Release circuit
    # objects between jobs and use the verified on-disk QPY cache for submission.
    for batch in window:
        pubs,rows=native_inputs(folder,source,data,batch,target,metadata,backend)
        del pubs,rows
    remote=batch_factory(backend=backend,max_time=policy['batch_max_time'])
    record=None;jobs=[]
    try:
        runtime_id=remote.session_id
        if not isinstance(runtime_id,str) or not runtime_id.strip():raise ValueError('Runtime batch returned no ID')
        record=dict(runtime_batch_id=runtime_id,job_indices=[b['index'] for b in window],status='open',
            max_time=policy['batch_max_time'],created_utc=datetime.now(timezone.utc).isoformat())
        manifest.setdefault('runtime_batches',[]).append(record)
        model.store.atomic_json(folder/'manifest.json',manifest)
        for batch in window:
            job=submit_job(folder,source,manifest,data,batch,target,metadata,backend,sampler_factory,remote,
                           runtime_batch_id=runtime_id)
            jobs.append((batch,job))
    finally:
        interrupted=sys.exc_info()[0] is not None
        try:
            if record is None:remote.close()
            else:close_runtime_batch(folder,manifest,record,remote)
        except Exception:
            if not interrupted:raise
            print('Could not close Runtime batch; its ID is retained for recovery.',flush=True)
    # No result wait occurs until all window jobs have been submitted and the
    # Runtime batch has stopped accepting new jobs.
    return jobs


def _execute(folder,account_file=None,max_jobs=None,*,execution_mode=None,jobs_per_batch=None,
             batch_max_time=None,service=None,backend=None,sampler_factory=None,batch_factory=None):
    folder=Path(folder);source,manifest,data=model.load_run(folder);cfg=source['configuration'];h=cfg['hardware']
    policy=model.execution_options(cfg,execution_mode,jobs_per_batch,batch_max_time)
    for name,digest in source['code'].items():
        source_integrity((ROOT/name).read_bytes(),digest)
    ambiguous=[b['index'] for b in manifest['batches'] if b['status']=='submitting' and not b.get('job_id')]
    if ambiguous:raise ValueError(f'Submission interrupted before a job ID was saved in batches {ambiguous}; reconcile on IBM before resuming. No automatic resubmission.')
    if all(b['status']=='collected' for b in manifest['batches']):
        from FinalModel.Model3Experiment.analysis import analyze
        return analyze(folder)
    print(f"Frozen run: N={cfg['physics']['N']}, steps={source['case']['steps']}, "
          f"R={cfg['sampling']['trajectories']}, S={cfg['sampling']['shots']}; "
          f"paths_per_job={h['paths_per_job']}, total_jobs={manifest['workload']['jobs']}; "
          f"execution_mode={policy['execution_mode']}, jobs_per_batch={policy['jobs_per_batch']}",flush=True)
    if service is None and backend is None:
        service=original_hardware.make_service(SimpleNamespace(account_file=account_file))
    if backend is None:backend=service.backend(h['backend'],use_fractional_gates=False)
    if backend.name!=h['backend']:raise ValueError('wrong live backend')
    live=folder/'live_target'
    if not (live/'target.json').exists():metadata=model.store.capture_snapshot(backend,live)
    target,metadata=model.store.load_snapshot(live)
    model.store.chain_layout(target,source['case']['N'],supplied=h['physical_path'])
    if metadata['backend']!=h['backend']:raise ValueError('target/backend mismatch')
    if sampler_factory is None:
        from qiskit_ibm_runtime import SamplerV2
        sampler_factory=SamplerV2
    open_groups=[g for g in manifest.get('runtime_batches',[]) if g['status']!='closed']
    if batch_factory is None and (policy['execution_mode']=='batch' or open_groups):
        from qiskit_ibm_runtime import Batch
        batch_factory=Batch
    for group in open_groups:
        if service is None:raise ValueError('service required to close an interrupted Runtime batch')
        remote=batch_factory.from_id(group['runtime_batch_id'],service=service)
        close_runtime_batch(folder,manifest,group,remote)
    pending=[b for b in manifest['batches'] if b['status']!='collected']
    if max_jobs is not None:pending=pending[:max_jobs]
    # Recover every selected known job before submitting anything new. A failed
    # or expired job remains recorded and is never silently replaced.
    for batch in pending:
        if not batch.get('job_id'):continue
        if service is None:raise ValueError('service required to recover an existing job')
        collect_job(folder,source,manifest,batch,service.job(batch['job_id']))
    planned=[b for b in pending if b['status']!='collected']
    if policy['execution_mode']=='job':
        for batch in planned:
            job=submit_job(folder,source,manifest,data,batch,target,metadata,backend,sampler_factory,backend)
            collect_job(folder,source,manifest,batch,job)
    else:
        size=policy['jobs_per_batch']
        for start in range(0,len(planned),size):
            jobs=submit_runtime_batch(folder,source,manifest,data,planned[start:start+size],target,metadata,
                                      backend,sampler_factory,batch_factory,policy)
            for batch,job in jobs:collect_job(folder,source,manifest,batch,job)
    if all(b['status']=='collected' for b in manifest['batches']):
        manifest['status']='collected';model.store.atomic_json(folder/'manifest.json',manifest)
        from FinalModel.Model3Experiment.analysis import analyze
        return analyze(folder)
    return manifest


def parse_arguments(arguments=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    plan=sub.add_parser('prepare');plan.add_argument('--config',type=Path,default=model.DIRECTORY/'config.json');plan.add_argument('--output',type=Path)
    comp=sub.add_parser('preview');comp.add_argument('--run',type=Path,required=True);comp.add_argument('--batch',type=int,default=0);comp.add_argument('--paths',type=int,default=1)
    execute=sub.add_parser('execute');execute.add_argument('--run',type=Path,required=True);execute.add_argument('--account-file',type=Path);execute.add_argument('--max-jobs',type=int)
    execute.add_argument('--execution-mode',choices=('job','batch'),help='Override hardware.execution_mode for this invocation')
    execute.add_argument('--jobs-per-batch',type=int,help='Jobs submitted together per Runtime batch (uses frozen configuration)')
    execute.add_argument('--batch-max-time',type=int,metavar='SECONDS',help='Runtime batch maximum TTL in seconds (uses frozen configuration)')
    analysis=sub.add_parser('analyze');analysis.add_argument('--run',type=Path,required=True)
    bundle=sub.add_parser('bundle');bundle.add_argument('--run',type=Path,required=True);bundle.add_argument('--output',type=Path,required=True)
    return parser.parse_args(arguments)


def main(arguments=None):
    options=parse_arguments(arguments)
    if options.action=='prepare':return model.prepare(options.config,options.output)
    if options.action=='preview':return preview(options.run,options.batch,options.paths)
    if options.action=='execute':return execute(options.run,options.account_file,options.max_jobs,
        execution_mode=options.execution_mode,jobs_per_batch=options.jobs_per_batch,batch_max_time=options.batch_max_time)
    if options.action=='bundle':return export_bundle(options.run,options.output)
    from FinalModel.Model3Experiment.analysis import analyze
    return analyze(options.run)


def export_bundle(folder,output):
    """Portable code and frozen input; excludes credentials, native caches and counts."""
    folder=Path(folder).resolve();source,manifest,_=model.load_run(folder)
    if any(b.get('job_id') for b in manifest['batches']):raise ValueError('Export a planned, unsubmitted run')
    files=list(model.DIRECTORY.glob('*.py'))+list(model.DIRECTORY.glob('*.md'))+[model.DIRECTORY/'config.json']
    files.extend(ROOT/name for name in (
        'FinalModel/Experiment1/phoenix_resources.py','FinalModel/Experiment1/study.py',
        'FinalModel/Experiment2/sampling.py','FinalModel/Experiment3/readout.py',
        'library/reservoir_scaling.py','Model3/Experiment5/circuit_depth.py','Model3/Experiment5/control_quality.py',
        'Model3/Experiment5/mitigation.py','Model3/Experiment5/run_hardware.py','Model3/Experiment5/analyze_hardware.py',
        'FinalModel/Experiment1/channel.cpp','FinalModel/Experiment2/trajectory.cpp','FinalModel/Experiment3/readout.cpp'))
    files.extend((ROOT/'IBMRuntime').rglob('*.py'))
    if source['configuration']['schedule']['mode']=='saved':
        files.append(ROOT/source['configuration']['schedule']['input'])
    files.extend(folder/name for name in ('source.json','paths.npz','manifest.json'))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(set(files)):
            archive.write(file,file.relative_to(ROOT).as_posix())
        archive.writestr('RUN_MODEL3.txt',
            f"Prepared Model 3 input, R={manifest['workload']['paths']} and S={manifest['workload']['shots_per_path_basis_factor']}. No hardware jobs submitted.\n"
            'From this extracted folder, in an environment with Qiskit and Runtime, run:\n'
            f'python FinalModel/Model3Experiment/run.py execute --run {folder.relative_to(ROOT).as_posix()}\n'
            f"For IBM Runtime batch execution, append: --execution-mode batch --jobs-per-batch {source['configuration']['hardware'].get('jobs_per_batch',4)}\n"
            'Batch lifetime can be set with --batch-max-time SECONDS (default 28800).\n'
            'Use --account-file FILE if your account is not already configured.\n'
            'See FinalModel/Model3Experiment/README.md for the protocol and workload.\n')
    print(json.dumps({'bundle':str(output),'sha256':model.store.file_digest(output),
                      'source_sha256':source['sha256'],'contains_credentials':False},indent=2),flush=True)
    return output


if __name__=='__main__':main()
