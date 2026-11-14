"""Physics, native mitigation, pair readout and recoverable Runtime execution."""
import contextlib
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import Mock,patch

import numpy as np
from qiskit import QuantumCircuit,qpy
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import DensityMatrix,Kraus
from FinalModel.Model3Experiment import model,circuits,analysis,run,portable


def config():
    cfg=model.load_config(model.DIRECTORY/'config.json')
    cfg['physics']['N']=2
    cfg['schedule']={'mode':'manual','step_dts':[.12,.23]}
    cfg['sampling'].update(trajectories=2,shots=16)
    cfg['mitigation']['control_shots']=16
    cfg['hardware'].update(physical_path=[0,1,2,3],paths_per_job=2,execution_mode='job')
    cfg['analysis']['bootstrap']=100
    return cfg


def backend():
    return GenericBackendV2(4,basis_gates=['rz','sx','x','cz'],
        coupling_map=[[0,1],[1,0],[1,2],[2,1],[2,3],[3,2]],seed=11)


def probabilities(qc,sites):
    rho=DensityMatrix.from_label('0'*qc.num_qubits)
    dephase=Kraus([np.diag([1,0]),np.diag([0,1])])
    for item in qc.data:
        op=item.operation;args=[qc.find_bit(q).index for q in item.qubits]
        if op.name=='barrier':continue
        if op.name=='measure':rho=rho.evolve(dephase,args)
        elif op.name=='reset':rho=rho.reset(args)
        else:rho=rho.evolve(op,args)
    return rho.probabilities(qargs=sites)


def runtime_result(rows):
    """Consistent small-system counts, with terminal masks still applied."""
    result=[]
    for row in rows:
        S=row['shots'];label='00' if row.get('control')=='all_zero' else '11'
        logical={label:S} if row.get('control') in ('all_zero','all_one') else {format(i,'02b'):S//4 for i in range(4)}
        physical={format(int(k,2)^row['mask'],'02b'):v for k,v in logical.items()}
        result.append(SimpleNamespace(data=SimpleNamespace(outcome=SimpleNamespace(get_counts=lambda c=physical:c))))
    return result


class Model3ExperimentTests(unittest.TestCase):
    def prepare(self,folder,cfg):
        path=Path(folder)/'config.json';path.write_text(json.dumps(cfg))
        with contextlib.redirect_stdout(io.StringIO()):return model.prepare(path,Path(folder)/'run')[0]

    def test_default_is_exact_scaling_schedule_and_user_chosen_budget(self):
        cfg=model.load_config(model.DIRECTORY/'config.json');case=model.make_case(cfg)
        self.assertEqual(case['steps'],72)
        self.assertAlmostEqual(case['physical_time'],14.86004146644756)
        self.assertEqual(cfg['hardware']['paths_per_job'],192)
        self.assertEqual(cfg['hardware']['execution_mode'],'batch')
        self.assertEqual(cfg['hardware']['jobs_per_batch'],8)
        self.assertEqual(model.workload(cfg)['jobs'],86)
        self.assertEqual(model.workload(cfg)['total_shots'],4202496)
        self.assertEqual(model.workload(cfg)['total_circuits'],640266)
        changed=copy.deepcopy(cfg);changed['physics']['gamma']=2
        with self.assertRaisesRegex(ValueError,'Saved schedule physics'):model.make_case(changed)

    def test_prepared_run_matches_agreed_packing_and_current_code(self):
        folder=model.DIRECTORY/'hardware_runs/model3_full_R16384_S16_P192_20261113'
        source,manifest,data=model.load_run(folder)
        self.assertEqual(source['configuration'],model.load_config(model.DIRECTORY/'config.json'))
        self.assertEqual(source['case']['steps'],72)
        self.assertEqual(data['choices'].shape,(16384,72))
        self.assertEqual(manifest['workload']['jobs'],86)
        self.assertEqual(manifest['workload']['total_shots'],4202496)
        self.assertEqual(sorted(i for b in manifest['batches'] for i in b['indices']),list(range(16384)))
        self.assertTrue(all(len(b['indices'])<=192 for b in manifest['batches']))
        revision=run.audit_execution_code(source)
        self.assertEqual(revision['patch_id'],'seekable-qpy-cache-v1')
        self.assertEqual(set(revision['files']),{'FinalModel/Model3Experiment/circuits.py','FinalModel/Model3Experiment/run.py'})

    def test_cached_qpy_uses_seekable_disk_and_closes_it(self):
        qc=QuantumCircuit(2);qc.h(0);qc.cx(0,1);qc.reset(1);qc.metadata={'case_id':'test'}
        loader=qpy.load;streams=[];temporary=tempfile.TemporaryFile
        def remember_temp(*args,**kwargs):
            stream=temporary(*args,**kwargs);streams.append(stream);return stream
        def inspect_stream(stream):
            self.assertNotIsInstance(stream,gzip.GzipFile)
            self.assertTrue(stream.seekable());self.assertEqual(stream.tell(),0)
            self.assertIsInstance(stream,io.BytesIO)
            self.assertLess(stream.getbuffer().nbytes,expanded_size)
            return loader(stream)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            path=Path(tmp)/'circuits.qpy.gz'
            with gzip.open(path,'wb') as f:qpy.dump([qc,qc],f)
            with gzip.open(path,'rb') as f:expanded_size=len(f.read())
            with patch.object(circuits.tempfile,'TemporaryFile',side_effect=remember_temp),patch.object(circuits.qpy,'load',side_effect=inspect_stream),patch.object(gzip.GzipFile,'seek',side_effect=AssertionError('QPY must not seek gzip')):
                loaded=circuits.load_cached_qpy(path)
            self.assertEqual(loaded,[qc,qc]);self.assertTrue(streams[0].closed)
            with patch.object(circuits.tempfile,'TemporaryFile',side_effect=remember_temp),patch.object(circuits.qpy,'load',side_effect=RuntimeError('bad QPY')):
                with self.assertRaisesRegex(RuntimeError,'bad QPY'):circuits.load_cached_qpy(path)
            self.assertTrue(streams[-1].closed)

    def test_cached_writer_avoids_whole_job_buffer_and_retains_payload(self):
        qc=QuantumCircuit(2);qc.rx(.37,0);qc.cx(0,1);qc.reset(1)
        writer=qpy.dump
        def inspect_writer(pubs,stream):
            self.assertNotIsInstance(stream,(gzip.GzipFile,io.BytesIO))
            self.assertIsInstance(stream.fileno(),int);return writer(pubs,stream)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            old=Path(tmp)/'old.qpy.gz';new=Path(tmp)/'new.qpy.gz'
            with gzip.open(old,'wb') as f:qpy.dump([qc,qc],f)
            with patch.object(circuits.qpy,'dump',side_effect=inspect_writer):circuits.write_cached_qpy(new,[qc,qc])
            with gzip.open(old,'rb') as a,gzip.open(new,'rb') as b:self.assertEqual(a.read(),b.read())
            self.assertEqual(circuits.load_cached_qpy(new),[qc,qc])
            legacy=Path(tmp)/'legacy.qpy.gz'
            with gzip.open(legacy,'wb') as f:qpy.dump([qc],f,version=15)
            self.assertEqual(circuits.load_cached_qpy(legacy),[qc])

    def test_cache_integrity_does_not_need_depth_recalculation(self):
        cfg=config();b=backend()
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            folder=self.prepare(tmp,cfg);source,manifest,data=model.load_run(folder)
            batch=manifest['batches'][0];metadata={'target_sha256':'local-test','compiler_methods':{}}
            original,rows=circuits.compile_batch(folder/'native',source,data,batch,b.target,metadata)
            with patch.object(circuits.old_circuits,'metrics',side_effect=AssertionError('repeated depth calculation')):
                loaded,again=circuits.compile_batch(folder/'native',source,data,batch,b.target,metadata)
            self.assertEqual(original,loaded);self.assertEqual(rows,again)
            path=folder/'native/batch_00000.qpy.gz'
            with path.open('ab') as f:f.write(b'changed')
            with patch.object(circuits,'load_cached_qpy',side_effect=AssertionError('must verify before loading')):
                with self.assertRaisesRegex(ValueError,'provenance changed'):
                    circuits.compile_batch(folder/'native',source,data,batch,b.target,metadata)

    def test_compatibility_rejects_unreviewed_code_changes(self):
        folder=model.DIRECTORY/'hardware_runs/model3_full_R16384_S16_P192_20261113'
        source,_,_=model.load_run(folder)
        changed=copy.deepcopy(source);changed['code']['FinalModel/Model3Experiment/model.py']='0'*64
        with self.assertRaisesRegex(ValueError,'Unsupported source change'):run.audit_execution_code(changed)
        rule=model.store.read_sealed(model.DIRECTORY/'cache_fix_compatibility.json')['files']['FinalModel/Model3Experiment/circuits.py']
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'ROOT',Path(tmp)):
            path=Path(tmp)/'FinalModel/Model3Experiment/circuits.py';path.parent.mkdir(parents=True);path.write_text('unreviewed code')
            fake=dict(sha256='original',code={'FinalModel/Model3Experiment/circuits.py':rule['previous_sha256']})
            with self.assertRaisesRegex(ValueError,'source hash mismatch'):run.audit_execution_code(fake)

    def test_compatibility_rejects_partial_patch_checkout(self):
        names=['FinalModel/Model3Experiment/circuits.py','FinalModel/Model3Experiment/run.py']
        old=[b'old loader',b'old runner'];new=[b'new loader',b'new runner']
        digest=lambda raw:hashlib.sha256(raw).hexdigest()
        rules={name:dict(previous_sha256=digest(a),replacement_sha256=digest(b)) for name,a,b in zip(names,old,new)}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'FinalModel/Model3Experiment';folder.mkdir(parents=True)
            (folder/'circuits.py').write_bytes(old[0]);(folder/'run.py').write_bytes(new[1])
            model.store.atomic_json(folder/'cache_fix_compatibility.json',model.store.sealed(dict(patch_id='seekable-qpy-cache-v1',files=rules)))
            source=dict(sha256='original',code={name:rule['previous_sha256'] for name,rule in rules.items()})
            with patch.object(run,'ROOT',root),patch.object(model,'DIRECTORY',folder),self.assertRaisesRegex(ValueError,'source hash mismatch'):
                run.audit_execution_code(source)

    def test_same_controls_and_matched_budgets_no_trajectory_split(self):
        cfg=config();records=model.records(cfg,model.batches(cfg)[0])
        self.assertEqual(len(records),93)
        for name in ('all_zero','all_one','initial_mixture'):
            self.assertEqual(sum(r['kind']=='control' and r['control']==name for r in records),5)
        for index in range(2):
            for basis in model.BASES:
                rows=[r for r in records if r.get('trajectory')==index and r['basis']==basis]
                self.assertEqual(sum(r['shots'] for r in rows if r['arm']=='raw'),16)
                for factor in range(3):self.assertEqual(sum(r['shots'] for r in rows if r['arm']=='twirled' and r['factor_index']==factor),16)
        cfg['sampling']['shots']=15
        with self.assertRaises(ValueError):model.validate_config(cfg)

    def test_asymmetric_local_response_corrects_pair_offsets(self):
        truth=np.array([.1,.2,.3,.4]);a=np.array([.03,.08]);b=np.array([.94,.90])
        observed=np.zeros(4)
        for x in range(4):
            probabilities_one=np.where([(x>>j)&1 for j in range(2)],b,a)
            for y in range(4):
                prob=np.prod([probabilities_one[j] if (y>>j)&1 else 1-probabilities_one[j] for j in range(2)])
                observed[y]+=truth[x]*prob
        z=np.array([[1-2*((x>>j)&1) for j in range(2)] for x in range(4)])
        features=np.c_[z,z[:,0]*z[:,1]]
        corrected=analysis.correct_features(observed@features,a,b)
        np.testing.assert_allclose(corrected,truth@features,atol=1e-14)
        np.testing.assert_allclose((1-corrected[:2])/2,
            circuits.original.corrected_population((1-(observed@features)[:2])/2,a,b),atol=1e-14)

    def test_current_decoder_sign_and_normalization(self):
        # YX parity +1 and XY parity -1 give a positive unit current for c=1/2.
        f=np.array([[0,0,0],[0,0,-1],[0,0,1.]])
        np.testing.assert_allclose(analysis.observables(f,.5,2),[.5,.5,1.])
        with self.assertRaises(ValueError):analysis.features({'0':16},2)

    def test_portable_stream_and_references_match_cpp_beyond_one_rng_block(self):
        self.assertEqual(portable.MT19937_64(5489).next(),14514284786278117030)
        cfg=config();cfg['physics'].update(J=.7,gamma=.4,prefactor=.37,p_left=.8,p_right=.3)
        cfg['schedule']['step_dts']=[.013]*330
        case=model.make_case(cfg);fast=model.references(cfg,case)
        cfg['sampling']['reference_engine']='portable';slow=model.references(cfg,case)
        np.testing.assert_array_equal(fast['choices'],slow['choices'])
        for key in ('digital','target','ideal_paths'):np.testing.assert_allclose(fast[key],slow[key],atol=3e-13)

    def test_execution_lock_blocks_a_second_controller(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'execution.lock'
            with run.execution_lock(path):
                with self.assertRaises(OSError):
                    with run.execution_lock(path):pass

    def test_full_native_variants_preserve_all_readout_distributions(self):
        cfg=config();case=model.make_case(cfg);data=model.references(cfg,case);source={'case':case,'configuration':cfg}
        b=backend();rows=model.records(cfg,model.batches(cfg)[0])
        pubs,records,_=circuits.compile_group(source,data,cfg,rows,b.target,{})
        expected={}
        for qc,row in zip(pubs,records):
            key=(row['kind'],row.get('trajectory',row.get('control')),row['basis'])
            if row['arm']=='raw':expected[key]=probabilities(qc,row['physical_sites'])
        for qc,row in zip(pubs,records):
            key=(row['kind'],row.get('trajectory',row.get('control')),row['basis'])
            actual=probabilities(qc,row['physical_sites'])
            decoded=actual[np.arange(4)^row['mask']]
            np.testing.assert_allclose(decoded,expected[key],atol=3e-12)
            self.assertEqual(qc.count_ops().get('reset',0),case['steps'] if row['kind']=='trajectory' else 0)
            self.assertEqual(qc.count_ops().get('cz',0),row['base_cz']+2*row['fold_count'])

    def test_plan_and_preview_never_access_an_account(self):
        cfg=config()
        with tempfile.TemporaryDirectory() as tmp,patch.object(run.original_hardware,'make_service',side_effect=AssertionError('no account')):
            folder=self.prepare(tmp,cfg);target=backend().target
            with contextlib.redirect_stdout(io.StringIO()):result=run.preview(folder,0,1,target=target,metadata={'target_sha256':'local-test','compiler_methods':{}})
            self.assertEqual(result['circuits'],54)
            self.assertEqual(json.loads((folder/'manifest.json').read_text())['status'],'planned')

    def test_saved_job_is_recovered_without_duplicate_submission(self):
        cfg=config();b=backend();cfg['hardware']['backend']=b.name
        rows=model.records(cfg,model.batches(cfg)[0])
        observations=[]
        for row in rows:
            S=row['shots'];label='00' if row.get('control')=='all_zero' else '11'
            logical={label:S} if row.get('control') in ('all_zero','all_one') else {format(i,'02b'):S//4 for i in range(4)}
            physical={format(int(k,2)^row['mask'],'02b'):v for k,v in logical.items()}
            observations.append(SimpleNamespace(data=SimpleNamespace(outcome=SimpleNamespace(get_counts=lambda c=physical:c))))
        job=Mock();job.job_id.return_value='saved-job';job.result.side_effect=[RuntimeError('interrupted collection'),observations]
        sampler=Mock();sampler.run.return_value=job;factory=Mock(return_value=sampler);service=Mock();service.job.return_value=job
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            folder=self.prepare(tmp,cfg)
            with self.assertRaisesRegex(RuntimeError,'interrupted'):run.execute(folder,service=service,backend=b,sampler_factory=factory)
            self.assertEqual(json.loads((folder/'manifest.json').read_text())['batches'][0]['job_id'],'saved-job')
            result=run.execute(folder,service=service,backend=b,sampler_factory=Mock(side_effect=AssertionError('duplicate submission')))
            self.assertIn('currents_over_J',result['profiles']['zne'])
            self.assertEqual(sampler.run.call_count,1)

    def test_ambiguous_submission_is_not_retried(self):
        cfg=config();b=backend();cfg['hardware']['backend']=b.name
        sampler=Mock();sampler.run.side_effect=RuntimeError('ambiguous network failure')
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            folder=self.prepare(tmp,cfg)
            with self.assertRaisesRegex(RuntimeError,'ambiguous'):run.execute(folder,backend=b,sampler_factory=Mock(return_value=sampler))
            with patch.object(run.original_hardware,'make_service',side_effect=AssertionError('no account')):
                with self.assertRaisesRegex(ValueError,'No automatic resubmission'):run.execute(folder)


    def test_execution_options_and_cli_do_not_change_path_grouping(self):
        cfg=config();before=model.workload(cfg)
        cfg['hardware'].update(execution_mode='batch',jobs_per_batch=2,batch_max_time=600)
        self.assertEqual(model.workload(cfg),before)
        self.assertEqual(model.execution_options(cfg),dict(execution_mode='batch',jobs_per_batch=2,batch_max_time=600))
        self.assertEqual(model.execution_options(cfg,execution_mode='job')['execution_mode'],'job')
        args=run.parse_arguments(['execute','--run','example','--execution-mode','batch','--jobs-per-batch','3','--batch-max-time','900'])
        self.assertEqual((args.execution_mode,args.jobs_per_batch,args.batch_max_time),('batch',3,900))
        for name,value in [('execution_mode','session'),('jobs_per_batch',0),('batch_max_time',-1)]:
            with self.assertRaises(ValueError):model.execution_options(cfg,**{name:value})
        with tempfile.TemporaryDirectory() as tmp,patch.object(run.original_hardware,'make_service',side_effect=AssertionError('no account')):
            folder=self.prepare(tmp,cfg)
            with self.assertRaisesRegex(ValueError,'jobs_per_batch'):run.execute(folder,jobs_per_batch=0)

    def test_batch_submits_whole_window_before_collection_and_honors_limit(self):
        cfg=config();cfg['sampling']['trajectories']=6;b=backend();cfg['hardware']['backend']=b.name
        events=[];remote=Mock();remote.session_id='runtime-group-0'
        remote.close.side_effect=lambda:events.append('close')
        compile_calls=[];compile_batch=circuits.compile_batch
        def compile_record(*args,**kwargs):
            compile_calls.append(args[3]['index']);return compile_batch(*args,**kwargs)
        def open_batch(**kwargs):
            self.assertEqual(compile_calls,[0,1]);self.assertEqual(kwargs['max_time'],600)
            events.append('open');return remote
        factory=Mock(side_effect=open_batch);samplers=[]
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            folder=self.prepare(tmp,cfg)
            def make_sampler(**kwargs):
                self.assertIs(kwargs['mode'],remote)
                self.assertFalse(kwargs['options']['twirling']['enable_gates'])
                index=len(samplers);sampler=Mock();job=Mock();job.job_id.return_value=f'job-{index}'
                def submit(pubs,**unused):
                    events.append(f'submit-{index}')
                    rows=model.records(cfg,model.batches(cfg)[index]);self.assertEqual(len(pubs),len(rows))
                    self.assertEqual([p[2] for p in pubs],[r['shots'] for r in rows])
                    def collect():
                        saved=json.loads((folder/'manifest.json').read_text())
                        self.assertEqual([r['job_id'] for r in saved['batches'][:2]],['job-0','job-1'])
                        self.assertEqual(saved['runtime_batches'][0]['status'],'closed')
                        events.append(f'collect-{index}');return runtime_result(rows)
                    job.result.side_effect=collect;return job
                sampler.run.side_effect=submit;samplers.append(sampler);return sampler
            with patch.object(circuits,'compile_batch',side_effect=compile_record):
                result=run.execute(folder,max_jobs=2,execution_mode='batch',jobs_per_batch=2,batch_max_time=600,
                    backend=b,sampler_factory=make_sampler,batch_factory=factory)
            self.assertEqual(events,['open','submit-0','submit-1','close','collect-0','collect-1'])
            self.assertEqual([g['status'] for g in result['batches']],['collected','collected','not_submitted'])
            self.assertEqual(result['runtime_batches'][0]['job_indices'],[0,1])
            self.assertTrue(all(g['runtime_batch_id']=='runtime-group-0' for g in result['batches'][:2]))
            self.assertFalse(any((folder/'native'/f'batch_{i:05d}.qpy.gz').exists() for i in [0,1]))

    def test_batch_collection_recovers_all_ids_without_resubmitting(self):
        cfg=config();cfg['sampling']['trajectories']=4;b=backend();cfg['hardware']['backend']=b.name
        jobs=[]
        for index,batch in enumerate(model.batches(cfg)):
            job=Mock();job.job_id.return_value=f'saved-{index}';job.result.return_value=runtime_result(model.records(cfg,batch));jobs.append(job)
        jobs[0].result.side_effect=[RuntimeError('interrupted batch collection'),jobs[0].result.return_value]
        samplers=[]
        def make_sampler(**kwargs):
            sampler=Mock();sampler.run.return_value=jobs[len(samplers)];samplers.append(sampler);return sampler
        remote=Mock();remote.session_id='closed-runtime-group';batch_factory=Mock(return_value=remote)
        service=Mock();service.job.side_effect=lambda job_id:jobs[int(job_id.split('-')[-1])]
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()),patch.object(analysis,'analyze',return_value='finished'):
            folder=self.prepare(tmp,cfg)
            with self.assertRaisesRegex(RuntimeError,'interrupted batch collection'):
                run.execute(folder,execution_mode='batch',jobs_per_batch=2,backend=b,sampler_factory=make_sampler,batch_factory=batch_factory)
            saved=json.loads((folder/'manifest.json').read_text())
            self.assertEqual([x['job_id'] for x in saved['batches']],['saved-0','saved-1'])
            self.assertEqual(saved['runtime_batches'][0]['status'],'closed');remote.close.assert_called_once()
            forbidden=Mock(side_effect=AssertionError('duplicate Runtime submission'))
            result=run.execute(folder,execution_mode='batch',backend=b,service=service,sampler_factory=forbidden,batch_factory=forbidden)
            self.assertEqual(result,'finished');forbidden.assert_not_called()
            self.assertEqual(service.job.call_count,2)

    def test_batch_windows_include_final_short_group(self):
        cfg=config();cfg['sampling']['trajectories']=6;b=backend();cfg['hardware']['backend']=b.name
        remotes=[]
        for index in range(2):
            remote=Mock();remote.session_id=f'runtime-{index}';remotes.append(remote)
        batch_factory=Mock(side_effect=remotes);modes=[]
        def make_sampler(**kwargs):
            index=len(modes);modes.append(kwargs['mode']);job=Mock();job.job_id.return_value=f'job-{index}'
            rows=model.records(cfg,model.batches(cfg)[index])
            def collect():
                remotes[index//2].close.assert_called_once()
                return runtime_result(rows)
            job.result.side_effect=collect;sampler=Mock();sampler.run.return_value=job;return sampler
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()),patch.object(analysis,'analyze',return_value='finished'):
            folder=self.prepare(tmp,cfg)
            self.assertEqual(run.execute(folder,execution_mode='batch',jobs_per_batch=2,backend=b,
                sampler_factory=make_sampler,batch_factory=batch_factory),'finished')
            saved=json.loads((folder/'manifest.json').read_text())
            self.assertEqual([g['job_indices'] for g in saved['runtime_batches']],[[0,1],[2]])
            self.assertEqual([g['status'] for g in saved['runtime_batches']],['closed','closed'])
            self.assertEqual(modes,[remotes[0],remotes[0],remotes[1]])

    def test_batch_compilation_failure_happens_before_open_or_submission(self):
        cfg=config();cfg['sampling']['trajectories']=4;b=backend();cfg['hardware']['backend']=b.name
        compiler=circuits.compile_batch
        def fail_second(*args,**kwargs):
            if args[3]['index']==1:raise ValueError('second job compilation failed')
            return compiler(*args,**kwargs)
        forbidden=Mock(side_effect=AssertionError('should not access Runtime'))
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            folder=self.prepare(tmp,cfg)
            with patch.object(circuits,'compile_batch',side_effect=fail_second),self.assertRaisesRegex(ValueError,'compilation failed'):
                run.execute(folder,execution_mode='batch',jobs_per_batch=2,backend=b,sampler_factory=forbidden,batch_factory=forbidden)
            forbidden.assert_not_called()
            saved=json.loads((folder/'manifest.json').read_text())
            self.assertTrue(all(g['status']=='not_submitted' and g['job_id'] is None for g in saved['batches']))

    def test_ambiguous_second_batch_submission_preserves_first_id(self):
        cfg=config();cfg['sampling']['trajectories']=4;b=backend();cfg['hardware']['backend']=b.name
        first=Mock();first.job_id.return_value='first-saved'
        sampler1=Mock();sampler1.run.return_value=first
        sampler2=Mock();sampler2.run.side_effect=RuntimeError('uncertain second submission')
        remote=Mock();remote.session_id='partial-group'
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            folder=self.prepare(tmp,cfg)
            with self.assertRaisesRegex(RuntimeError,'uncertain second submission'):
                run.execute(folder,execution_mode='batch',jobs_per_batch=2,backend=b,
                    sampler_factory=Mock(side_effect=[sampler1,sampler2]),batch_factory=Mock(return_value=remote))
            remote.close.assert_called_once();first.result.assert_not_called()
            saved=json.loads((folder/'manifest.json').read_text())
            self.assertEqual(saved['batches'][0]['job_id'],'first-saved')
            self.assertEqual(saved['batches'][1]['status'],'submitting')
            with patch.object(run.original_hardware,'make_service',side_effect=AssertionError('no account')):
                with self.assertRaisesRegex(ValueError,'No automatic resubmission'):run.execute(folder)

    def test_interrupted_batch_close_is_recovered_without_replacing_jobs(self):
        cfg=config();cfg['sampling']['trajectories']=4;b=backend();cfg['hardware']['backend']=b.name
        jobs=[];samplers=[]
        for index,batch in enumerate(model.batches(cfg)):
            job=Mock();job.job_id.return_value=f'job-{index}';job.result.return_value=runtime_result(model.records(cfg,batch));jobs.append(job)
            sampler=Mock();sampler.run.return_value=job;samplers.append(sampler)
        remote=Mock();remote.session_id='unclosed-group';remote.close.side_effect=RuntimeError('close interrupted')
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()),patch.object(analysis,'analyze',return_value='finished'):
            folder=self.prepare(tmp,cfg)
            with self.assertRaisesRegex(RuntimeError,'close interrupted'):
                run.execute(folder,execution_mode='batch',jobs_per_batch=2,backend=b,
                    sampler_factory=Mock(side_effect=samplers),batch_factory=Mock(return_value=remote))
            self.assertTrue(all(job.result.call_count==0 for job in jobs))
            recovered=Mock();batch_factory=Mock();batch_factory.from_id.return_value=recovered
            service=Mock();service.job.side_effect=lambda job_id:jobs[int(job_id.split('-')[-1])]
            forbidden=Mock(side_effect=AssertionError('duplicate submission'))
            self.assertEqual(run.execute(folder,backend=b,service=service,sampler_factory=forbidden,batch_factory=batch_factory),'finished')
            batch_factory.assert_not_called();batch_factory.from_id.assert_called_once_with('unclosed-group',service=service)
            recovered.close.assert_called_once();forbidden.assert_not_called()
            saved=json.loads((folder/'manifest.json').read_text())
            self.assertEqual(saved['runtime_batches'][0]['status'],'closed')

    def test_legacy_run_resumes_after_cache_interrupt_without_duplicate_jobs(self):
        cfg=config();cfg['sampling']['trajectories']=6;b=backend();cfg['hardware']['backend']=b.name
        jobs=[];samplers=[]
        for index,batch in enumerate(model.batches(cfg)):
            job=Mock();job.job_id.return_value=f'original-{index}';job.result.return_value=runtime_result(model.records(cfg,batch));jobs.append(job)
            sampler=Mock();sampler.run.return_value=job;samplers.append(sampler)
        remote=Mock();remote.session_id='interrupted-original-group'
        loader=circuits.load_cached_qpy
        def interrupt_third(path):
            if Path(path).name=='batch_00002.qpy.gz':raise KeyboardInterrupt()
            return loader(path)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()),patch.object(analysis,'analyze',return_value='finished'):
            folder=self.prepare(tmp,cfg);source,manifest,_=model.load_run(folder)
            rules=model.store.read_sealed(model.DIRECTORY/'cache_fix_compatibility.json')['files']
            source.pop('sha256')
            for name,rule in rules.items():source['code'][name]=rule['previous_sha256']
            source=model.store.sealed(source);manifest['source_sha256']=source['sha256']
            model.store.atomic_json(folder/'source.json',source);model.store.atomic_json(folder/'manifest.json',manifest)
            original_source=(folder/'source.json').read_bytes();original_paths=model.store.file_digest(folder/'paths.npz')
            first_factory=Mock(side_effect=samplers[:2])
            with patch.object(circuits,'load_cached_qpy',side_effect=interrupt_third),self.assertRaises(KeyboardInterrupt):
                run.execute(folder,execution_mode='batch',jobs_per_batch=3,backend=b,sampler_factory=first_factory,batch_factory=Mock(return_value=remote))
            saved=json.loads((folder/'manifest.json').read_text())
            self.assertEqual([g['job_id'] for g in saved['batches']],['original-0','original-1',None])
            self.assertEqual(saved['batches'][2]['status'],'not_submitted')
            remote.close.assert_called_once();self.assertEqual(first_factory.call_count,2)
            service=Mock();service.job.side_effect=lambda job_id:jobs[int(job_id.split('-')[-1])]
            next_remote=Mock();next_remote.session_id='resumed-group';new_factory=Mock(return_value=samplers[2])
            with patch.object(circuits,'load_cached_qpy',wraps=loader) as reads:
                self.assertEqual(run.execute(folder,execution_mode='batch',jobs_per_batch=3,backend=b,service=service,
                    sampler_factory=new_factory,batch_factory=Mock(return_value=next_remote)),'finished')
                self.assertEqual(reads.call_count,1)  # Only the genuinely unsubmitted job is loaded.
            self.assertEqual(service.job.call_count,2);new_factory.assert_called_once()
            saved=json.loads((folder/'manifest.json').read_text())
            self.assertEqual([g['job_id'] for g in saved['batches']],['original-0','original-1','original-2'])
            self.assertEqual(len(saved['execution_updates']),1)
            self.assertEqual((folder/'source.json').read_bytes(),original_source)
            self.assertEqual(model.store.file_digest(folder/'paths.npz'),original_paths)
            for batch in saved['batches']:
                compiled=model.store.read_sealed(folder/batch['records_file'])
                self.assertEqual(compiled['source_sha256'],source['sha256'])


if __name__=='__main__':unittest.main()
